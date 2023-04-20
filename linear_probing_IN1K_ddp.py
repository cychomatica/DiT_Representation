import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision import transforms
from torchvision.utils import save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from timm.utils import accuracy
import numpy as np
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from PIL import Image
from IPython.display import display

import argparse, os, logging
from glob import glob
from time import time

from lmdb_utils import imagenet_lmdb_dataset

class Head(torch.nn.Module):
    def __init__(self, dim, num_classes) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.norm_layer = torch.nn.BatchNorm1d(self.dim, affine=False, eps=1e-6)
        self.fc = torch.nn.Linear(self.dim, self.num_classes)

    def forward(self, x):
        x = self.norm_layer(x)
        x = self.fc(x)
        return x

class Latent(torch.nn.Module):

    def __init__(self, vae:torch.nn.Module, dit:torch.nn.Module, 
                 hidden_size=1152, t:int=0) -> None:
        super().__init__()
        self.vae = vae
        self.dit = dit
        self.norm_layer = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t = t
    
    def forward(self, x, t=None):
        '''
        x: (N, 3, 256, 256)
        image_size = 256
        latent_size = image_size // 8 = 32
        in_channels = 4
        patch_size = 2
        hidden_size = 1152
        out_channels = in_channels * 2 if learn_sigma else in_channels
        T = image_size**2 / patch_size**2 = 256
        '''

        # VAE latent
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)   # (N, 4, latent_size, latent_size)

        # conditioning (only timestep)
        if t is None:
            t = torch.tensor([self.t]*x.shape[0]).to(x.device)
        else:
            t = torch.tensor([t]*x.shape[0]).to(x.device)
        t = self.dit.t_embedder(t)                          # (N, hidden_size)
        c = t

        # DiT latent
        x = self.dit.x_embedder(x) + self.dit.pos_embed     # (N, T, hidden_size)
        for block in self.dit.blocks:
            x = block(x, c)                                 # (N, T, hidden_size)
        
        # avg pooling
        x = x.mean(dim=1)
        # x = self.dit.final_layer(x, c)                      # (N, T, patch_size ** 2 * out_channels)
        # x = self.dit.unpatchify(x)                          # (N, out_channels, latent_size, latent_size)

        return x

    def forward_return_n_last_blocks(self, x, t=None, representation_layers=[], return_patch_avgpool=False):

        # VAE latent
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)   # (N, 4, latent_size, latent_size)

        # conditioning (only timestep)
        if t is None:
            t = torch.tensor([self.t]*x.shape[0]).to(x.device)
        else:
            t = torch.tensor([t]*x.shape[0]).to(x.device)
        t = self.dit.t_embedder(t)                          # (N, hidden_size)
        y = torch.tensor([1000]*x.shape[0]).to(x.device)    # equivalent to cfg label dropping
        y = self.dit.y_embedder(y, self.training)           # (N, D)
        c = t + y

        # DiT latent
        if len(representation_layers) == 0:
            representation_layers = [len(self.dit.blocks)]  # if representation_layers empty, use the representation from the last layer by default
        output = []
        x = self.dit.x_embedder(x) + self.dit.pos_embed     # (N, T, hidden_size)
        for i, block in enumerate(self.dit.blocks):
            x = block(x, c) 
            if i+1 in representation_layers:
                output.append(x.mean(dim=1))

        # not used
        # if return_patch_avgpool:
        #     x = self.norm_layer(x)
        #     output.append(torch.mean(x[:, 1:], dim=1))

        output = torch.cat(output, dim=-1)
        return output


class DiffRep(torch.nn.Module):

    def __init__(self, 
                 vae:torch.nn.Module, 
                 dit:torch.nn.Module, 
                 hidden_size:int=1152, 
                 t:int=0, 
                 representation_layers=[], 
                 return_patch_avgpool:bool=False, 
                 num_classes:int=1000) -> None:
        super().__init__()

        self.t = t
        self.representation_layers = representation_layers
        self.return_patch_avgpool = return_patch_avgpool
        self.latent = Latent(vae, dit, hidden_size, t)

        head_dim_in = hidden_size * (len(representation_layers) + int(return_patch_avgpool))
        self.head = Head(head_dim_in, num_classes)
        # print(hidden_size * (n_last_blocks + int(return_patch_avgpool)))

    def forward(self, x):

        with torch.no_grad():
            x = self.latent.forward_return_n_last_blocks(x, self.t, self.representation_layers, self.return_patch_avgpool)
        x = self.head(x)
        return x
    
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) for k in topk]

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    # setip device
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # region
    # setip DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.dit_model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    # endregion

    # create model
    if args.ckpt is None:
        assert args.dit_model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    
    latent_size = int(args.image_size) // 8
    representation_layers = args.representation_layers if len(args.representation_layers) > 0 else np.arange(len(DiT_model.blocks) - args.shift_from_last - args.n_blocks + 1, len(DiT_model.blocks) - args.shift_from_last + 1).tolist()

    # Load pretrained model:
    DiT_model = DiT_models[args.dit_model](input_size=latent_size, num_classes=args.num_classes)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    DiT_model.load_state_dict(state_dict)
    VAE_model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    model = DiffRep(vae=VAE_model, 
                    dit=DiT_model, 
                    t=0,
                    representation_layers=representation_layers,
                    return_patch_avgpool=False, 
                    num_classes=args.num_classes
                    ).to(device)
    model = DDP(model, device_ids=[rank])

    logger.info('Representations from DiT block: {}'.format(representation_layers))
    logger.info('Linear head dim: {}'.format([model.module.head.dim, model.module.head.num_classes]))
    # return 0
    criterion = torch.nn.CrossEntropyLoss()

    assert args.training in ['linear_probe', 'fine_tune']
    logger.info(f"Training mode: {args.training}")

    if args.training == 'linear_probe':
        requires_grad(model.module.latent, False)
        requires_grad(model.module.head, True)
        opt = torch.optim.AdamW(model.module.head.parameters(), lr=args.lr, weight_decay=0)
    elif args.training == 'fine_tune':
        requires_grad(model.module.latent, True)
        requires_grad(model.module.head, True)
        opt = torch.optim.AdamW(model.module.parameters(), lr=args.lr, weight_decay=0)
    # opt = torch.optim.SGD(model.module.head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    transform_train = transforms.Compose([
                                        transforms.RandomResizedCrop(256, scale=(0.2, 1.0), interpolation=3),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                                        ])
    transform_val = transforms.Compose([
                                        transforms.Resize(292, interpolation=3),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                                        ])
    dataset_train = imagenet_lmdb_dataset(root=os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = imagenet_lmdb_dataset(root=os.path.join(args.data_path, 'val'), transform=transform_val)

    sampler_train = DistributedSampler(
                    dataset_train,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=True,
                    seed=args.global_seed
                    )
    sampler_val = DistributedSampler(
                    dataset_val,
                    num_replicas=dist.get_world_size(),
                    rank=rank,
                    shuffle=True,
                    seed=args.global_seed
                    )
    loader_train = DataLoader(
                    dataset_train,
                    batch_size=int(args.global_batch_size // dist.get_world_size()),
                    shuffle=False,
                    sampler=sampler_train,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True
                    )
    loader_val = DataLoader(
                    dataset_val,
                    batch_size=int(args.global_batch_size // dist.get_world_size()),
                    shuffle=False,
                    sampler=sampler_val,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False
                    )
    print(loader_train.batch_size, loader_val.batch_size)
    logger.info(f"Training set contains {len(dataset_train):,} images")
    logger.info(f"Validation set contains {len(dataset_val):,} images")

    logger.info(f"Training for {args.epochs} epochs...")

    for epoch in range(args.epochs):

        loader_train.sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch} training...")

        if args.training == 'linear_probe':
            model.module.latent.eval()
            model.module.head.train()
        elif args.training == 'fine_tune':
            model.module.train()

        train_steps = 0
        log_steps = 0
        training_loss = 0

        train_total = 0
        train_correct1, train_correct5 = 0, 0

        start_time = time()

        for x, y in loader_train:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_correct1_batch, train_correct5_batch = correct(y_pred, y, topk=(1, 5))

            training_loss += loss.item()
            train_correct1 += train_correct1_batch.item()
            train_correct5 += train_correct5_batch.item()
            train_total += x.shape[0]

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(training_loss / log_steps, device=device)
                avg_acc1 = torch.tensor(train_correct1 / train_total * 100, device=device)
                avg_acc5 = torch.tensor(train_correct5 / train_total * 100, device=device)

                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc1, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc5, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_acc1 = avg_acc1.item() / dist.get_world_size()
                avg_acc5 = avg_acc5.item() / dist.get_world_size()

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, \
                            Train Acc1 : {avg_acc1:.4f}%, Train Acc5 : {avg_acc5:.4f}%, \
                            Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                training_loss = 0
                log_steps = 0
                train_total = 0
                train_correct1, train_correct5 = 0, 0
                start_time = time()
        
        # Save linear-probing head checkpoint:
        if rank == 0:
            if args.training == 'linear_probe':
                checkpoint = {
                    "head_model": model.module.head.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
            elif args.training == 'fine_tune':
                checkpoint = {
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
            checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()

        # validation

        model.module.eval()
        val_total = 0
        val_correct1, val_correct5 = 0, 0

        logger.info(f"Beginning epoch {epoch} validation...")
        for x, y in loader_val:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            val_correct1_batch, val_correct5_batch = correct(y_pred, y, topk=(1, 5))
            val_correct1 += val_correct1_batch.item()
            val_correct5 += val_correct5_batch.item()
            val_total += x.shape[0]
        val_acc1 = torch.tensor(val_correct1 / val_total * 100, device=device)
        val_acc5 = torch.tensor(val_correct5 / val_total * 100, device=device)
        dist.all_reduce(val_acc1, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc5, op=dist.ReduceOp.SUM)
        val_acc1 = val_acc1.item() / dist.get_world_size()
        val_acc5 = val_acc5.item() / dist.get_world_size()
        logger.info(f"(epoch={epoch}) Val Acc1: {val_acc1:.4f}%, Val Acc5: {val_acc5:.4f}%")
    
    logger.info("Training Finished.")
    cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/home/data/ILSVRC2012')
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dit_model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--n_blocks", type=int, default=10)
    parser.add_argument("--shift_from_last", type=int, default=0)
    parser.add_argument("--representation_layers", type=int, nargs='+', default=[])
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--training", type=str, choices=['linear_probe', 'fine_tune'], default='linear_probe')
    parser.add_argument("--lr", type=float, default=1e-4),
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    main(args)