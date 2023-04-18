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

class Head(torch.nn.Module):
    def __init__(self, dim, num_classes) -> None:
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.norm_layer = torch.nn.BatchNorm1d(self.dim, affine=False, eps=1e-6)
        # self.norm_layer = torch.nn.LayerNorm(self.dim)
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

    def forward_return_n_last_blocks(self, x, t=None, n_last_blocks=1, return_patch_avgpool=False, depths=[]):
        # VAE latent
        x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)   # (N, 4, latent_size, latent_size)

        # conditioning (only timestep)
        if t is None:
            t = torch.tensor([self.t]*x.shape[0]).to(x.device)
        else:
            t = torch.tensor([t]*x.shape[0]).to(x.device)
        t = self.dit.t_embedder(t)                          # (N, hidden_size)
        y = torch.tensor([1000]*x.shape[0]).to(x.device)
        y = self.dit.y_embedder(y, self.training)           # (N, D)
        c = t + y
        # DiT latent
        output = []
        x = self.dit.x_embedder(x) + self.dit.pos_embed     # (N, T, hidden_size)
        for i, block in enumerate(self.dit.blocks):
            x = block(x, c) 
            if len(self.dit.blocks) - i <= n_last_blocks:
                # output.append(self.norm_layer(x)[:,0])
                # output.append(self.norm_layer(x).mean(dim=1))
                output.append(x.mean(dim=1))

        if return_patch_avgpool:
            x = self.norm_layer(x)
            output.append(torch.mean(x[:, 1:], dim=1))

        output = torch.cat(output, dim=-1)
        return output

class DiffRep(torch.nn.Module):

    def __init__(self, vae:torch.nn.Module, dit:torch.nn.Module, 
                 hidden_size:int=1152, t:int=0, n_last_blocks:int=1, 
                 return_patch_avgpool:bool=True, num_classes:int=1000) -> None:
        super().__init__()

        self.t = t
        self.n_last_blocks = n_last_blocks
        self.return_patch_avgpool = return_patch_avgpool
        self.latent = Latent(vae, dit, hidden_size, t)
        self.head = Head(hidden_size * (n_last_blocks + int(return_patch_avgpool)), num_classes)
        print(hidden_size * (n_last_blocks + int(return_patch_avgpool)))
        # freeze the params of latent extractor
        for param in self.latent.parameters():
            param.requires_grad = False
        
    def forward(self, x):

        with torch.no_grad():
            x = self.latent.forward_return_n_last_blocks(x, self.t, self.n_last_blocks, self.return_patch_avgpool)
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
    batch_size = target.size(0)
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
    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # create model
    if args.ckpt is None:
        assert args.dit_model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    
    latent_size = int(args.image_size) // 8

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.dit_model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Experiment directory created at {experiment_dir}")

    # Load pretrained model:
    DiT_model = DiT_models[args.dit_model](input_size=latent_size, num_classes=args.num_classes)
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    DiT_model.load_state_dict(state_dict)
    VAE_model = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    hidden_size = 1152
    n_last_blocks = args.n_last_blocks
    return_patch_avgpool = False
    Feature_Extractor = Latent(vae=VAE_model, dit=DiT_model)
    Linear_Classifier = Head(dim=hidden_size * (n_last_blocks + int(return_patch_avgpool)), num_classes=1000)
    Feature_Extractor.to(device)
    Feature_Extractor.eval()
    Linear_Classifier.to(device)

    # model = DiffRep(vae=VAE_model, dit=DiT_model, t=0, n_last_blocks=4, return_patch_avgpool=False, num_classes=args.num_classes).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(Linear_Classifier.parameters(), lr=args.lr, weight_decay=0)

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
    dataset_train = ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)

    loader_train = DataLoader(
                    dataset_train,
                    batch_size=args.global_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True
                    )
    loader_val = DataLoader(
                    dataset_val,
                    batch_size=args.global_batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=False
                    )

    for epoch in range(args.epochs):

        print(f"Beginning epoch {epoch} training...")

        Linear_Classifier.train()

        train_steps = 0
        log_steps = 0
        training_loss = 0

        train_total = 0
        train_correct1, train_correct5 = 0, 0

        start_time = time()

        for x, y in loader_train:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = Feature_Extractor.forward_return_n_last_blocks(x,t=0,n_last_blocks=n_last_blocks,return_patch_avgpool=return_patch_avgpool)
            y_pred = Linear_Classifier(x)
            loss = criterion(y_pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            train_correct1_batch, train_correct5_batch = correct(y_pred, y, topk=(1, 5))

            training_loss += loss.item()
            train_correct1 += train_correct1_batch
            train_correct5 += train_correct5_batch
            train_total += x.shape[0]

            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:

                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                # Reduce loss history over all processes:
                avg_loss = training_loss / log_steps
                avg_acc1 = train_correct1 / train_total * 100
                avg_acc5 = train_correct5 / train_total * 100

                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, \
                        Train Acc1 : {avg_acc1:.4f}%, Train Acc5 : {avg_acc5:.4f}%, \
                        Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                training_loss = 0
                log_steps = 0
                train_total = 0
                train_correct1, train_correct5 = 0, 0
                start_time = time()

        checkpoint = {
            "head_model": Linear_Classifier.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        Linear_Classifier.eval()
        val_total = 0
        val_correct1, val_correct5 = 0, 0

        print(f"Beginning epoch {epoch} validation...")
        for x, y in loader_val:
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                x = Feature_Extractor.forward_return_n_last_blocks(x,t=0,n_last_blocks=n_last_blocks,return_patch_avgpool=return_patch_avgpool)
                y_pred = Linear_Classifier(x)
                val_correct1_batch, val_correct5_batch = correct(y_pred, y, topk=(1, 5))
                val_correct1 += val_correct1_batch
                val_correct5 += val_correct5_batch
                val_total += x.shape[0]
        val_acc1 = val_correct1 / val_total * 100
        val_acc5 = val_correct5 / val_total * 100

        print(f"(epoch={epoch:07d}) Val Acc1: {val_acc1:.4f}%, Val Acc5: {val_acc5:.4f}%")
    
    print("Training Finished.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/home/data/ILSVRC2012')
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--dit_model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--ckpt", type=str, help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--n_last_blocks", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    main(args)