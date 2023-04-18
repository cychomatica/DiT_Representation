CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 linear_probing_IN1K_ddp.py --global_batch_size=192 --lr=1e-4
