CUDA_VISIBLE_DEVICES=0,1,2,4 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 linear_probing_IN1K_ddp.py --global_batch_size=2048 --lr=1e-4
