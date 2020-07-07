export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens3
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8000"
export CUDA_VISIBLE_DEVICES='0,1,2,3'
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch \
--nproc_per_node=1 \
--nnodes=4 \
--node_rank=0 \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
./imagenet_code/imagenet.py \
	--dataset IMAGENET \
	--data_path /homes/wwangbc/wwangbc/data/Imagenet/ILSVRC/Data/CLS-LOC  \
	--world_size 4 \
	--num_epochs 100 \
	--learning_rate 1e-1 \
	--weight_decay 5e-4 \
	--batch_size 25 \
    --sync ds_sync \
	--base_batch_size 128 2>&1 | tee log_imagenet_rsp
