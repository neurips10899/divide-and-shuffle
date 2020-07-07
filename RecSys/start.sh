export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens3
export MASTER_ADDR="172.31.32.162"
export MASTER_PORT="8000"
export CUDA_VISIBLE_DEVICES='0'
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=0 ./code/deepfm.py \
	--num_epochs 20 \
	--data_scale full \
	--learning_rate 1e-3 \
	--weight_decay 1e-5 \
	--dropout 0.5 \
	--sync ds_sync \
	--batch_size 81920