export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens3
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8000"
export CUDA_VISIBLE_DEVICES='0'
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch \
--nproc_per_node=1 \
--nnodes=4 \
--node_rank=0 \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
./cifar_code/cifar10.py \
--epoch 200 \
--batch-size 32 \
--lr 0.1 \
--momentum 0.9 \
--wd 5e-4 \
-ct 10 \
--sync ds_sync \
--world_size 4  2>&1 | tee log_cifar10_rsp
