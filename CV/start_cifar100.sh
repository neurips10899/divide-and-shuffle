export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens3
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="8001"
export CUDA_VISIBLE_DEVICES='0'
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch \
--nproc_per_node=4 \
--nnodes=1 \
--node_rank=0 \
--master_addr=${MASTER_ADDR} \
--master_port=${MASTER_PORT} \
./cifar_code/cifar100.py \
--epoch 200 \
--batch-size 32 \
--lr 0.1 \
--momentum 0.9 \
--wd 5e-4 \
-ct 100 \
--sync ds_sync \
--world_size 4  2>&1 | tee log_cifar100_rsp
