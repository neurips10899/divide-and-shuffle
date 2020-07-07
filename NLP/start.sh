set -x
export NCCL_SOCKET_IFNAME=ens3
export MASTER_ADDR="172.31.32.162"
export MASTER_PORT="8000"
CUDA_VISIBLE_DEVICES="0"
export NCCL_IB_DISABLE=1
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=4 --node_rank=0 --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ./code/run_squad.py \
    --model_type bert \
    --overwrite_output_dir \
    --evaluate_during_training \
    --model_name_or_path bert-large-uncased-whole-word-masking \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_file ./SQuAD/train-v1.1.json \
    --predict_file ./SQuAD/dev-v1.1.json \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./models/wwm_uncased_finetuned_squad/ \
    --per_gpu_eval_batch_size=64 \
    --per_gpu_train_batch_size=6   \
    --sync ds_sync \
    --logging_steps 200 \
    --save_steps 2000
