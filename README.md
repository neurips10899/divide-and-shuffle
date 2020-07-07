## Divide-and-Shuffle Synchronization for Distributed Machine Learning
This is the anonymous code repository for NeurIPS 2020 submition 10899 Divide-and-Shuffle Synchronization for Distributed Machine Learning
### Introduction

We use CIFAR10/100, ImageNet, SQuADv1.1 and Criteo, 4 famous data sets, which stands for applications in Computer Vision, Natural Language Processing, and Recommendation System. We follow the standard pre-process and data split. Specifically, CIFAR10/100 has 50000 training images and 10000 test images. ImageNet has 1.28 million training images and 50k validation images to evaluate. SQuAD v1.1 is a collection of over 100,000 question-answer pairs on over 500 articles. Criteo data set includes 45840617 users’ click records. For both SQuAD v1.1 and Criteo data sets, we split it into two parts: 90% is for training, while the rest 10% is for testing.

We use PyTorch to develop our DS-Sync and all baselines, including BSP and ASP. NCCL library is the default communication method for DS-Sync and BSP, while ASP has to use the send and receive functions in Gloo to realize the pull-push operation. We train CIFAR10/100 on WRN-26-10 and ImageNet with WRN-50-2 by SGD momentum and share the same hyperparameters of initial learning rate(LR): 0.1, momentum: 0.9, and weight decay: 0.0001 for convenience. In Cifar10/100, LR decrease by 0.2 in epochs 60, 120, 160. Cifar10 sets 256 as the batch size per GPU, while Cifar100 uses 128. But in ImageNet, LR decrease by 0.1 in epochs 30 and 60, and batch size per GPU is 30. We fine-tuned the SQuADv1.1 dataset on the pre-trained BERT-Large model(bert-large-uncased-whole-word-masking). We set the batch size per GPU as 6 to fulfill GPU memory. AdamW optimizer is used with LR: 3e-5, epsilon equals: 1e-8, weight decay: 0, and betas: 0.9 and 0.999. We trained a DeepFM model on Criteo data with Adam. We set batch size per GPU as 20480. Other hyperparameters includes LR: 1e-3, weight decay: 1e-5 and betas: 0.9 and 0.999. If not specified, all experiments are repeated by five times for 4-node experiments and three times for 16-node experiments except that ImageNet only runs for once.

### Training and evaluation

To run and repo the experiment, first, you should put the data into the right file path. We already contruct the data directory in the source code directory.

Then for each model and dataset, we have a start script for running the model and experiments. The related python packages are in the requirements.txt file. After install the required packages. For example, if we wants to run the DeepFM model. The start.sh file will be like the following.

~~~sh
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
~~~

Noted that for each rank of the worker, we should change the node rank in the scripts. Besides the environment parameters are also need to be modified according to the experiment environment.

Then the results will be stored in the result directory or in the tensorboard file.

### Results: Converged Accuracy Comparision in 4 nodes

+ Cifar10

  | Sync method | Accuracy           | LogLoss             |
  | ----------- | ------------------ | ------------------- |
  | BSP         | 90.00%(±1.69%)     | 0.4655(±0.0881)     |
  | DS-Sync     | **91.63%(±0.76%)** | **0.4082(±0.0424)** |

+ Cifar100

  | Sync method | Accuracy           | LogLoss             |
  | ----------- | ------------------ | ------------------- |
  | BSP         | **75.86%(±0.28%)** | 1.1406(±0.0148)     |
  | DS-Sync     | 75.78%(±0.39%)     | **1.1357(±0.0200)** |

+ ImageNet

  | Sync method | Accuracy                    | LogLoss    |
  | ----------- | --------------------------- | ---------- |
  | BSP         | top1:73.10% top5:91.34%     | 1.1743     |
  | DS-Sync     | **top1:73.25% top5:91.35%** | **1.1574** |

+ SQuAD

  | Sync method | F1 Score           | Exact Match        |
  | ----------- | ------------------ | ------------------ |
  | BSP         | **93.06%(±0.22%)** | **86.85%(±0.53%**) |
  | DS-Sync     | 93.01%(±0.26%)     | 86.83%((±0.56%)    |

+ Criteo

  | Sync method | AUC                | LogLoss             |
  | ----------- | ------------------ | ------------------- |
  | BSP         | **80.51%(±0.03%)** | **0.4469(±0.0005)** |
  | DS-Sync     | 80.50%(±0.02%)     | **0.4469(±0.0004)** |

  
