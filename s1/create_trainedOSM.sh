#!/bin/bash
# bash ./s1/create_trainedOSM.sh cifar10 gpu outputs epochs model_name alt_split

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3
batch_size=96
valid_batch_size=1024
layers=8
epochs=$4
model_name=$5
alt_cifar100_split=$6

data_path="../../../data"
config_path="configs/CMAES-NAS.config"

if [ -z "$6" ]
then
  python ./s1/create_trainedOSM.py --gpu ${gpu} --dataset ${dataset} --batch_size ${batch_size} --config_path ${config_path} \
                     --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} --model_name ${model_name} \
                     --init_channel ${channel} --layers ${layers} --valid_batch_size ${valid_batch_size} --cutout
else
  # with alternate cifar100 split for training
  python ./s1/create_trainedOSM.py --gpu ${gpu} --dataset ${dataset} --batch_size ${batch_size} --config_path ${config_path} \
                     --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} --model_name ${model_name} \
                     --init_channel ${channel} --layers ${layers} --valid_batch_size ${valid_batch_size} --cutout \
                     --alt_cifar100_split ${alt_cifar100_split}
fi
