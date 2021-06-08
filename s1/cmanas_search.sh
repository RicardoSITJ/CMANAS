#!/bin/bash
# bash ./s1/cmanas_search.sh cifar10 gpu output_dir pre_trained_model epochs alt_C100_split

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3
tr_model_dir="$output_dir"
batch_size=64
valid_batch_size=2048
layers=8
n_trained_models=1
pre_tr_model=$4
epochs=$5
alt_cifar100_split=$6

data_path="../../../data"
config_path="configs/CMAES-NAS.config"

if [ -z $6 ]
then
  python ./s1/cmanas_search.py --gpu ${gpu} --dataset ${dataset} --batch_size ${batch_size} --config_path ${config_path} \
                         --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} --tr_model_dir ${tr_model_dir} \
                         --init_channel ${channel} --layers ${layers} --valid_batch_size ${valid_batch_size} --cutout  \
                         --n_trained_models ${n_trained_models} --pre_tr_model ${pre_tr_model}
else
  # For CIFAR100 with 80-20 split
  echo ${alt_cifar100_split}
  python ./s1/cmanas_search.py --gpu ${gpu} --dataset ${dataset} --batch_size ${batch_size} --config_path ${config_path} \
                         --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} --tr_model_dir ${tr_model_dir} \
                         --init_channel ${channel} --layers ${layers} --valid_batch_size ${valid_batch_size} --cutout  \
                         --n_trained_models ${n_trained_models} --pre_tr_model ${pre_tr_model} --alt_cifar100_split ${alt_cifar100_split}
fi
