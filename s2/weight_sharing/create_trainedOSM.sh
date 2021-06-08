#!/bin/bash
# bash ./s2/weight_sharing/create_trained_OSM1_nodiscrete.sh dataset gpu outputs epochs data_path model_name

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir=$3
batch_size=256
valid_batch_size=1024
epochs=$4
data_path=$5
model_name=$6

config_path="configs/CMAES-NAS.config"

python ./s2/weight_sharing/create_trainedOSM.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                               --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} \
                               --config_path ${config_path} --batch_size ${batch_size} --model_name ${model_name} \
                               --valid_batch_size ${valid_batch_size} --track_running_stats
