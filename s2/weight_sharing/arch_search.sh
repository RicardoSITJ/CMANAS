#!/bin/bash
# bash ./s2/weight_sharing/arch_search.sh dataset gpu output_dir_idx model_name

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
channel=16
num_cells=5
max_nodes=4
output_dir="./outputs/s2/exp-${dataset}-weight-sharing-search-$3"
batch_size=256
valid_batch_size=1024
epochs=2
model_name="$4.pt"

if [ -d "${output_dir}" ]; then
  echo "Change the directory name for the experiment as it already exist"
  exit 0
else
  echo ${output_dir}
fi

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="../../../data"
else
  data_path="../../../data/ImageNet16"
fi
config_path="configs/CMAES-NAS.config"
api_path="../../NAS-Bench-201-v1_1-096897.pth"

# First create a trained One Shot Model
bash ./s2/weight_sharing/create_trainedOSM.sh ${dataset} ${gpu} ${output_dir} ${epochs} ${data_path} ${model_name}

# Architecture search using CMANAS algorithm
pr_trained_model="${output_dir}/${model_name}"
echo ${pr_trained_model}
python ./s2/weight_sharing/cmanas_search.py --gpu ${gpu} --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                             --api_path ${api_path} --pr_trained_model ${pr_trained_model} \
                             --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} \
                             --config_path ${config_path} --batch_size ${batch_size} --n_trained_models 1 \
                             --valid_batch_size ${valid_batch_size} --track_running_stats
