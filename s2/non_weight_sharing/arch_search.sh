#!/bin/bash
# bash ./s2/non_weight_sharing/arch_search.sh dataset output_directory epochs ntrials
echo script name: $0
echo $# arguments

dataset=$1
channel=16
num_cells=5
max_nodes=4
output_dir=$2
epochs=$3
ntrials=$4

if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
  data_path="../../../data"
else
  data_path="../../../data/ImageNet16"
fi

if [ "$dataset" == "cifar10" ];
then
  csv_path="${output_dir}/non-weight-sharing-cifar10-${epochs}epochs"
elif [ "$dataset" == "cifar100" ];
then
  csv_path="${output_dir}/non-weight-sharing-cifar100-${epochs}epochs"
else
  csv_path="${output_dir}/non-weight-sharing-ImageNet16-120-${epochs}epochs"
fi

api_path="../../NAS-Bench-201-v1_1-096897.pth"
config_path="configs/CMAES-NAS.config"

echo csv_path

python ./s2/non_weight_sharing/cmanas_search.py --max_nodes ${max_nodes} --init_channel ${channel} --num_cells ${num_cells} \
                               --dataset ${dataset} --data_dir ${data_path} --output_dir ${output_dir} --epochs ${epochs} \
                               --api_path ${api_path} --config_path ${config_path} --ntrials ${ntrials} \
                               --track_running_stats --record_filename ${csv_path}

python ./s2/non_weight_sharing/cal_stats.py --csv_file "${csv_path}.csv"
