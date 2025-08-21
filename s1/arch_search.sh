#!/bin/bash
# bash ./s1/arch_search.sh dataset gpu output_dir model_name

dataset=$1
gpu=$2
output_dir="./outputs/s1/exp-${dataset}-search-$3"
model_name="$4.pt"
epochs=100
# cmanas_epochs=100
cmanas_epochs=100
C100_split=2

if [ -d "${output_dir}" ]; then
  echo "Change the directory name for the experiment as it already exist"
  exit 0
else
  echo ${output_dir}
fi

# First create a trained One Shot Model
bash ./s1/create_trainedOSM.sh ${dataset} ${gpu} ${output_dir} ${epochs} ${model_name} ${C100_split}

pre_trained_model="${output_dir}/${model_name}"
# Architecture search using CMANAS algorithm
bash ./s1/cmanas_search.sh ${dataset} ${gpu} ${output_dir} ${pre_trained_model} ${cmanas_epochs} ${C100_split} 

# Evaluate the searched architecture
bash ./s1/eval_arch.sh ${dataset} ${gpu} ${output_dir}
