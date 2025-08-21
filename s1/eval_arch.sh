#!/bin/bash
# bash ./s1/eval_arch.sh cifar10 gpu arch_dir

echo script name: $0
echo $# arguments

dataset=$1
gpu=$2
arch_dir=$3
# epochs=600
epochs=600

data_path="../../../data"

if [ "$dataset" == "cifar10" ] ; then
  python ./s1/train_cifar10.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu}
elif [ "$dataset" == "jaffe7" ] ; then
  python ./s1/train_jaffe7.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu}
elif [ "$dataset" == "ckplus" ] ; then
  python ./s1/train_ckplus.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu}
elif [ "$dataset" == "kdef" ] ; then
  python ./s1/train_kdef.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu}
else
  python ./s1/train_cifar100.py --cutout --auxiliary --epochs ${epochs} --dir ${arch_dir} --data ${data_path} --gpu ${gpu}
fi
