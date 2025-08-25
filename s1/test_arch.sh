# bash ./s1/test_arch.sh dataset gpu data_path batch_size genotype_dir model_path log_path

dataset=$1
gpu=$2
data_path=$3
batch_size=$4
genotype_dir=$5
model_path=$6
log_path=$7

if [ "$dataset" == "cifar10" ]; then
  python ./s1/test_cifar10.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "cifar100" ]; then
  python ./s1/test_cifar100.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "jaffe7" ]; then
  python ./s1/test_jaffe7.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "ckplus_2" ]; then
  python ./s1/test_ckplus_2.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "ckplus_3" ]; then
  python ./s1/test_ckplus_3.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "ckplus_7" ]; then
  python ./s1/test_ckplus_7.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
elif [ "$dataset" == "kdef" ]; then
  python ./s1/test_kdef.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100 --log_path ${log_path} --gpu ${gpu}
else
  python ./s1/test_imagenet.py --batch_size ${batch_size} --data ${data_path} --dir ${genotype_dir} --auxiliary --model_path ${model_path} --report_freq 100
fi
