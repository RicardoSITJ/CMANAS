import sys
sys.path.insert(0,'./')

import argparse
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import torch

from cma_es                  import CMAES
from architecture            import Architecture
from cell_operations         import NAS_BENCH_201
from nas_201_api             import NASBench201API as API
from procedures              import prepare_seed
from search_cells            import NAS201SearchCell as SearchCell
from search_model_cmaes_nas  import TinyNetworkCMAES_NAS as CMAES_NAS
from utils                   import create_exp_dir

parser = argparse.ArgumentParser("CMAES search with NAS201 api")
parser.add_argument('--api_path', type = str, default = None, help = 'path to the NAS201 api')
parser.add_argument('--config_path', type=str, help='The config path.')
parser.add_argument('--data_dir', type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--epochs', type = int, default = 100, help = 'num of architecture search epochs')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')
parser.add_argument('--max_nodes', type = int, default = 4, help = 'maximim nodes in the cell for NAS201 network')
parser.add_argument('--num_cells', type = int, default = 5, help = 'number of cells for NAS201 network')
parser.add_argument('--ntrials', type = int, default = 500, help = 'number of trials')
parser.add_argument('--output_dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--pop_size', type = int, default = None, help = 'population size')
parser.add_argument('--record_filename', type = str, default = None, help = 'filename of the csv file for recording the final results')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--seed', type = int, default=-1, help = 'random seed')
parser.add_argument('--track_running_stats', action = 'store_true', default = False, help = 'use track_running_stats in BN layer')
args = parser.parse_args()

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
assert args.api_path is not None, 'NAS201 data path has not been provided'
args.edges = 6

if args.output_dir is not None:
  if not os.path.exists(args.output_dir):
    create_exp_dir(args.output_dir)
# Configuring the logger
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.output_dir, 'non-weight-sharing-{}-log.txt'.format(args.dataset)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def get_arch_score(api, arch_str, dataset, acc_type=None, use_012_epoch_training=False):
  arch_index = api.query_index_by_arch( arch_str )
  assert arch_index >= 0, 'can not find this arch : {:}'.format(arch_str)
  if use_012_epoch_training:
    info = api.get_more_info(arch_index, dataset, iepoch=None, hp='12', is_random=True)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    return valid_acc, time_cost
  else:
    return api.query_by_index(arch_index=arch_index, hp = '200').get_metrics(dataset, acc_type)['accuracy']

def main(args, api):
  # Configuring the logger
  log_format = '%(asctime)s %(message)s'

  logging.info(f'[INFO] torch version {torch.__version__}')
  logging.info(f'[INFO] {args}')

  # Reproducibility
  prepare_seed(args.seed)

  logging.info(f'length of api: {len(api)}')

  # Configuring dataset and dataloader
  if args.dataset == 'cifar10':
    acc_type     = 'ori-test'
    val_acc_type = 'x-valid'
  else:
    acc_type     = 'x-test'
    val_acc_type = 'x-valid'
  
  if args.dataset == 'cifar10': num_classes = 10
  elif args.dataset == 'cifar100': num_classes = 100
  else: num_classes = 120
  logging.info("#classes: {}".format(num_classes))

  # Model Initialization
  model = CMAES_NAS(C = args.init_channels, N = args.num_cells, max_nodes = args.max_nodes,
                      num_classes = num_classes, search_space = NAS_BENCH_201, affine = False,
                      track_running_stats = args.track_running_stats)
  if not args.track_running_stats:
    logging.info(model)
  
  # logging the initialized architecture
  best_arch_per_epoch = []
  arch_str = model.genotype().tostr()
  if args.dataset == 'cifar10':
    #def get_arch_score(api, arch_str, dataset, acc_type, use_012_epoch_training=False):
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
  else:
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
  tmp = (arch_str, test_acc, valid_acc)
  best_arch_per_epoch.append(tmp)

  cmaes_optimizer = CMAES(mean=np.zeros(args.edges * len(NAS_BENCH_201), dtype=np.float64), sigma=1.3,
                          bounds = None, seed = args.seed, n_max_resampling=100, population_size = args.pop_size)
  
  mean_list, cov_list = [], []
  mean_list.append(cmaes_optimizer._mean.copy())
  cov_list.append(cmaes_optimizer._C.copy())
  logging.info(f'[INFO] mean {mean_list[-1]}, covariance: {cov_list[-1]}')
  
  arch_df = pd.DataFrame(columns=['genotype', 'generation', 'test_acc', 'valid_acc', 'arch', 'arch_score', 'time_cost'])

  total_epochs = args.epochs
  epoch_start = time.time()
  for epoch in range(total_epochs):
    logging.info('='*100)
    logging.info(f'[INFO] Epoch ({epoch + 1}/{total_epochs})')
    
    train_start = time.time()
    
    logging.info("POPULATION EVALUATION")
    solutions = []
    eval_start = time.time()
    for ind in range(cmaes_optimizer.population_size):
      # Ask for a architecture parameter
      ind_start = time.time()
      x = cmaes_optimizer.ask()
      tx = torch.tensor(x).reshape(args.edges, -1).type(torch.float32)
      
      # Evaluating the sampled architecture
      model.update_alphas(tx)
      assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
      arch_str = model.genotype().tostr()
      
      if not ( arch_df[ arch_df['genotype']==arch_str ].empty ):
        row = arch_df[ arch_df['genotype']==arch_str ]
        valid_acc, time_cost = row['arch_score'].values[0], row['time_cost'].values[0]
        test_acc_tmp, valid_acc_tmp = row['test_acc'].values[0], row['valid_acc'].values[0]
      else:
        if args.dataset == 'cifar10':
          valid_acc, time_cost = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', use_012_epoch_training=True)
          test_acc_tmp = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
          valid_acc_tmp = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
        else:
          valid_acc, time_cost = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, use_012_epoch_training=True)
          test_acc_tmp = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
          valid_acc_tmp = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
      
      logging.info(f'[INFO] Evaluating ({ind+1:03d}/{cmaes_optimizer.population_size:03d}) valid_acc: {valid_acc:.5f} finished in {(time.time()-ind_start):.5f} seconds')
      solutions.append((x, valid_acc))
      
      # Updating the main dataframe
      if ( arch_df[ arch_df['genotype']==arch_str ].empty ):
        d_tmp = {'genotype':arch_str,'generation':epoch+1,'arch_score':valid_acc,'test_acc':test_acc_tmp,
                 'valid_acc':valid_acc_tmp,'arch':x.copy(), 'time_cost': time_cost}
        arch_df = arch_df.append(d_tmp, ignore_index=True)
      
    logging.info('[INFO] Epoch ({}/{})Evaluation finished in {:.5f} seconds'.format(epoch + 1, total_epochs, (time.time()-eval_start)))

    # Updating the architecture parameters according to the fitness estimated    
    cmaes_optimizer.tell(solutions)
    mean_list.append(cmaes_optimizer._mean.copy())
    cov_list.append(cmaes_optimizer._C.copy())
    
    #logging.info("EVALUATING the MEAN ARCHITECTURE")
    x_mean = cmaes_optimizer._mean
    tx_mean = torch.tensor(x_mean).reshape(args.edges, -1).type(torch.float32)
    model.update_alphas(tx_mean)
    assert model.check_alphas(tx_mean), "Given alphas has not been copied successfully to the model"

    _, df_max, _ = model.show_alphas_dataframe()
    logging.info(f'\n{df_max}')
  
    genotype = model.genotype()
    logging.info(f'Genotype: {genotype}')

    arch_str = genotype.tostr()    
    if args.dataset == 'cifar10':
      valid_acc_h12, time_cost = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', use_012_epoch_training=True)
      test_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
      valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
    else:
      valid_acc_h12, time_cost = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, use_012_epoch_training=True)
      test_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
      valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
    tmp = (arch_str, test_acc, valid_acc)
    best_arch_per_epoch.append(tmp)
    logging.info(f'[INFO] Architecture: {arch_str} with test accuracy: {test_acc:.3f} and validation accuracy: {valid_acc:.3f}')
    
    # Updating the main dataframe
    if ( arch_df[ arch_df['genotype']==arch_str ].empty ):
      d_tmp = {'genotype':arch_str,'generation':epoch+1,'arch_score':valid_acc_h12,'test_acc':test_acc,
               'valid_acc':valid_acc,'arch':x_mean.copy(), 'time_cost': time_cost}
      arch_df = arch_df.append(d_tmp, ignore_index=True)
   
    logging.info(f'[INFO] length main dataframe: {len(arch_df)}, solutions: {len(solutions)}')
    
    logging.info(f'[INFO] Epoch finished in {(time.time()-train_start):.5f} seconds')
  
  logging.info(f'[INFO] Architecture search finished in {(time.time()-epoch_start):.5f} seconds')
  
  logging.info(f'[INFO] Best Architecture after the search: {best_arch_per_epoch[-1]}')
  logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}')
  

  tmp_a = {'run': args.run, 'valid': best_arch_per_epoch[-1][2], 'test': best_arch_per_epoch[-1][1], 'time': time.time()-epoch_start}
  
  df = pd.DataFrame()
  df = df.append(tmp_a, ignore_index=True)
  df = df.set_index('run')
  if args.run == 1:
    df.to_csv(args.record_filename, mode='a')
  else:
    df.to_csv(args.record_filename, mode='a', header=False)

if __name__ == '__main__':
  api = API(args.api_path, verbose = False)
  args.record_filename = '{}.csv'.format(args.record_filename)
  if args.seed is None or args.seed < 0:
    for index in range(args.ntrials):
      args.run = index+1
      args.seed = random.randint(1, 100000)
      main(args, api)
  else:
    main(args, api)

