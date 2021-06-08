import sys
sys.path.insert(0,'./')

import argparse
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import torch
import torchvision

from cma_es                  import CMAES
from architecture            import Architecture
from cell_operations         import NAS_BENCH_201
from config_utils            import load_config
from datasets                import get_datasets, get_nas_search_loaders
from nas_201_api             import NASBench201API as API
from procedures              import prepare_seed
from procedures              import get_optim_scheduler
from search_cells            import NAS201SearchCell as SearchCell
from search_model_cmaes_nas  import TinyNetworkCMAES_NAS as CMAES_NAS
from utils                   import create_exp_dir, model_load

parser = argparse.ArgumentParser("CMANAS search for Search Space 2")
parser.add_argument('--api_path', type = str, default = None, help = 'path to the NAS201 api')
parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
parser.add_argument('--config_path', type=str, help='The config path.')
parser.add_argument('--cutout', action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length', type = int, default = 16, help = 'cutout length')
parser.add_argument('--data_dir', type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--epochs', type = int, default = 50, help = 'num of architecture search epochs')
parser.add_argument('--gpu', type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--max_nodes', type = int, default = 4, help = 'maximim nodes in the cell for NAS201 network')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--n_trained_models', type = int, default = 1, help = 'number of trained models used for search')
parser.add_argument('--num_cells', type = int, default = 5, help = 'number of cells for NAS201 network')
parser.add_argument('--output_dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--pr_trained_model', type = str, default = None, help = 'pre trained model')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--seed', type = int, help = 'random seed')
parser.add_argument('--track_running_stats', action = 'store_true', default = False, help = 'use track_running_stats in BN layer')
parser.add_argument('--valid_batch_size', type = int, default = 1024, help = 'batch size')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
args = parser.parse_args()

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
assert args.api_path is not None, 'NAS201 data path has not been provided'
args.edges = 6
api = API(args.api_path, verbose = False)

# Configuring the Output directory
DIR = args.output_dir
if not os.path.exists(args.output_dir):
  create_exp_dir(DIR)

# Configuring the logger
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'CMANAS_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

tmp=''
for arg in sys.argv:
  tmp+=' {}'.format(arg)
logging.info(f'python{tmp}')
logging.info(f'[INFO] torch version {torch.__version__}, torchvision version: {torchvision.__version__}')
logging.info(f'[INFO] {args}')

if args.dataset == 'cifar10':
  acc_type     = 'ori-test'
  val_acc_type = 'x-valid'
else:
  acc_type     = 'x-test'
  val_acc_type = 'x-valid'

def get_arch_score(api, arch_str, dataset, acc_type=None, use_012_epoch_training=False):
  arch_index = api.query_index_by_arch( arch_str )
  assert arch_index >= 0, 'can not find this arch : {:}'.format(arch_str)
  if use_012_epoch_training:
    info = api.get_more_info(arch_index, dataset, iepoch=None, hp='12', is_random=True)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    return valid_acc, time_cost
  else:
    return api.query_by_index(arch_index=arch_index, hp = '200').get_metrics(dataset, acc_type)['accuracy']

def evaluate(valid_loader, criterion, df, arch, model, gen, trained_models, device, api, ind=None, pop_size=None, pop_flag=True):
  '''
  Evaluate the given architecture using the pre-trained models
  '''
  tx = arch.get().cpu().numpy()
  arch.cuda(device)
  # Query the accuracy from NAS201 for each sampled architecture
  model.update_alphas(arch.get())
  assert model.check_alphas(arch.get()), "Given alphas has not been copied successfully to the model"
  arch_str_tmp = model.genotype().tostr()    
  if args.dataset == 'cifar10':
    test_acc_tmp = get_arch_score(api, arch_str_tmp, 'cifar10', acc_type)
    valid_acc_tmp = get_arch_score(api, arch_str_tmp, 'cifar10-valid', val_acc_type)
  else:
    test_acc_tmp = get_arch_score(api, arch_str_tmp, args.dataset, acc_type)
    valid_acc_tmp = get_arch_score(api, arch_str_tmp, args.dataset, val_acc_type)

  # Evaluation process
  # Check if the architecture has already been evaluated
  if (not df.empty) and (not df[ df['genotype']==arch_str_tmp ].empty ):
    series = df[ df['genotype']==arch_str_tmp ]
    arch_loss_list, arch_top1_list, arch_top5_list = series['arch_loss'].values[0], series['arch_top1'].values[0], series['arch_top5'].values[0]
    if pop_flag:
      logging.info(f'[INFO] ({ind+1:03d}/{pop_size:03d}) already evaluated in generation {series["generation"].values[0]}')
    else:
      logging.info(f'[INFO] Mean architecture already evaluated in generation {series["generation"].values[0]}')
  else:
    # Discretizing the architecture
    _, arch_max, _ = arch.show_alphas_dataframe()
    new_arch = torch.tensor(arch_max.to_numpy()).type(torch.float).cuda()
    arch.update(new_arch)
        
    # Evaluating the sampled architecture
    arch_top1_list, arch_loss_list, arch_top5_list = [], [], []
    for trained_model in trained_models:
      pre_start = time.time()
      model_load(model=model, model_path=trained_model, gpu=args.gpu)
      arch_loss, arch_top1, arch_top5 = arch.evaluate(data_loader=valid_loader, model=model, criterion=criterion, device=device)
    
      if pop_flag:
        logging.info(f'[INFO] Evaluating ({ind+1:03d}/{pop_size:03d}) Losses: {arch_loss:.5f}, top1: {arch_top1:.5f}, top5: {arch_top5:.5f} finished in {(time.time()-pre_start)/60:.3f} minutes')
      else:
        logging.info(f'[INFO] Evaluating (mean architecture) Losses: {arch_loss:.5f}, top1: {arch_top1:.5f}, top5: {arch_top5:.5f} finished in {(time.time()-pre_start)/60:.3f} minutes')
      arch_top1_list.append(arch_top1)
      arch_loss_list.append(arch_loss)
      arch_top5_list.append(arch_top5)
    
    # Appending the newly sample architecture to the main dataframe
    d_tmp = {'genotype': arch_str_tmp, 'arch_loss': arch_loss_list, 'arch_top1': arch_top1_list, 'arch_score': arch_top1_list[0],
             'arch_top5': arch_top5_list, 'test_acc': test_acc_tmp, 'valid_acc': valid_acc_tmp, 'arch': tx.copy(), 'generation': gen}
    df = df.append(d_tmp, ignore_index=True)
    
  logging.info(f'[INFO] architecture stats: Loss {arch_loss_list} with top5: {arch_top5_list}')
  logging.info(f'[INFO] Score (top1) of the architecture, {arch_top1_list}')
  
  if not pop_flag:
    return df, arch_loss_list, arch_top1_list, arch_top5_list
  else:
    return df, arch_top1_list[0]

def main(args):
  # Reproducibility
  device = torch.device("cuda:{}".format(args.gpu))
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  prepare_seed(args.seed)
  torch.cuda.set_device(args.gpu)

  logging.info(f'length of api: {len(api)}')

  # Loading the trained models
  trained_models = [args.pr_trained_model]
  logging.info(f'trained model: {trained_models}')

  # Configuring dataset and dataloader
  if args.cutout:
    train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=args.cutout_length)
  else:
    train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=-1)
  logging.info("train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(len(train_data), len(valid_data), xshape, num_classes))
  config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
  logging.info(f'config: {config}')
  _, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data, dataset=args.dataset,
                                                        config_root='configs', batch_size=(args.batch_size, args.valid_batch_size),
                                                        workers=args.workers)
  logging.info('train_loader length: {}, valid_loader length: {}'.format(len(train_loader), len(valid_loader)))

  # Model Initialization
  model = CMAES_NAS(C = args.init_channels, N = args.num_cells, max_nodes = args.max_nodes,
                      num_classes = num_classes, search_space = NAS_BENCH_201, affine = False,
                      track_running_stats = args.track_running_stats)
  model = model.to(device)
  if not args.track_running_stats:
    logging.info(model)
  
  # logging the initialized architecture
  best_arch_per_epoch = []
  arch_str = model.genotype().tostr()
  if args.dataset == 'cifar10':
    test_acc = get_arch_score(api, arch_str, 'cifar10', acc_type)
    valid_acc = get_arch_score(api, arch_str, 'cifar10-valid', val_acc_type)
  else:
    test_acc = get_arch_score(api, arch_str, args.dataset, acc_type)
    valid_acc = get_arch_score(api, arch_str, args.dataset, val_acc_type)
  tmp = (arch_str, test_acc, valid_acc)
  best_arch_per_epoch.append(tmp)

  # Optimizer and criterion intialization
  _, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
  criterion = criterion.cuda()
  logging.info(f'Criterion: {criterion}')
  cmaes_optimizer = CMAES(mean=np.zeros(args.edges * len(NAS_BENCH_201), dtype=np.float64), sigma=1.3, bounds = None, n_max_resampling=100)
  mean_list, cov_list = [], []
  mean_list.append(cmaes_optimizer._mean.copy())
  cov_list.append(cmaes_optimizer._C.copy())
  logging.info(f'[INFO] mean {mean_list[-1]}, covariance: {cov_list[-1]}')

  df = pd.DataFrame(columns=['genotype', 'generation', 'arch_loss', 'arch_top1', 'arch_top5','test_acc', 'valid_acc', 'arch', 'arch_score'])

  total_epochs = args.epochs
  epoch_start = time.time()
  for epoch in range(total_epochs+1):
    logging.info('='*100)

    train_start = time.time()
    if epoch < total_epochs:
      logging.info(f'[INFO] Epoch ({epoch + 1}/{total_epochs})')
      logging.info("POPULATION EVALUATION")
      solutions = []
      eval_start = time.time()
      
      for ind in range(cmaes_optimizer.population_size):
        # Ask for a architecture parameter
        ind_start = time.time()
        x = cmaes_optimizer.ask()
        tx = torch.tensor(x).type(torch.float).reshape(args.edges, -1)
        arch = Architecture(num_edges=args.edges, search_space=NAS_BENCH_201, value=tx)

        df, arch_score = evaluate(valid_loader=valid_loader, criterion=criterion,
                                          df=df, arch=arch,
                                          model=model, gen=epoch+1,
                                          trained_models=trained_models,
                                          ind=ind, pop_size=cmaes_optimizer.population_size,
                                          device=device, api=api)
        
        logging.info(f'[INFO] Score of the architecture: {arch_score:.5f} finished in {(time.time()-ind_start)/60:.3f} minutes')
        
        solutions.append((x, arch_score))
      logging.info('[INFO] Epoch ({}/{})Evaluation finished in {:.3f} minutes'.format(epoch + 1, total_epochs, (time.time()-eval_start) / 60))

      if epoch == total_epochs - 1:
        df.to_json(os.path.join(DIR, 'all_sampled_architectures.json'))

      logging.info('[INFO] UPDATING the CMAES optimizer')
      cmaes_optimizer.tell(solutions)
      mean_list.append(cmaes_optimizer._mean.copy())
      cov_list.append(cmaes_optimizer._C.copy())
      
      logging.info(f'[INFO] mean: {mean_list[-1]}')
      
      logging.info("EVALUATING the MEAN ARCHITECTURE")
      x = cmaes_optimizer._mean
      tx = torch.tensor(x).type(torch.float).reshape(args.edges, -1)
      arch = Architecture(num_edges=args.edges, search_space=NAS_BENCH_201, value=tx)
      df, mean_loss, mean_top1, mean_top5 = evaluate(valid_loader=valid_loader, criterion=criterion,
                                                     df=df, arch=arch,
                                                     model=model, gen=epoch+1,
                                                     trained_models=trained_models, api=api,
                                                     device=device, pop_flag=False)
      logging.info(f'[INFO] length of main dataframe: {len(df)}, solutions: {len(solutions)}')
      
      assert model.check_alphas(arch.get()), "Given alphas has not been copied successfully to the model"
      _, df_max, _ = model.show_alphas_dataframe()
      logging.info(f'\n{df_max}')
      genotype = model.genotype()
      logging.info(f'Genotype: {genotype}')

      arch_str = genotype.tostr()    
      if args.dataset == 'cifar10':
        test_acc = get_arch_score(api, arch_str, 'cifar10', acc_type)
        valid_acc = get_arch_score(api, arch_str, 'cifar10-valid', val_acc_type)
      else:
        test_acc = get_arch_score(api, arch_str, args.dataset, acc_type)
        valid_acc = get_arch_score(api, arch_str, args.dataset, val_acc_type)
      tmp = (arch_str, test_acc, valid_acc)
      best_arch_per_epoch.append(tmp)
      logging.info(f'[INFO] Architecture: {arch_str} with test accuracy: {test_acc:.3f} and validation accuracy: {valid_acc:.3f}, top1: {mean_top1}')
    
    if epoch < total_epochs: logging.info(f'[INFO] Epoch finished in {(time.time()-train_start) / 60:.5f} minutes')
  
  logging.info(f'[INFO] Architecture search finished in {(time.time()-epoch_start) / 3600:.5f} hours')
  
  logging.info(f'[INFO] Mean Architecture after the search: {best_arch_per_epoch[-1]}')
  logging.info(f'length best_arch_per_epoch: {len(best_arch_per_epoch)}')
  
  with open(os.path.join(DIR, "mean_list.pickle"), 'wb') as f:
    pickle.dump(mean_list, f)

  with open(os.path.join(DIR, "cov_list.pickle"), 'wb') as f:
    pickle.dump(cov_list, f)

if __name__ == '__main__':
  main(args)

