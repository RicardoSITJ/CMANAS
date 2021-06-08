import sys
sys.path.insert(0,'./')

import argparse
import json
import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import utils
import visualize

from cma_es                  import CMAES
from config_utils            import load_config, dict2config
from datasets                import get_datasets, get_nas_search_loaders
from log_utils               import AverageMeter, time_string, convert_secs2time
from procedures              import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint
from procedures              import get_optim_scheduler
from model_search            import Network
from torch.utils.data        import DataLoader
from torch.autograd          import Variable
from utils                   import get_model_infos, obtain_accuracy, create_exp_dir, model_save, model_load, get_arch_score, discretize

parser = argparse.ArgumentParser("CMANAS for Search Space 1")
parser.add_argument('--alt_cifar100_split',type = int, default = 0, help = 'What alternate split to use for CIFAR100')
parser.add_argument('--batch_size',       type = int, default = 64, help = 'batch size')
parser.add_argument('--config_path',      type=str, help='The config path.')
parser.add_argument('--cutout',           action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length',    type = int, default = 16, help = 'cutout length')
parser.add_argument('--dataset',          type = str, default = 'cifar10', help = 'dataset name')
parser.add_argument('--data_dir',         type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--epochs',           type = int, default = None , help = 'num of training epochs')
parser.add_argument('--gpu',              type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip',        type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels',    type = int, default = 16, help = 'num of init channels')
parser.add_argument('--layers',           type = int, default = 8, help = 'total number of layers')
parser.add_argument('--learning_rate',    type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min',type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum',         type = float, default = 0.9, help = 'momentum')
parser.add_argument('--n_trained_models', type = int, default = 1, help = 'number of trained models used for search')
parser.add_argument('--output_dir',       type = str, default = None, help = 'location of trials')
parser.add_argument('--pop_size',         type = int, default = None, help = 'population size')
parser.add_argument('--pre_tr_model',     type = str, default = None, help = 'pre trained model')
parser.add_argument('--report_freq',      type = float, default = 100, help = 'report frequency')
parser.add_argument('--run_id',           type=int, default=None, help='running id for the experiment')
parser.add_argument('--seed',             type = int, default = None, help = 'random seed')
parser.add_argument('--train_discrete',   action='store_true', default = False, help='Whether to use discretization during training')
parser.add_argument('--train_portion',    type = float, default = 0.5, help = 'portion of training data')
parser.add_argument('--tr_model_dir',     type = str, help = 'directory of trained models')
parser.add_argument('--valid_batch_size', type = int, default = 1024, help = 'batch size')
parser.add_argument('--ind_epochs',          type = int, default = 0, help = 'num of epochs for training per epoch of CMAES')
parser.add_argument('--weight_decay',     type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--workers',          type=int, default=2, help='number of data loading workers (default: 2)')
args = parser.parse_args()
if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
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

def eval_arch(data_loader, model, criterion):
  '''
    Evaluate the architecture
  '''
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  
  model.eval()
  with torch.no_grad():
    for step, (inputs, targets) in enumerate(data_loader):
      inputs = inputs.cuda(non_blocking=True)
      targets = targets.cuda(non_blocking=True)
      
      # prediction
      logits = model(inputs)
      arch_loss = criterion(logits, targets)
      
      # record
      arch_prec1, arch_prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
      arch_losses.update(arch_loss.item(),  inputs.size(0))
      arch_top1.update  (arch_prec1.item(), inputs.size(0))
      arch_top5.update  (arch_prec5.item(), inputs.size(0))
      #break
      
  return arch_losses.avg, arch_top1.avg, arch_top5.avg

def evaluate(valid_loader, criterion, df, alphas, model, gen, trained_models, alphax, ind=None, pop_size=None, pop_flag=True):
  '''
  Evaluate the given architecture using the pre-trained models
  '''
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  
  # Copying the given alphas to the model
  model.update_alphas(alphas)
  assert model.check_alphas(alphas), "Given alphas has not been copied successfully to the model"
  genotype_tmp = model.genotype()
  
  # Evaluation process
  # Check if the architecture has already been evaluated
  series = utils.search_dataframe(df, genotype_tmp)
  #logging.info(f'series: {series}')
  if (series is not None):
    arch_loss_list, arch_top1_list, arch_top5_list = series['arch_loss'], series['arch_top1'], series['arch_top5']
    if pop_flag:
      logging.info(f'[INFO] ({ind+1:03d}/{pop_size:03d}) already evaluated in generation {series["generation"]}')
    else:
      logging.info(f'[INFO] Mean architecture already evaluated in generation {series["generation"]}')
  else:
    # Discretizing the architecture
    assert model.check_alphas(alphas), "Given alphas has not been copied successfully to the model"
    discrete_alphas = utils.discretize(alphas=alphas, arch_genotype=model.genotype())
    model.update_alphas(discrete_alphas)
    assert model.check_alphas(discrete_alphas), "Given alphas has not been copied successfully to the model"

    # Evaluating the sampled architecture
    arch_top1_list, arch_loss_list, arch_top5_list = [], [], []
    for trained_model in trained_models:
      pre_start = time.time()
      model_load(model=model, model_path=trained_model, gpu=args.gpu)
      #def eval_arch(data_loader, model, criterion):
      arch_loss, arch_top1, arch_top5 = eval_arch(data_loader=valid_loader, model=model, criterion=criterion)
    
      if pop_flag:
        logging.info(f'[INFO] Evaluating ({ind+1:03d}/{pop_size:03d}) Losses: {arch_loss:.5f}, top1: {arch_top1:.5f}, top5: {arch_top5:.5f} finished in {(time.time()-pre_start)/60:.3f} minutes')
      else:
        logging.info(f'[INFO] Evaluating (mean architecture) Losses: {arch_loss:.5f}, top1: {arch_top1:.5f}, top5: {arch_top5:.5f} finished in {(time.time()-pre_start)/60:.3f} minutes')
      arch_top1_list.append(arch_top1)
      arch_loss_list.append(arch_loss)
      arch_top5_list.append(arch_top5)
    
    # Appending the newly sample architecture to the main dataframe
    d_tmp = {'genotype': genotype_tmp, 'arch_loss': arch_loss_list, 'arch_top1': arch_top1_list, 'arch_score': arch_top1_list[0],
             'arch_top5': arch_top5_list, 'arch': alphax, 'generation': gen}
    df = df.append(d_tmp, ignore_index=True)
    
  logging.info(f'[INFO] architecture stats: Loss: {arch_loss_list} with top5: {arch_top5_list}')
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
  #torch.set_default_dtype(torch.float64)
  torch.cuda.set_device(args.gpu)

  logging.info('CMANAS with pre-trained one shot models for evaluation')

  # Loading the trained models
  if args.pre_tr_model == 'None':
    tmp_files = os.listdir(args.tr_model_dir)
    trained_models = [os.path.join(args.tr_model_dir, pt_file) for pt_file in tmp_files if 'pt' in pt_file]
    logging.info(f'trained models: {trained_models}')
    trained_models = random.sample(trained_models, args.n_trained_models)
  else:
    trained_models = [args.pre_tr_model]
  logging.info(f'Selected trained models: {trained_models}')

  # Configuring dataset and dataloader
  if args.cutout:
    train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=args.cutout_length)
  else:
    train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data_dir, cutout=-1)
  
  if args.dataset == 'cifar100' and (args.alt_cifar100_split == 1):
    logging.info('[INFO] Using similar split for CIFAR100 as CIFAR10 for S1')
    with open('configs/cifar100-split.txt', 'r') as f:
      cifar100_split = json.load(f)
    cifar100_split = dict2config(cifar100_split, logger=None)
    logging.info(f'cifar100_split.train: {len(cifar100_split.train)}, cifar100_split.valid: {len(cifar100_split.valid)}')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory = True, num_workers = args.workers,
                                sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.train))
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.valid_batch_size, pin_memory = True, num_workers = args.workers,
                                sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.valid))
  elif args.dataset == 'cifar100' and (args.alt_cifar100_split == 2):
    logging.info('[INFO] Using 80-20 split for CIFAR100')
    with open('configs/cifar100_80-split.txt', 'r') as f:
      cifar100_split = json.load(f)
    cifar100_split = dict2config(cifar100_split, logger=None)
    logging.info(f'cifar100_split.train: {len(cifar100_split.train)}, cifar100_split.valid: {len(cifar100_split.valid)}')
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, pin_memory = True, num_workers = args.workers,
                                sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.train))
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.valid_batch_size, pin_memory = True, num_workers = args.workers,
                                sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.valid))
  else:
    _, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data, dataset=args.dataset,
                                                           config_root='configs', batch_size=(args.batch_size, args.valid_batch_size),
                                                           workers=args.workers)
  logging.info('train_loader length: {}, valid_loader length: {}'.format(len(train_loader), len(valid_loader)))
  
  # Model Initialization
  model = Network(args.init_channels, num_classes, args.layers, device)
  model = model.to(device)
  
  # Optimizer and criterion intialization
  config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
  logging.info(f'config: {config}')
  _, _, criterion = get_optim_scheduler(parameters=model.parameters(), config=config)
  criterion = criterion.cuda()
  logging.info(f'Criterion: {criterion}')
  
  # Initializing the CMA-ES optimizer for the architecture search
  cmaes_optimizer = CMAES(mean=np.zeros(model.arch_parameters()[0].view(-1).shape[0] * 2, dtype=np.float32),
                        sigma=1.3, bounds = None, seed = args.seed, n_max_resampling=100, population_size = args.pop_size)
  mean_list, cov_list, genotype_list = [], [], []
  mean_list.append(cmaes_optimizer._mean.copy())
  cov_list.append(cmaes_optimizer._C.copy())
  logging.info(f'[INFO] mean {mean_list[-1]}, covariance: {cov_list[-1]}')
  
  df = pd.DataFrame(columns=['genotype', 'generation', 'arch_loss', 'arch_top1', 'arch_top5', 'arch', 'arch_score'])

  total_epochs = args.epochs
  epoch_start = time.time()
  for epoch in range(total_epochs + 1):
    train_start = time.time()
    # Start architecture search after the warmup
    if epoch < total_epochs:
      logging.info('='*100)
      logging.info(f'[INFO] Epoch ({epoch + 1}/{total_epochs})')
      logging.info("POPULATION EVALUATION")
      solutions = []
      eval_start = time.time()

      for ind in range(cmaes_optimizer.population_size):
        # Evaluating the architecture in the population
        ind_start = time.time()
        x = cmaes_optimizer.ask()
        #if ((ind+1) % 3) == 0: x = solutions[ind-1][0] # For testing the dataframe working
        tx = torch.tensor(x).type(torch.float).cuda()
        tx = list(torch.chunk(tx, 2))
        tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]
        
        df, arch_score = evaluate(valid_loader=valid_loader, criterion=criterion, df=df, alphas=tx,
                                                   model=model, gen=epoch+1, trained_models=trained_models,
                                                   alphax=x, ind=ind, pop_size=cmaes_optimizer.population_size)
        logging.info(f'[INFO] Score of the architecture, {arch_score}:{arch_score:.5f} finished in {(time.time()-ind_start)/60:.3f} minutes')
        solutions.append((x, arch_score))

      if epoch < total_epochs:
        logging.info('[INFO] Epoch ({}/{})Evaluation finished in {:.3f} minutes'.format(epoch + 1, total_epochs, (time.time()-eval_start) / 60))

      if epoch == total_epochs - 1:
        df.to_json(os.path.join(DIR, 'all_sampled_architectures.json'))

      # Updating the architecture parameters according to the fitness estimated
      logging.info('[INFO] UPDATING the CMAES optimizer')
      cmaes_optimizer.tell(solutions)
      mean_list.append(cmaes_optimizer._mean.copy())
      cov_list.append(cmaes_optimizer._C.copy())

    # Evaluating the mean architecture
    logging.info("EVALUATING the MEAN ARCHITECTURE")
    x = cmaes_optimizer._mean
    tx = torch.tensor(x).cuda()
    tx = list(torch.chunk(tx, 2))
    tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]
    model.update_alphas(tx)
    assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
    mean_start = time.time()
    df, mean_loss, mean_top1, mean_top5 = evaluate(valid_loader=valid_loader, criterion=criterion, df=df, alphas=tx,
                                                 model=model, gen=epoch+1, trained_models=trained_models,
                                                 alphax=x, pop_flag=False)
    logging.info(f'[INFO] Score of the mean architecture, {mean_top1} finished in {(time.time()-mean_start)/60:.3f} minutes')

    logging.info(f'[INFO] length of the main dataframe: {len(df)}, solutions: {len(solutions)}')
    
    genotype = model.genotype()
    logging.info(f'Genotype: {genotype}')
    genotype_list.append(genotype)

    logging.info(f'[INFO] Mean architecture: {genotype_list[-1]} with top1: {mean_top1[0]:.3f}')
    
    logging.info(f'[INFO] Epoch finished in {(time.time()-train_start) / 60:.5f} minutes')
  
  logging.info(f'length of genotype_list: {len(genotype_list)}, mean_list: {len(mean_list)}, cov_list: {len(cov_list)}')
    
  logging.info(f'[INFO] Architecture search finished in {(time.time()-epoch_start) / 3600:.5f} hours')
  mean_arch = genotype_list[-1]
  logging.info(f'[INFO] Mean architecture: {mean_arch}')
  with open(os.path.join(DIR, "genotype.pickle"), 'wb') as f:
    pickle.dump(mean_arch, f)
  
  normal_cell, reduce_cell = mean_arch.normal, mean_arch.reduce
  visualize.plot(genotype=normal_cell, filename=os.path.join(DIR, 'normal'))
  visualize.plot(genotype=reduce_cell, filename=os.path.join(DIR, 'reduction'))

  with open(os.path.join(DIR, "mean_list.pickle"), 'wb') as f:
    pickle.dump(mean_list, f)
  
  with open(os.path.join(DIR, "cov_list.pickle"), 'wb') as f:
    pickle.dump(cov_list, f)
  
  with open(os.path.join(DIR, "genotype_list.pickle"), 'wb') as f:
    pickle.dump(genotype_list, f)

if __name__ == '__main__':
  main(args)
