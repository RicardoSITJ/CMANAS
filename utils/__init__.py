##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

import genotypes
import os
import torch
import pandas as pd

from .configure_utils    import load_config, dict2config, configure2str
from .basic_args         import obtain_basic_args
from .attention_args     import obtain_attention_args
from .random_baseline    import obtain_RandomSearch_args
from .cls_kd_args        import obtain_cls_kd_args
from .cls_init_args      import obtain_cls_init_args
from .search_single_args import obtain_search_single_args
from .search_args        import obtain_search_args
# for network pruning
from .pruning_args       import obtain_pruning_args

# utility function
from .flop_benchmark   import get_model_infos, count_parameters_in_MB
from .evaluation_utils import obtain_accuracy

# Custom functions added
def create_exp_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))
    
def model_save(model, model_path):
  torch.save(model.state_dict(), model_path)

def model_load(model, model_path, gpu = 0):
  model.load_state_dict(torch.load(model_path, map_location = 'cuda:{}'.format(gpu)), strict=False)

def get_arch_score(df, arch_str, dataset, acc_type):
  '''Gets the accuracy for given dataset and architecture from a customized pandas dataframe created from NAS201
  which contains only accuracy informations of all 15625 architectures in NAS201. This dataframe was created to
  reduce the RAM requirement for accessing the accuracy information from the original NAS201'''
  series = df[df['arch'] == arch_str]
  return series[dataset+'-'+acc_type].values[0]

def discretize(alphas, arch_genotype):
  genotype = genotypes.PRIMITIVES
  normal_cell = arch_genotype.normal
  reduction_cell = arch_genotype.reduce
  
  # Discretizing the normal cell
  index = 0
  offset = 0
  new_normal = torch.zeros_like(alphas[0])
  while index < len(normal_cell):
    op, cell = normal_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_normal[int(offset + cell)][idx] = 1
    index += 1
    op, cell = normal_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_normal[int(offset + cell)][idx] = 1
    offset += (index // 2) + 2
    index += 1
  
  # Discretizing the reduction cell
  index = 0
  offset = 0
  new_reduce = torch.zeros_like(alphas[1])
  while index < len(reduction_cell):
    op, cell = reduction_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_reduce[int(offset + cell)][idx] = 1
    index += 1
    op, cell = reduction_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_reduce[int(offset + cell)][idx] = 1
    offset += (index // 2) + 2
    index += 1
  return [new_normal, new_reduce]

def compare_s1genotype(g1, g2):
  for index, node1 in enumerate(g1):
    tmp_list = g2[int(index/2)*2: (int(index/2) + 1)*2]
    if node1 not in tmp_list:
      return False
  return True

def search_dataframe(df, g):
  if (not df.empty):
    for index, row in df.iterrows():
      if compare_s1genotype(row['genotype'].normal, g.normal):
        if compare_s1genotype(row['genotype'].reduce, g.reduce):
          return row
  return None
