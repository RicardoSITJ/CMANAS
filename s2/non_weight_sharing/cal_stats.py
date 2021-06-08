import argparse
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser("CMAES search with OSM1 with random Training (NAS201)")
parser.add_argument('--csv_file', type = str, help = 'csv file name')
args = parser.parse_args()

#args.csv_file = '{}.csv'.format(args.csv_file)
print(args)
if os.path.exists(args.csv_file):
  df = pd.read_csv(args.csv_file)
  #print(df)
  print(f'[INFO] length of {args.csv_file}: {len(df)}')

  mean_valid = np.mean(df['valid'])
  std_valid = np.std(df['valid'])
  
  mean_test = np.mean(df['test'])
  std_test = np.std(df['test'])

  mean_time = np.mean(df['time'])

  print(f'Validation: {mean_valid:.5f}+-{std_valid:.5f}\t\t Test: {mean_test:.5f}+-{std_test:.5f} in mean time: {mean_time:.5f}')
