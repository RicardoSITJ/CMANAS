import sys
sys.path.insert(0, './')

# EMU EXP-004: multi-objective BASELINES over NAS-Bench-201 (S2) — NSGA-II (pymoo) and Random Search.
# Same discrete search space, same oracle queries and same output format as EMU's cmanas_search.py,
# so the fronts (accuracy up, FLOPs down, params down) are directly comparable by hypervolume.
# This script does NOT use CMANAS's CMA-ES/model; it queries the NB201 API directly with the
# identical genotype encoding, so objective values match EMU exactly.

import argparse
import logging
import os
import random
import time

import numpy as np
import pandas as pd

from nas_201_api import NASBench201API as API

try:
    from cell_operations import NAS_BENCH_201            # ['none','skip_connect','nor_conv_1x1','nor_conv_3x3','avg_pool_3x3']
except Exception:
    NAS_BENCH_201 = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']

N_EDGES = 6
N_OPS = len(NAS_BENCH_201)

parser = argparse.ArgumentParser("Multi-objective NAS baselines (NSGA-II / Random) on NAS-Bench-201")
parser.add_argument('--api_path', type=str, required=True, help='path to the NAS201 .pth')
parser.add_argument('--dataset', type=str, default='cifar10', help='["cifar10","cifar100","ImageNet16-120"]')
parser.add_argument('--method', type=str, default='nsga2', choices=['nsga2', 'random'], help='baseline to run')
parser.add_argument('--pop_size', type=int, default=35, help='NSGA-II population size (also block size for random)')
parser.add_argument('--n_gen', type=int, default=10, help='NSGA-II generations (budget = pop_size*n_gen evals)')
parser.add_argument('--ntrials', type=int, default=3, help='independent runs (random seed each)')
parser.add_argument('--seed', type=int, default=-1, help='fixed seed (<0 -> ntrials random seeds)')
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--record_filename', type=str, required=True, help='base for the per-run summary csv')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p')


def genotype_from_ops(ops):
    # NB201 arch string (6 edges: node1<-0 ; node2<-0,1 ; node3<-0,1,2). Matches model.genotype().tostr().
    o = [NAS_BENCH_201[int(i)] for i in ops]
    return '|{}~0|+|{}~0|{}~1|+|{}~0|{}~1|{}~2|'.format(o[0], o[1], o[2], o[3], o[4], o[5])


def _acc_types(dataset):
    if dataset == 'cifar10':
        return 'cifar10-valid', 'cifar10', 'ori-test', 'x-valid'      # h12/valid ds, test ds, test acc, valid acc
    return dataset, dataset, 'x-test', 'x-valid'


def eval_arch(api, ops, dataset, cache):
    key = tuple(int(o) for o in ops)
    if key in cache:
        return cache[key]
    gstr = genotype_from_ops(key)
    idx = api.query_index_by_arch(gstr)
    assert idx >= 0, 'arch not found: {}'.format(gstr)
    h12_ds, test_ds, test_acc_type, val_acc_type = _acc_types(dataset)
    # selection signal = h12 valid accuracy (identical to EMU's get_arch_score(use_012_epoch_training=True))
    info12 = api.get_more_info(idx, h12_ds, iepoch=None, hp='12', is_random=True)
    valid_h12 = info12['valid-accuracy']
    # reporting metrics (hp=200) + cost, identical to EMU
    test_acc = api.query_by_index(arch_index=idx, hp='200').get_metrics(test_ds, test_acc_type)['accuracy']
    valid_acc = api.query_by_index(arch_index=idx, hp='200').get_metrics(h12_ds, val_acc_type)['accuracy']
    cost = api.get_cost_info(idx, test_ds)
    rec = {'genotype': gstr, 'arch_score': valid_h12, 'test_acc': test_acc, 'valid_acc': valid_acc,
           'flops': cost.get('flops'), 'params': cost.get('params'), 'latency': cost.get('latency')}
    cache[key] = rec
    return rec


def pareto_front(df):
    # non-dominated over (arch_score up, flops down, params down) — same convention as EMU
    d = df.dropna(subset=['arch_score', 'flops', 'params']).reset_index(drop=True)
    if d.empty:
        return d
    obj = d[['arch_score', 'flops', 'params']].to_numpy(float) * np.array([1.0, -1.0, -1.0])
    n = len(obj); keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and np.all(obj[j] >= obj[i]) and np.any(obj[j] > obj[i]):
                keep[i] = False; break
    return d[keep].sort_values('arch_score', ascending=False).reset_index(drop=True)


def run_random(api, dataset, n_evals, seed):
    rng = random.Random(seed)
    cache = {}
    while len(cache) < n_evals:
        ops = [rng.randint(0, N_OPS - 1) for _ in range(N_EDGES)]
        eval_arch(api, ops, dataset, cache)
    return pd.DataFrame(list(cache.values()))


def run_nsga2(api, dataset, pop_size, n_gen, seed):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.sampling.rnd import IntegerRandomSampling
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.repair.rounding import RoundingRepair
    from pymoo.optimize import minimize

    cache = {}

    class NB201Problem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=N_EDGES, n_obj=3, xl=0, xu=N_OPS - 1, vtype=int)

        def _evaluate(self, x, out, *a, **kw):
            ops = [min(N_OPS - 1, max(0, int(round(v)))) for v in x]
            r = eval_arch(api, ops, dataset, cache)
            out['F'] = [-r['arch_score'], r['flops'], r['params']]   # minimize

    algo = NSGA2(pop_size=pop_size,
                 sampling=IntegerRandomSampling(),
                 crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                 mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                 eliminate_duplicates=True)
    minimize(NB201Problem(), algo, ('n_gen', n_gen), seed=seed, verbose=False)
    return pd.DataFrame(list(cache.values()))


def one_run(api, run_idx, seed):
    t0 = time.time()
    logging.info('[EXP-004] method={} run={} seed={} dataset={}'.format(args.method, run_idx, seed, args.dataset))
    if args.method == 'nsga2':
        archdf = run_nsga2(api, args.dataset, args.pop_size, args.n_gen, seed)
    else:
        archdf = run_random(api, args.dataset, args.pop_size * args.n_gen, seed)
    front = pareto_front(archdf)
    base = args.record_filename[:-4] if args.record_filename.endswith('.csv') else args.record_filename
    archdf.to_csv('{}-archdf-run{}.csv'.format(base, run_idx), index=False)
    front.to_csv('{}-front-run{}.csv'.format(base, run_idx), index=False)
    best = archdf.sort_values('test_acc', ascending=False).iloc[0]
    logging.info('[EXP-004] run {} done: {} evals, front={}, best_test={:.3f}, {:.1f}s'.format(
        run_idx, len(archdf), len(front), best['test_acc'], time.time() - t0))
    return {'run': run_idx, 'valid': best['valid_acc'], 'test': best['test_acc'], 'time': time.time() - t0}


if __name__ == '__main__':
    api = API(args.api_path, verbose=False)
    rec = '{}.csv'.format(args.record_filename)
    rows = []
    if args.seed is None or args.seed < 0:
        for i in range(args.ntrials):
            rows.append(one_run(api, i + 1, random.randint(1, 100000)))
    else:
        rows.append(one_run(api, 1, args.seed))
    pd.DataFrame(rows).set_index('run').to_csv(rec)
    logging.info('[EXP-004] wrote summary {}'.format(rec))
