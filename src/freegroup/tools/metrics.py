from nltk.metrics.distance import *
from multiprocess import Pool
from tqdm.auto import tqdm
import numpy as np
from .tools import (
    determine_fdim, generators, permute_generators, reduce_modulo_normal_closure_step, substitute_generators, reciprocal
)
from .helper import remove_prefix

def dyck_path(word, closure):
    stack, path = [], [0]
    substitutions = {
        closure[0]: reciprocal(closure[1:]),
        -closure[0]: closure[1:],
    }
    for f in substitute_generators(word, substitutions):
        reduce_modulo_normal_closure_step(stack, f, closure)
        path.append(len(stack))
        
    return path

def number_of_valleys(path = None, word = None, closure = None):
    if path is None:
        path = dyck_path(word, closure)

    result = 0
    for idx in range(1, len(path) - 1):
        if path[idx - 1] > path[idx] and path[idx] < path[idx + 1] and path[idx] > 0:
            result += 1
    return result


def cycle_shift_invariant_similarity(
    s1, s2, metric_name = 'edit_distance', fdim = None, reduction = None, **metric_kwargs
):
    fdim = max(determine_fdim(s1, fdim), determine_fdim(s2, fdim))
    
    metric_fn = {
        'edit_distance': edit_distance,
        'edit_distance_align': edit_distance_align,
        'jaro_similarity': jaro_similarity,
        'jaro_winkler_similarity': jaro_winkler_similarity,
    }[metric_name]
    
    results = [
        metric_fn(s1, permute_generators(s2, fdim = fdim, shift = shift), **metric_kwargs)
        for shift in range(fdim)
    ]
    
    results = np.array(results)
    
    if reduction is None:
        return results
    
    if isinstance(reduction, str):    
        return {
            'min': np.min,
            'max': np.max,
            'mean': np.mean,
            'sum': np.sum,
        }[reduction](results)
    
    return reduction(results)


def batch_cycle_shift_invariant_similarity(
    words, fdim = None, **kwargs
):
    fdim = max([determine_fdim(w, fdim) for w in words])
    
    pool_kwargs = remove_prefix('pool', kwargs)
    pool_workers = pool_kwargs.pop('workers', 1)
    
    map_kwargs = remove_prefix('map', kwargs)
    map_chunksize = map_kwargs.pop('chunksize', pool_workers * 10)
    
    tqdm_kwargs = remove_prefix('tqdm', kwargs)
    
    metric_kwargs = remove_prefix('metric', kwargs)
    metric_name = metric_kwargs.pop('name', 'edit_distance')
    metric_reduction = metric_kwargs.pop('reduction', None)
    metric_self = metric_kwargs.pop('self', 0)
    
    if kwargs: raise ValueError(f'Unknown arguments: {kwargs}')
    
    def handle(task_configuration):
        i, j = task_configuration
        return i, j, cycle_shift_invariant_similarity(
            words[i], words[j], fdim = fdim, metric_name = metric_name, reduction = metric_reduction, **metric_kwargs
        )
    
    tasks = [(i, j) for i in range(len(words)) for j in range(i + 1, len(words))]
    
    results = [[None for _ in range(len(words))] for _ in range(len(words))]
    
    with tqdm(total = len(tasks), **tqdm_kwargs) as pbar, Pool(pool_workers, **pool_kwargs) as pool:
        
        for i, j, result in pool.imap_unordered(handle, tasks, chunksize = map_chunksize, **map_kwargs):
            results[i][j] = result
            results[j][i] = result
            pbar.update(1)

    for i in range(len(words)):
        results[i][i] = [0] * fdim if metric_reduction is None else metric_self
        
    return np.array(results)
    
