from typing import List, Tuple, Iterable

import math
from iteration_utilities import repeatfunc
import numpy as np

from ..tools import (
    reciprocal, normalize, conjugate, Comm, Mult
)

from ..tools.helper import remove_prefix
from .helper import get_rng

from functools import reduce

from copy import deepcopy


def geometric(low: int = 0, high: int = 10, p: float = 0.5, size = None, rng = 0, **kwargs):
    x = get_rng(rng).geometric(p = p, size = size)
    return np.where(x <= high + 1 - low, high + 1 - x, low)

def uniform(low: int = 0, high: int = 10, size = None, rng = 0, **kwargs):
    return get_rng(rng).integers(low = low, high = high, size = size)

def constant(value: int, size = None, rng = 0, **kwargs):
    return value if size is None else value * np.ones(size)

def random_length(method = 'geom', *args, **kwargs):
    if not isinstance(method, str):
        return method(*args, **kwargs)
    if method in ['geometric', 'geom', 'g']:
        return geometric(*args, **kwargs)
    if method in ['uniform', 'u']:
        return uniform(*args, **kwargs)
    if method in ['constant', 'c']:
        return constant(*args, **kwargs)
    raise ValueError('Unknown distribution over lengths')

    
def freegroup(fdim: int, prefix: List[int] = None, rng = 0, **kwargs):
    
    kwargs = deepcopy(kwargs)
    
    rng = get_rng(rng)
    
    length_kwargs = remove_prefix('length', kwargs)
    length = random_length(rng = rng, **length_kwargs)
    
    def generators_index(generator):
        if generator < 0:
            return abs(generator) - 1
        return fdim + abs(generator) - 1
    
    p = 1 / (2 * fdim - 1)
    
    dist = [p for _ in range(2 * fdim)]
    generators = [-x for x in range(1, fdim + 1)] +\
        [x for x in range(1, fdim + 1)]
    
    result = prefix[::] if not prefix is None else []
    if not result:
        result.append(rng.choice(generators).item())
    for _ in range(length - len(result)):
        last, _last = generators_index(result[-1]), generators_index(-result[-1])
        dist[_last], dist[last] = 0, p
        result.append(rng.choice(generators, p = dist).item())
        dist[_last] = p
        
    return result[:length]


def freegroup_generator(*args, **kwargs):
    return repeatfunc(lambda: freegroup(*args, **kwargs))


def normal_closure_via_conjugation(
    closure: List[int], fdim: int,
    rng = 0, **kwargs,
):  
    kwargs = deepcopy(kwargs)
  
    rng = get_rng(rng)
    
    length_kwargs = remove_prefix('length', kwargs)
    length = random_length(rng = rng, **length_kwargs)
    
    previous_suffix_kwargs = remove_prefix('suffix_length', kwargs)
    previous_suffix_kwargs['allow_zero'] = True
    previous_suffix_radius = previous_suffix_kwargs.get('radius', length)
    
    conjugator_kwargs = remove_prefix('conjugator', kwargs)
    conjugator_length_radius = min((length - len(closure)) // 2, conjugator_kwargs.get('length_radius', length))
    conjugator_kwargs['length_allow_zero'] = False
    
    if kwargs: raise ValueError(f'Unknown arguments: {kwargs}')
    
    result, previous_conjugator = [], []
    while True:
        factor = closure if rng.random() > 0.5 else reciprocal(closure)
                
        previous_suffix_kwargs['radius'] = min(
            len(previous_conjugator),
            previous_suffix_radius
        )
        
        k = random_length(rng = rng, **previous_suffix_kwargs)
        
        conjugator_kwargs['length_radius'] = min(
            conjugator_length_radius,
            (length - len(result) - len(closure) + 2 * k) // 2
        )
        previous_conjugator = freegroup(
            fdim = fdim, rng = rng,
            prefix = previous_conjugator[-k:][::-1], **conjugator_kwargs
        )[::-1]
                
        new_result = normalize(result + conjugate(factor, previous_conjugator))
        
        if len(new_result) > length:
            break
        
        result = new_result

    return result


def __random_bracket_sequence(n, rng = 0):
    """Generates a balanced sequence of n +1s and n -1s corresponding to correctly nested brackets."""
    # Source: https://gist.github.com/rygorous/d57941fa5ae6beb59f17bc30793d3d75
    # "Generating binary trees at random", Atkinson & Sack, 1992
    
    rng = get_rng(rng)

    seq = [-1, 1] * n
    rng.shuffle(seq)
    
    prefix, suffix, word = [], [], []
    partial_sum = 0

    for s in seq:
        word.append(s)
        partial_sum += s
        if partial_sum == 0:
            if s == -1:
                prefix += word
            else:
                prefix.append(1)
                suffix = [-1] + [-x for x in word[1:-1]] + suffix
            word = []
    return prefix + suffix

def normal_closure_via_brackets(
    closure: List[int],
    fdim: int,
    rng = 0,
    mind_reduction: bool = True,
    **kwargs,
):
    
    rng = get_rng(rng)
    
    kwargs = deepcopy(kwargs)
    
    proba_kwargs = remove_prefix('proba', kwargs)
    pconjugation, pclosure =\
        proba_kwargs.get('conjugation', None), proba_kwargs.get('closure', None)
    if pconjugation is None and pclosure is None:
        pconjugation = 1.
    if not pconjugation is None:
        pclosure = 1 - pconjugation
    if not pclosure is None:
        pconjugation = 1 - pclosure
    
    depth_kwargs = remove_prefix('depth', kwargs)
    depth = random_length(rng = rng, **depth_kwargs)
    seq = __random_bracket_sequence(n = depth, rng = rng)
    
    match, stack = [None] * len(seq), []

    for i, c in enumerate(seq):
        stack.append((i, c))
        if len(stack) < 2:
            continue
        (i1, c1), (i2, c2) = stack[-2], stack[-1]
        if c1 == -c2:
            del stack[-2:]
            match[i1] = i2
            match[i2] = i1

    sampled = [None] * len(seq)
    
    for idx, match_idx in enumerate(match):
        if match_idx < idx: continue
        
        coin = rng.random()
        
        if (mind_reduction and idx + 1 == match_idx) or (coin <= pclosure):
            split = rng.integers(low = 0, high = len(closure))
            left, right = closure[:split], closure[split:]
            if rng.random() < 0.5: left, right = right, left
            if rng.random() < 0.5: left, right = reciprocal(left), reciprocal(right)
            sampled[idx] = left
            sampled[match_idx] = right
            continue
        coin -= pclosure

        if coin <= pconjugation:
            probas = np.ones(2 * fdim + 1)
            probas[0 + fdim] = 0
            
            if mind_reduction:
                i = idx - 1
                while i >= 0 and not sampled[i] is None:
                    if not sampled[i]: i -= 1; continue
                    probas[-sampled[i][-1] + fdim] = 0; break

                i = idx + 1
                while i <= len(seq) - 1 and not sampled[i] is None:
                    if not sampled[i]: i += 1; continue
                    probas[-sampled[i][0] + fdim] = 0

                i = match_idx - 1
                while i >= 0 and not sampled[i] is None:
                    if not sampled[i]: i -=1; continue
                    probas[sampled[i][-1] + fdim] = 0; break

                i = match_idx + 1
                while i <= len(seq) - 1 and not sampled[i] is None:
                    if not sampled[i]: i += 1; continue
                    probas[sampled[i][0] + fdim] = 0; break
            
            probas /= probas.sum()
            conjugator = rng.choice(2 * fdim + 1, p = probas)
            sampled[idx] = [conjugator - fdim]
            sampled[match_idx] = [fdim - conjugator]
            continue
        coin -= pconjugation
        
    return normalize(reduce(lambda x, y: x + y, sampled))
    
    
def normal_closure(method = 'conjugation', *args, **params):
    if method in ['conjugation', 'conj']:
        return normal_closure_via_conjugation(*args, **params)
    if method in ['brackets', 'br']:
        return normal_closure_via_brackets(*args, **params)
    raise ValueError('Unknown method')
    
def normal_closure_generator(method = 'conjugation', *args, **params):
    if method in ['conjugation', 'conj']:
        return repeatfunc(lambda: normal_closure_via_conjugation(*args, **params))
    if method in ['brackets', 'br']:
        return repeatfunc(lambda: normal_closure_via_brackets(*args, **params))
    raise ValuesError('Unknown method')



def random_tree(
    words: List[List[int]],
    rng = 0,
    **kwargs,
):
    rng = get_rng(rng)
    
    pmult = kwargs.get('pmult', 0.)
    pcomm = kwargs.get('pcomm', 1.)
    
    assert pmult + pcomm == 1.
    
    if len(words) == 0: return []
    if len(words) == 1: return words[0]
    
    coin = rng.random()
    if coin <= pmult:
        return Mult(words)
    coin -= pmult
    
    if coin <= pcomm:
        idx = rng.integers(1, len(words))
        return Comm([
            random_tree(words[:idx], rng = rng, **kwargs),
            random_tree(words[idx:], rng = rng, **kwargs)
        ])
    
    coin -= pcomm
    