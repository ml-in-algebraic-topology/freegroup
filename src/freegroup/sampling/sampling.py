from typing import List, Tuple, Iterable

import math
from iteration_utilities import repeatfunc
import numpy as np

from ..tools import (
    reciprocal, normalize, conjugate, Comm, Mult
)

from ..tools.helper import value_or_default, remove_prefix

from functools import reduce

DEFAULT_RNG = np.random.default_rng()


def uniform_hyperbolic(radius: float, rng = None, **kwargs):
    rng = value_or_default(rng, DEFAULT_RNG)
    return max(1, int(round(math.acosh(1 + rng.random() * (np.cosh(radius) - 1)))))

def almost_uniform_hyperbolic(radius: float, rng = None, **kwargs):
    rng = value_or_default(rng, DEFAULT_RNG)
    return max(1, int(round(math.asinh(rng.random() * np.cosh(radius - 1)))))

def uniform(radius: float, rng = None, **kwargs):
    rng = value_or_default(rng, DEFAULT_RNG)
    return max(1, int(round(rng.random() * radius)))

def constant(radius: float, **kwargs): 
    return max(1, int(radius))

def random_length(method = 'uniform_hyperbolic', *args, **kwargs):
    if not isinstance(method, str):
        return method(*args, **kwargs)
    if method in ['uniform_hyperbolic', 'uh']:
        return uniform_hyperbolic(*args, **kwargs)
    if method in ['almost_uniform_hyperbolic', 'auh']:
        return almost_uniform_hyperbolic(*args, **kwargs)
    if method in ['uniform', 'u']:
        return uniform(*args, **kwargs)
    if method in ['constant', 'c']:
        return constant(*args, **kwargs)

    
def freegroup(fdim, rng = None, **kwargs):
    
    rng = value_or_default(rng, DEFAULT_RNG)
    
    length_kwargs = remove_prefix("length", kwargs)
    
    def generators_index(generator):
        if generator < 0:
            return abs(generator) - 1
        return fdim + abs(generator) - 1
    
    p = 1 / (2 * fdim - 1)
    
    dist = [p for _ in range(2 * fdim)]
    generators = [-x for x in range(1, fdim + 1)] +\
        [x for x in range(1, fdim + 1)]
    
    result = [rng.choice(generators).item()]
    for _ in range(1, random_length(rng = rng, **length_kwargs)):
        last, _last = generators_index(result[-1]), generators_index(-result[-1])
        dist[_last], dist[last] = 0, p
        result.append(rng.choice(generators, p = dist).item())
        dist[_last] = p
        
    return result

def freegroup_generator(*args, **kwargs):
    return repeatfunc(lambda: freegroup(*args, **kwargs))
        
    
def normal_closure_via_conjugation(
    closure: List[int],
    fdim: int = 2,
    rng = None,
    **kwargs,
):
  
    rng = value_or_default(rng, DEFAULT_RNG)
    
    length_kwargs = remove_prefix('length', kwargs)
    length = random_length(rng = rng, **length_kwargs)
    
    conjugator_kwargs = remove_prefix('conjugator', kwargs)
    
    result = []
    while True:
        factor = closure if rng.random() > 0.5 else reciprocal(closure)
        conjugator = freegroup(fdim = fdim, rng = rng, **conjugator_kwargs)
        new_result = result + conjugate(factor, conjugator)
        new_result = normalize(new_result)
        if len(new_result) > length:
            break
        result = new_result

    return result


def __random_bracket_sequence(n, rng = None):
    """Generates a balanced sequence of n +1s and n -1s corresponding to correctly nested brackets."""
    # Source: https://gist.github.com/rygorous/d57941fa5ae6beb59f17bc30793d3d75
    # "Generating binary trees at random", Atkinson & Sack, 1992
    
    rng = value_or_default(rng, DEFAULT_RNG)

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
    rng = None,
    mind_reduction: bool = True,
    **kwargs,
):
    
    rng = value_or_default(rng, DEFAULT_RNG)
    
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
    rng = None,
    **kwargs,
):
    rng = value_or_default(rng, DEFAULT_RNG)
    
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
    