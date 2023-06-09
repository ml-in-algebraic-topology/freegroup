from typing import List, Tuple, Iterable

import math
from iteration_utilities import repeatfunc
from numpy import random, ones

from ..tools import (
    reciprocal, normalize, conjugate, Comm, Mult
)
from functools import reduce

def uniform_hyperbolic(radius: float):
    return max(1, int(round(math.acosh(1 + random.random() * (math.cosh(radius) - 1)))))

def almost_uniform_hyperbolic(radius: float):
    return max(1, int(round(math.asinh(random.random() * math.cosh(radius - 1)))))

def uniform(radius: float):
    return max(1, int(round(random.random() * radius)))

def constant(radius: float):
    return max(1, int(radius))

def random_length(method = "uniform_hyperbolic", *args, **kwargs):
    if not isinstance(method, str):
        return method(*args, **kwargs)
    if method in ["uniform_hyperbolic", "uh"]:
        return uniform_hyperbolic(*args, **kwargs)
    if method in ["almost_uniform_hyperbolic", "auh"]:
        return almost_uniform_hyperbolic(*args, **kwargs)
    if method in ["uniform", "u"]:
        return uniform(*args, **kwargs)
    if method in ["constant", "c"]:
        return constant(*args, **kwargs)
    

    
def freegroup(freegroup_dimension, length_method, length_parameters):
    def generators_index(generator):
        if generator < 0:
            return abs(generator) - 1
        return freegroup_dimension + abs(generator) - 1
    
    p = 1 / (2 * freegroup_dimension - 1)
    
    dist = [p for _ in range(2 * freegroup_dimension)]
    generators = [-x for x in range(1, freegroup_dimension + 1)] +\
        [x for x in range(1, freegroup_dimension + 1)]
    
    result = [random.choice(generators)]
    for _ in range(1, random_length(length_method, **length_parameters)):
        last, _last = generators_index(result[-1]), generators_index(-result[-1])
        dist[_last], dist[last] = 0, p
        result.append(random.choice(generators, p = dist))
        dist[_last] = p
        
    return result

def freegroup_generator(*args, **kwargs):
    return repeatfunc(lambda: freegroup(*args, **kwargs))
        
    
def normal_closure_via_conjugation(
    closure: List[int],
    freegroup_dimension: int = 2,
    length_method: str = 'uh',
    length_parameters = {'radius': 5},
    conjugator_length_method: str ='uh',
    conjugator_length_parameters = {'radius': 5}
):
    length = random_length(length_method, **length_parameters)
    result = []
    while True:
        factor = closure if random.random() > 0.5 else reciprocal(closure)
        conjugator = freegroup(freegroup_dimension, conjugator_length_method, conjugator_length_parameters)
        new_result = result + conjugate(factor, conjugator)
        new_result = normalize(new_result)
        if len(new_result) > length:
            break
        result = new_result

    return result


def __random_bracket_sequence(n):
    """Generates a balanced sequence of n +1s and n -1s corresponding to correctly nested brackets."""
    # Source: https://gist.github.com/rygorous/d57941fa5ae6beb59f17bc30793d3d75
    # "Generating binary trees at random", Atkinson & Sack, 1992

    seq = [-1, 1] * n
    random.shuffle(seq)
    
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
    freegroup_dimension: int, 
    depth_method: str = 'uniform',
    depth_parameters = {'radius': 5},
    proba_conjugation: float = None,
    proba_closure: float = None,
    mind_reduction: bool = True,
):
    assert not proba_conjugation is None or not proba_closure is None
    if not proba_conjugation is None:
        proba_closure = 1 - proba_conjugation
    if not proba_closure is None:
        proba_conjugation = 1 - proba_closure    
    
    seq = __random_bracket_sequence(random_length(depth_method, **depth_parameters))
    
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
        
        coin = random.random()
        
        if (mind_reduction and idx + 1 == match_idx) or (coin <= proba_closure):
            split = random.randint(low = 0, high = len(closure))
            left, right = closure[:split], closure[split:]
            if random.random() < 0.5: left, right = right, left
            if random.random() < 0.5: left, right = reciprocal(left), reciprocal(right)
            sampled[idx] = left
            sampled[match_idx] = right
            continue
        coin -= proba_closure

        if coin <= proba_conjugation:
            probas = ones(2 * freegroup_dimension + 1)
            probas[0 + freegroup_dimension] = 0
            
            if mind_reduction:
                i = idx - 1
                while i >= 0 and not sampled[i] is None:
                    if not sampled[i]: i -= 1; continue
                    probas[-sampled[i][-1] + freegroup_dimension] = 0; break

                i = idx + 1
                while i <= len(seq) - 1 and not sampled[i] is None:
                    if not sampled[i]: i += 1; continue
                    probas[-sampled[i][0] + freegroup_dimension] = 0

                i = match_idx - 1
                while i >= 0 and not sampled[i] is None:
                    if not sampled[i]: i -=1; continue
                    probas[sampled[i][-1] + freegroup_dimension] = 0; break

                i = match_idx + 1
                while i <= len(seq) - 1 and not sampled[i] is None:
                    if not sampled[i]: i += 1; continue
                    probas[sampled[i][0] + freegroup_dimension] = 0; break
            
            probas /= probas.sum()
            conjugator = random.choice(2 * freegroup_dimension + 1, p = probas)
            sampled[idx] = [conjugator - freegroup_dimension]
            sampled[match_idx] = [freegroup_dimension - conjugator]
            continue
        coin -= proba_conjugation
        
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
    **params,
):
    def p_mult(): return params.get('p_mult', 0.)
    def p_comm(): return params.get('p_comm', 1.)
    
    if len(words) == 0: return []
    if len(words) == 1: return words[0]
    
    coin = random.random()
    if coin <= p_mult():
        return Mult(words)
    coin -= p_mult()
    
    if coin <= p_comm():
        idx = random.randint(1, len(words))
        return Comm([random_tree(words[:idx], **params), random_tree(words[idx:], **params)])
    coin -= p_comm()
    
    assert p_mult() + p_comm() == 1.
