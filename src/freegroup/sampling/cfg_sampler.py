import numpy as np
from nltk.grammar import CFG, Nonterminal, Production
from freegroup.tools import generators, reciprocal
from freegroup.tools.helper import remove_prefix
from itertools import product
import pickle
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from typing import List, Dict, Union, Tuple


class CFGNormalClosureSamplerError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CFGNormalClosureSampler:
    
    Tree = List[List[Union[Tuple, int]]]
    PrecomputedProbas = List[float]
    
    def _generate_filaname(closure: List[int], fdim: int):
        return f'<{",".join(map(str, closure))}>_{fdim}.pickle'
    
    def _default_cache_dir_and_filename(closure: List[int], fdim: int, filename: str = None, cache_dir: Path = None):
        if cache_dir is None: cache_dir = Path(__file__).parents[0]
        if filename is None: filename = CFGNormalClosureSampler._generate_filaname(closure, fdim)
        return filename, cache_dir
    
    def load(closure: List[int], fdim: int, filename: str = None, cache_dir: Path = None, **init_kwargs):
        filename, cache_dir = CFGNormalClosureSampler._default_cache_dir_and_filename(closure, fdim, filename, cache_dir)
        
        if not (cache_dir / filename).is_file():
            raise ValueError(f'There is no file: {cache_dir / filename}')
        
        with open(cache_dir / filename, 'rb') as file:
            tree, closure, by_nonterminals, by_productions, by_splits = pickle.load(file)
        
        return CFGNormalClosureSampler(tree, closure, by_nonterminals, by_productions, by_splits, **init_kwargs)
        
    def build(
        closure: List[int], fdim: int, max_length: int = 50,
        filename: str = None, cache_dir: Path = None, load_cache: bool = True, overwrite: bool = True, save: bool = True,
        **kwargs
    ):
        kwargs = deepcopy(kwargs)
        precompute_kwargs = remove_prefix("precompute", kwargs)
        init_kwargs = remove_prefix("init", kwargs)
        
        filename, cache_dir = CFGNormalClosureSampler._default_cache_dir_and_filename(closure, fdim, filename, cache_dir)
        
        if (cache_dir / filename).is_file() and load_cache:
            return CFGNormalClosureSampler.load(closure, fdim, filename = filename, cache_dir = cache_dir, **init_kwargs)
        
        if (cache_dir / filename).is_file() and not load_cache and not overwrite and save:
            raise ValueError('Either set `overwrite` to `True` or set `save` to `False` or provide another path')
        
        grammar = __build_grammar__(closure, fdim)
        tree = __grammar_to_primitive_tree__(grammar.start(), __reachable__(grammar.chomsky_normal_form()))
        by_nonterminals, by_productions, by_splits = __precompute_derivations__(tree, max_length = max_length, **precompute_kwargs)
        
        init_kwargs['normalize'] = True
        sampler = CFGNormalClosureSampler(tree, closure, by_nonterminals, by_productions, by_splits, **init_kwargs)
        
        if not save: return sampler
        
        with open(cache_dir / filename, 'wb') as file:
            pickle.dump((sampler._tree, sampler._closure, sampler._by_nonterminals, sampler._by_productions, sampler._by_splits), file)
        
        return sampler
    
    def _validate(by_nonterminals, by_productions, by_splits):
        assert len(by_nonterminals) == len(by_productions) and \
            len(by_productions) == len(by_splits),\
            "All lengths accross `length` dimensions should coincide"
        
        n_nonterminals = len(by_nonterminals[0])
        assert all([len(x) == n_nonterminals for x in by_nonterminals]), \
            "Number of nonterminals should coincide accorss various `length`s"
        
        # TODO
    
    def _normalize(by_nonterminals, by_productions, by_splits):
        
        max_length = len(by_nonterminals)
        n_nonterminals = len(by_nonterminals[0])
        
        for length in range(max_length):
            for nt in range(n_nonterminals):
                if by_nonterminals[length][nt] == 0: continue
                n_productions = len(by_productions[length][nt])
                for idx in range(n_productions):
                    if by_productions[length][nt][idx] == 0: continue
                    for k in range(length):
                        by_splits[length][nt][idx][k] /= by_productions[length][nt][idx]
                    by_productions[length][nt][idx] /= by_nonterminals[length][nt]
    
    def __init__(
        self,
        tree: Tree,
        closure: List[int],
        by_nonterminals: List[List[int]],
        by_productions: List[List[PrecomputedProbas]],
        by_splits: List[List[List[PrecomputedProbas]]],
        normalize = False,
    ):
        self._tree = tree
        self._closure = closure
        self._by_nonterminals = by_nonterminals
        self._by_productions = by_productions
        self._by_splits = by_splits
        
        CFGNormalClosureSampler._validate(
            self._by_nonterminals,
            self._by_productions,
            self._by_splits,
        )
        
        self._max_length = len(self._by_nonterminals)
        
        if normalize:
            
            self._by_nonterminals = deepcopy(by_nonterminals)
            self._by_productions = deepcopy(by_productions)
            self._by_splits = deepcopy(by_splits)
            
            CFGNormalClosureSampler._normalize(
                self._by_nonterminals,
                self._by_productions,
                self._by_splits
            )
            
        
    def __call__(self, length: int, **kwargs):
        return self.sample(length, nt = 0, **kwargs)
    
    
    def _overlength_choice(self, length: int, nt: int, rng):
        possible_transitions = []
        for k, p in product(range(length - 1), self._tree[nt]):
            if isinstance(p, int):
                continue
            if len(self._closure) % 2 == 0 and (k + 1) % 2 != 0:
                continue

            b, c = p
            if self._by_nonterminals[min(self._max_length, k + 1) - 1][b] == 0 or\
                self._by_nonterminals[min(self._max_length, length - k - 1) - 1][c] == 0:
                continue
            possible_transitions.append((b, c, k))
        
        return rng.choice(possible_transitions)
        
        
    def sample(self, length: int, nt: int, verbose: int = 0, rng = None):
        
        if length <= 0:
            raise CFGNormalClosureSamplerError('length should be greater than 0')
            
        if self._by_nonterminals[min(self._max_length, length) - 1][nt] == 0:
            raise CFGNormalClosureSamplerError(f'there is no words of length: {length}')

        if rng is None:
            rng = np.random.default_rng()
        if isinstance(rng, int):
            rng = np.random.default_rng(seed = rng)

        if length == 1:
            return [rng.choice([p for p in self._tree[nt] if isinstance(p, int)])]
        
        if length > self._max_length:
            if verbose > 0:
                print('WARNING: Using overlength sampling strategy')
        
            b, c, k = self._overlength_choice(length, nt, rng)
        else:
            p_idx = rng.choice(len(self._tree[nt]), p = self._by_productions[length - 1][nt])
            k     = rng.choice(length - 1, p = self._by_splits[length - 1][nt][p_idx])
            b, c = self._tree[nt][p_idx]
            
        left = self.sample(k + 1, b, verbose = verbose, rng = rng)
        right = self.sample(length - (k + 1), c, verbose = verbose, rng = rng)
        return left + right

        


def __grammar_to_primitive_tree__(start: Nonterminal, productions: List[Production]) -> CFGNormalClosureSampler.Tree:
    nt2idx, nonterminals = {}, set()
    
    for p in productions:
        nonterminals.add(p.lhs())
        
    for idx, nt in enumerate(sorted(nonterminals, key = lambda x: 0 if start == x else 1)):
        nt2idx[nt] = idx
        
    tree = [[] for _ in range(len(nonterminals))]
    
    for p in productions:
        if len(p.rhs()) == 1:
            tree[nt2idx[p.lhs()]].append(p.rhs()[0])
        else:
            tree[nt2idx[p.lhs()]].append(tuple(map(nt2idx.get, p.rhs())))
            
    assert nt2idx[start] == 0
    
    return tree
    

def __precompute_derivations__(tree: CFGNormalClosureSampler.Tree, max_length: int = 10, dtype = int):
    by_nonterminals, by_productions, by_splits = [], [], []
        
    for length in range(max_length):
        by_nonterminals.append([None for _ in range(len(tree))])
        by_productions.append([None for _ in range(len(tree))])
        by_splits.append([None for _ in range(len(tree))])
        
        for a, ps in enumerate(tree):
            by_nonterminals[-1][a] = dtype(0)
            by_productions[-1][a] = [dtype(0) for p in ps]
            by_splits[-1][a] = [[dtype(0)] * length for p in ps]
            for idx, p in enumerate(ps):
                
                if length == 0 and isinstance(p, int):
                    by_nonterminals[-1][a] += dtype(1)
                    by_productions[-1][a][idx] += dtype(1)
                elif length > 0 and not isinstance(p, int):
                    b, c = p
                    for k in range(length):
                        num = by_nonterminals[k][b] * by_nonterminals[length - (k + 1)][c]
                        by_splits[-1][a][idx][k] = num
                        by_productions[-1][a][idx] += num
                        by_nonterminals[-1][a] += num
                
    return by_nonterminals, by_productions, by_splits
        

def __reachable__(g: CFG, v = None, used = None, productions_by_nonterminal=None):
    if productions_by_nonterminal is None:
        productions_by_nonterminal = defaultdict(lambda: [])
        for p in g.productions():
            productions_by_nonterminal[p.lhs()].append(p)
    
    productions = []
    
    if v is None: v = g.start()
    
    if used is None: used = set()
    if v in used: return []
    used.add(v)
    
    for p in productions_by_nonterminal[v]:
        productions.append(p)
        for u in p.rhs(): productions.extend(__reachable__(g, u, used, productions_by_nonterminal))
    
    return productions


def __build_grammar__(closure: List[int], fdim: int = 2) -> CFG:
    productions = []
    
    n = len(closure)
    splits = []
    _closure = reciprocal(closure)
    for shift, split in product(range(n), range(n)):
        shifted = closure[shift:] + closure[:shift]
        _shifted = _closure[shift:] + _closure[:shift]
        splits.append((shifted[:split + 1], shifted[split + 1:]))
        splits.append((_shifted[:split + 1], _shifted[split + 1:]))
        
    
    configurations = [(None, None)]
    for x in generators(fdim):
        configurations.append((x, None))
    for xi, xj in product(generators(fdim), generators(fdim)):
        configurations.append((xi, xj))
    
    
    nonterminals = {
        x: Nonterminal(f'{x}') for x in generators(fdim)
    }
    
    nonterminals.update({
        (l, r) : Nonterminal(f'{l}_{r}') for (l, r) in configurations
    })
    
    for x in generators(fdim):
        productions.append(Production(nonterminals[x], [x]))
    
    for (l, r) in configurations:
        
        # For every configuration of left letter and right
        # we first add productions associated
        # with certain shift and split of `closure`
        for splitl, splitr in splits:
            # We do not need rules like ySY -> Y YSy y
            # since 'y's would reduce
            if l == -splitl[0]: continue
                
            nl, nr = splitl[-1], None if not splitr else splitr[0]
            
            # S -> xyz
            if not splitr and r != -splitl[-1]:
                productions.append(Production(
                    nonterminals[(l, r)],
                    [nonterminals[x] for x in splitl]
                ))
            
            
            # S -> x xSy yz | xy ySz z | ... | zxy YS'r' ...
            if not splitr or -splitr[-1] != r:
                productions.append(Production(
                    nonterminals[(l, r)],
                    [
                        *[nonterminals[x] for x in splitl],
                        nonterminals[(nl, nr if not nr is None else r)],
                        *[nonterminals[x] for x in splitr]
                    ]
                ))

            # S -> x xSy yz zS | xy ySz z zS | ... | zxy YS ...
            if splitr:
                productions.append(Production(
                    nonterminals[(l, r)],
                    [
                        *[nonterminals[x] for x in splitl],
                        nonterminals[(nl, nr)],
                        *[nonterminals[x] for x in splitr],
                        nonterminals[(splitr[-1], r)]
                    ]
                ))
            
            
        # Then we add conjugation rules
        for x in generators(fdim):
            
            # If |closure| == 1, we do not want to add conjugation rules
            if n == 1 and (x == closure[0] or -x == closure[0]):
                continue
                
            # We do not need rules like ySY -> Y YSy y
            # since 'y's would reduce
            if (not l is None and l == -x) or (not r is None and r == x):
                continue
            
            productions.append(Production(
                nonterminals[(l, r)],
                [nonterminals[x], nonterminals[(x, -x)], nonterminals[-x]]
            ))
            
            productions.append(Production(
                nonterminals[(l, r)],
                [nonterminals[x], nonterminals[(x, -x)], nonterminals[-x], nonterminals[(-x, r)]]
            ))
    
    
    return CFG(start = nonterminals[(None, None)], productions = productions)
    
