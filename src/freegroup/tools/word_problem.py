from functools import reduce
from .tools import (
    normalize, reduce_modulo_normal_closure,
    generators
)
from collections import defaultdict
from copy import deepcopy


#current_depth = 0

class LetterWithSubscript:
    """Immutable letter with subscript"""
    
    def __init__(self, x, *subscript):
        self._x = x
        self._subscript = tuple(subscript)

    def __hash__(self):
        return hash((self._x, *self._subscript))

    @property
    def head(self):
        return self._x

    @property
    def sign(self):
        return 1 if self._x > 0 else -1

    @property
    def abs(self):
        return -self if self.sign < 0 else self

    @property
    def subscript(self):
        return self._subscript

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self._x == other._x and self._subscript == other._subscript

    def __neg__(self):
        return LetterWithSubscript(-self._x, *self._subscript)

    def __repr__(self):
        str_subscript = '_'.join([str(x) for x in self.subscript])
        return f"{self._x}" if not str_subscript else f"{self._x}_{str_subscript}"

    def __add__(self, other):
        return LetterWithSubscript(self.head, *self.subscript[:-1], self.subscript[-1] + other)

    def __sub__(self, other):
        return LetterWithSubscript(self.head, *self.subscript[:-1], self.subscript[-1] - other)

    def remove_subscript(self):
        return LetterWithSubscript(self.head, *self.subscript[:-1])

    def add_subscript(self, subscript):
        return LetterWithSubscript(self.head, *self.subscript, subscript)


def sign(x):
    return 1 if x > 0 else -1

def occurence(x, word):
    return sum([v.abs == x for v in word])

def exponent(x, word):
    return sum([1 if x == v else -1 if x == -v else 0 for v in word])

def split_by_T(word, T):
    if not word:
        return [[]], []
    ts, splits = [[]], []
    previous_t = True
    for v in word:
        if v.abs in T and previous_t:
            ts[-1].append(v)
        elif v.abs in T:
            ts.append([v])
        elif previous_t:
            splits.append([v])
        else:
            splits[-1].append(v)
        previous_t = v.abs in T
    if not word[-1].abs in T:
        ts.append([])

    return ts, splits

def cyclicaly_reduce(word):
    n = len(word)
    stack = []
    opt = None
    for idx, x in enumerate(word + word):
        ###print(idx, x, stack)
        if not stack:
            stack.append((idx, x))
        else:
            jdx, y = stack[-1]
            if y == -x:
                stack.pop()
            else:
                stack.append((idx, x))
        if not stack and idx >= n:
            return []
        elif stack:
            jdx, y = stack[0]
            if idx - jdx >= n:
                stack.pop(0)
            if idx >= n:
                opt = opt if opt is not None and len(stack) > len(opt) else stack[::]
    return [v for _, v in opt]


def _add_t_n_times(t, n, src):
    if n > 0:
        for _ in range(n): src.append(deepcopy(t))
    else:
        for _ in range(-n): src.append(deepcopy(-t))

def remove_subscript(word, t):
    w = []
    for v in word:
        _add_t_n_times(t, v.subscript[-1], w)
        w.append(v.remove_subscript())
        _add_t_n_times(-t, v.subscript[-1], w)
        
    return w



def propogate_t(cumsum, word, relator, t, x, a, b, cache=None):
    """HNN Normalization, remove pins"""
    #print('\t' * #current_depth +f'cumsum={cumsum}, word={word}, relator={relator}, t={t}, x={x}, a={a}, b={b}')

    if cache is None or not cache:
        cache = {'a': {}, 'b': {}}

    x_a, x_b = x.add_subscript(a), x.add_subscript(b)
    A = {v.abs for v in word + relator if v.abs != x_a}
    B = {v.abs for v in word + relator if v.abs != x_b}
    
    carry, iteration = 0, 0
    while cumsum != 0:
        A.update([v.abs for v in word if v.abs != x_a])
        B.update([v.abs for v in word if v.abs != x_b])

        #print('\t' * #current_depth +f'Starting {iteration} iteration...')
        
        if cumsum > 0 and any([v.abs == x_b for v in word]):
            #print('\t' * #current_depth +f"Falling into rewriting x_b with elements from B")
            if not tuple(word) in cache['b']:
                cache['b'][tuple(word)] = impl_reduce_word_problem(word, relator, B)
            word_prime = cache['b'][tuple(word)]
            #print('\t' * #current_depth +f"Got w'= {word_prime}")
            
            if word_prime and all([v.abs in B for v in word_prime]):
                word = word_prime
            else:
                break
        elif cumsum > 0:
            #print('\t' * #current_depth +f"Increasing subscript")
            
            word = [v + 1 for v in word]
            cumsum -= 1; carry += 1
        elif cumsum < 0 and any([v.abs == x_a for v in word]):
            #print('\t' * #current_depth +f"Falling into rewriting x_a with elements from A")
            if not tuple(word) in cache['a']:
                cache['a'][tuple(word)] = impl_reduce_word_problem(word, relator, A)
            word_prime = cache['a'][tuple(word)]
            #print('\t' * #current_depth +f"Got w'= {word_prime}")
            
            if word_prime and all([v.abs in A for v in word_prime]):
                cache[tuple(word)] = word_prime
                word = word_prime
            else:
                break
        elif cumsum < 0:
            #print('\t' * #current_depth +f"Decreasing subscript")
            
            word = [v - 1 for v in word]
            cumsum += 1; carry -= 1

        iteration += 1
        
    #print('\t' * #current_depth +f'Result: carry={carry}, word={word}')
    return carry, word


def psi(word, t, x, alpha, beta):
    w = []
    for v in word:
        if v.abs == x.abs:
            if v.sign > 0:
                w.append(x)
                _add_t_n_times(t, -alpha , w)
            else:
                _add_t_n_times(t, alpha, w)
                w.append(-x)
        elif v.abs == t.abs:
            _add_t_n_times(v, beta, w)
        else:
            w.append(v)
    return w


def psi_pseudo_inverse(word, t, x, alpha, beta):
    w = []
    for v in word:
        if not v.abs == x.abs:
            w.append(v)
        else:
            if v.sign > 0:
                w.append(x)
                _add_t_n_times(t, alpha, w)
            else:
                _add_t_n_times(t, -alpha, w)
                w.append(-x)
    return w
            

def magnus_is_from_normal_closure(word, relator, T=None):
    w = magnus_reduce_modulo_normal_closure(word, relator, T)
    return not w or all([x in T for x in w])


def magnus_reduce_modulo_normal_closure(word, relator, T=None):
    word = [LetterWithSubscript(x) for x in word]
    relator = [LetterWithSubscript(x) for x in relator]
    T = set() if T is None else set([abs(LetterWithSubscript(x)) for x in T])

    return impl_reduce_word_problem(word, relator, T)


def impl_reduce_word_problem(word, relator, T=None):
    # global #current_depth
    #current_depth += 1

    #print('\t' * #current_depth +f'word={word}, relator={relator}, T={T}')
    
    relator = cyclicaly_reduce(deepcopy(relator))
    word = normalize(deepcopy(word))

    T = deepcopy(T) if not T is None else set()

    if (not word) or all([v.abs in T for v in word]):
        return word

    #print('\t' * #current_depth +f'After normalizing word={word}, relator={relator}, T={T}')
    

    letter_to_positions = defaultdict(lambda: [])
    for idx, v in enumerate(relator):
        letter_to_positions[v.abs].append(idx)

    ##print('---- Call impl reduce word problem ----')
    ##print(word, relator, T, letter_to_positions)
        
    
    if all([v in T for v in letter_to_positions.keys()]):
        ##print('Case 3.5.1')
        flags, splits = zip(*split_by_T(word, T))
        redcued_by_T = [
            impl_reduce_word_problem(w, relator, T=set(), fdim=fdim) if flag else w for flag, w in zip(flags, splits)
        ]
        ##print(redcued_by_T)
        #current_depth -= 1
        return normalize(reduce(lambda x, y: x + y, redcued_by_T))
        

    if len(set(relator)) == 1:
        #print('\t' * #current_depth +f'Falling into base case with {relator}')
        #current_depth -= 1
        return reduce_modulo_normal_closure(word, relator)

    appropriate_ts = [v.abs for v in relator
                     if exponent(v, relator) == 0 and occurence(v, relator) > 0]

    if appropriate_ts:

        t = appropriate_ts[0]
        #print('\t' * #current_depth +f'Found letter with sigma = 0, t = {t}')
        #print('\t' * #current_depth +'Falling into case 3.5.2.1')
            
        # 3.5.2.1
        if t in T:
            # Find first position of 
            # letter that is not in T
            x_idx = [not v.abs in T for v in relator].index(True)
        else:
            # Find first position of 
            # letter that is not equal to 't'
            x_idx = [v.abs != t for v in relator].index(True)
        relator = relator[x_idx:] + relator[:x_idx]
        x = relator[0].abs
        assert t != x

        #print('\t' * #current_depth +f'Found x = {x}')
        #print('\t' * #current_depth +f'Now relator = {relator}')

        r_prime, carry = [], 0
        t_splits, splits = split_by_T(relator, {t})
        for ts, split in zip(t_splits, splits):
            carry += len(ts) if ts and ts[0].sign > 0 else -len(ts)
            r_prime.extend([v.add_subscript(carry) for v in split])    
        #print('\t' * #current_depth +f"r'={r_prime}")
        assert normalize(remove_subscript(r_prime, t)) == relator

        a = min([v.subscript[-1] for v in r_prime if v.abs.remove_subscript() == x])
        b = max([v.subscript[-1] for v in r_prime if v.abs.remove_subscript() == x])
        assert a <= 0 and b >= 0

        # cache = {}
        t_splits, splits = split_by_T(word, {t})
        cumsums = [len(w) if w and w[0].sign > 0 else -len(w) for w in t_splits]
        splits = [[v.add_subscript(0) for v in w] for w in splits]
        
        iteration = 0
        while True:
            #print('\t' * #current_depth +f'Removing pins before {iteration} iteration: cumsums={cumsums}, splits={splits}')
            
            carry, flag, cache = 0, False, {}
            cumsums_, splits_ = [], []
            for idx, (cumsum, split) in enumerate(zip(cumsums, splits)):
                carry_, w = propogate_t(carry + cumsum, split, r_prime, t, x, a, b, cache)
                
                if splits_ and carry + cumsum == carry_:
                    #print('\t' * #current_depth +f"Propogated all t's, so merging to splits")
                    splits_[-1].extend(w)
                else:
                    #print('\t' * #current_depth +f"Propogated NOT all t's")
                    cumsums_.append(carry + cumsum - carry_)
                    splits_.append(w)
                
                carry = carry_
                flag = flag or (carry_ != 0)

            cumsums_.append(
                carry + cumsums[-1] if len(cumsums) > len(splits) else 0)
            cumsums, splits = cumsums_, splits_
        
            if not flag:
                break
                
            iteration += 1

        #print('\t' * #current_depth +"Preparing recursive call")
        if t.abs in T:
            T_prime = {
                v.abs for w in [*splits, r_prime] for v in w
                if v.remove_subscript().abs in T
            }
        else:
            T_prime = {LetterWithSubscript(v.head, *v.subscript, 0) for v in T}
        #print('\t' * #current_depth +f"t in T? = {t.abs in T}, so T'={T_prime}")

        splits = [impl_reduce_word_problem(w, r_prime, T_prime) for w in splits]

        #print('\t' * #current_depth +"Combining all together")
        
        result = []
        for ts, w in zip(cumsums, splits):
            _add_t_n_times(t, ts, result)
            result.extend(remove_subscript(w, t))
        _add_t_n_times(t, cumsums[-1], result)
        result = normalize(result)
        
        #print('\t' * #current_depth +f"Result={result}")
        #current_depth -= 1
        
        return result

    #print('\t' * #current_depth +'There is no letter with sigma = 0')
    #print('\t' * #current_depth +'Falling into case 2')
        
    try:
        t, x = [v for v in letter_to_positions.keys() if not v in T][:2]
    except ValueError:
        t = [v for v in letter_to_positions.keys() if v in T][0]
        x = [v for v in letter_to_positions.keys() if not v == t][0]

    alpha = exponent(t, relator)
    beta  = exponent(x, relator)

    #print('\t' * #current_depth +f'Found t={t} with alpha={alpha}, x={x} with beta={beta}')

    r_prime = psi(relator, t, x, alpha, beta)
    w_prime = psi(word, t, x, alpha, beta)

    #print('\t' * #current_depth +f'psi(r)={r_prime}, psi(w)={w_prime}')
    
    result = impl_reduce_word_problem(w_prime, r_prime, T)
    if not t in T:
        # x not in result and t not in result, so no need for inverse
        result = normalize(result)
        
        #print('\t' * #current_depth +f't not in T, so returning {result}')
        #current_depth -= 1
        
        return result

    ws = []
    result = normalize(psi_pseudo_inverse(result, t, x, alpha, beta))
    ts, splits = split_by_T(result, {t})
    new_splits = [None] * (len(ts) + len(splits))
    new_splits[::2] = ts
    new_splits[1::2] = splits
    #print('\t'*#current_depth + f'Merging splits: {new_splits}')
    for i, split in enumerate(new_splits):
        if not split: continue
            
        if i % 2 != 0:
            ws.extend(split)
        
        # alpha? beta?
        elif split and (len(split) % abs(beta) == 0):
            _add_t_n_times(split[0], sign(beta) * len(split) // abs(beta), ws)
            # ws.extend(split[:(len(split) // abs(beta))])
        else:
            #print('\t'*#current_depth + 'Returning None, since len(split) is not divisible by beta')
            #current_depth -= 1
            return None
    ##print('Case 2 return:', ws)
    #print('\t' * #current_depth +f'Result: {ws}')
    #current_depth -= 1
    
    return ws
            