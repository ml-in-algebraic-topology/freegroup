import freegroup._derivatives as cpp
from ..tools import normalize, flatten, batch_normalize, batch_flatten, from_string, batch_from_string

from numpy import array, pad, ndarray, isscalar


def to_numpy(arg, keepdim = True, **kwargs):
    if isinstance(arg, ndarray):
        return arg
    
    method = kwargs.pop('from_string_method', 'lu')
    ndim = 0
    
    if isinstance(arg, str): ndim = 1; words = [from_string(arg, method = method)]
        
    elif isinstance(arg, list) or isinstance(arg, tuple):
        assert len(arg) > 0, "empty word!"
        
        if isinstance(arg[0], str): ndim = 2; words = batch_from_string(arg, method = method)
        
        elif isscalar(arg[0]): ndim = 1; words = [arg]
        
        else: ndim = 2; words = arg
    
    words = batch_normalize(batch_flatten(words))
    n = max(map(len, words))
    arr = array(list(map(lambda v: pad(v, (0, n - len(v))), words)))
    
    return arr if keepdim else arr.reshape(-1) if ndim == 1 else arr
    

def magnus_coefficients(X, fdim, gamma, keepdim = True, **kwargs):
    X = to_numpy(X, **kwargs)

    result = cpp.magnus_coefficients(X, fdim, gamma)
    if not keepdim and result.shape[0] == 1:
        return result.reshape(-1)
    return result


def derivative(X, fdim, wrt, keepdim = True, **kwargs):
    X = to_numpy(X, **kwargs)
    wrt = to_numpy(wrt, **kwargs)
    
    assert (wrt > 0).all(), "`with respect to` array should contain only positive indexes"
    
    result = cpp.derivative(X, fdim, wrt)
    if not keepdim and result.shape[0] == 1:
        return result.reshape(-1)
    return result
    

def max_gamma_contains(X, fdim, keepdim = True, **kwargs):
    X = to_numpy(X, **kwargs)
    
    result = cpp.max_gamma_contains(X, fdim)
    if not keepdim and result.shape[0] == 1:
        return result[0]
    return result





    