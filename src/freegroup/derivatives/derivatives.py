import freegroup._derivatives as cpp
from ..tools import normalize, flatten, batch_normalize, batch_flatten, from_string, batch_from_string

from numpy import array, pad, ndarray


def to_numpy(arg, keepdim = False, **kwargs):
    if isinstance(arg, ndarray):
        return arg
    
    method = kwargs.pop('from_string_method', 'lu')
    ndim = 0
    
    if isinstance(arg, str): ndim = 1; words = [from_string(arg, method = method)]
        
    elif isinstance(arg, list) or isinstance(arg, tuple):
        assert len(arg) > 0, "empty word!"
            
        if isinstance(arg[0], int): ndim = 1; words = [arg]
            
        if isinstance(arg[0], str): ndim = 2; words = batch_from_string(arg, method = method)
            
        if isinstance(arg[0], list) or isinstance(arg[0], tuple): ndim = 2; words = arg
    
    words = batch_normalize(batch_flatten(words))
    n = max(map(len, words))
    arr = array(list(map(lambda v: pad(v, (0, n - len(v))), words)))
    
    return arr if not keepdim else arr.reshape(-1) if ndim == 1 else arr
    

def magnus_coefficients(X, fdim, gamma, keepdim = False, **kwargs):
    X = to_numpy(X, **kwargs)

    result = cpp.magnus_coefficients(X, fdim, gamma)
    if keepdim and result.shape[0] == 1:
        return result.reshape(-1)
    return result


def derivative(X, fdim, wrt, keepdim = False, **kwargs):
    X = to_numpy(X, **kwargs)
    wrt = to_numpy(wrt, **kwargs)
    
    assert (wrt > 0).all(), "`with respect to` array should contain only positive indexes"
    
    result = cpp.derivative(X, fdim, wrt)
    if keepdim and result.shape[0] == 1:
        return result.reshape(-1)
    return result
    
    





    