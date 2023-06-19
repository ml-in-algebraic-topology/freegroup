# Installation

- `setuptools`.
    - `cd` to the cloned repository.
    - Run `python -m pip install -r requirements.txt`
    - Run `python -m pip install .`

# Definitions

- **generator** is a free group *letter*, word of length 1, i. e. `1` is a free group word `x`, and `-1` is a free group word `X`
- **word** is a `list` of either **generator** or **commutator**, i.e. `list(1, -2, 3)` is a free group word `xYz`
- also **word** is either **Mult** or **Comm**. Mult is a multiplication of words' list, Comm is an iterable commutator of words' list. And **Mult** and **Comm** can be nested. 

Examples:
```py
Comm([1, 2], [2])          # [xy, y] = YXyxyyy
Mult([-1],Comm([2], [-1])) # X[y, X] = XYxyX
```

# `freegroup.tools`
All methods have `batch_` version that accepts a list of `word`s
- methods `to_string(word, method)` and `from_string(word, method)`. Operate with string representations of a given **word** using chosen *method*. Methods: 
    - 'lu', 'lowerupper'. `i` maps to a lowercase latin letter, `-i` maps to an uppercase latin letter.
    - 'int', 'integer'. `i` and `-i` map to string representations of these numbers
    - 'su', 'superscript'. `i` maps to a lowercase latin letter, `-i` maps to this latin letter with superscript of -1.

- method `flatten(word)` computes all commutators inside given **word** and flattens inner multiplications.
    ```py
    from freegroup.tools import flatten, from_string
    a = from_string('Z[X, y]xz', method='lu')
    assert flatten(a) == [-3, 1, -2, -1, 2, 1, 3]
    ``` 
- method `reciprocal(word)`. Intverts the given **word**.
  ```py
  from freegroup.tools import reciprocal
  a = from_string('Z[X, y]xz', method = 'lu')
  assert reciprocal(a) == Mult([[-3, -1], Comm([[2], [-1]]), [3]])
  ```
 - method `normalize(word)`. Normalizes the given **word**, i. e. reduces `i` and `-i`, ...
    ```py
    from freegroup.tools import normalize
    assert normalize(Mult([[-1], [1, 2], [1]])) == [2, 1]
    ```
 - method `reduce_modulo_singleton_normal_closure(word, closure)`. Reduces and removes all trivial words by modulo of `closure`, which is a `list` of **generator**s
    ```py
    from freegroup.tools import reduce_modulo_singleton_normal_closure
    assert reduce_modulo_singleton_normal_closure([2, 1, 1, 2, 3, -2, 2, 3, 1, 1], [1, 2, 3]) == [2, 1, -2, 1]
    ```
- method `is_from_singleton_normal_closure(word, closure)`. Checks wether the given word is from normal closure.
  ```py
  from freegroup.tools import is_from_singleton_normal_closure
  assert is_from_singleton_normal_closure([-3, 1, -2, -1, 2, 3], [1]) == True
  ```

# `freegroup.sampling`
All samplers have `_generator` version for infinite iterable
This module helps to build **word** samplers for generating datasets
- `random_length(method = Either ['uniform', 'uniform_radius', 'constant'] or custom_function, **params)`. Returns a number from the given distribution. One can pass custom distribuiton in `method` parameter.
- `freegroup(freegroup_dimension, length_method, length_parameters)`. Infinite generator of non-reducible words from free group on `freegroup_dimension` **generator**s
- `normal_closure(method = ['conjugation', 'brackets'], closure, freegroup_dimension, **params)`. Random word from the normal closure `<closure>`
  ```py
  from freegroup.sampling import normal_closure
  generator = normal_closure('brackets', [1], 4, depth_method = 'uniform', depth_parameters = {'radius': 10})  
  ```
  `generator` will produce **word**s from `<x>` with uniform length from 2 to 20
