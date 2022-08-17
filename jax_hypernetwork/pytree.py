"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import math
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

PyTree = Any


class PytreeReshaper:
    def __init__(self, tree_shapes: PyTree):
        self.shapes, self.treedef = jtu.tree_flatten(
            tree_shapes, is_leaf=is_tuple_of_ints
        )
        sizes = [math.prod(shape) for shape in self.shapes]

        self.split_indeces = list(np.cumsum(sizes)[:-1])
        self.num_elements = sum(sizes)

    def __call__(self, array_flat: jnp.ndarray):
        arrays_split = jnp.split(array_flat, self.split_indeces)
        arrays_reshaped = [a.reshape(shape) for a, shape in zip(arrays_split, self.shapes)]

        return jtu.tree_unflatten(self.treedef, arrays_reshaped)


def is_tuple_of_ints(x: Any):
    return isinstance(x, tuple) and all(isinstance(v, int) for v in x)
