"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from typing import Optional, Tuple, Union


def dict_filter(d: dict, key: str, all_but_key: Optional[bool] = False):
    """
    Filter a dictionary by key returing only entries that contain the key.
    Returns the complement if all_but_key=True.
    """

    def match_key_tuples(key1: Union[str, Tuple[str]], key2: Tuple[str]):
        """
        Check if key1 is contained in key2.
        """
        if isinstance(key1, str):
            return key1 in key2
        else:
            return all(k1 in k2 for k1, k2 in zip(key1, key2))

    d_flat = flatten_dict(d)
    if not all_but_key:
        d_flat_filtered = {k: v for k, v in d_flat.items() if match_key_tuples(key, k)}
    else:
        d_flat_filtered = {k: v for k, v in d_flat.items() if not match_key_tuples(key, k)}

    d_filtered = unflatten_dict(d_flat_filtered)

    return d_filtered


def flatten_dict(d: dict, parent_key: Optional[str] = ''):
    """
    Flatten nested dictionary combining keys into tuples.
    """
    items = []
    for k, v in d.items():
        new_key = (parent_key, k) if parent_key else k

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten_dict(d_flat: dict):
    """
    Unflatten a dictionary from tuple keys.
    """
    assert isinstance(d_flat, dict)
    result = dict()
    for path, value in d_flat.items():
        cursor = result
        for key in path[:-1]:
            if key not in cursor:
                cursor[key] = dict()
            cursor = cursor[key]

        cursor[path[-1]] = value

    return result
