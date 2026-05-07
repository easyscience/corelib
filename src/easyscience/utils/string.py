# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

from fractions import Fraction
from typing import Any

"""
This module provides utility classes for string operations.
"""


def transformation_to_string(
    matrix: Any,
    translation_vec: tuple[Any, Any, Any] = (0, 0, 0),
    components: tuple[str, str, str] = ('x', 'y', 'z'),
    c: str = '',
    delim: str = ',',
) -> str:
    """
    Convenience method.

    Given matrix returns string, e.g. x+2y+1/4

    Parameters
    ----------
    matrix : Any
        Transformation matrix.
    translation_vec : tuple[Any, Any, Any], default=(0, 0, 0)
        By default, (0, 0, 0).
    components : tuple[str, str, str], default=('x', 'y', 'z')
        Either ('x', 'y', 'z') or ('a', 'b', 'c'). By default, ('x',
        'y', 'z').
    c : str, default=''
        Optional additional character to print (used for magmoms). By
        default, ''.
    delim : str, default=','
        Delimiter. By default, ','.

    Returns
    -------
    str
        Xyz string.
    """
    parts = []
    for i in range(3):
        s = ''
        m = matrix[i]
        t = translation_vec[i]
        for j, dim in enumerate(components):
            if m[j] != 0:
                f = Fraction(m[j]).limit_denominator()
                if s != '' and f >= 0:
                    s += '+'
                if abs(f.numerator) != 1:
                    s += str(f.numerator)
                elif f < 0:
                    s += '-'
                s += c + dim
                if f.denominator != 1:
                    s += '/' + str(f.denominator)
        if t != 0:
            s += ('+' if (t > 0 and s != '') else '') + str(Fraction(t).limit_denominator())
        if s == '':
            s += '0'
        parts.append(s)
    return delim.join(parts)
