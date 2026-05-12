# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from easyscience import global_object
from easyscience.models.polynomial import Polynomial


@pytest.mark.fast
def test_polynomial_smoke() -> None:
    """Exercise a minimal user-facing model workflow."""
    global_object.map._clear()

    model = Polynomial(name='smoke', coefficients=[1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])

    assert [coefficient.name for coefficient in model.coefficients] == ['c0', 'c1', 'c2']
    assert np.allclose(model(x), np.polyval([1.0, 2.0, 3.0], x))
