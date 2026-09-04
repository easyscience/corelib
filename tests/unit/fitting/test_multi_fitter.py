# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from easyscience import ObjBase
from easyscience import Parameter
from easyscience import global_object
from easyscience.base_classes import EasyList
from easyscience.base_classes import ModelBase
from easyscience.fitting.fitter import Fitter
from easyscience.fitting.multi_fitter import MultiFitter


class Line(ModelBase):
    def __init__(self, m_val: float, c_val: float):
        super().__init__()
        self._m = Parameter('m', m_val)
        self._c = Parameter('c', c_val)

    @property
    def m(self) -> Parameter:
        return self._m

    @m.setter
    def m(self, value: float) -> None:
        self._m.value = value

    @property
    def c(self) -> Parameter:
        return self._c

    @c.setter
    def c(self, value: float) -> None:
        self._c.value = value

    def __call__(self, x):
        return self.m.value * x + self.c.value


class LegacyLine(ObjBase):
    """Deprecated-hierarchy model; no longer accepted by MultiFitter."""

    m: Parameter
    c: Parameter

    def __init__(self, m_val: float, c_val: float):
        super().__init__('line', m=Parameter('m', m_val), c=Parameter('c', c_val))

    def __call__(self, x):
        return self.m.value * x + self.c.value


class TestMultiFitter:
    @pytest.fixture
    def multi_fitter(self, monkeypatch):
        monkeypatch.setattr(Fitter, '_update_minimizer', MagicMock())
        fit_object_1 = Line(1.0, 0.5)
        fit_object_2 = Line(2.0, 1.5)
        return MultiFitter([fit_object_1, fit_object_2], [fit_object_1, fit_object_2])

    def test_fit_progress_callback(self, multi_fitter: MultiFitter):
        # When
        multi_fitter._precompute_reshaping = MagicMock(
            return_value=('x_fit', 'x_new', 'y_new', 'weights', 'dims')
        )
        multi_fitter._fit_function_wrapper = MagicMock(return_value='wrapped_fit_function')
        multi_fitter._post_compute_reshaping = MagicMock(return_value='fit_result')
        multi_fitter._minimizer = MagicMock()
        multi_fitter._minimizer.fit = MagicMock(return_value='result')
        progress_callback = MagicMock()

        # Then
        result = multi_fitter.fit(
            ['x_1', 'x_2'],
            ['y_1', 'y_2'],
            ['weights_1', 'weights_2'],
            progress_callback=progress_callback,
        )

        # Expect
        assert result == 'fit_result'
        multi_fitter._minimizer.fit.assert_called_once_with(
            'x_fit',
            'y_new',
            weights='weights',
            tolerance=None,
            max_evaluations=None,
            progress_callback=progress_callback,
        )


# ===================================================================
# MultiFitter._post_compute_reshaping
# ===================================================================


class TestPostComputeReshaping:
    def test_splits_single_result_into_two(self):
        """Verify _post_compute_reshaping splits combined result by dataset."""
        import numpy as np

        from easyscience.fitting.minimizers import FitResults

        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        # Simulate a combined result
        combined = FitResults()
        combined.success = True
        combined.x = np.array([0, 1, 2])
        combined.y_obs = np.array([0.5, 1.0, 1.5])
        combined.y_calc = np.array([0.51, 0.99, 1.51])
        combined.y_err = np.array([0.01, 0.02, 0.03])
        combined.p = {'pm': 1.0, 'pc': 0.5}
        combined.p0 = {'pm': 0.0, 'pc': 0.0}
        combined.n_evaluations = 100
        combined.iterations = 50
        combined.message = 'success'
        combined.minimizer_engine = 'bumps'
        combined.engine_result = 'engine_result'

        # Set dependent_dims as _precompute_reshaping would
        mf._dependent_dims = [(2,), (1,)]

        x_input = [np.array([0.0, 1.0]), np.array([2.0])]
        y_input = [np.array([0.5, 1.0]), np.array([1.5])]

        results = mf._post_compute_reshaping(combined, x_input, y_input)

        assert len(results) == 2
        assert results[0].success is True
        assert results[1].success is True
        assert len(results[0].y_obs) == 2
        assert len(results[1].y_obs) == 1
        assert results[0].total_results is combined
        assert results[1].total_results is combined
        assert np.allclose(results[0].y_calc, [0.51, 0.99])

    def test_handles_single_dataset(self):
        import numpy as np

        from easyscience.fitting.minimizers import FitResults

        fit_objects = [Line(1.0, 0.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        combined = FitResults()
        combined.success = True
        combined.y_obs = np.array([1.0, 2.0, 3.0])
        combined.y_calc = np.array([1.1, 2.1, 3.1])
        combined.y_err = np.array([0.1, 0.1, 0.1])
        combined.p = {'pm': 1.0}
        combined.p0 = {'pm': 0.0}
        combined.n_evaluations = 50
        combined.iterations = 25
        combined.message = 'ok'
        combined.minimizer_engine = 'bumps'
        combined.engine_result = 'er'

        mf._dependent_dims = [(3,)]
        x_input = [np.array([0.0, 1.0, 2.0])]
        y_input = [np.array([1.0, 2.0, 3.0])]

        results = mf._post_compute_reshaping(combined, x_input, y_input)

        assert len(results) == 1
        assert np.allclose(results[0].y_calc, [1.1, 2.1, 3.1])


# ===================================================================
# MultiFitter._fit_function_wrapper
# ===================================================================


class TestFitFunctionWrapper:
    def test_fit_function_restored_with_multiple_datasets(self):
        """Wrapping must not leave ``fit_function`` pointing at the last
        dataset's function (regression: sampling on a 2+ dataset MultiFitter
        silently swapped the user-visible fit function)."""
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        mf = MultiFitter(fit_objects, fit_objects)
        original = mf.fit_function
        assert original is fit_objects[0]

        x = [np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0])]
        mf._dependent_dims = [(3,), (2,)]
        wrapped = mf._fit_function_wrapper(x, flatten=True)

        assert mf.fit_function is original

        # Each wrapped section still evaluates its own dataset's function.
        y = wrapped(np.zeros(5))
        expected = np.hstack([fit_objects[0](x[0]), fit_objects[1](x[1])])
        assert np.allclose(y, expected)


# ===================================================================
# MultiFitter._precompute_reshaping with weights=None
# ===================================================================


class TestPrecomputeReshaping:
    def test_weights_all_none_returns_none(self):
        """When all weights are None, _precompute_reshaping should return None for weights."""
        import numpy as np

        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        mf = MultiFitter(fit_objects, fit_objects)

        x = [np.array([1.0, 2.0]), np.array([3.0])]
        y = [np.array([1.5, 2.5]), np.array([4.5])]
        weights = [None, None]

        x_fit, x_new, y_new, w_new, dims = MultiFitter._precompute_reshaping(
            x, y, weights, vectorized=False
        )

        assert w_new is None
        assert len(dims) == 2


# ===================================================================
# The EasyList container replacing the deprecated CollectionBase
# ===================================================================


class TestFitObjectContainer:
    def test_no_collection_base_deprecation_warning(self, caplog):
        """Building a MultiFitter must not warn about CollectionBase.

        The assertion is on message content rather than on the logger,
        because other deprecated classes warn on the very same logger.
        """
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]

        with caplog.at_level(logging.WARNING, logger='easyscience'):
            MultiFitter(fit_objects, fit_objects)

        assert not [r for r in caplog.records if 'CollectionBase is deprecated' in r.message]

    def test_container_is_an_easy_list(self):
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]

        assert isinstance(MultiFitter(fit_objects, fit_objects).fit_object, EasyList)

    def test_fit_objects_are_not_retyped(self):
        """The container must not reclassify the caller's fit objects.

        The old CollectionBase dummy re-typed every fit object as
        'created_internal', hiding the caller's own objects from the
        map's 'created' set.
        """
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        types_before = [global_object.map.find_type(obj) for obj in fit_objects]

        MultiFitter(fit_objects, fit_objects)

        assert [global_object.map.find_type(obj) for obj in fit_objects] == types_before
        assert all('created_internal' not in types for types in types_before)

    def test_sequence_contract(self):
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        container = MultiFitter(fit_objects, fit_objects).fit_object

        assert container[0] is fit_objects[0]
        assert container[1] is fit_objects[1]
        assert container[-1] is fit_objects[1]
        assert list(container[0:1]) == [fit_objects[0]]
        assert len(container) == 2
        assert list(container) == fit_objects
        assert fit_objects[0] in container
        assert container[fit_objects[0].unique_name] is fit_objects[0]

    def test_get_fit_parameters_concatenates_in_order(self):
        fit_objects = [Line(1.0, 0.5), Line(2.0, 1.5)]
        container = MultiFitter(fit_objects, fit_objects).fit_object

        assert container.get_fit_parameters() == [
            *fit_objects[0].get_fit_parameters(),
            *fit_objects[1].get_fit_parameters(),
        ]

    def test_rejects_foreign_object(self):
        with pytest.raises(TypeError, match='Items must be one of'):
            MultiFitter([Line(1.0, 0.5), 'not a model'], [None, None])

    def test_rejects_bare_parameter(self):
        """CollectionBase accepted bare parameters; EasyList does not."""
        model = Line(1.0, 0.5)
        with pytest.raises(TypeError, match='Items must be one of'):
            MultiFitter([model, Parameter('p', 1.0)], [model, None])

    def test_flattens_nested_list(self):
        """Nested lists are flattened, as CollectionBase did."""
        models = [Line(1.0, 0.5), Line(2.0, 1.5)]

        container = MultiFitter([models[0], [models[1]]], models).fit_object

        assert list(container) == models

    def test_accepts_fit_objects_as_a_tuple(self):
        """Any sequence works, not just a list.

        easyreflectometry passes its models as a tuple; a container that
        only flattened lists wrapped the whole tuple as a single item.
        """
        models = (Line(1.0, 0.5), Line(2.0, 1.5))

        container = MultiFitter(models, list(models)).fit_object

        assert list(container) == list(models)

    def test_rejects_legacy_obj_base_fit_objects(self):
        """Support for the deprecated ObjBase hierarchy was dropped."""
        legacy = [LegacyLine(1.0, 0.5), LegacyLine(2.0, 1.5)]

        with pytest.raises(TypeError, match='Items must be one of'):
            MultiFitter(legacy, legacy)
