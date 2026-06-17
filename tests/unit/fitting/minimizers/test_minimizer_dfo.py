# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

import easyscience.fitting.minimizers.minimizer_dfo
from easyscience.fitting.minimizers.minimizer_dfo import DFO
from easyscience.fitting.minimizers.minimizer_dfo import DFOCallbackState
from easyscience.fitting.minimizers.utils import FitError
from easyscience.variable import Parameter


class TestDFOFit:
    @pytest.fixture
    def minimizer(self) -> DFO:
        minimizer = DFO(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=MagicMock(package='dfo', method='leastsq'),
        )
        return minimizer

    def test_init(self, minimizer: DFO) -> None:
        assert minimizer._p_0 == {}
        assert minimizer.package == 'dfo'

    def test_init_exception(self) -> None:
        with pytest.raises(FitError):
            DFO(
                obj='obj',
                fit_function='fit_function',
                minimizer_enum=MagicMock(package='dfo', method='not_leastsq'),
            )

    def test_supported_methods(self, minimizer: DFO) -> None:
        # When Then Expect
        assert minimizer.supported_methods() == ['leastsq']

    def test_supported_methods(self, minimizer: DFO) -> None:
        # When Then Expect
        assert minimizer.supported_methods() == ['leastsq']

    def test_fit(self, minimizer: DFO) -> None:
        # When
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        cached_pars = {'mock_parm_1': cached_par}
        minimizer._cached_pars = cached_pars

        # Then
        result = minimizer.fit(x=1.0, y=2.0, weights=1)

        # Expect
        assert result == 'gen_fit_results'
        minimizer._dfo_fit.assert_called_once_with(
            cached_pars, mock_model, user_params={'logging.save_diagnostic_info': True}
        )
        minimizer._make_model.assert_called_once_with(
            parameters=None,
            callback=None,
        )
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit', 1)
        mock_model_function.assert_called_once_with(1.0, 2.0, 1)

    def test_fit_passes_callback_to_model_builder(self, minimizer: DFO) -> None:
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        minimizer._cached_pars = {'mock_parm_1': cached_par}

        callback = MagicMock()

        minimizer.fit(x=1.0, y=2.0, weights=1, callback=callback)

        minimizer._make_model.assert_called_once_with(
            parameters=None,
            callback=callback,
        )

    def test_fit_wraps_supplied_model_with_explicit_callback(self, minimizer: DFO) -> None:
        from easyscience import global_object

        global_object.stack.enabled = False

        supplied_model = MagicMock()
        wrapped_model = MagicMock()
        explicit_callback = MagicMock()

        minimizer._make_model = MagicMock()
        minimizer._wrap_model_with_callback = MagicMock(return_value=wrapped_model)
        minimizer._get_callback_parameter_names = MagicMock(return_value=['palpha'])
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')
        minimizer._cached_pars = {'alpha': MagicMock(value=1.0)}

        result = minimizer.fit(
            x=np.array([1.0]),
            y=np.array([2.0]),
            weights=np.array([1.0]),
            model=supplied_model,
            callback=explicit_callback,
        )

        assert result == 'gen_fit_results'
        minimizer._make_model.assert_not_called()
        minimizer._wrap_model_with_callback.assert_called_once_with(
            supplied_model,
            ['palpha'],
            explicit_callback,
        )
        minimizer._dfo_fit.assert_called_once_with(
            minimizer._cached_pars,
            wrapped_model,
            user_params={'logging.save_diagnostic_info': True},
        )

    def test_fit_uses_supplied_model_without_callback(self, minimizer: DFO) -> None:
        from easyscience import global_object

        global_object.stack.enabled = False

        supplied_model = MagicMock()

        minimizer._make_model = MagicMock()
        minimizer._wrap_model_with_callback = MagicMock()
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')
        minimizer._cached_pars = {'alpha': MagicMock(value=1.0)}

        result = minimizer.fit(
            x=np.array([1.0]),
            y=np.array([2.0]),
            weights=np.array([1.0]),
            model=supplied_model,
        )

        assert result == 'gen_fit_results'
        minimizer._make_model.assert_not_called()
        minimizer._wrap_model_with_callback.assert_not_called()
        minimizer._dfo_fit.assert_called_once_with(
            minimizer._cached_pars,
            supplied_model,
            user_params={'logging.save_diagnostic_info': True},
        )

    def test_generate_fit_function(self, minimizer: DFO) -> None:
        # When
        minimizer._original_fit_function = MagicMock(return_value='fit_function_result')

        minimizer._object = MagicMock()
        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        mock_parm_1.value = 1.0
        mock_parm_1.error = 0.1
        mock_parm_2 = MagicMock()
        mock_parm_2.unique_name = 'mock_parm_2'
        mock_parm_2.value = 2.0
        mock_parm_2.error = 0.2
        minimizer._object.get_fit_parameters = MagicMock(return_value=[mock_parm_1, mock_parm_2])

        # Then
        fit_function = minimizer._generate_fit_function()
        fit_function_result = fit_function([10.0])

        # Expect
        assert 'fit_function_result' == fit_function_result
        minimizer._original_fit_function.assert_called_once_with([10.0])
        assert minimizer._cached_pars['mock_parm_1'] == mock_parm_1
        assert minimizer._cached_pars['mock_parm_2'] == mock_parm_2

    @pytest.mark.parametrize(
        'weights',
        [
            np.array([1, 2, 3, 4]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.repeat(np.nan, 3),
            np.zeros(3),
            np.repeat(np.inf, 3),
            -np.ones(3),
        ],
        ids=['wrong_length', 'multidimensional', 'NaNs', 'zeros', 'Infs', 'negative'],
    )
    def test_fit_weight_exceptions(self, minimizer: DFO, weights) -> None:
        # When Then Expect
        with pytest.raises(ValueError):
            minimizer.fit(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), weights=weights)

    def test_make_model(self, minimizer: DFO) -> None:
        # When
        mock_fit_function = MagicMock(return_value=np.array([11, 22]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        mock_parm_1.value = 1000.0
        mock_parm_2 = MagicMock()
        mock_parm_2.unique_name = 'mock_parm_2'
        mock_parm_2.value = 2000.0

        # Then
        model = minimizer._make_model(parameters=[mock_parm_1, mock_parm_2])
        residuals_for_model = model(
            x=np.array([1, 2]),
            y=np.array([10, 20]),
            weights=np.array([1 / 100, 1 / 200]),
        )

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        assert all(np.array([-0.01, -0.01]) == residuals_for_model(np.array([1111, 2222])))
        assert all(mock_fit_function.call_args[0][0] == np.array([1, 2]))
        assert mock_fit_function.call_args[1] == {
            'pmock_parm_1': 1111,
            'pmock_parm_2': 2222,
        }

    def test_make_model_callback(self, minimizer: DFO) -> None:
        mock_fit_function = MagicMock(return_value=np.array([11, 22]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        mock_parm_1.value = 1000.0
        mock_parm_2 = MagicMock()
        mock_parm_2.unique_name = 'mock_parm_2'
        mock_parm_2.value = 2000.0

        callback = MagicMock()

        model = minimizer._make_model(parameters=[mock_parm_1, mock_parm_2], callback=callback)
        residuals_for_model = model(
            x=np.array([1, 2]),
            y=np.array([10, 20]),
            weights=np.array([1 / 100, 1 / 200]),
        )

        residuals = residuals_for_model(np.array([1111, 2222]))

        assert all(np.array([-0.01, -0.01]) == residuals)
        callback.assert_called_once()
        state = callback.call_args[0][0]
        assert isinstance(state, DFOCallbackState)
        assert state.evaluation == 1
        assert state.improved == True
        assert state.objective == pytest.approx(0.0002)
        assert all(state.xk == np.array([1111, 2222]))
        assert all(state.residuals == np.array([-0.01, -0.01]))
        assert state.parameters == {
            'pmock_parm_1': 1111.0,
            'pmock_parm_2': 2222.0,
        }
        assert all(state.best_xk == np.array([1111, 2222]))
        assert state.best_parameters == {
            'pmock_parm_1': 1111.0,
            'pmock_parm_2': 2222.0,
        }

    def test_make_model_without_parameters_uses_cached_parameters(self, minimizer: DFO) -> None:
        mock_fit_function = MagicMock(return_value=np.array([11.0]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)
        minimizer._cached_pars = {'alpha': MagicMock(value=1000.0)}

        model = minimizer._make_model()
        residuals_for_model = model(
            x=np.array([1.0]),
            y=np.array([10.0]),
            weights=np.array([0.5]),
        )

        residuals_for_model(np.array([1111.0]))

        assert mock_fit_function.call_args.kwargs == {'palpha': 1111.0}

    def test_set_parameter_fit_result_no_stack_status(self, minimizer: DFO):
        # When
        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
        }
        minimizer._cached_pars['a'].value = 'a'
        minimizer._cached_pars['b'].value = 'b'

        mock_fit_result = MagicMock()
        mock_fit_result.x = [1.0, 2.0]
        mock_fit_result.jacobian = 'jacobian'
        mock_fit_result.resid = 'resid'

        minimizer._error_from_jacobian = MagicMock(return_value=np.array([[0.1, 0.0], [0.0, 0.2]]))

        # Then
        minimizer._set_parameter_fit_result(mock_fit_result, False)

        # Expect
        assert minimizer._cached_pars['a'].value == 1.0
        assert minimizer._cached_pars['a'].error == 0.1
        assert minimizer._cached_pars['b'].value == 2.0
        assert minimizer._cached_pars['b'].error == 0.2
        minimizer._error_from_jacobian.assert_called_once_with('jacobian', 'resid', 0.95)

    def test_gen_fit_results(self, minimizer: DFO, monkeypatch):
        # When
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.EXIT_SUCCESS = 0
        mock_fit_result.flag = 0
        mock_fit_result.nf = 12
        mock_fit_result.diagnostic_info = {'iters_total': [3]}
        mock_fit_result.msg = 'Maximum function evaluations reached'

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'par_value_1'
        mock_cached_par_2 = MagicMock()
        mock_cached_par_2.value = 'par_value_2'
        minimizer._cached_pars = {
            'par_1': mock_cached_par_1,
            'par_2': mock_cached_par_2,
        }

        minimizer._p_0 = 'p_0'
        minimizer.evaluate = MagicMock(return_value='evaluate')

        # Then
        weights = np.array([2.0, 4.0])
        domain_fit_results = minimizer._gen_fit_results(
            mock_fit_result, weights, **{'kwargs_set_key': 'kwargs_set_val'}
        )

        # Expect
        assert domain_fit_results == mock_domain_fit_results
        assert domain_fit_results.kwargs_set_key == 'kwargs_set_val'
        assert domain_fit_results.success == True
        assert domain_fit_results.y_obs == 'y'
        assert domain_fit_results.x == 'x'
        assert domain_fit_results.p == {
            'ppar_1': 'par_value_1',
            'ppar_2': 'par_value_2',
        }
        assert domain_fit_results.p0 == 'p_0'
        assert domain_fit_results.y_calc == 'evaluate'
        np.testing.assert_array_equal(domain_fit_results.y_err, 1 / weights)
        assert domain_fit_results.n_evaluations == 12
        assert domain_fit_results.iterations == 3
        assert domain_fit_results.message == 'Maximum function evaluations reached'
        assert domain_fit_results.engine_result == mock_fit_result
        assert (
            str(domain_fit_results.minimizer_engine)
            == "<class 'easyscience.fitting.minimizers.minimizer_dfo.DFO'>"
        )

    def test_gen_fit_results_maxfun_warning_sets_success_false_and_warns(
        self, minimizer: DFO, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ):
        """When DFO returns EXIT_MAXFUN_WARNING, _gen_fit_results must warn and set success=False."""
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.EXIT_SUCCESS = 0
        mock_fit_result.EXIT_MAXFUN_WARNING = 1
        mock_fit_result.flag = 1  # MAXFUN_WARNING
        mock_fit_result.nf = 50
        mock_fit_result.msg = 'Objective has been called MAXFUN times'

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'v1'
        minimizer._cached_pars = {'par_1': mock_cached_par_1}
        minimizer._p_0 = 'p_0'
        minimizer.evaluate = MagicMock(return_value='evaluate')

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.dfo'):
            domain_fit_results = minimizer._gen_fit_results(mock_fit_result, np.array([1.0]))

        assert 'Objective has been called MAXFUN times' in caplog.text
        fit_results = MagicMock()
        fit_results.diagnostic_info = {'iters_total': [1, 2, 5]}

        assert DFO._extract_iterations(fit_results) == 5

    def test_gen_fit_results_success_does_not_warn(
        self, minimizer: DFO, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ):
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.EXIT_SUCCESS = 0
        mock_fit_result.EXIT_MAXFUN_WARNING = 1
        mock_fit_result.flag = 1  # MAXFUN_WARNING
        mock_fit_result.nf = 50
        mock_fit_result.msg = 'Objective has been called MAXFUN times'

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'v1'
        minimizer._cached_pars = {'par_1': mock_cached_par_1}
        minimizer._p_0 = 'p_0'
        minimizer.evaluate = MagicMock(return_value='evaluate')

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.dfo'):
            domain_fit_results = minimizer._gen_fit_results(mock_fit_result, np.array([1.0]))

        assert 'Objective has been called MAXFUN times' in caplog.text
        assert domain_fit_results.success == False
        assert domain_fit_results.n_evaluations == 50
        assert domain_fit_results.message == 'Objective has been called MAXFUN times'

    def test_dfo_fit_allows_maxfun_warning(self, minimizer: DFO, monkeypatch) -> None:
        mock_result = MagicMock()
        mock_result.EXIT_SUCCESS = 0
        mock_result.EXIT_MAXFUN_WARNING = 1
        mock_result.flag = 1

        mock_solve = MagicMock(return_value=mock_result)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo.dfols, 'solve', mock_solve
        )

    def test_gen_fit_results_success_does_not_warn(
        self, minimizer: DFO, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ):
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.EXIT_SUCCESS = 0
        mock_fit_result.EXIT_MAXFUN_WARNING = 1
        mock_fit_result.flag = 0
        mock_fit_result.nf = 12
        mock_fit_result.msg = 'Success'

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'v1'
        minimizer._cached_pars = {'par_1': mock_cached_par_1}
        minimizer._p_0 = 'p_0'
        minimizer.evaluate = MagicMock(return_value='evaluate')

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.dfo'):
            domain_fit_results = minimizer._gen_fit_results(mock_fit_result, np.array([1.0]))

        assert len(caplog.records) == 0
        assert domain_fit_results.success == True

    def test_dfo_fit_allows_maxfun_warning(self, minimizer: DFO, monkeypatch) -> None:
        mock_result = MagicMock()
        mock_result.EXIT_SUCCESS = 0
        mock_result.EXIT_MAXFUN_WARNING = 1
        mock_result.flag = 1

        mock_solve = MagicMock(return_value=mock_result)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo.dfols, 'solve', mock_solve
        )

        parameter = MagicMock(min=0.0, max=1.0, value=0.5)

        result = minimizer._dfo_fit({'par': parameter}, MagicMock())

        assert result == mock_result

    def test_dfo_fit_raises_for_non_maxfun_failure(self, minimizer: DFO, monkeypatch) -> None:
        mock_result = MagicMock()
        mock_result.EXIT_SUCCESS = 0
        mock_result.EXIT_MAXFUN_WARNING = 1
        mock_result.flag = 4
        mock_result.msg = 'linear algebra error'

        mock_solve = MagicMock(return_value=mock_result)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_dfo.dfols, 'solve', mock_solve
        )

        parameter = MagicMock(min=0.0, max=1.0, value=0.5)

        with pytest.raises(FitError, match='linear algebra error'):
            minimizer._dfo_fit({'par': parameter}, MagicMock())

    def test_dfo_fit(self, minimizer: DFO, monkeypatch):
        # When
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = 0.1
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = 0.2
        mock_parm_2.max = 20.0
        pars = {1: mock_parm_1, 2: mock_parm_2}

        kwargs = {'kwargs_set_key': 'kwargs_set_val'}

        mock_dfols = MagicMock()
        mock_results = MagicMock()
        mock_results.EXIT_SUCCESS = 0
        mock_results.EXIT_MAXFUN_WARNING = 1
        mock_results.flag = 0
        mock_results.msg = 'Success'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, 'dfols', mock_dfols)

        # Then
        results = minimizer._dfo_fit(pars, 'model', **kwargs)

        # Expect
        assert results == mock_results
        assert mock_dfols.solve.call_args[0][0] == 'model'
        assert all(mock_dfols.solve.call_args[0][1] == np.array([1.0, 2.0]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][0] == np.array([0.1, 0.2]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][1] == np.array([10.0, 20.0]))
        assert mock_dfols.solve.call_args[1]['scaling_within_bounds'] is True
        assert mock_dfols.solve.call_args[1]['kwargs_set_key'] == 'kwargs_set_val'

    def test_dfo_fit_no_scaling(self, minimizer: DFO, monkeypatch):
        # When
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = -np.inf
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = 0.2
        mock_parm_2.max = 20.0
        pars = {1: mock_parm_1, 2: mock_parm_2}

        kwargs = {'kwargs_set_key': 'kwargs_set_val'}

        mock_dfols = MagicMock()
        mock_results = MagicMock()
        mock_results.EXIT_SUCCESS = 0
        mock_results.EXIT_MAXFUN_WARNING = 1
        mock_results.flag = 0
        mock_results.msg = 'Success'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, 'dfols', mock_dfols)

        # Then
        results = minimizer._dfo_fit(pars, 'model', **kwargs)

        # Expect
        assert results == mock_results
        assert mock_dfols.solve.call_args[0][0] == 'model'
        assert all(mock_dfols.solve.call_args[0][1] == np.array([1.0, 2.0]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][0] == np.array([-np.inf, 0.2]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][1] == np.array([10.0, 20.0]))
        assert 'scaling_within_bounds' not in list(mock_dfols.solve.call_args[1].keys())
        assert 'kwargs_set_key' in list(mock_dfols.solve.call_args[1].keys())
        assert mock_dfols.solve.call_args[1]['kwargs_set_key'] == 'kwargs_set_val'

    def test_fit_generic_exception_resets_parameters_and_raises_fit_error(
        self, minimizer: DFO
    ) -> None:
        """When _dfo_fit raises a non-FitError exception, fit() must reset
        parameter values to cached originals and re-raise as FitError."""
        from easyscience import global_object

        global_object.stack.enabled = True

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(side_effect=RuntimeError('solver crashed'))

        cached_par_1 = MagicMock()
        cached_par_1.value = 5.0
        cached_par_2 = MagicMock()
        cached_par_2.value = 10.0
        minimizer._cached_pars = {'a': cached_par_1, 'b': cached_par_2}
        minimizer._cached_pars_vals = {'a': (1.0, 0.1), 'b': (2.0, 0.2)}

        with pytest.raises(FitError):
            minimizer.fit(x=np.array([1.0]), y=np.array([1.0]), weights=np.array([1.0]))

        assert cached_par_1.value == 1.0
        assert cached_par_2.value == 2.0
        assert global_object.stack.enabled is True

    def test_fit_fit_error_resets_parameters_and_reraises(self, minimizer: DFO) -> None:
        """When _dfo_fit raises FitError, fit() must reset parameter values and re-raise it."""
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(side_effect=FitError(RuntimeError('solver failed')))

        cached_par_1 = MagicMock()
        cached_par_1.value = 5.0
        cached_par_2 = MagicMock()
        cached_par_2.value = 10.0
        minimizer._cached_pars = {'a': cached_par_1, 'b': cached_par_2}
        minimizer._cached_pars_vals = {'a': (1.0, 0.1), 'b': (2.0, 0.2)}

        with pytest.raises(FitError):
            minimizer.fit(x=np.array([1.0]), y=np.array([1.0]), weights=np.array([1.0]))

        assert cached_par_1.value == 1.0
        assert cached_par_2.value == 2.0

    def test_dfo_fit_exception(self, minimizer: DFO, monkeypatch):
        # When
        pars = {1: MagicMock(Parameter)}
        kwargs = {'kwargs_set_key': 'kwargs_set_val'}

        mock_dfols = MagicMock()
        mock_results = MagicMock()
        mock_results.EXIT_SUCCESS = 0
        mock_results.EXIT_MAXFUN_WARNING = 1
        mock_results.flag = 3
        mock_results.msg = 'Failed'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, 'dfols', mock_dfols)

        # Then Expect
        with pytest.raises(FitError):
            minimizer._dfo_fit(pars, 'model', **kwargs)

    def test_progress_callback_creates_adapter_when_no_explicit_callback(
        self, minimizer: DFO
    ) -> None:
        """When progress_callback is provided without an explicit callback,
        fit() should auto-create a DFO callback adapter."""
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        progress_cb = MagicMock()

        minimizer.fit(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, 1.0, 1.0]),
            progress_callback=progress_cb,
        )

        # The adapter should have been passed as callback to _make_model
        call_kwargs = minimizer._make_model.call_args[1]
        assert call_kwargs['callback'] is not None
        assert callable(call_kwargs['callback'])

        call_kwargs['callback'](
            DFOCallbackState(
                evaluation=1,
                xk=np.array([1.0]),
                residuals=np.array([1.0, 1.0, 2.0]),
                objective=6.0,
                parameters={'pmock_parm_1': 1.0},
                best_xk=np.array([1.0]),
                best_objective=6.0,
                best_parameters={'pmock_parm_1': 1.0},
                improved=True,
            )
        )

        payload = progress_cb.call_args[0][0]
        assert payload['reduced_chi2'] == pytest.approx(3.0)
        assert payload['parameter_values'] == {'mock_parm_1': 1.0}

    def test_progress_callback_not_used_when_explicit_callback_given(self, minimizer: DFO) -> None:
        """When both progress_callback and callback are given, the explicit
        callback takes precedence and progress_callback is ignored."""
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._dfo_fit = MagicMock(return_value='fit')
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        minimizer._cached_pars = {'mock_parm_1': cached_par}

        progress_cb = MagicMock()
        explicit_cb = MagicMock()

        minimizer.fit(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([1.0, 2.0, 3.0]),
            weights=np.array([1.0, 1.0, 1.0]),
            progress_callback=progress_cb,
            callback=explicit_cb,
        )

        call_kwargs = minimizer._make_model.call_args[1]
        assert call_kwargs['callback'] is explicit_cb

    @pytest.mark.parametrize(
        ('parameters', 'expected_names'),
        [
            ([MagicMock(unique_name='alpha')], ['palpha']),
            (None, ['pbeta']),
        ],
    )
    def test_get_callback_parameter_names_optional_parameters(
        self, minimizer: DFO, parameters, expected_names
    ) -> None:
        minimizer._cached_pars = {'beta': MagicMock(value=1.0)}

        parameter_names = minimizer._get_callback_parameter_names(parameters)

        assert parameter_names == expected_names

    def test_wrap_model_with_callback_invokes_on_each_evaluation(self, minimizer: DFO) -> None:
        callback = MagicMock()
        wrapped_model = minimizer._wrap_model_with_callback(
            lambda pars_values: np.asarray([pars_values[0] - 1.0]),
            ['palpha'],
            callback,
        )

        wrapped_model([0.5])

        callback.assert_called_once()
        assert callback.call_args.args[0].improved is True

    def test_prepare_kwargs_with_optional_arguments(self, minimizer: DFO) -> None:
        kwargs = minimizer._prepare_kwargs(tolerance=0.05, max_evaluations=11, keep=True)

        assert kwargs == {
            'keep': True,
            'maxfun': 11,
            'rhoend': 0.05,
            'user_params': {'logging.save_diagnostic_info': True},
        }

    def test_prepare_kwargs_rejects_large_tolerance(self, minimizer: DFO) -> None:
        with pytest.raises(ValueError, match='Tolerance must be equal or smaller than 0.1'):
            minimizer._prepare_kwargs(tolerance=0.2)

    def test_make_progress_adapter_payload_format(self) -> None:
        """The adapter must produce the standard progress payload dict."""
        progress_cb = MagicMock()

        adapter = DFO._make_progress_adapter(progress_cb)

        state = DFOCallbackState(
            evaluation=10,
            xk=np.array([1.0, 2.0]),
            residuals=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            objective=0.05,
            parameters={'pmock_parm_1': 1.0, 'pmock_parm_2': 2.0},
            best_xk=np.array([1.0, 2.0]),
            best_objective=0.04,
            best_parameters={'pmock_parm_1': 1.0, 'pmock_parm_2': 2.0},
            improved=True,
        )

        adapter(state)

        progress_cb.assert_called_once()
        payload = progress_cb.call_args[0][0]
        assert payload['iteration'] == 10
        assert payload['chi2'] == 0.04  # best_objective
        assert payload['reduced_chi2'] == pytest.approx(0.04 / 5)
        assert payload['parameter_values'] == {'mock_parm_1': 1.0, 'mock_parm_2': 2.0}
        assert payload['refresh_plots'] is False
        assert payload['finished'] is False
