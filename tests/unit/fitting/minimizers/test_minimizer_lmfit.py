# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import ANY
from unittest.mock import MagicMock

import numpy as np
import pytest
from lmfit import Parameter as LMParameter

import easyscience.fitting.minimizers.minimizer_lmfit
from easyscience import Parameter
from easyscience.fitting.minimizers.minimizer_lmfit import LMFit
from easyscience.fitting.minimizers.utils import FitError


class TestLMFit:
    @pytest.fixture
    def minimizer(self) -> LMFit:
        minimizer = LMFit(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=MagicMock(package='lm', method='leastsq'),
        )
        return minimizer

    def test_init(self, minimizer: LMFit) -> None:
        assert minimizer.package == 'lmfit'

    def test_init_exception(self) -> None:
        with pytest.raises(FitError):
            LMFit(
                obj='obj',
                fit_function='fit_function',
                minimizer_enum=MagicMock(package='dfo', method='not_leastsq'),
            )

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
    def test_fit_weight_exceptions(self, minimizer: LMFit, weights) -> None:
        # When Then Expect
        with pytest.raises(ValueError):
            minimizer.fit(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), weights=weights)

    def test_make_model(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'LMModel', mock_LMModel
        )
        minimizer._generate_fit_function = MagicMock(return_value='model')
        mock_parm_1 = MagicMock(LMParameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = -10.0
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(LMParameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = -20.0
        mock_parm_2.max = 20.0
        pars = {'key_1': mock_parm_1, 'key_2': mock_parm_2}

        # Then
        model = minimizer._make_model(pars=pars)

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        mock_LMModel.assert_called_once_with(
            'model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2']
        )
        mock_lm_model.set_param_hint.assert_called_with('pkey_2', value=2.0, min=-20.0, max=20.0)
        assert mock_lm_model.set_param_hint.call_count == 2
        assert model == mock_lm_model

    def test_make_model_no_pars(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'LMModel', mock_LMModel
        )
        minimizer._generate_fit_function = MagicMock(return_value='model')
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_1.min = -10.0
        mock_parm_1.max = 10.0
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        mock_parm_2.min = -20.0
        mock_parm_2.max = 20.0
        minimizer._cached_pars = {'key_1': mock_parm_1, 'key_2': mock_parm_2}

        # Then
        model = minimizer._make_model()

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        mock_LMModel.assert_called_once_with(
            'model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2']
        )
        mock_lm_model.set_param_hint.assert_called_with('pkey_2', value=2.0, min=-20.0, max=20.0)
        assert mock_lm_model.set_param_hint.call_count == 2
        assert model == mock_lm_model

    def test_fit(self, minimizer: LMFit) -> None:
        # When
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        result = minimizer.fit(x=1.0, y=2.0, weights=1)

        # Expect
        assert result == 'gen_fit_results'
        mock_model.fit.assert_called_once_with(
            2.0, x=1.0, weights=1, max_nfev=None, iter_cb=ANY, fit_kws={}, method='leastsq'
        )
        assert callable(mock_model.fit.call_args.kwargs['iter_cb'])
        minimizer._make_model.assert_called_once_with()
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit', iterations=None)

    def test_fit_model(self, minimizer: LMFit) -> None:
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, weights=1, model=mock_model)

        # Expect
        mock_model.fit.assert_called_once_with(
            2.0, x=1.0, weights=1, max_nfev=None, iter_cb=ANY, fit_kws={}, method='leastsq'
        )
        assert callable(mock_model.fit.call_args.kwargs['iter_cb'])
        minimizer._make_model.assert_not_called()

    def test_fit_method(self, minimizer: LMFit) -> None:
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')
        minimizer.supported_methods = MagicMock(return_value=['method_passed'])
        minimizer.all_methods = MagicMock(return_value=['method_passed'])

        # Then
        minimizer.fit(x=1.0, y=2.0, weights=1, method='method_passed')

        # Expect
        mock_model.fit.assert_called_once_with(
            2.0,
            x=1.0,
            weights=1,
            max_nfev=None,
            iter_cb=ANY,
            fit_kws={},
            method='method_passed',
        )
        assert callable(mock_model.fit.call_args.kwargs['iter_cb'])
        minimizer.supported_methods.assert_called_once_with()

    def test_fit_kwargs(self, minimizer: LMFit) -> None:
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(
            x=1.0,
            y=2.0,
            weights=1,
            minimizer_kwargs={'minimizer_key': 'minimizer_val'},
            engine_kwargs={'engine_key': 'engine_val'},
        )

        # Expect
        mock_model.fit.assert_called_once_with(
            2.0,
            x=1.0,
            weights=1,
            max_nfev=None,
            iter_cb=ANY,
            fit_kws={'minimizer_key': 'minimizer_val'},
            method='leastsq',
            engine_key='engine_val',
        )
        assert callable(mock_model.fit.call_args.kwargs['iter_cb'])

    def test_fit_progress_callback(self, minimizer: LMFit) -> None:
        # When
        progress_callback = MagicMock(return_value=True)
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, weights=1, progress_callback=progress_callback)

        # Expect
        assert mock_model.fit.call_count == 1
        iter_cb = mock_model.fit.call_args.kwargs['iter_cb']
        assert callable(iter_cb)

    def test_fit_progress_callback_with_supplied_model_uses_iter_params(
        self, minimizer: LMFit
    ) -> None:
        progress_callback = MagicMock(return_value=True)
        mock_model = MagicMock()
        mock_param_alpha = MagicMock()
        mock_param_alpha.value = 1.0
        mock_param_alpha.vary = True
        params = {'palpha': mock_param_alpha}

        mock_model.fit = MagicMock(
            side_effect=lambda *args, **kwargs: (
                kwargs['iter_cb'](params, 4, np.array([3.0, 4.0])) or 'fit'
            )
        )
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        result = minimizer.fit(
            x=np.array([1.0, 2.0]),
            y=np.array([1.0, 2.0]),
            weights=np.array([1.0, 1.0]),
            model=mock_model,
            progress_callback=progress_callback,
        )

        assert result == 'gen_fit_results'
        minimizer._make_model.assert_not_called()
        payload = progress_callback.call_args[0][0]
        assert payload['parameter_values'] == {'alpha': 1.0}
        assert payload['chi2'] == 25.0
        assert payload['reduced_chi2'] == 25.0

    def test_create_iter_callback_no_callback(self, minimizer: LMFit) -> None:
        # When
        iter_cb = minimizer._create_iter_callback(None)

        # Then Expect
        assert callable(iter_cb)
        assert iter_cb(MagicMock(), 5, np.array([1.0, -2.0])) is False
        assert minimizer._last_iteration == 5

    def test_create_iter_callback_invokes_progress(self, minimizer: LMFit) -> None:
        # When
        progress_callback = MagicMock(return_value=False)
        iter_cb = minimizer._create_iter_callback(progress_callback)

        # Then
        result = iter_cb(MagicMock(), 5, np.array([1.0, -2.0]))

        # Expect — progress callback is notified, but its return value is ignored
        progress_callback.assert_called_once()
        assert result is False

    def test_build_progress_payload(self, minimizer: LMFit) -> None:
        # When
        parameter_a = MagicMock(Parameter)
        parameter_a.value = 1.5
        parameter_b = MagicMock(Parameter)
        parameter_b.value = 2.5
        minimizer._cached_pars = {'alpha': parameter_a, 'beta': parameter_b}

        mock_param_alpha = MagicMock()
        mock_param_alpha.value = 1.0
        mock_param_alpha.vary = True
        mock_param_beta = MagicMock()
        mock_param_beta.value = 2.0
        mock_param_beta.vary = False
        params = {'palpha': mock_param_alpha, 'pbeta': mock_param_beta}

        # Then
        payload = minimizer._build_progress_payload(params, 7, np.array([3.0, 4.0]))

        # Expect
        assert payload == {
            'iteration': 7,
            'chi2': 25.0,
            'reduced_chi2': 25.0,
            'parameter_values': {'alpha': 1.0, 'beta': 2.0},
            'refresh_plots': False,
            'finished': False,
        }

    def test_build_progress_payload_without_cached_pars(self, minimizer: LMFit) -> None:
        mock_param_alpha = MagicMock()
        mock_param_alpha.value = 1.0
        mock_param_alpha.vary = True
        params = {'palpha': mock_param_alpha}

        payload = minimizer._build_progress_payload(params, 4, np.array([3.0, 4.0]))

        assert payload == {
            'iteration': 4,
            'chi2': 25.0,
            'reduced_chi2': 25.0,
            'parameter_values': {'alpha': 1.0},
            'refresh_plots': False,
            'finished': False,
        }

    def test_fit_exception(self, minimizer: LMFit) -> None:
        # When
        minimizer._make_model = MagicMock(side_effect=Exception('Exception'))
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then Expect
        with pytest.raises(FitError):
            minimizer.fit(x=1.0, y=2.0, weights=1)

    def test_gen_fit_results_populates_evaluation_metadata(
        self, minimizer: LMFit, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        fit_results = MagicMock()
        fit_results.success = False
        fit_results.data = 'data'
        fit_results.userkws = {'x': 'x'}
        fit_results.values = {'p1': 1.0}
        fit_results.init_values = {'p1': 0.5}
        fit_results.best_fit = 'best_fit'
        fit_results.weights = 2
        fit_results.nfev = 9
        fit_results.message = 'max evaluations reached'

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.lmfit'):
            result = minimizer._gen_fit_results(fit_results, iterations=4)

        assert 'max evaluations reached' in caplog.text

        assert result.success is False
        assert result.n_evaluations == 9
        assert result.iterations == 4
        assert result.message == 'max evaluations reached'
        assert result.engine_result == fit_results

    def test_gen_fit_results_success_does_not_warn(
        self, minimizer: LMFit, caplog: "pytest.LogCaptureFixture"
    ) -> None:
        fit_results = MagicMock()
        fit_results.success = True
        fit_results.data = 'data'
        fit_results.userkws = {'x': 'x'}
        fit_results.values = {'p1': 1.0}
        fit_results.init_values = {'p1': 0.5}
        fit_results.best_fit = 'best_fit'
        fit_results.weights = 2
        fit_results.nfev = 3
        fit_results.message = 'success'

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.lmfit'):
            result = minimizer._gen_fit_results(fit_results)

        assert len(caplog.records) == 0
        assert result.success is True

    def test_convert_to_pars_obj(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        minimizer._object = MagicMock()
        minimizer._object.get_fit_parameters = MagicMock(return_value=['parm_1', 'parm_2'])

        minimizer.convert_to_par_object = MagicMock(return_value='convert_to_par_object')

        mock_lm_parameter = MagicMock()
        mock_lm_parameter.add_many = MagicMock(return_value='add_many')
        mock_LMParameters = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'LMParameters', mock_LMParameters
        )

        # Then
        pars = minimizer.convert_to_pars_obj()

        # Expect
        assert pars == 'add_many'
        assert minimizer.convert_to_par_object.call_count == 2
        minimizer._object.get_fit_parameters.assert_called_once_with()
        minimizer.convert_to_par_object.assert_called_with('parm_2')
        mock_lm_parameter.add_many.assert_called_once_with([
            'convert_to_par_object',
            'convert_to_par_object',
        ])

    def test_convert_to_pars_obj_with_parameters(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        minimizer.convert_to_par_object = MagicMock(return_value='convert_to_par_object')

        mock_lm_parameter = MagicMock()
        mock_lm_parameter.add_many = MagicMock(return_value='add_many')
        mock_LMParameters = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'LMParameters', mock_LMParameters
        )

        # Then
        pars = minimizer.convert_to_pars_obj(['parm_1', 'parm_2'])

        # Expect
        assert pars == 'add_many'
        assert minimizer.convert_to_par_object.call_count == 2
        minimizer.convert_to_par_object.assert_called_with('parm_2')
        mock_lm_parameter.add_many.assert_called_once_with([
            'convert_to_par_object',
            'convert_to_par_object',
        ])

    def test_convert_to_par_object(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_parameter = MagicMock()
        mock_LMParameter = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'LMParameter', mock_LMParameter
        )

        mock_parm = MagicMock(Parameter)
        mock_parm.value = 1.0
        mock_parm.fixed = True
        mock_parm.min = -10.0
        mock_parm.max = 10.0
        mock_parm.unique_name = 'key_converted'

        # Then
        par = minimizer.convert_to_par_object(mock_parm)

        # Expect
        assert par == mock_lm_parameter
        mock_LMParameter.assert_called_once_with(
            'pkey_converted',
            value=1.0,
            vary=False,
            min=-10.0,
            max=10.0,
            expr=None,
            brute_step=None,
        )

    def test_set_parameter_fit_result_no_stack_status(self, minimizer: LMFit) -> None:
        # When
        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
        }
        minimizer._cached_pars['a'].value = 'a'
        minimizer._cached_pars['b'].value = 'b'

        mock_param_a = MagicMock()
        mock_param_a.value = 1.0
        mock_param_a.stderr = 0.1
        mock_param_b = MagicMock
        mock_param_b.value = 2.0
        mock_param_b.stderr = 0.2
        mock_fit_result = MagicMock()
        mock_fit_result.params = {'pa': mock_param_a, 'pb': mock_param_b}
        mock_fit_result.errorbars = True

        # Then
        minimizer._set_parameter_fit_result(mock_fit_result, False)

        # Expect
        assert minimizer._cached_pars['a'].value == 1.0
        assert minimizer._cached_pars['a'].error == 0.1
        assert minimizer._cached_pars['b'].value == 2.0
        assert minimizer._cached_pars['b'].error == 0.2

    def test_set_parameter_fit_result_no_stack_status_no_error(self, minimizer: LMFit) -> None:
        # When
        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
        }
        minimizer._cached_pars['a'].value = 'a'
        minimizer._cached_pars['b'].value = 'b'

        mock_param_a = MagicMock()
        mock_param_a.value = 1.0
        mock_param_a.stderr = 0.1
        mock_param_b = MagicMock
        mock_param_b.value = 2.0
        mock_param_b.stderr = 0.2
        mock_fit_result = MagicMock()
        mock_fit_result.params = {'pa': mock_param_a, 'pb': mock_param_b}
        mock_fit_result.errorbars = False

        # Then
        minimizer._set_parameter_fit_result(mock_fit_result, False)

        # Expect
        assert minimizer._cached_pars['a'].value == 1.0
        assert minimizer._cached_pars['a'].error == 0.0
        assert minimizer._cached_pars['b'].value == 2.0
        assert minimizer._cached_pars['b'].error == 0.0

    def test_gen_fit_results(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_lmfit, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.success = 'success'
        mock_fit_result.data = 'data'
        mock_fit_result.userkws = {'x': 'x_val'}
        mock_fit_result.values = 'values'
        mock_fit_result.init_values = 'init_values'
        mock_fit_result.best_fit = 'best_fit'
        mock_fit_result.weights = 10

        # Then
        domain_fit_results = minimizer._gen_fit_results(
            mock_fit_result, **{'kwargs_set_key': 'kwargs_set_val'}
        )

        # Expect
        assert domain_fit_results == mock_domain_fit_results
        assert domain_fit_results.kwargs_set_key == 'kwargs_set_val'
        assert domain_fit_results.success == 'success'
        assert domain_fit_results.y_obs == 'data'
        assert domain_fit_results.x == 'x_val'
        assert domain_fit_results.p == 'values'
        assert domain_fit_results.p0 == 'init_values'
        assert domain_fit_results.y_calc == 'best_fit'
        assert domain_fit_results.y_err == 0.1
        assert (
            str(domain_fit_results.minimizer_engine)
            == "<class 'easyscience.fitting.minimizers.minimizer_lmfit.LMFit'>"
        )
        assert domain_fit_results.fit_args is None
