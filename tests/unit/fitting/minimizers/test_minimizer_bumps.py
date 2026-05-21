# SPDX-FileCopyrightText: 2026 EasyScience contributors <https://github.com/easyscience>
# SPDX-License-Identifier: BSD-3-Clause

import logging
from unittest.mock import ANY
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

import easyscience.fitting.minimizers.minimizer_bumps
from easyscience.fitting.minimizers.bumps_utils import BumpsProgressMonitor
from easyscience.fitting.minimizers.minimizer_bumps import Bumps
from easyscience.fitting.minimizers.utils import FitError


class TestBumpsFit:
    @pytest.fixture
    def minimizer(self) -> Bumps:
        minimizer = Bumps(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=MagicMock(package='bumps', method='amoeba'),
        )
        return minimizer

    def test_init(self, minimizer: Bumps) -> None:
        assert minimizer._p_0 == {}
        assert minimizer.package == 'bumps'

    def test_init_exception(self) -> None:
        with pytest.raises(FitError):
            Bumps(
                obj='obj',
                fit_function='fit_function',
                minimizer_enum=MagicMock(package='bumps', method='not_amoeba'),
            )

    def test_all_methods(self, minimizer: Bumps) -> None:
        # When Then Expect
        assert minimizer.all_methods() == ['amoeba', 'de', 'dream', 'newton', 'lm']

    def test_supported_methods(self, minimizer: Bumps) -> None:
        # When Then Expect
        assert set(minimizer.supported_methods()) == set(['newton', 'lm', 'amoeba'])

    def test_fit(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        from easyscience import global_object

        global_object.stack.enabled = False

        # Mock FitDriver. driver.fit() returns (x, fx); driver.stderr() returns dx.
        mock_driver_instance = MagicMock()
        mock_driver_instance.clip = MagicMock()
        mock_driver_instance.fit = MagicMock(return_value=(np.array([42.0]), 0.0))
        mock_driver_instance.stderr = MagicMock(return_value=np.array([0.1]))
        mock_driver_instance.monitor_runner.history.step = [0]
        mock_FitDriver = MagicMock(return_value=mock_driver_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitDriver', mock_FitDriver
        )

        # Prepare a mock parameter with .name = 'pmock_parm_1'
        mock_bumps_param = MagicMock()
        mock_bumps_param.name = 'pmock_parm_1'
        # Patch FitProblem to have _parameters attribute as expected
        mock_FitProblem_instance = MagicMock()
        mock_FitProblem_instance._parameters = [mock_bumps_param]
        mock_FitProblem = MagicMock(return_value=mock_FitProblem_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitProblem', mock_FitProblem
        )

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        cached_pars = {'mock_parm_1': cached_par}
        minimizer._cached_pars = cached_pars
        minimizer._cached_pars_vals = {'mock_parm_1': (1, 0.0)}

        # Patch _set_parameter_fit_result
        def fake_set_parameter_fit_result(fit_result, stack_status, par_list):
            for index, name in enumerate([par.name for par in par_list]):
                dict_name = name[len('p') :]
            minimizer._cached_pars[dict_name].value = fit_result.x[index]

        minimizer._set_parameter_fit_result = fake_set_parameter_fit_result

        mock_fitclass = MagicMock()
        mock_fitclass.id = 'amoeba'
        minimizer._resolve_fitclass = MagicMock(return_value=mock_fitclass)

        # Then
        result = minimizer.fit(x=1.0, y=2.0, weights=1)

        # Expect
        assert result == 'gen_fit_results'
        mock_FitDriver.assert_called_once()
        mock_driver_instance.clip.assert_called_once()
        mock_driver_instance.fit.assert_called_once()
        minimizer._make_model.assert_called_once_with(parameters=None)
        # _gen_fit_results is called with the OptimizeResult built from driver.fit()
        minimizer._gen_fit_results.assert_called_once()
        passed_result = minimizer._gen_fit_results.call_args.args[0]
        assert np.array_equal(passed_result.x, np.array([42.0]))
        assert np.array_equal(passed_result.dx, np.array([0.1]))
        assert minimizer._gen_fit_results.call_args.kwargs == {
            'max_evaluations': None,
            'tolerance': None,
        }
        mock_model_function.assert_called_once_with(1.0, 2.0, 1)
        mock_FitProblem.assert_called_once_with(mock_model)

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
    def test_fit_weight_exceptions(self, minimizer: Bumps, weights) -> None:
        # When Then Expect
        with pytest.raises(ValueError):
            minimizer.fit(x=np.array([1, 2, 3]), y=np.array([1, 2, 3]), weights=weights)

    def test_make_model(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        mock_fit_function = MagicMock(return_value=np.array([11, 22]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        minimizer.convert_to_par_object = MagicMock(return_value='converted_parm_1')

        mock_Curve = MagicMock(return_value='curve')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, 'Curve', mock_Curve)

        # Then
        model = minimizer._make_model(parameters=[mock_parm_1])
        curve_for_model = model(
            x=np.array([1, 2]), y=np.array([10, 20]), weights=np.array([100, 200])
        )
        wrapped_fit_function = mock_Curve.call_args[0][0]
        wrapped_fit_function(np.array([1, 2]), pmock_parm_1=3)

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        assert minimizer._eval_counter is wrapped_fit_function
        assert minimizer._eval_counter.count == 1
        assert all(mock_Curve.call_args[0][1] == np.array([1, 2]))
        assert all(mock_Curve.call_args[0][2] == np.array([10, 20]))
        assert curve_for_model == 'curve'

    def test_set_parameter_fit_result_no_stack_status(self, minimizer: Bumps):
        # When
        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
        }
        minimizer._cached_pars['a'].value = 'a'
        minimizer._cached_pars['b'].value = 'b'

        mock_cached_model = MagicMock()
        mock_cached_model.pars = {'pa': 0, 'pb': 0}
        minimizer._cached_model = mock_cached_model

        mock_fit_result = MagicMock()
        mock_fit_result.x = np.array([1.0, 2.0])
        mock_fit_result.dx = np.array([0.1, 0.2])

        # The new argument: par_list (list of mock parameters)
        mock_par_a = MagicMock()
        mock_par_a.name = 'pa'
        mock_par_b = MagicMock()
        mock_par_b.name = 'pb'
        par_list = [mock_par_a, mock_par_b]

        # Then
        minimizer._set_parameter_fit_result(mock_fit_result, False, par_list)

        # Expect
        assert minimizer._cached_pars['a'].value == 1.0
        assert minimizer._cached_pars['a'].error == 0.1
        assert minimizer._cached_pars['b'].value == 2.0
        assert minimizer._cached_pars['b'].error == 0.2

    def test_gen_fit_results(
        self, minimizer: Bumps, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ):
        # When
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.success = True
        mock_fit_result.nit = 2

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        mock_cached_model.dy = 'dy'
        mock_cached_model.pars = {'ppar_1': 0, 'ppar_2': 0}
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'par_value_1'
        mock_cached_par_2 = MagicMock()
        mock_cached_par_2.value = 'par_value_2'
        minimizer._cached_pars = {'par_1': mock_cached_par_1, 'par_2': mock_cached_par_2}

        minimizer._p_0 = 'p_0'
        minimizer._eval_counter = MagicMock(count=7)
        minimizer.evaluate = MagicMock(return_value='evaluate')

        # Then
        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.bumps'):
            domain_fit_results = minimizer._gen_fit_results(
                mock_fit_result,
                max_evaluations=3,
                **{'kwargs_set_key': 'kwargs_set_val'},
            )
        assert 'maximum optimizer steps of 3' in caplog.text

        # Expect
        assert domain_fit_results == mock_domain_fit_results
        assert domain_fit_results.kwargs_set_key == 'kwargs_set_val'
        assert domain_fit_results.success == False
        assert domain_fit_results.y_obs == 'y'
        assert domain_fit_results.x == 'x'
        assert domain_fit_results.p == {'ppar_1': 'par_value_1', 'ppar_2': 'par_value_2'}
        assert domain_fit_results.p0 == 'p_0'
        assert domain_fit_results.y_calc == 'evaluate'
        assert domain_fit_results.y_err == 'dy'
        assert domain_fit_results.n_evaluations == 7
        assert domain_fit_results.iterations == 3
        assert (
            domain_fit_results.message
            == 'Fit stopped: reached maximum optimizer steps (3); objective evaluated 7 times'
        )
        assert (
            str(domain_fit_results.minimizer_engine)
            == "<class 'easyscience.fitting.minimizers.minimizer_bumps.Bumps'>"
        )
        assert domain_fit_results.fit_args is None
        assert domain_fit_results.engine_result == mock_fit_result
        minimizer.evaluate.assert_called_once_with(
            'x', minimizer_parameters={'ppar_1': 'par_value_1', 'ppar_2': 'par_value_2'}
        )

    @pytest.mark.parametrize(
        'n_evaluations, max_evaluations, expected_success',
        [
            (1, 3, True),  # last step (1) < budget-1 (2) => success
            (2, 3, False),  # last step (2) == budget-1 (2) => budget consumed => failure
            (3, 3, False),  # last step (3) > budget-1 (2) => failure
            (0, 1, False),  # 0 >= 0 => failure (budget of 1, step counter 0-indexed)
            (5, 1000, True),  # well below budget => success
        ],
    )
    def test_gen_fit_results_max_evaluations_boundary(
        self, minimizer: Bumps, monkeypatch, n_evaluations, max_evaluations, expected_success
    ):
        """Bumps step counter is 0-indexed so the last step of a budget
        of N is N-1.  Verify the boundary condition in _gen_fit_results."""
        mock_domain_fit_results = MagicMock()
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps,
            'FitResults',
            MagicMock(return_value=mock_domain_fit_results),
        )

        mock_cached_model = MagicMock()
        mock_cached_model.pars = {'ppar_1': 0}
        minimizer._cached_model = mock_cached_model

        mock_par = MagicMock()
        mock_par.value = 1.0
        minimizer._cached_pars = {'par_1': mock_par}
        minimizer._p_0 = 'p_0'
        minimizer._eval_counter = MagicMock(count=n_evaluations)
        minimizer.evaluate = MagicMock(return_value='evaluate')

        mock_fit_result = MagicMock()
        mock_fit_result.success = True
        mock_fit_result.nit = n_evaluations

        minimizer._gen_fit_results(mock_fit_result, max_evaluations=max_evaluations)

        assert mock_domain_fit_results.success is expected_success

    def test_resolve_fitclass_valid(self, minimizer: Bumps) -> None:
        # When Then
        fitclass = Bumps._resolve_fitclass('lm')

        # Expect
        assert fitclass.id == 'lm'

    def test_resolve_fitclass_invalid(self, minimizer: Bumps) -> None:
        # When Then Expect
        with pytest.raises(FitError):
            Bumps._resolve_fitclass('nonexistent_method')

    def test_fit_progress_callback(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        from easyscience import global_object

        global_object.stack.enabled = False

        progress_callback = MagicMock(return_value=True)

        mock_driver_instance = MagicMock()
        mock_driver_instance.clip = MagicMock()
        mock_driver_instance.fit = MagicMock(return_value=(np.array([42.0]), 0.0))
        mock_driver_instance.stderr = MagicMock(return_value=np.array([0.1]))
        mock_driver_instance.monitor_runner.history.step = [0]
        mock_FitDriver = MagicMock(return_value=mock_driver_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitDriver', mock_FitDriver
        )

        mock_bumps_param = MagicMock()
        mock_bumps_param.name = 'pmock_parm_1'
        mock_FitProblem_instance = MagicMock()
        mock_FitProblem_instance._parameters = [mock_bumps_param]
        mock_FitProblem = MagicMock(return_value=mock_FitProblem_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitProblem', mock_FitProblem
        )

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        minimizer._cached_pars = {'mock_parm_1': cached_par}
        minimizer._cached_pars_vals = {'mock_parm_1': (1, 0.0)}

        minimizer._resolve_fitclass = MagicMock(return_value=MagicMock(id='amoeba'))

        # Then
        result = minimizer.fit(x=1.0, y=2.0, weights=1, progress_callback=progress_callback)

        # Expect - FitDriver was called with a monitor list containing our monitor
        assert result == 'gen_fit_results'
        driver_call_kwargs = mock_FitDriver.call_args
        monitors = driver_call_kwargs.kwargs.get('monitors', driver_call_kwargs[1].get('monitors'))
        assert len(monitors) == 1
        assert isinstance(monitors[0], BumpsProgressMonitor)
        assert monitors[0]._problem is mock_FitProblem_instance
        assert monitors[0]._callback is progress_callback
        assert monitors[0]._payload_builder == minimizer._build_progress_payload

    def test_fit_uses_supplied_model_and_optional_kwargs(
        self, minimizer: Bumps, monkeypatch
    ) -> None:
        from easyscience import global_object

        global_object.stack.enabled = False

        mock_driver_instance = MagicMock()
        mock_driver_instance.clip = MagicMock()
        mock_driver_instance.fit = MagicMock(return_value=(np.array([3.0]), 0.0))
        mock_driver_instance.stderr = MagicMock(return_value=np.array([0.1]))
        mock_driver_instance.monitor_runner.history.step = [0]
        mock_FitDriver = MagicMock(return_value=mock_driver_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitDriver', mock_FitDriver
        )

        mock_bumps_param = MagicMock()
        mock_bumps_param.name = 'pmock_parm_1'
        mock_problem = MagicMock()
        mock_problem._parameters = [mock_bumps_param]
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps,
            'FitProblem',
            MagicMock(return_value=mock_problem),
        )

        minimizer._make_model = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')
        minimizer._resolve_fitclass = MagicMock(return_value=MagicMock(id='amoeba'))
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._cached_pars = {'mock_parm_1': MagicMock(value=1.0)}
        minimizer._cached_pars_vals = {'mock_parm_1': (1.0, 0.0)}

        supplied_model = MagicMock()
        minimizer_kwargs = {'existing_option': 'minimizer'}
        engine_kwargs = {'engine_option': 'engine'}

        result = minimizer.fit(
            x=np.array([1.0]),
            y=np.array([2.0]),
            weights=np.array([1.0]),
            model=supplied_model,
            tolerance=0.25,
            max_evaluations=7,
            minimizer_kwargs=minimizer_kwargs,
            engine_kwargs=engine_kwargs,
        )

        assert result == 'gen_fit_results'
        minimizer._make_model.assert_not_called()
        fit_driver_kwargs = mock_FitDriver.call_args.kwargs
        assert fit_driver_kwargs['problem'] is mock_problem
        assert fit_driver_kwargs['existing_option'] == 'minimizer'
        assert fit_driver_kwargs['engine_option'] == 'engine'
        assert fit_driver_kwargs['ftol'] == 0.25
        assert fit_driver_kwargs['xtol'] == 0.25
        assert fit_driver_kwargs['steps'] == 7
        mock_driver_instance.fit.assert_called_once()

    def test_fit_rejects_non_callable_progress_callback(
        self, minimizer: Bumps, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps,
            'FitProblem',
            MagicMock(return_value=MagicMock()),
        )
        minimizer._resolve_fitclass = MagicMock(return_value=MagicMock(id='amoeba'))

        with pytest.raises(ValueError, match='progress_callback must be callable'):
            minimizer.fit(
                x=np.array([1.0]),
                y=np.array([2.0]),
                weights=np.array([1.0]),
                model=MagicMock(),
                progress_callback='not-callable',
            )

    def test_build_progress_payload(self, minimizer: Bumps) -> None:
        # When
        mock_problem = MagicMock()
        mock_problem.chisq.side_effect = [25.0, 12.5]
        mock_problem.labels.return_value = ['palpha', 'pbeta']
        mock_problem.getp.return_value = np.array([1.0, 2.0])

        point = np.array([1.0, 2.0])
        nllf = 12.5

        # Then
        payload = minimizer._build_progress_payload(mock_problem, 7, point, nllf)

        # Expect
        assert payload == {
            'iteration': 7,
            'chi2': 25.0,
            'reduced_chi2': 12.5,
            'parameter_values': {'alpha': 1.0, 'beta': 2.0},
            'refresh_plots': False,
            'finished': False,
        }
        mock_problem.chisq.assert_any_call(nllf=nllf, norm=False)
        mock_problem.chisq.assert_any_call(nllf=nllf, norm=True)
        # setp should NOT be called – the monitor avoids model re-evaluation
        mock_problem.setp.assert_not_called()

    def test_build_progress_payload_keys_match_lmfit(self, minimizer: Bumps) -> None:
        # When
        mock_problem = MagicMock()
        mock_problem.chisq.side_effect = [10.0, 5.0]
        mock_problem.labels.return_value = ['pa']
        mock_problem.getp.return_value = np.array([5.0])

        minimizer._cached_pars = {'a': MagicMock(value=5.0)}

        # Then
        payload = minimizer._build_progress_payload(mock_problem, 1, np.array([5.0]), nllf=5.0)

        # Expect - same keys as LMFit payload
        expected_keys = {
            'iteration',
            'chi2',
            'reduced_chi2',
            'parameter_values',
            'refresh_plots',
            'finished',
        }
        assert set(payload.keys()) == expected_keys
        assert isinstance(payload['iteration'], int)
        assert isinstance(payload['chi2'], float)
        assert isinstance(payload['reduced_chi2'], float)
        assert isinstance(payload['parameter_values'], dict)
        assert payload['refresh_plots'] is False
        assert payload['finished'] is False

    def test_build_progress_payload_reduced_chi2_positive_dof(self, minimizer: Bumps) -> None:
        # When - use BUMPS chisq helpers for raw and normalized values
        mock_problem = MagicMock()
        mock_problem.chisq.side_effect = [10.0, 5.0]
        mock_problem.labels.return_value = ['pa']
        mock_problem.getp.return_value = np.array([5.0])

        minimizer._cached_pars = {'a': MagicMock(value=5.0)}

        # Then
        payload = minimizer._build_progress_payload(mock_problem, 1, np.array([5.0]), nllf=5.0)

        # Expect
        assert payload['chi2'] == 10.0
        assert payload['reduced_chi2'] == 5.0
        assert mock_problem.chisq.call_args_list == [
            ((), {'nllf': 5.0, 'norm': False}),
            ((), {'nllf': 5.0, 'norm': True}),
        ]

    def test_current_parameter_snapshot(self, minimizer: Bumps) -> None:
        # When
        mock_problem = MagicMock()
        mock_problem.labels.return_value = ['palpha', 'pbeta']

        point = np.array([1.5, 2.5])

        # Then
        snapshot = minimizer._current_parameter_snapshot(mock_problem, point)

        # Expect
        assert snapshot == {'alpha': 1.5, 'beta': 2.5}

    @pytest.mark.parametrize('par_list', [None, [MagicMock(unique_name='alpha')]])
    def test_convert_to_pars_obj_optional_parameter_list(
        self, minimizer: Bumps, par_list, monkeypatch
    ) -> None:
        object_parameters = [MagicMock(unique_name='beta')]
        minimizer._object = MagicMock()
        minimizer._object.get_fit_parameters = MagicMock(return_value=object_parameters)
        monkeypatch.setattr(
            Bumps,
            'convert_to_par_object',
            staticmethod(lambda parameter: parameter.unique_name),
        )

        converted = minimizer.convert_to_pars_obj(par_list)

        expected_parameters = object_parameters if par_list is None else par_list
        assert converted == [parameter.unique_name for parameter in expected_parameters]
        if par_list is None:
            minimizer._object.get_fit_parameters.assert_called_once_with()
        else:
            minimizer._object.get_fit_parameters.assert_not_called()

    def test_make_model_without_parameters_uses_cached_parameters(
        self, minimizer: Bumps, monkeypatch
    ) -> None:
        minimizer._generate_fit_function = MagicMock(
            return_value=MagicMock(return_value=np.array([2.0]))
        )
        minimizer._cached_pars = {'alpha': MagicMock(value=1.0)}
        minimizer.convert_to_par_object = MagicMock(return_value='converted-alpha')

        mock_curve = MagicMock(return_value='curve')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, 'Curve', mock_curve)

        model = minimizer._make_model()
        curve = model(np.array([1.0]), np.array([2.0]), np.array([3.0]))

        assert curve == 'curve'
        minimizer.convert_to_par_object.assert_called_once_with(minimizer._cached_pars['alpha'])
        assert mock_curve.call_args.kwargs['palpha'] == 'converted-alpha'

    def test_bumps_progress_monitor_calls_callback(self, minimizer: Bumps) -> None:
        # When
        callback = MagicMock(return_value=True)
        mock_problem = MagicMock()
        payload_builder = MagicMock(return_value={'iteration': 1})

        monitor = BumpsProgressMonitor(mock_problem, callback, payload_builder)

        mock_history = MagicMock()
        mock_history.step = [5]
        mock_history.point = [np.array([1.0])]
        mock_history.value = [42.0]

        # Then
        monitor(mock_history)

        # Expect
        callback.assert_called_once_with({'iteration': 1})
        payload_builder.assert_called_once_with(
            problem=mock_problem,
            iteration=5,
            point=ANY,
            nllf=42.0,
        )

    def test_fit_exception_restores_values(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        from easyscience import global_object

        global_object.stack.enabled = False

        from easyscience.variable import Parameter

        parameter = MagicMock(Parameter)
        parameter.value = 10.0
        minimizer._cached_pars = {'alpha': parameter}
        minimizer._cached_pars_vals = {'alpha': (1.0, None)}

        mock_driver_instance = MagicMock()
        mock_driver_instance.fit.side_effect = RuntimeError('something broke')
        mock_driver_instance.clip = MagicMock()
        mock_FitDriver = MagicMock(return_value=mock_driver_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitDriver', mock_FitDriver
        )

        mock_FitProblem_instance = MagicMock()
        mock_FitProblem_instance._parameters = []
        mock_FitProblem = MagicMock(return_value=mock_FitProblem_instance)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitProblem', mock_FitProblem
        )

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._resolve_fitclass = MagicMock(return_value=MagicMock(id='amoeba'))

        # Then Expect
        with pytest.raises(FitError):
            minimizer.fit(x=1.0, y=2.0, weights=1)

        assert parameter.value == 1.0

    def test_gen_fit_results_uses_nit_for_budget_check(
        self, minimizer: Bumps, monkeypatch, caplog: 'pytest.LogCaptureFixture'
    ):
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(
            easyscience.fitting.minimizers.minimizer_bumps, 'FitResults', mock_FitResults
        )

        mock_fit_result = MagicMock()
        mock_fit_result.success = True
        mock_fit_result.nit = 99

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        mock_cached_model.dy = 'dy'
        mock_cached_model.pars = {'ppar_1': 0}
        minimizer._cached_model = mock_cached_model

        mock_cached_par = MagicMock()
        mock_cached_par.value = 'par_value_1'
        minimizer._cached_pars = {'par_1': mock_cached_par}

        minimizer._p_0 = 'p_0'
        minimizer._eval_counter = MagicMock(count=2)
        minimizer.evaluate = MagicMock(return_value='evaluate')

        with caplog.at_level(logging.WARNING, logger='easyscience.fitting.bumps'):
            domain_fit_results = minimizer._gen_fit_results(mock_fit_result, max_evaluations=3)

        assert 'maximum optimizer steps of 3' in caplog.text

        assert domain_fit_results.success == False
        assert domain_fit_results.n_evaluations == 2
        assert domain_fit_results.iterations == 100
        assert (
            domain_fit_results.message
            == 'Fit stopped: reached maximum optimizer steps (3); objective evaluated 2 times'
        )
