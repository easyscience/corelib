import pytest

from unittest.mock import MagicMock

import easyscience.fitting.minimizers.minimizer_lmfit
from easyscience.fitting.minimizers.minimizer_lmfit import LMFit
from easyscience.Objects.variable import Parameter
from lmfit import Parameter as LMParameter
from easyscience.fitting.minimizers.utils import FitError


class TestLMFit():
    @pytest.fixture
    def minimizer(self) -> LMFit:
        minimizer = LMFit(
            obj='obj',
            fit_function='fit_function', 
            minimizer_enum=MagicMock(package='lm', method='leastsq')
        )
        return minimizer

    def test_init(self, minimizer: LMFit) -> None:
        assert minimizer.package == 'lmfit'

    def test_init_exception(self) -> None:
        with pytest.raises(FitError):
            LMFit(
                obj='obj',
                fit_function='fit_function', 
                minimizer_enum=MagicMock(package='dfo', method='not_leastsq')
            )

    def test_make_model(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMModel", mock_LMModel)
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
        mock_LMModel.assert_called_once_with('model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2'])
        mock_lm_model.set_param_hint.assert_called_with('pkey_2', value=2.0, min=-20.0, max=20.0)
        assert mock_lm_model.set_param_hint.call_count == 2
        assert model == mock_lm_model

    def test_make_model_no_pars(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_model = MagicMock()
        mock_LMModel = MagicMock(return_value=mock_lm_model)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMModel", mock_LMModel)
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
        mock_LMModel.assert_called_once_with('model', independent_vars=['x'], param_names=['pkey_1', 'pkey_2'])
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
        result = minimizer.fit(x=1.0, y=2.0)

        # Expect
        assert result == 'gen_fit_results'
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, max_nfev=None, fit_kws={}, method='leastsq')
        minimizer._make_model.assert_called_once_with()
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit')

    def test_fit_model(self, minimizer: LMFit) -> None:
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, model=mock_model)

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, max_nfev=None, fit_kws={}, method='leastsq')
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
        minimizer.fit(x=1.0, y=2.0, method='method_passed')

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, max_nfev=None, fit_kws={}, method='method_passed')
        minimizer.supported_methods.assert_called_once_with()

    def test_fit_kwargs(self, minimizer: LMFit) -> None:
        # When
        mock_model = MagicMock()
        mock_model.fit = MagicMock(return_value='fit')
        minimizer._make_model = MagicMock(return_value=mock_model)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then
        minimizer.fit(x=1.0, y=2.0, minimizer_kwargs={'minimizer_key': 'minimizer_val'}, engine_kwargs={'engine_key': 'engine_val'})

        # Expect
        mock_model.fit.assert_called_once_with(2.0, x=1.0, weights=0.7071067811865475, max_nfev=None, fit_kws={'minimizer_key': 'minimizer_val'}, method='leastsq', engine_key='engine_val')

    def test_fit_exception(self, minimizer: LMFit) -> None:
        # When
        minimizer._make_model = MagicMock(side_effect=Exception('Exception'))
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        # Then Expect
        with pytest.raises(FitError):
            minimizer.fit(x=1.0, y=2.0)

    def test_convert_to_pars_obj(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        minimizer._object = MagicMock()
        minimizer._object.get_fit_parameters = MagicMock(return_value = ['parm_1', 'parm_2'])

        minimizer.convert_to_par_object = MagicMock(return_value='convert_to_par_object')

        mock_lm_parameter = MagicMock()
        mock_lm_parameter.add_many = MagicMock(return_value='add_many')
        mock_LMParameters = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMParameters", mock_LMParameters)

        # Then
        pars = minimizer.convert_to_pars_obj()

        # Expect
        assert pars == 'add_many'
        assert minimizer.convert_to_par_object.call_count == 2
        minimizer._object.get_fit_parameters.assert_called_once_with()
        minimizer.convert_to_par_object.assert_called_with('parm_2')
        mock_lm_parameter.add_many.assert_called_once_with(['convert_to_par_object', 'convert_to_par_object'])

    def test_convert_to_pars_obj_with_parameters(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        minimizer.convert_to_par_object = MagicMock(return_value='convert_to_par_object')

        mock_lm_parameter = MagicMock()
        mock_lm_parameter.add_many = MagicMock(return_value='add_many')
        mock_LMParameters = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMParameters", mock_LMParameters)

        # Then
        pars = minimizer.convert_to_pars_obj(['parm_1', 'parm_2'])

        # Expect
        assert pars == 'add_many'
        assert minimizer.convert_to_par_object.call_count == 2
        minimizer.convert_to_par_object.assert_called_with('parm_2')
        mock_lm_parameter.add_many.assert_called_once_with(['convert_to_par_object', 'convert_to_par_object'])

    def test_convert_to_par_object(self, minimizer: LMFit, monkeypatch) -> None:
        # When
        mock_lm_parameter = MagicMock()
        mock_LMParameter = MagicMock(return_value=mock_lm_parameter)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "LMParameter", mock_LMParameter)

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
        mock_LMParameter.assert_called_once_with('pkey_converted', value=1.0, vary=False, min=-10.0, max=10.0, expr=None, brute_step=None)

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
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_lmfit, "FitResults", mock_FitResults)

        mock_fit_result = MagicMock()
        mock_fit_result.success ='success'
        mock_fit_result.data = 'data'
        mock_fit_result.userkws = {'x': 'x_val'}
        mock_fit_result.values = 'values'
        mock_fit_result.init_values = 'init_values'
        mock_fit_result.best_fit = 'best_fit'
        mock_fit_result.weights = 10

        # Then
        domain_fit_results = minimizer._gen_fit_results(mock_fit_result, **{'kwargs_set_key': 'kwargs_set_val'})

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
        assert str(domain_fit_results.minimizer_engine) == "<class 'easyscience.fitting.minimizers.minimizer_lmfit.LMFit'>"
        assert domain_fit_results.fit_args is None

