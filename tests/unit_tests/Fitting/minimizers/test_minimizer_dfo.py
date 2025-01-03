import pytest

from unittest.mock import MagicMock
import numpy as np

import easyscience.fitting.minimizers.minimizer_dfo
from easyscience.Objects.variable import Parameter

from easyscience.fitting.minimizers.minimizer_dfo import DFO
from easyscience.fitting.minimizers.utils import FitError


class TestDFOFit():
    @pytest.fixture
    def minimizer(self) -> DFO:
        minimizer = DFO(
            obj='obj',
            fit_function='fit_function', 
            minimizer_enum=MagicMock(package='dfo', method='leastsq')
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
                minimizer_enum=MagicMock(package='dfo', method='not_leastsq')
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
        cached_par.raw_value = 1
        cached_pars = {'mock_parm_1': cached_par}
        minimizer._cached_pars = cached_pars

        # Then
        result = minimizer.fit(x=1.0, y=2.0)

        # Expect
        assert result == 'gen_fit_results'
        minimizer._dfo_fit.assert_called_once_with(cached_pars, mock_model)
        minimizer._make_model.assert_called_once_with(parameters=None)
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit', 1.4142135623730951)
        mock_model_function.assert_called_once_with(1.0, 2.0, 1.4142135623730951)

    def test_generate_fit_function(self, minimizer: DFO) -> None:
        # When
        minimizer._original_fit_function = MagicMock(return_value='fit_function_result')

        mock_fit_constraint = MagicMock()
        minimizer.fit_constraints = MagicMock(return_value=[mock_fit_constraint])

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
        mock_fit_constraint.assert_called_once_with()
        minimizer._original_fit_function.assert_called_once_with([10.0])
        assert minimizer._cached_pars['mock_parm_1'] == mock_parm_1
        assert minimizer._cached_pars['mock_parm_2'] == mock_parm_2

    def test_make_model(self, minimizer: DFO) -> None:
        # When
        mock_fit_function = MagicMock(return_value=np.array([11, 22]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        mock_parm_1.raw_value = 1000.0
        mock_parm_2 = MagicMock()
        mock_parm_2.unique_name = 'mock_parm_2'
        mock_parm_2.raw_value = 2000.0

        # Then
        model = minimizer._make_model(parameters=[mock_parm_1, mock_parm_2])
        residuals_for_model = model(x=np.array([1, 2]), y=np.array([10, 20]), weights=np.array([100, 200]))

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        assert all(np.array([-0.01, -0.01]) == residuals_for_model(np.array([1111, 2222])))
        assert all(mock_fit_function.call_args[0][0] == np.array([1, 2]))
        assert mock_fit_function.call_args[1] == {'pmock_parm_1': 1111, 'pmock_parm_2': 2222}

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
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, "FitResults", mock_FitResults)

        mock_fit_result = MagicMock()
        mock_fit_result.flag = False

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.raw_value = 'par_raw_value_1'
        mock_cached_par_2 = MagicMock()
        mock_cached_par_2.raw_value = 'par_raw_value_2'
        minimizer._cached_pars = {'par_1': mock_cached_par_1, 'par_2': mock_cached_par_2}

        minimizer._p_0 = 'p_0' 
        minimizer.evaluate = MagicMock(return_value='evaluate')

        # Then
        domain_fit_results = minimizer._gen_fit_results(mock_fit_result, 'weights', **{'kwargs_set_key': 'kwargs_set_val'})

        # Expect
        assert domain_fit_results == mock_domain_fit_results
        assert domain_fit_results.kwargs_set_key == 'kwargs_set_val'
        assert domain_fit_results.success == True 
        assert domain_fit_results.y_obs == 'y'
        assert domain_fit_results.x == 'x'
        assert domain_fit_results.p == {'ppar_1': 'par_raw_value_1', 'ppar_2': 'par_raw_value_2'}
        assert domain_fit_results.p0 == 'p_0'
        assert domain_fit_results.y_calc == 'evaluate'
        assert domain_fit_results.y_err == 'weights'
        assert str(domain_fit_results.minimizer_engine) == "<class 'easyscience.fitting.minimizers.minimizer_dfo.DFO'>"
        assert domain_fit_results.fit_args is None
        minimizer.evaluate.assert_called_once_with('x', minimizer_parameters={'ppar_1': 'par_raw_value_1', 'ppar_2': 'par_raw_value_2'})

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
        mock_results.msg = 'Success'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, "dfols", mock_dfols)

        # Then
        results = minimizer._dfo_fit(pars, 'model', **kwargs)
        
        # Expect
        assert results == mock_results
        assert mock_dfols.solve.call_args[0][0] == 'model'
        assert all(mock_dfols.solve.call_args[0][1] == np.array([1., 2.]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][0] == np.array([0.1, 0.2]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][1] == np.array([10., 20.]))
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
        mock_results.msg = 'Success'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, "dfols", mock_dfols)

        # Then
        results = minimizer._dfo_fit(pars, 'model', **kwargs)
        
        # Expect
        assert results == mock_results
        assert mock_dfols.solve.call_args[0][0] == 'model'
        assert all(mock_dfols.solve.call_args[0][1] == np.array([1., 2.]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][0] == np.array([-np.inf, 0.2]))
        assert all(mock_dfols.solve.call_args[1]['bounds'][1] == np.array([10., 20.]))
        assert not 'scaling_within_bounds' in list(mock_dfols.solve.call_args[1].keys())
        assert 'kwargs_set_key' in list(mock_dfols.solve.call_args[1].keys())
        assert mock_dfols.solve.call_args[1]['kwargs_set_key'] == 'kwargs_set_val'

    def test_dfo_fit_exception(self, minimizer: DFO, monkeypatch):
        # When
        pars = {1: MagicMock(Parameter)}
        kwargs = {'kwargs_set_key': 'kwargs_set_val'}

        mock_dfols = MagicMock()
        mock_results = MagicMock()
        mock_results.msg = 'Failed'
        mock_dfols.solve = MagicMock(return_value=mock_results)

        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_dfo, "dfols", mock_dfols)

        # Then Expect
        with pytest.raises(FitError):
            minimizer._dfo_fit(pars, 'model', **kwargs)