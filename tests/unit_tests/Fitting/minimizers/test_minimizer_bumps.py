import pytest

from unittest.mock import MagicMock
import numpy as np

import easyscience.fitting.minimizers.minimizer_bumps

from easyscience.fitting.minimizers.minimizer_bumps import Bumps
from easyscience.fitting.minimizers.utils import FitError


class TestBumpsFit():
    @pytest.fixture
    def minimizer(self) -> Bumps:
        minimizer = Bumps(
            obj='obj',
            fit_function='fit_function', 
            minimizer_enum=MagicMock(package='bumps', method='amoeba')
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
                minimizer_enum=MagicMock(package='bumps', method='not_amoeba')
            )

    def test_all_methods(self, minimizer: Bumps) -> None:
        # When Then Expect
        assert minimizer.all_methods() == ['amoeba', 'de', 'dream', 'newton', 'scipy.leastsq', 'lm']

    def test_supported_methods(self, minimizer: Bumps) -> None:
        # When Then Expect
        assert set(minimizer.supported_methods()) == set(['scipy.leastsq','newton', 'lm', 'amoeba'])

    def test_fit(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        from easyscience import global_object
        global_object.stack.enabled = False

        mock_bumps_fit = MagicMock(return_value='fit')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, "bumps_fit", mock_bumps_fit)

        mock_FitProblem = MagicMock(return_value='fit_problem')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, "FitProblem", mock_FitProblem)

        mock_model = MagicMock()
        mock_model_function = MagicMock(return_value=mock_model)
        minimizer._make_model = MagicMock(return_value=mock_model_function)
        minimizer._set_parameter_fit_result = MagicMock()
        minimizer._gen_fit_results = MagicMock(return_value='gen_fit_results')

        cached_par = MagicMock()
        cached_par.value = 1
        cached_pars = {'mock_parm_1': cached_par}
        minimizer._cached_pars = cached_pars

        # Then
        result = minimizer.fit(x=1.0, y=2.0)

        # Expect
        assert result == 'gen_fit_results'
        mock_bumps_fit.assert_called_once_with('fit_problem', method='amoeba')
        minimizer._make_model.assert_called_once_with(parameters=None)
        minimizer._set_parameter_fit_result.assert_called_once_with('fit', False)
        minimizer._gen_fit_results.assert_called_once_with('fit')
        mock_model_function.assert_called_once_with(1.0, 2.0, 1.4142135623730951)
        mock_FitProblem.assert_called_once_with(mock_model)
 

    def test_make_model(self, minimizer: Bumps, monkeypatch) -> None:
        # When
        mock_fit_function = MagicMock(return_value=np.array([11, 22]))
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        mock_parm_1 = MagicMock()
        mock_parm_1.unique_name = 'mock_parm_1'
        minimizer.convert_to_par_object = MagicMock(return_value='converted_parm_1') 

        mock_Curve = MagicMock(return_value='curve')
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, "Curve", mock_Curve)

        # Then
        model = minimizer._make_model(parameters=[mock_parm_1])
        curve_for_model = model(x=np.array([1, 2]), y=np.array([10, 20]), weights=np.array([100, 200]))

        # Expect
        minimizer._generate_fit_function.assert_called_once_with()
        assert mock_Curve.call_args[0][0] == mock_fit_function
        assert all(mock_Curve.call_args[0][1] == np.array([1,2]))
        assert all(mock_Curve.call_args[0][2] == np.array([10,20]))
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
        mock_cached_model._pnames = ['pa', 'pb']
        minimizer._cached_model = mock_cached_model

        mock_fit_result = MagicMock()
        mock_fit_result.x = [1.0, 2.0]
        mock_fit_result.dx = [0.1, 0.2]

        # Then
        minimizer._set_parameter_fit_result(mock_fit_result, False)

        # Expect
        assert minimizer._cached_pars['a'].value == 1.0
        assert minimizer._cached_pars['a'].error == 0.1
        assert minimizer._cached_pars['b'].value == 2.0
        assert minimizer._cached_pars['b'].error == 0.2

    def test_gen_fit_results(self, minimizer: Bumps, monkeypatch):
        # When
        mock_domain_fit_results = MagicMock()
        mock_FitResults = MagicMock(return_value=mock_domain_fit_results)
        monkeypatch.setattr(easyscience.fitting.minimizers.minimizer_bumps, "FitResults", mock_FitResults)

        mock_fit_result = MagicMock()
        mock_fit_result.success = True

        mock_cached_model = MagicMock()
        mock_cached_model.x = 'x'
        mock_cached_model.y = 'y'
        mock_cached_model.dy = 'dy'
        mock_cached_model._pnames = ['ppar_1', 'ppar_2']
        minimizer._cached_model = mock_cached_model

        mock_cached_par_1 = MagicMock()
        mock_cached_par_1.value = 'par_value_1'
        mock_cached_par_2 = MagicMock()
        mock_cached_par_2.value = 'par_value_2'
        minimizer._cached_pars = {'par_1': mock_cached_par_1, 'par_2': mock_cached_par_2}

        minimizer._p_0 = 'p_0' 
        minimizer.evaluate = MagicMock(return_value='evaluate')

        # Then
        domain_fit_results = minimizer._gen_fit_results(mock_fit_result, **{'kwargs_set_key': 'kwargs_set_val'})

        # Expect
        assert domain_fit_results == mock_domain_fit_results
        assert domain_fit_results.kwargs_set_key == 'kwargs_set_val'
        assert domain_fit_results.success == True 
        assert domain_fit_results.y_obs == 'y'
        assert domain_fit_results.x == 'x'
        assert domain_fit_results.p == {'ppar_1': 'par_value_1', 'ppar_2': 'par_value_2'}
        assert domain_fit_results.p0 == 'p_0'
        assert domain_fit_results.y_calc == 'evaluate'
        assert domain_fit_results.y_err == 'dy'
        assert str(domain_fit_results.minimizer_engine) == "<class 'easyscience.fitting.minimizers.minimizer_bumps.Bumps'>"
        assert domain_fit_results.fit_args is None
        minimizer.evaluate.assert_called_once_with('x', minimizer_parameters={'ppar_1': 'par_value_1', 'ppar_2': 'par_value_2'})
