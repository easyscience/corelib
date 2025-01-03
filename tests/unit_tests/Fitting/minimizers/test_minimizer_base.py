import pytest

from unittest.mock import MagicMock

from inspect import Parameter as InspectParameter
from inspect import Signature
from inspect import _empty

from easyscience.fitting.minimizers.minimizer_base import MinimizerBase
from easyscience.fitting.minimizers.utils import FitError
from easyscience.Objects.variable import Parameter

class TestMinimizerBase():
    @pytest.fixture
    def minimizer(self):
        # This avoids the error: TypeError: Can't instantiate abstract class with abstract methods __init__
        MinimizerBase.__abstractmethods__ = set()
        MinimizerBase.supported_methods = MagicMock(return_value=['method'])

        self._mock_minimizer_enum = MagicMock(package='package', method='method') 
        minimizer = MinimizerBase(
            obj='obj',
            fit_function='fit_function',
            minimizer_enum=self._mock_minimizer_enum
        )
        return minimizer
    
    def test_init_exception(self):
        # When Then
        MinimizerBase.__abstractmethods__ = set()
        MinimizerBase.supported_methods = MagicMock(return_value=['method'])

        # Expect
        with pytest.raises(FitError):
            MinimizerBase(
                obj='obj',
                fit_function='fit_function', 
                minimizer_enum=MagicMock(package='package', method='not_a_method')
            )

    def test_init(self, minimizer: MinimizerBase):
        assert minimizer._object == 'obj'
        assert minimizer._original_fit_function == 'fit_function'
        assert minimizer._method == 'method'
        assert minimizer._cached_pars == {}
        assert minimizer._cached_pars_vals == {}
        assert minimizer._cached_model == None
        assert minimizer._fit_function == None
        assert minimizer._constraints == []
    
    def test_enum(self, minimizer: MinimizerBase):
        assert minimizer.enum == self._mock_minimizer_enum

    def test_evaluate(self, minimizer: MinimizerBase):
        # When
        minimizer._fit_function = MagicMock(return_value='fit_function_return')
        minimizer._prepare_parameters = MagicMock(return_value={'prepared_parms_key': 'prepared_parms_val'})

        # Then
        result = minimizer.evaluate('x', minimizer_parameters={'parms_key': 'parms_val'}, kwargs={'kwargs_key': 'kwargs_val'})

        # Expect
        assert result == 'fit_function_return'
        minimizer._fit_function.assert_called_once_with('x', prepared_parms_key='prepared_parms_val', kwargs={'kwargs_key': 'kwargs_val'})
        minimizer._prepare_parameters.assert_called_once_with({'parms_key': 'parms_val'})

    def test_evaluate_no_fit_function(self, minimizer: MinimizerBase):
        # When
        mock_fit_function = MagicMock()
        minimizer._fit_function = None
        minimizer._prepare_parameters = MagicMock(return_value={'prepared_parms_key': 'prepared_parms_val'})
        minimizer._generate_fit_function = MagicMock(return_value=mock_fit_function)

        # Then
        minimizer.evaluate('x', minimizer_parameters={'parms_key': 'parms_val'}, kwargs={'kwargs_key': 'kwargs_val'})

        # Expect
        mock_fit_function.assert_called_once_with('x', prepared_parms_key='prepared_parms_val', kwargs={'kwargs_key': 'kwargs_val'})
        minimizer._prepare_parameters.assert_called_once_with({'parms_key': 'parms_val'})

    def test_evaluate_no_parameters(self, minimizer: MinimizerBase):
        # When
        minimizer._fit_function = MagicMock(return_value='fit_function_return')
        minimizer._prepare_parameters = MagicMock(return_value={'parms_key': 'parms_val'})

        # Then
        minimizer.evaluate('x')

        # Expect
        minimizer._prepare_parameters.assert_called_once_with({})
        minimizer._fit_function.assert_called_once_with('x', parms_key='parms_val')

    def test_evaluate_exception(self, minimizer: MinimizerBase):
        # When
        minimizer_parameters = 'not dict type'

        # Then Expect
        with pytest.raises(TypeError):
            minimizer.evaluate('x', minimizer_parameters=minimizer_parameters)

    def test_prepare_parameters(self, minimizer: MinimizerBase):
        # When
        parameters = {
            'pa': 1,
            'pb': 2
        }

        minimizer._cached_pars = {
            'a': MagicMock(),
            'b': MagicMock(),
            'c': MagicMock()
        }
        minimizer._cached_pars['a'].raw_value = 3
        minimizer._cached_pars['b'].raw_value = 4
        minimizer._cached_pars['c'].raw_value = 5

        # Then
        parameters = minimizer._prepare_parameters(parameters)

        # Expect
        assert parameters == {
            'pa': 1,
            'pb': 2,
            'pc': 5
        }

    def test_generate_fit_function(self, minimizer: MinimizerBase) -> None:
        # When
        minimizer._original_fit_function = MagicMock(return_value='fit_function_result')

        mock_fit_constraint = MagicMock()
        minimizer.fit_constraints = MagicMock(return_value=[mock_fit_constraint])

        minimizer._object = MagicMock()
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.unique_name = 'mock_parm_1'
        mock_parm_1.value = 1.0
        mock_parm_1.error = 0.1
        mock_parm_2 = MagicMock(Parameter)
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
        assert str(fit_function.__signature__) == '(x, pmock_parm_1=1.0, pmock_parm_2=2.0)'

    def test_create_signature(self, minimizer: MinimizerBase) -> None:
        # When
        mock_parm_1 = MagicMock(Parameter)
        mock_parm_1.value = 1.0
        mock_parm_2 = MagicMock(Parameter)
        mock_parm_2.value = 2.0
        pars = {1: mock_parm_1, 2: mock_parm_2}

        # Then
        signature = minimizer._create_signature(pars)
        
        # Expect
        wrapped_parameters = [
            InspectParameter('x', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty),
            InspectParameter('p1', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty, default=1.0),
            InspectParameter('p2', InspectParameter.POSITIONAL_OR_KEYWORD, annotation=_empty, default=2.0)
        ]
        expected_signature = Signature(wrapped_parameters)
        assert signature == expected_signature

    def test_get_method_dict(self, minimizer: MinimizerBase) -> None:
        # When Then
        result = minimizer._get_method_kwargs()

        # Expect
        assert result == {'method': 'method'}

    def test_get_method_dict_no_self(self, minimizer: MinimizerBase) -> None:
        # When 
        minimizer._method = None

        # Then
        result = minimizer._get_method_kwargs()

        # Expect
        assert result == {}

    def test_get_method_dict_supported_method(self, minimizer: MinimizerBase) -> None:
        # When
        minimizer.supported_methods = MagicMock(return_value=['supported_method'])

        # Then
        result = minimizer._get_method_kwargs('supported_method')

        # Expect
        assert result == {'method': 'supported_method'}

    def test_get_method_dict_not_supported_method(self, minimizer: MinimizerBase) -> None:
        # When
        minimizer.supported_methods = MagicMock(return_value=['supported_method'])

        # Then Expect
        with pytest.raises(FitError):
            result = minimizer._get_method_kwargs('not_supported_method')

