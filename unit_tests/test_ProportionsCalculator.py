# Appending module to sys directory list through relative path
import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))), 'bayesian_proportions_calculator'))

import pytest
import pandas as pd
import numpy as np

# from bayesian_calculator import PosteriorGenerator
from bayesian_calculator.bayesian_proportions_calculator import ProportionsCalculator as PC

# General fixtures -----------------------------------------------------------------------------------------
# String fixture
@pytest.fixture
def make_str():
    return 'foobar'


# String fixture
@pytest.fixture
def make_str_integer():
    return '1'


# Decimal fixture
@pytest.fixture
def make_str_decimal():
    return 1.5


# 0 fixture
@pytest.fixture
def make_zero():
    return 0


# Negative fixture
@pytest.fixture
def make_negative():
    return -1


# Tests -----------------------------------------------------------------------------------------
# ----------------------------------
def test_PosteriorGenerator_init():
    # Ensuring default values are correct
    d = PC(N_ctrl=1, N_test=1)
    assert d.N_test == 1
    assert d.ctrl_successes == 0
    assert d.N_ctrl == 1
    assert d.test_successes == 0
    assert d.ctrl_name == 'Ctrl'
    assert d.test_name == 'Test'
    assert d.ctrl_failures == 1
    assert d.test_failures == 1
    assert d.ctrl_success_rate == 0
    assert d.test_success_rate == 0
    assert d.prior_alpha_ctrl == 1
    assert d.prior_beta_ctrl == 1
    assert d.prior_alpha_test == 1
    assert d.prior_beta_test == 1

    # Ensuring that passing values are correct
    d = PC(N_ctrl=1, N_test=1, ctrl_successes=1, test_successes=1
                           , ctrl_name='Control_name', test_name='Test_name'
                           , prior_alpha_ctrl=5, prior_beta_ctrl=10
                           , prior_alpha_test=15, prior_beta_test=100)
    assert d.N_test == 1
    assert d.ctrl_successes == 1
    assert d.N_ctrl == 1
    assert d.test_successes == 1
    assert d.ctrl_name == 'Control_name'
    assert d.test_name == 'Test_name'
    assert d.ctrl_failures == 0
    assert d.test_failures == 0
    assert d.ctrl_success_rate == 100
    assert d.test_success_rate == 100
    assert d.prior_alpha_ctrl == 5
    assert d.prior_beta_ctrl == 10
    assert d.prior_alpha_test == 15
    assert d.prior_beta_test == 100


# ----------------------------------
def test_checks_init_integer_type(make_str, make_str_integer, make_str_decimal):
    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=make_str, N_test=make_str, ctrl_successes=make_str, test_successes=make_str)
        d.checks_init_integer_type()
    assert str(e.value) == str(make_str) + ' should be an integer'

    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=make_str_integer, N_test=make_str_integer, ctrl_successes=make_str_integer,
                               test_successes=make_str_integer)
        d.checks_init_integer_type()
    assert str(e.value) == str(make_str_integer) + ' should be an integer'

    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=make_str_decimal, N_test=make_str_decimal, ctrl_successes=make_str_decimal,
                               test_successes=make_str_decimal)
        d.checks_init_integer_type()
    assert str(e.value) == str(make_str_decimal) + ' should be an integer'


# ----------------------------------
def test_checks_init_samples_greater_0(make_zero, make_negative):
    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=make_zero, N_test=make_zero)
        d.checks_init_samples_greater_0()
    assert str(e.value) == 'N_ctrl and N_test must be greater than 0'

    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=make_negative, N_test=make_negative)
        d.checks_init_samples_greater_0()
    assert str(e.value) == 'N_ctrl and N_test must be greater than 0'


# ----------------------------------
def test_check_init_successes_positive(make_negative):
    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=1, N_test=1, ctrl_successes=make_negative, test_successes=make_negative)
        d.check_init_successes_positive()
    assert str(e.value) == 'ctrl_successes and test_successes cant be negative'


# ----------------------------------
def test_check_init_samples_greater_successes():
    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=1, N_test=1, ctrl_successes=2, test_successes=2)
        d.check_init_samples_greater_successes()
    assert str(e.value) == 'Successes cant be bigger than Samples'


# ----------------------------------
def test_check_init_alpha_beta_greater_0(make_zero, make_negative):
    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=2, N_test=2
                               , prior_alpha_ctrl=make_zero, prior_beta_ctrl=make_zero, prior_alpha_test=make_zero,
                               prior_beta_test=make_zero)
        d.check_init_alpha_beta_greater_0()
    assert str(e.value) == 'Alpha and betas must be greater than 0'

    with pytest.raises(ValueError) as e:
        d = PC(N_ctrl=2, N_test=2
                               , prior_alpha_ctrl=make_negative, prior_beta_ctrl=make_negative,
                               prior_alpha_test=make_negative, prior_beta_test=make_negative)
        d.check_init_alpha_beta_greater_0()
    assert str(e.value) == 'Alpha and betas must be greater than 0'


# ----------------------------------
def test_input_results_as_dataframe():
    d = PC(N_ctrl=1234, N_test=1432, ctrl_successes=123, test_successes=321
                           , ctrl_name='experiment_123', test_name='experiment_321')
    df = d.inputResultsAsDataframe()

    should_equal_df = pd.DataFrame([['Ctrl', 'experiment_123', 1234, 123, 1111, 9.97]
                                       , ['Test', 'experiment_321', 1432, 321, 1111, 22.42]]
                                   , columns=['Experiment', 'Name', 'Impressions', 'Successes', 'Failures',
                                              'Success_Rate'])

    assert df.equals(should_equal_df)

# ----------------------------------
def test_posterior_beta_distribution():
    d = PC(N_ctrl=1234, N_test=1432)
    beta_distrib = d.PosteriorBetaDistribution(prior_a = 1, prior_b = 1, N = 5, successes = 2, interval_percentile = 0.95)

    assert beta_distrib.get('updated_alpha') == 3
    assert beta_distrib.get('updated_beta') == 4
    assert np.array_equal(beta_distrib.get('x_samples'),np.array([0., 0.25, 0.5, 0.75, 1.]))
    assert np.array_equal(np.round(beta_distrib.get('beta_pdf'), 4), np.array([0., 158.2031, 187.5, 52.7344, 0.]))
    assert np.array_equal(np.round(beta_distrib.get('beta_cdf'), 4), np.array([0., 16.9434, 65.625, 96.2402, 100.]))
    assert np.round(beta_distrib.get('beta_median'), 4) == 42.1407
    assert np.round(beta_distrib.get('beta_mean'), 4) == 42.8571
    assert np.round(beta_distrib.get('beta_variance'), 4) == 3.0612
    assert np.round(beta_distrib.get('beta_std'), 4) == 17.4964
    assert np.round(beta_distrib.get('beta_interval')[0], 4) == 0.1181
    assert np.round(beta_distrib.get('beta_interval')[1], 4) == 0.7772

    dict_keys = dict.fromkeys(['updated_alpha', 'updated_beta', 'x_samples', 'beta_distribution_object', 'beta_pdf'
                                  , 'beta_cdf', 'beta_median', 'beta_mean', 'beta_variance', 'beta_std', 'beta_interval'])
    for key_ in beta_distrib.keys():
        assert key_ in dict_keys