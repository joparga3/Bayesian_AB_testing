# Appending module to sys directory list through relative path
import sys
from os.path import dirname, abspath, join
sys.path.append(join(dirname(dirname(abspath(__file__))),'bayesian_calculator'))

# Importing required modules
import numpy as np
import unittest
from bayesian_calculator import PosteriorGenerator

# Class for unit testing
class TestPosteriorGenerator(unittest.TestCase):

    # ----------------------------------------------------------------------------------------------------------------
    def test_init(self):
        # Ensuring default values are correct
        d = PosteriorGenerator.PosteriorGenerator(N_ctrl = 1, N_test = 1)
        self.assertEqual(d.N_ctrl, 1)
        self.assertEqual(d.N_test, 1)
        self.assertEqual(d.ctrl_successes, 0)
        self.assertEqual(d.test_successes, 0)
        self.assertEqual(d.ctrl_name, 'Ctrl')
        self.assertEqual(d.ctrl_failures, 1)
        self.assertEqual(d.test_failures, 1)
        self.assertEqual(d.ctrl_success_rate, 0)
        self.assertEqual(d.test_success_rate, 0)
        self.assertEqual(d.prior_alpha_ctrl, 1)
        self.assertEqual(d.prior_beta_ctrl, 1)
        self.assertEqual(d.prior_alpha_test, 1)
        self.assertEqual(d.prior_beta_test, 1)

        # Ensuring that passing values are correct
        d = PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, ctrl_successes=1, test_successes=1
                                                  , ctrl_name='Control_name', test_name='Test_name'
                                                  , prior_alpha_ctrl=5, prior_beta_ctrl=10
                                                  , prior_alpha_test=15, prior_beta_test=100)
        self.assertEqual(d.N_ctrl, 1)
        self.assertEqual(d.N_test, 1)
        self.assertEqual(d.ctrl_successes, 1)
        self.assertEqual(d.test_successes, 1)
        self.assertEqual(d.ctrl_name, 'Control_name')
        self.assertEqual(d.test_name, 'Test_name')
        self.assertEqual(d.ctrl_failures, 0)
        self.assertEqual(d.test_failures, 0)
        self.assertEqual(d.ctrl_success_rate, 100)
        self.assertEqual(d.test_success_rate, 100)
        self.assertEqual(d.prior_alpha_ctrl, 5)
        self.assertEqual(d.prior_beta_ctrl, 10)
        self.assertEqual(d.prior_alpha_test, 15)
        self.assertEqual(d.prior_beta_test, 100)

        # Ensuring assert statements work
        with self.assertRaises(ValueError):
            PosteriorGenerator.PosteriorGenerator(N_ctrl=0, N_test=0)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=0, N_test=1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=0)

            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, ctrl_successes=-1, test_successes=-1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, ctrl_successes=-1, test_successes=1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, ctrl_successes=1, test_successes=-1)

            PosteriorGenerator.PosteriorGenerator(N_ctrl=5, N_test=5, ctrl_successes=6, test_successes=6)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=5, N_test=1, ctrl_successes=6, test_successes=1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=5, ctrl_successes=1, test_successes=6)

            PosteriorGenerator.PosteriorGenerator(N_ctrl='5', N_test=5, ctrl_successes=6, test_successes=6)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=5, N_test='5', ctrl_successes=6, test_successes=6)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=5, N_test=5, ctrl_successes='6', test_successes=6)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=5, N_test=5, ctrl_successes=6, test_successes='6')

            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_alpha_ctrl=-1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_alpha_ctrl=0)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_beta_ctrl=-1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_beta_ctrl=0)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_alpha_test=-1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_alpha_test=0)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_beta_test=-1)
            PosteriorGenerator.PosteriorGenerator(N_ctrl=1, N_test=1, prior_beta_test=0)


    # ----------------------------------------------------------------------------------------------------------------
    def test_inputResultsAsDataframe(self):
        d = PosteriorGenerator.PosteriorGenerator(N_ctrl=100, N_test=100, ctrl_successes=50, test_successes=33
                                                  , ctrl_name='Control_name', test_name='Test_name')
        df = d.inputResultsAsDataframe()

        self.assertEqual(df[df['Experiment'] == 'Ctrl']['Name'].values, 'Control_name')
        self.assertEqual(df[df['Experiment'] == 'Ctrl']['Impressions'].values, 100)
        self.assertEqual(df[df['Experiment'] == 'Ctrl']['Successes'].values, 50)
        self.assertEqual(df[df['Experiment'] == 'Ctrl']['Failures'].values, 50)
        self.assertEqual(df[df['Experiment'] == 'Ctrl']['Success_Rate'].values, 50.0)

        self.assertEqual(df[df['Experiment'] == 'Test']['Name'].values, 'Test_name')
        self.assertEqual(df[df['Experiment'] == 'Test']['Impressions'].values, 100)
        self.assertEqual(df[df['Experiment'] == 'Test']['Successes'].values, 33)
        self.assertEqual(df[df['Experiment'] == 'Test']['Failures'].values, 67)
        self.assertEqual(df[df['Experiment'] == 'Test']['Success_Rate'].values, 33.0)

    # ----------------------------------------------------------------------------------------------------------------
    def test_PosteriorBetaDistribution(self):
        d = PosteriorGenerator.PosteriorGenerator(N_ctrl=100, N_test=100, ctrl_successes=50, test_successes=33
                               , prior_alpha_ctrl=5, prior_beta_ctrl=16
                               , prior_alpha_test=20, prior_beta_test=67
                               )

        # Checking updated alpha and beta returned results
        d_post = d.PosteriorBetaDistribution(prior_a = d.prior_alpha_ctrl, prior_b = d.prior_beta_ctrl
                                             , N = d.N_ctrl, successes = d.ctrl_successes)
        self.assertEqual(d_post.get('updated_alpha'), 55)
        self.assertEqual(d_post.get('updated_beta'), 66)
        self.assertEqual(np.round(d_post.get('beta_median'),4), 0.4543)
        self.assertEqual(np.round(d_post.get('beta_mean'), 4), 0.4545)
        self.assertEqual(np.round(d_post.get('beta_variance'), 4), 0.002)
        self.assertEqual(np.round(d_post.get('beta_std'), 4), 0.0451)
        self.assertEqual(np.round(d_post.get('beta_interval')[0], 4), 0.3671)
        self.assertEqual(np.round(d_post.get('beta_interval')[1], 4), 0.5435)

        # Is there a way to test a distribution??
        # Idea 1. Manually input the expected results...

        # Test that the correct names of return objects






# So that we can run this directly from the script (and not call through the terminal)
if __name__ == '__main__':
    unittest.main()