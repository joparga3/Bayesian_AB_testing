import numpy as np
import pandas as pd
from scipy.stats import beta
import arviz as az

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------------------------------------------------
class ProportionsCalculator:
    # ------------------------------------------------------------------------------------------------------
    def __init__(self, N_ctrl, N_test, ctrl_successes=0, test_successes=0, ctrl_name='Ctrl', test_name='Test'
                 , prior_alpha_ctrl=1, prior_beta_ctrl=1
                 , prior_alpha_test=1, prior_beta_test=1):
        # '''
        #
        # Input parameters:
        # * N_ctrl => number of samples in the control segment
        # * N_test => number of samples in the test segment
        # * ctrl_successes => number of successes in the control segment
        # * test_successes => number of successes in the test segment
        # * ctrl_name => name for the control segment (will be used for plotting and results summary purposes)
        # * test_name => name for the test segment (will be used for plotting and results summary purposes)
        # * prior_alpha_ctrl => prior alpha for the beta distribution representing the control segment
        # * prior_beta_ctrl => prior beta for the beta distribution representing the control segment
        # * prior_alpha_test => prior alpha for the beta distribution representing the test segment
        # * prior_beta_test => prior beta for the beta distribution representing the test segment
        #
        # This method also performs:
        # 1. Various sanity checks on the input data
        #
        # 2. Calculates extra fields
        #   * ctrl_failures => number of failures in the control segment
        #   * test_failures => number of failures in the test segment
        #   * ctrl_success_rate => control success rate
        #   * test_success_rate => test success rate
        #
        # 3. Generate extra [self] instance variables
        # -> Given that we work with simulations (that may not be reproducible), we run the simulation once, and
        #    pass the results to any method that requires them through the [self] instances.
        # -> Defined as None and superceeded (or updated) by BayesianMCMCsimulation() method
        #   * simulation_ctrl_samples => success rate for control segment for all the MCMC simulations
        #   * simulation_test_samples => success rate for test segment for all the MCMC simulations
        #   * simulation_success_rate_difference => difference in success rate between control and test for the MCMC simulations
        #   * simulation_success_rate_ratio => ratio between control and test for the MCMC simulations
        #
        # '''

        # # Ensure that samples and successes are INTs --
        # for i in [N_ctrl, N_test, ctrl_successes, test_successes]:
        #     if isinstance(i, int) == False:
        #         raise ValueError(str(i) + ' should be an integer')
        #     else:
        #         pass
        #
        # # Ensure that samples are greater than 0  --
        # if (N_ctrl <= 0) or (N_test <= 0):
        #     raise ValueError('N_ctrl and N_test must be greater than 0')
        # else:
        #     pass
        #
        # # Ensure that successes are positive --
        # if (ctrl_successes < 0) or (test_successes < 0):
        #     raise ValueError('ctrl_successes and test_successes cant be negative')
        # else:
        #     pass
        #
        # # Ensure that successes are less than samples --
        # if (ctrl_successes > N_ctrl) or (test_successes > N_test):
        #     raise ValueError('Successes cant be bigger than Samples')
        # else:
        #     pass
        #
        # # Ensure that prior alpha and beta are bigger than 0
        # if (prior_alpha_ctrl < 0) or (prior_beta_ctrl < 0) or (prior_alpha_test < 0) or (prior_beta_test < 0):
        #     raise ValueError('ctrl_successes and test_successes cant be negative')
        # else:
        #     pass

        self.ctrl_name = str(ctrl_name)
        self.test_name = str(test_name)
        self.N_ctrl = N_ctrl
        self.N_test = N_test
        self.ctrl_successes = ctrl_successes
        self.test_successes = test_successes
        self.prior_alpha_ctrl = prior_alpha_ctrl
        self.prior_beta_ctrl = prior_beta_ctrl
        self.prior_alpha_test = prior_alpha_test
        self.prior_beta_test = prior_beta_test

        # Checks
        self.checks_init_integer_type()
        self.checks_init_samples_greater_0()
        self.check_init_successes_positive()
        self.check_init_samples_greater_successes()
        self.check_init_alpha_beta_greater_0()

        # Calculate extra fields
        self.ctrl_failures = N_ctrl - ctrl_successes
        self.test_failures = N_test - test_successes
        self.ctrl_success_rate = np.round(100*(1.0*ctrl_successes)/(1.0*N_ctrl),2)
        self.test_success_rate = np.round(100*(1.0*test_successes)/(1.0*N_test),2)

        # Self vars that will be populated later through different methods
        # Initiate them as None in self, and update as required later on.
        self.simulation_ctrl_samples = None
        self.simulation_test_samples = None
        self.simulation_success_rate_difference = None
        self.simulation_success_rate_ratio = None
        # --> run simulation to set the self.simulation_XXX variables for global usage
        self.Bayesian_sampling_simulations()

        # --> run ROPE, HDI and CI
        self.decisionCriteria_assign_intervals()

    # ------------------------------------------------------------------------------------------------------
    def checks_init_integer_type(self):
        # Ensure that samples and successes are INTs --
        for i in [self.N_ctrl, self.N_test, self.ctrl_successes, self.test_successes]:
            if isinstance(i, int) == False:
                raise ValueError(str(i) + ' should be an integer')
            else:
                pass

    # ------------------------------------------------------------------------------------------------------
    def checks_init_samples_greater_0(self):
        # Ensure that samples are greater than 0  --
        if (self.N_ctrl <= 0) or (self.N_test <= 0):
            raise ValueError('N_ctrl and N_test must be greater than 0')
        else:
            pass

    # ------------------------------------------------------------------------------------------------------
    def check_init_successes_positive(self):
        # Ensure that successes are positive --
        if (self.ctrl_successes < 0) or (self.test_successes < 0):
            raise ValueError('ctrl_successes and test_successes cant be negative')
        else:
            pass

    # ------------------------------------------------------------------------------------------------------
    def check_init_samples_greater_successes(self):
        # Ensure that successes are less than samples --
        if (self.ctrl_successes > self.N_ctrl) or (self.test_successes > self.N_test):
            raise ValueError('Successes cant be bigger than Samples')
        else:
            pass

    # ------------------------------------------------------------------------------------------------------
    def check_init_alpha_beta_greater_0(self):
        # Ensure that prior alpha and beta are bigger than 0
        if (self.prior_alpha_ctrl <= 0) or (self.prior_beta_ctrl <= 0) or (self.prior_alpha_test <= 0) or (self.prior_beta_test <= 0):
            raise ValueError('Alpha and betas must be greater than 0')
        else:
            pass

    # ------------------------------------------------------------------------------------------------------
    def inputResultsAsDataframe(self):
        # '''
        # Method to prettify the input results into a dataframe
        # '''

        df = pd.DataFrame([ ['Ctrl', self.ctrl_name , self.N_ctrl, self.ctrl_successes, self.ctrl_failures, self.ctrl_success_rate]
                          , ['Test' , self.test_name, self.N_test, self.test_successes, self.test_failures, self.test_success_rate] ]
                        , columns=['Experiment', 'Name', 'Impressions', 'Successes', 'Failures', 'Success_Rate']
                        )

        return df

    # ----------------------------------------------------------------------------------------------------------------
    def PosteriorBetaDistribution(self, prior_a, prior_b, N, successes, interval_percentile = 0.95):
        # '''
        # Method to generate beta distributions and extract relevant metrics
        #
        # Input:
        # * prior_a => alpha parameter
        # * prior_b => beta parameter
        # * N => number of samples
        # * successes => number of successes
        # * interval_percentile => percentile (equally tailed). Used to calculate the extremes at either side as minimum and maximum range.
        #
        # Output (dictionary form):
        # * Updated alpha and beta parameters
        # * x_samples => success samples from 0 to 1 based on the input sample volumes
        # * beta_distribution_object => beta object from scipy.stats
        # * beta_pdf => beta probability distribution function
        # * beta_cdf => beta cumulative distribution function
        # * Metrics such as median, mean, variance, standard deviation, interval
        #
        # '''

        # Update input alpha and beta values
        updated_alpha = prior_a + successes
        updated_beta  = prior_b + (N - successes)

        # Instantiate a beta object
        beta_distribution_object = beta(updated_alpha, updated_beta)

        # Generate samples representing success rates from 0 to 1, with number of occurrences equal to N
        x_samples = np.linspace(0.0, 1.0, N)

        # Summary metrics for beta distribution
        median_ = 100*beta_distribution_object.median()
        mean_ = 100*beta_distribution_object.mean()
        variance_ = 100*beta_distribution_object.var()
        standard_dev_ = 100*beta_distribution_object.std()
        interval_ = 100*beta_distribution_object.interval(alpha = interval_percentile)

        # Return results
        return {'updated_alpha': updated_alpha
                , 'updated_beta': updated_beta
                , 'x_samples': x_samples
                , 'beta_distribution_object': beta_distribution_object
                , 'beta_pdf': 100*beta_distribution_object.pdf(x_samples)
                , 'beta_cdf': 100*beta_distribution_object.cdf(x_samples)
                , 'beta_median': median_
                , 'beta_mean': mean_
                , 'beta_variance': variance_
                , 'beta_std': standard_dev_
                , 'beta_interval': interval_
                }

    # ----------------------------------------------------------------------------------------------------------------
    def Bayesian_sampling_simulations(self, num_trials = 10000):
        # '''
        #
        # Method to run simulations based on the posterior distributions (which in turn depend on the input experiment results).
        #
        # Output:
        # As mentioned in the __init__ method, this will update the simulation self instance variables.
        #
        # '''

        # print('... Running ' + str(num_trials) + ' simulations ...')

        # Control beta object
        posterior_control = self.PosteriorBetaDistribution(prior_a = self.prior_alpha_ctrl
                                                           , prior_b = self.prior_beta_ctrl
                                                           , N = self.N_ctrl
                                                           , successes = self.ctrl_successes).get('beta_distribution_object')

        # Test beta object
        posterior_test = self.PosteriorBetaDistribution(prior_a=self.prior_alpha_test
                                                        , prior_b=self.prior_beta_test
                                                        , N=self.N_test
                                                        , successes=self.test_successes).get('beta_distribution_object')

        # Running simulation -> success rate for control and test based on sampling from a beta distribution
        control_samples = pd.Series([posterior_control.rvs() for _ in range(num_trials)])
        test_samples = pd.Series([posterior_test.rvs() for _ in range(num_trials)])

        # Differences
        success_rate_difference = test_samples - control_samples
        success_rate_ratio = test_samples / control_samples

        # Assigning to object
        self.simulation_ctrl_samples = control_samples
        self.simulation_test_samples = test_samples
        self.simulation_success_rate_difference = success_rate_difference
        self.simulation_success_rate_ratio = success_rate_ratio

    # ----------------------------------------------------------------------------------------------------------------
    def Bayesian_sampling_simulations_dataframe(self):
        # '''
        # Method to prettify results from the simulation into a dataframe
        # '''

        df = pd.DataFrame(
            [[self.simulation_ctrl_samples, self.simulation_test_samples
             , self.simulation_success_rate_difference, self.simulation_success_rate_ratio]]
            , columns=['Ctrl_samples', 'Test_samples', 'Difference', 'Ratio']
            )

        return df

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_probability(self):
        # '''
        # Method to calculate the probability that test is better than control
        # How many times did test win over control?
        # '''

        num_trials = len(self.simulation_ctrl_samples)
        test_wins = sum(self.simulation_test_samples > self.simulation_ctrl_samples)
        percentage_wins = 100*test_wins/num_trials

        return percentage_wins

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_difference(self):
        # '''
        # Method to calculate metrics for the differences between success rates of test and control
        # Output (dictionary form)
        # * Calculates average, median and standard deviation
        # '''

        return {'simulation_mean_diff': np.mean(self.simulation_success_rate_difference)
                , 'simulation_median_diff': np.median(self.simulation_success_rate_difference)
                , 'simulation_std_diff': np.std(self.simulation_success_rate_difference)
                }

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_relative(self):
        # '''
        # Method to calculate metrics for the ratio between success rates of test and control
        # Output (dictionary form)
        # * Calculates average, median and standard deviation
        # '''

        return {'simulation_mean_relative': 100*(np.mean(self.simulation_success_rate_ratio) - 1)
                , 'simulation_median_relative': 100*(np.median(self.simulation_success_rate_ratio) - 1)
                , 'simulation_std_relative': 100*(np.std(self.simulation_success_rate_ratio))
                }

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_summary(self):
        # '''
        # Method to summarise all findings into a dataframe
        # '''

        # Extract results of a beta distribution based on the control metrics
        control_posterior = self.PosteriorBetaDistribution(prior_a = self.prior_alpha_ctrl
                                                           , prior_b = self.prior_beta_ctrl
                                                           , N = self.N_ctrl
                                                           , successes = self.ctrl_successes)

        # Extract results of a beta distribution based on the test metrics
        test_posterior = self.PosteriorBetaDistribution(prior_a=self.prior_alpha_test
                                                        , prior_b=self.prior_beta_test
                                                        , N=self.N_test
                                                        , successes=self.test_successes)

        # Extract simulation results
        results_diff = self.simulationResults_difference()
        relative_diff = self.simulationResults_relative()

        # Prettify in dataframe
        data = pd.DataFrame([['Median Control success rate & [interval]', str(np.round(control_posterior.get('beta_median'), 2)) + '%, [' + str(np.round(100*control_posterior.get('beta_interval')[0], 2)) + '%, ' + str(np.round(100*control_posterior.get('beta_interval')[1], 2)) + '%]']
                            , ['Median Test success rate & [interval]', str(np.round(test_posterior.get('beta_median'), 2)) + '%, [' + str(np.round(100*test_posterior.get('beta_interval')[0], 2)) + '%, ' + str(np.round(100*test_posterior.get('beta_interval')[1], 2)) + '%]']
                            , ['Probability that test is better than control', str(np.round(self.simulationResults_probability(), 2)) + '%']
                            , ['Expected success rate change (test - control)', str(np.round(100 * results_diff.get('simulation_mean_diff'), 2)) + ' +/-' + str(np.round(100 * results_diff.get('simulation_std_diff'), 2)) + '% points']
                            , ['Expected relative success rate change (test/control)', str(np.round(relative_diff.get('simulation_mean_relative'), 2)) + '+/-' + str(np.round(relative_diff.get('simulation_std_relative'), 2)) + '%']
                            ]
                            , columns=['Description', 'Value'])

        return data

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_percentiles(self, metric, samples):
        # '''
        # Method to calculate equally distributed tails based on the percentage volume.
        # '''

        def percentile_interval(p, values):
            # p => proportion of volume. If p = 99, we want the min and max range that contain 99% of the volume
            #      CENTERED around the median.
            min_perc = 0 + (100-p)/2
            max_perc = 100 - (100-p)/2

            min_val = np.round(np.percentile(values, min_perc),2)
            max_val = np.round(np.percentile(values, max_perc),2)

            return{'min_val': min_val
                    , 'max_val': max_val}

        # Append results in a dataframe
        data = pd.DataFrame(columns=['Metric', 'Interval', 'From', 'To'])

        for i in [99, 95, 90, 75, 50, 25]:
            pi = percentile_interval(p = i, values = samples)
            df_support = pd.DataFrame([[metric, i, pi.get('min_val'), pi.get('max_val')]]
                                      ,columns=['Metric', 'Interval', 'From', 'To'])
            data = data.append(df_support)

        return data

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_percentilesDiff(self):
        return self.simulationResults_percentiles(metric='Difference', samples=100*self.simulation_success_rate_difference)

    # ----------------------------------------------------------------------------------------------------------------
    def simulationResults_percentilesRatio(self):
        return self.simulationResults_percentiles(metric='Ratio', samples=self.simulation_success_rate_ratio)

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_BayesFactor_directional(self):
        # '''
        #
        # '''

        delta = self.simulation_success_rate_difference
        p_h1 = float(len(delta[delta>0])/len(delta))
        p_h2 = float(max(0.001,1.0 - p_h1))

        bayes_factor = p_h1 / p_h2

        return bayes_factor

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_BayesFactor_null_hypothesis(self):
        # '''
        #
        # '''

        print('Not yet implemented')

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_BayesFactor(self, method = 'directional'):
        # '''
        #
        # '''

        if method == 'directional':
            result = np.round(self.decisionCriteria_BayesFactor_directional(),2)
        else:
            raise ValueError('BayesFactor is currently only implemented with directional testing')

        if result < 1:
            # rec = 'Negative'
            rec = 'Keep testing.'
        elif 1 <= result < 3.2:
            # rec = 'Positive'
            rec = 'Keep testing.'
        elif 3.2 <= result < 10:
            # rec = 'Substantial'
            rec = 'Keep testing.'
        elif 10 <= result < 100:
            # rec = 'Strong'
            rec = 'Stop. High confidence.'
        else:
            # rec = 'Decisive'
            rec = 'Stop. High confidence.'

        return {'bayes_factor': result
                , 'bayes_criteria': rec
                , 'bayes_message': 'Bayes Factor: ' + str(result) + ', decision criteria => ' + str(rec)
        }

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_definition_intervals(self, rope_threshold=0.01, hdi=0.95, confidence_interval=95):
        # '''
        # Method to calculate the lower and upper bounds of different intervals
        # ROPE = Region of practical equivalence
        # HDI = high density interval
        # Confidence_interval = confidence interval
        # '''
        delta = self.simulation_success_rate_difference

        lower_rope = -abs(rope_threshold)
        upper_rope = abs(rope_threshold)
        range_rope = [lower_rope, upper_rope]

        range_hdi = az.hdi((100*delta).to_numpy(), credible_interval=hdi)
        lower_hdi = range_hdi[0]
        upper_hdi = range_hdi[1]

        percentiles = (100 - confidence_interval)/2
        range_ci = np.percentile(delta, [0+percentiles, 100-percentiles])
        lower_ci = range_ci[0]
        upper_ci = range_ci[1]

        return {'range_rope': range_rope
                , 'lower_rope': lower_rope
                , 'upper_rope': upper_rope
                , 'range_hdi': range_hdi
                , 'lower_hdi': lower_hdi
                , 'upper_hdi': upper_hdi
                , 'range_ci': range_ci
                , 'lower_ci': lower_ci
                , 'upper_ci': upper_ci
                }

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_assign_intervals(self, rope_threshold=0.01, hdi=0.95, confidence_interval=95):
        d = self.decisionCriteria_definition_intervals(rope_threshold, hdi, confidence_interval)
        self.range_rope = d.get('range_rope')
        self.lower_rope = d.get('lower_rope')
        self.upper_rope = d.get('upper_rope')
        self.range_hdi = d.get('range_hdi')
        self.lower_hdi = d.get('lower_hdi')
        self.upper_hdi = d.get('upper_hdi')
        self.range_ci = d.get('range_ci')
        self.lower_ci = d.get('lower_ci')
        self.upper_ci = d.get('upper_ci')

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_ROPE_probCalculations(self):
        # '''
        # Methd to calculate the percentage of samples that lie outside and within the ROPE interval
        # '''

        # self.decisionCriteria_assign_intervals()
        delta = self.simulation_success_rate_difference

        # What is the probability that the samples lie outside the ROPE? - upper bound (positive effect)
        rope_h1 = np.round(100*(len(delta[delta > self.upper_rope]) / len(delta)),2)

        # What is the probability that the samples lie outside the ROPE? - lower bound (negative effect)
        rope_h2 = np.round(100*(len(delta[delta < self.lower_rope]) / len(delta)),2)

        # Probability that the sample differences can be identified? - both positive and negative
        p_sum_identify = np.round(100*(rope_h1 + rope_h2),2)

        # Probability tht the sample differences can't be identified?
        p_sum_not_identify = np.round(100*(1 - p_sum_identify),2)

        return {'Probability_positive_detection': rope_h1
                , 'Probability_negative_detection': rope_h2
                , 'Probability_overall_detection': p_sum_identify
                , 'Probability_no_detection': p_sum_not_identify}

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_ROPE_only(self):
        # '''
        # Method to recommend stopping or continuing experiment based on ROPE only results
        # '''

        result = self.decisionCriteria_ROPE_probCalculations()

        # If all the data is positively affected and outside ROPE:
        if result.get('Probability_positive_detection') == 1:
            rec = 'Stop experiment. Strong evidence: Implement variant. Variant is better than control.'
        # If all the data is negatively affected and outside ROPE:
        elif result.get('Probability_negative_detection') == 1:
            rec = 'Stop experiment. Strong evidence: Implement control. Variant is worse than control.'
        # If all the data is contained inside the ROPE
        elif result.get('Probability_no_detection') == 0:
            rec = 'Stop experiment. Strong evidence: Implement either. Variant doesnt improve or worsen over control.'
        # If some part of the samples is still inside the ROPE
        else:
            rec = 'Keep testing.'

        return {'ROPE_criteria': rec}

    # ----------------------------------------------------------------------------------------------------------------
    def decisionCriteria_ROPE_with_HDI(self):
        # '''
        #
        # '''

        # If ROPE is completely below HDI
        if (self.lower_rope <= self.lower_hdi) and (self.upper_rope <= self.lower_hdi):
            rec = 'Stop Experiment. Strong evidence: Implement variant. Variant is better than control.'
        # If ROPE is completely above HDI
        elif (self.lower_rope >= self.upper_hdi) and (self.upper_rope >= self.upper_hdi):
            rec = 'Stop Experiment. Strong evidence: Implement control. Variant is worse than control.'
        # If HID is contained in the ROPE
        elif (self.lower_rope <= self.lower_hdi) and (self.upper_rope >= self.lower_hdi):
            rec = 'Stop Experiment. Strong evidence: Implement either. Variant doesnt improve or worsen over control.'
        else:
            rec = 'Keep testing.'

        return {'ROPE_with_HDI_criteria': rec}

    # ----------------------------------------------------------------------------------------------------------------
    def bin_grouping(self, x_axis_values, buckets=100):
        # Calculate bin groups and max volume
        # * This is used to control for the bar chart
        # * But primarily to get the volume of the biggest bin to plot other graphical objects
        def num_of_zeros(n):
            s = '{:.16f}'.format(n).split('.')[1]
            return len(s) - len(s.lstrip('0'))

        # Calculate the range between the min and max range of the success rate
        range_values = np.max(x_axis_values) - np.min(x_axis_values)

        # Calculate bin size
        bin_size = range_values / buckets

        # Find first non-zero number after decimal point
        # * This is used to add an extra level of rounding (or not) for the binning
        bin_size_ = str(float(bin_size))
        non_zero = int(bin_size_[bin_size_.index('.') + 1:].lstrip('0')[0])
        if non_zero < 5:
            nn = 1
        else:
            nn = 0

        # Calculate the different bins and volumes
        unique_vals, counts_ = np.unique(np.round(x_axis_values, num_of_zeros(bin_size) + nn),
                                         return_counts=True)
        max_volume = np.max(counts_)

        return {'unique_vals': unique_vals
                , 'counts_': counts_
                , 'max_volume': max_volume}

    # ----------------------------------------------------------------------------------------------------------------
    def plotPosteriorDistributions(self):
        # '''
        # Method to plot the  theoretical posterior distributions
        # The plot will show:
        # * Subplot probability distribution
        # * Subplot cumulative distribution
        # '''

        # Extract results of a beta distribution based on the control metrics
        control_posterior = self.PosteriorBetaDistribution(prior_a=self.prior_alpha_ctrl
                                                           , prior_b=self.prior_beta_ctrl
                                                           , N=self.N_ctrl
                                                           , successes=self.ctrl_successes)

        # Extract results of a beta distribution based on the test metrics
        test_posterior = self.PosteriorBetaDistribution(prior_a=self.prior_alpha_test
                                                        , prior_b=self.prior_beta_test
                                                        , N=self.N_test
                                                        , successes=self.test_successes)

        # Subplot 1
        trace1 = go.Scatter(x=100 * control_posterior.get('x_samples')
                            , y=control_posterior.get('beta_pdf') / 100
                            , mode='lines'
                            , name='Probability Distribution - Control'
                            , marker=dict(color='blue')
                            , hovertemplate='Relative volume: %{y:.2f}'
                            )

        control_max_density = np.round(np.max(control_posterior.get('beta_pdf')), 2)
        test_max_density = np.round(np.max(test_posterior.get('beta_pdf')), 2)
        max_density = np.max([control_max_density, test_max_density]) / 100

        trace1_1 = go.Scatter(x=[control_posterior.get('beta_median'), control_posterior.get('beta_median')]
                              , y=[0, 1.1 * max_density]
                              , mode='lines'
                              , name='Most common success rate - Control: ' + str(
                np.round(control_posterior.get('beta_median'), 3))
                              , line=dict(color='blue', dash='dash')
                              )

        trace2 = go.Scatter(x=100 * test_posterior.get('x_samples')
                            , y=test_posterior.get('beta_pdf') / 100
                            , mode='lines'
                            , name='Probability Distribution - Test'
                            , marker=dict(color='red')
                            , hovertemplate='Relative volume: %{y:.2f}'
                            )

        trace2_1 = go.Scatter(x=[test_posterior.get('beta_median'), test_posterior.get('beta_median')]
                              , y=[0, 1.1 * max_density]
                              , mode='lines'
                              , name='Most common success rate - Test: ' + str(
                np.round(test_posterior.get('beta_median'), 3))
                              , line=dict(color='red', dash='dash')
                              )

        # Subplot 2
        trace3 = go.Scatter(x=100 * control_posterior.get('x_samples')
                            , y=control_posterior.get('beta_cdf')
                            , mode='lines'
                            , name='Cumulative Distribution - Control'
                            , line=dict(color='blue', dash='dot')
                            , hovertemplate='Relative volume: %{y:.2f}'
                            )

        trace4 = go.Scatter(x=100 * test_posterior.get('x_samples')
                            , y=test_posterior.get('beta_cdf')
                            , mode='lines'
                            , name='Cumulative Distribution - Test'
                            , line=dict(color='red', dash='dot')
                            , hovertemplate='Relative volume: %{y:.2f}'
                            )

        # Putting subplots together
        fig = make_subplots(rows=2, cols=1
                            , shared_xaxes=False, vertical_spacing=0.1
                            , subplot_titles=(
            '<b>Posterior distributions of Control (' + str(self.ctrl_name) + ') and Test (' + str(
                self.test_name) + ')</b>'
            , '<b>Posterior cumulative distributions of Control (' + str(self.ctrl_name) + ') and Test (' + str(
                self.test_name) + ')</b><br>')
                            )

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace1_1, 1, 1)
        fig.append_trace(trace2, 1, 1)
        fig.append_trace(trace2_1, 1, 1)

        fig.append_trace(trace3, 2, 1)
        fig.append_trace(trace4, 2, 1)

        lower_x_range_ctrl = 100 * np.max([np.min(control_posterior.get('beta_interval')) / 2, 0])
        upper_x_range_ctrl = 100 * np.min([np.max(control_posterior.get('beta_interval')) * 2, 100])
        lower_x_range_test = 100 * np.max([np.min(test_posterior.get('beta_interval')) / 2, 0])
        upper_x_range_test = 100 * np.min([np.max(test_posterior.get('beta_interval')) * 2, 100])
        lower_x_range = min(lower_x_range_ctrl, lower_x_range_test)
        upper_x_range = max(upper_x_range_ctrl, upper_x_range_test)

        fig.update_xaxes(title_text='', row=1, col=1, range=[lower_x_range, upper_x_range])
        fig.update_xaxes(title_text='Distribution of Success Rate (%) -- Range [0,100]', row=2, col=1,
                         range=[lower_x_range, upper_x_range])
        fig.update_yaxes(title_text='Relative volume', row=1, col=1)
        fig.update_yaxes(title_text='Percentage of volume', row=2, col=1)
        fig.update_layout(title='', height=700, width=1500
                          , hovermode='x unified')

        return fig

    # ----------------------------------------------------------------------------------------------------------------
    def plotSimulations(self):
        # '''
        # Method to plot the results of the simulations in a graphical form.
        # The plot will show:
        # * Subplot with the distribution of simulation results based on the difference in success rate between test and control
        # * Subplot with the distribution of simulation results based on the ratio in success rate between test and control
        # '''

        # Extract simulations results
        results_diff = self.simulationResults_difference()
        relative_diff = self.simulationResults_relative()

        # Plotting <--
        # Subplot 1
        sp1 = self.bin_grouping(100*self.simulation_success_rate_difference)
        trace1 = go.Bar(x=sp1.get('unique_vals')
                        , y=sp1.get('counts_')
                        , marker_color='orange'
                        , name = 'Difference distribution'
                        , hovertemplate = '<i>Difference bin</i>: %{x}' + '<br><i>Volume</i>: %{y}'
                        )

        r = 100*results_diff.get('simulation_mean_diff')
        trace1_1 = go.Scatter(x = [r, r]
                              , y = [0, sp1.get('max_volume')]
                              , mode = 'lines'
                              , name = 'Average difference between Test and Control: ' + str(np.round(r,2)) + '% points'
                              , line = dict(color = 'rgb(204, 102, 0)', dash = 'dash')
                              )

        # Subplot 2
        sp2 = self.bin_grouping(self.simulation_success_rate_ratio)
        trace2 = go.Bar(x=sp2.get('unique_vals')
                        , y=sp2.get('counts_')
                        , marker_color='rgb(51, 204, 51)'
                        , name='Ratio distribution'
                        , hovertemplate='<i>Ratio bin</i>: %{x}' + '<br><i>Volume</i>: %{y}'
                        )

        r = (relative_diff.get('simulation_mean_relative')/100)+1
        trace2_1 = go.Scatter(x=[r, r]
                              , y=[0, sp2.get('max_volume')]
                              , mode='lines'
                              , name='Average ratio difference between Test and Control: ' + str(np.round(r,2))
                              , line = dict(color = 'rgb(0, 153, 51)', dash = 'dash')
                              )

        # Putting plots together
        fig = make_subplots(rows = 2, cols = 1
                            , shared_xaxes=False, vertical_spacing=0.2
                            , subplot_titles=('<b>Difference between Control (' + str(self.ctrl_name) + ') and Test (' + str(self.test_name) + ')</b> -- Range [0,100]'
                                              ,'<b>Relative difference (ratio) between Control (' + str(self.ctrl_name) + ') and Test (' + str(self.test_name) + ')</b> -- Range [0, inf)')
                            )

        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace1_1, 1, 1)

        fig.append_trace(trace2, 2, 1)
        fig.append_trace(trace2_1, 2, 1)

        fig.update_xaxes(title_text=str(self.test_name) + ' rate - ' + str(self.ctrl_name) + ' rate (% points)', row=1, col=1)
        fig.update_xaxes(title_text=str(self.test_name) + ' rate / ' + str(self.ctrl_name) + ' rate', row=2, col=1)
        fig.update_yaxes(title_text='Number of occurrences', row=1, col=1)
        fig.update_yaxes(title_text='Number of occurrences', row=2, col=1)
        fig.update_layout(title='<b>Performance of Test (' + str(self.test_name) + ') vs Control (' + str(
            self.ctrl_name) + ') simulated 100,000 times</b>')

        return fig

    # ----------------------------------------------------------------------------------------------------------------
    def plotSimulations_ROPE_HDI(self):

        sp1 = self.bin_grouping(100 * self.simulation_success_rate_difference)
        max_v = sp1.get('max_volume')

        # New trace
        lr = self.lower_rope
        ur = self.upper_rope
        trace1 = go.Scatter(x=[lr, lr, ur, ur]
                            , y=[0, max_v, max_v, 0]
                            , fill='toself'
                            , fillcolor='rgb(255, 153, 153)'
                            , opacity=0.3
                            , line=dict(color='rgb(255, 153, 153)', width=2)
                            , name='ROPE'
                          )

        # New trace
        lhdi = self.lower_hdi
        uhdi = self.upper_hdi
        trace2 = go.Scatter(x=[lhdi, lhdi, uhdi, uhdi]
                            , y=[0, max_v, max_v, 0]
                            , fill='toself'
                            , fillcolor='lightblue'
                            , opacity=0.3
                            , line=dict(color='lightblue', width=2)
                            , name='HDI'
                            )

        # Importing figure plot
        fig = self.plotSimulations()

        # Appending trace
        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 1, 1)

        return fig