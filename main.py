from bayesian_calculator import PosteriorGenerator

d = PosteriorGenerator.PosteriorGenerator(N_ctrl=100, N_test=100, ctrl_successes=33, test_successes=50
                       , ctrl_name='control', test_name='variant'
                       , prior_alpha_ctrl = 5
                       , prior_beta_ctrl = 16)

print('---------------------------------------------------')
print('Simulation Summary Results:')
print('---------------------------------------------------')
print(d.simulationResults_summary())

print('')
print('---------------------------------------------------')
print('Success rate difference percentile table (% points')
print('---------------------------------------------------')
print(d.simulationResults_percentilesDiff())

print('')
print('---------------------------------------------------')
print('Success rate ratio percentile table')
print('---------------------------------------------------')
print(d.simulationResults_percentilesRatio())

d.plotSimulations().show()
d.plotPosteriorDistributions().show()
