from bayesian_calculator.bayesian_proportions_calculator import ProportionsCalculator

d = ProportionsCalculator(N_ctrl=97124, N_test=94711, ctrl_successes=264, test_successes=245
                          , ctrl_name='ctrl', test_name='variant')

print('---------------------------------------------------')
print('Simulation Summary Results:')
print('---------------------------------------------------')
print(d.simulationResults_summary())

print('')
print('---------------------------------------------------')
print('Decision Criteria results:')
print('---------------------------------------------------')
print(d.decisionCriteria_BayesFactor().get('bayes_criteria'))
print(d.decisionCriteria_ROPE_only().get('ROPE_criteria'))
print(d.decisionCriteria_ROPE_with_HDI().get('ROPE_with_HDI_criteria'))

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

d.plotPosteriorDistributions().show()
d.plotSimulations().show()
d.plotSimulations_ROPE_HDI().show()
