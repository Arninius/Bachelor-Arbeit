from brian2 import *
from scipy import stats
import simulation as sm
import analysis as nl
import plotting as pl

neurons = sm.setup_neurons(n = 1000)
params = sm.get_realistic_params()
signal = sm.gen_bounded_signal(len = 10*second, dt = 0.1*ms)
spikes = sm.simulate(neurons, params, signal)
print("simulation complete")
#info2 = nl.calc_stepwise(signal, spikes, nl.info)**2
#plot(info2)
#x = arange(len(info2))
#r = stats.linregress(x, info2)
#plot(r.slope * x + r.intercept)

#plot(signal)
#show()

nl.calc_overall(signal, spikes, ones(len(spikes)), 0)
nl.calc_overall(signal, spikes, ones(len(spikes)), 1)
nl.calc_overall(signal, spikes, nl.smart_weights(len(spikes)), 1)
legend()
show()

#plot(nl.calc_stepwise(signal, spikes, ones(len(spikes)), 0), label = "no delays")
#plot(nl.calc_stepwise(signal, spikes, ones(len(spikes)), 1), label = "uniform weights")
#plot(nl.calc_stepwise(signal, spikes, nl.smart_weights(len(spikes)), 1), label = "smart weights")
#legend()
#show()