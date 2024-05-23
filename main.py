from brian2 import *
from scipy import stats
import simulation as sm
import analysis as nl
import plotting as pl
from IPython import embed

neurons = sm.setup_neurons(n = 2**10)
params = sm.get_realistic_params()
signal = sm.gen_realistic_signal(len = 1*second, dt = 0.1*ms)
spikes = sm.simulate(neurons, params, signal)
print("simulation complete")
#info2 = nl.calc_stepwise(signal, spikes, nl.info)**2
#plot(info2)
#x = arange(len(info2))
#r = stats.linregress(x, info2)
#plot(r.slope * x + r.intercept)
plot(nl.calc_stepwise(signal, spikes, ones(len(spikes)), 0))
plot(nl.calc_stepwise(signal, spikes, ones(len(spikes)), 1))
plot(nl.calc_stepwise(signal, spikes, nl.smart_weights(len(spikes)), 1))
show()