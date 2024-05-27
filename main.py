from brian2 import *
from scipy import stats
import simulation as sm
import analysis as nl
import plotting as pl

neurons = sm.setup_neurons(n = 100)
params = sm.get_realistic_params()
signal = sm.gen_bounded_signal(len = 1*second, dt = 0.01*ms, min = 100, max = 1000)
#signal = sm.gen_pink_signal(len = 1*second, dt = 0.01*ms)
spikes = sm.simulate(neurons, params, signal)
print("simulation complete")

func = nl.info_at(100)
uni_weights = ones(len(spikes))
tuned_weights = nl.curvy_weights(100, 10)

#nl.calc_overall(signal, spikes, uni_weights, 0)
#nl.calc_overall(signal, spikes, uni_weights, 10)
#nl.calc_overall(signal, spikes, tuned_weights, 10)
#show()
plot(nl.calc_stepwise(signal, spikes, uni_weights, 0, func), label = "no delays")
plot(nl.calc_stepwise(signal, spikes, uni_weights, 10, func), label = "naive sum")
plot(nl.calc_stepwise(signal, spikes, tuned_weights, 10, func), label = "freq-wise")

legend()
show()