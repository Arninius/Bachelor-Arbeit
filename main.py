from brian2 import *
from scipy import ndimage
import simulation as sm
import analysis as nl
import plotting as pl
from IPython import embed

NEURON_NUM = 1000
SIM_LENGTH = 10*second
RESOLUTION = 0.01*ms

neurons = sm.setup_neurons(NEURON_NUM)
params = sm.get_realistic_params()
signal = sm.gen_bounded_signal(SIM_LENGTH, RESOLUTION, min = 100/second, max = 1000/second)
spikes = sm.simulate(neurons, params, signal)
print("simulation complete")

#pl.plot_infos_per_window(NEURON_NUM, RESOLUTION, signal, spikes)

#print("First plot done")

n = NEURON_NUM
dt = RESOLUTION
delays = [nl.linear_delays(n, 0*ms, 0*ms),
          nl.linear_delays(n, 5*ms, 5*ms),
          nl.linear_delays(n, 10*ms, 10*ms),
          nl.linear_delays(n, 0*ms, 10*ms),
          nl.linear_delays(n, 0*ms, 10*ms),
          nl.linear_delays(n, 0*ms, 10*ms)]
labels = ["no delays",
          "5ms delays",
          "10ms delays",
          "0-10ms delays, fast",
          "0-10ms delays, compromise",
          "0-10ms delays, precise"]
colors = ["lightgray",
          "darkgray",
          "dimgray",
          "goldenrod",
          "forestgreen",
          "royalblue"]
windows = np.around(10 ** np.linspace(0, 2, 10)) * 10*ms
evals = np.empty(len(windows))

for i in range(len(labels)):
    output = np.zeros(round(SIM_LENGTH/dt))
    for freq in np.linspace(100/second, 1000/second, 10):
        if i < 4: dels = delays[i]
        elif i == 4: dels = nl.improve_delays(delays[i], freq, 0.75)
        elif i == 5: dels = nl.improve_delays(delays[i], freq, 1)
        if i < 3: weights = nl.ones(n)
        else: weights = nl.optimal_weights(freq, dels)
        output += nl.calc_output(n, dt, signal, spikes, dels, weights)
    evals = np.empty(len(windows))
    for w in range(len(windows)):
        evals[w] = nl.evaluate(dt, signal, output, np.max(delays[i]), 100/second, 1000/second, windows[w])
    print(labels[i])
    plot(windows/ms, evals, label = labels[i], color = colors[i])

xlabel("evaluation window in ms")
xscale("log")
xlim(10, 1000)
ylabel("mutual information")
legend()
savefig("Global-Comparison")