from brian2 import *
import simulation as sm
import analysis as nl
import plotting as pl
from IPython import embed

n = 100
dt = 0.01*ms
dur = 10*second
steps = round(dur/dt)
min_f = 100/second
max_f = 1000/second
params = sm.realistic_params

print("preparation")
signal = sm.normal_noise(dt, steps, min_f, max_f)
# TODO: SIMILAR NOISE FOR NEARBY NEURONS
neurons = sm.sensor_neurons(n, dt, params)
monitor = SpikeMonitor(neurons)

print("simulation")
run(dur, namespace = {'signal': signal})
spikes = monitor.spike_trains()

print("integration")
delays = np.linspace(0*ms, 10*ms, n, endpoint = False)
models = [#{'label': "min delay",
          # 'color': "silver",
          # 'output': nl.const_delay_sum(n, dt, steps, spikes, np.min(delays))},
          #{'label': "max delay",
          # 'color': "gray",
          # 'output': nl.const_delay_sum(n, dt, steps, spikes, np.max(delays))},
          {'label': "fourier combine",
           'color': "magenta",
           'output': nl.fourier_combine(n, dt, dur, steps, spikes, delays, min_f, max_f, 0.1*ms, 0.1)},     
          {'label': "ideal field",
           'color': "gold",
           'output': nl.ideal_field_sum(dt, steps, signal, spikes, delays, 100*ms, min_f, max_f)},
         # {'label': "freqs delay",
         #  'color': "green",
         #  'output': nl.freqs_delay_sum(n, dt, steps, spikes, delays, min_f, max_f)},
          {'label': "freq weight",
           'color': "royalblue",
           'output': nl.freq_weight_sum(n, dt, steps, spikes, delays, min_f, max_f)}
         ]

print("evaluation")
pl.spect_plot(dt, signal, models, min_f, max_f, 100*ms)
