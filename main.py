from brian2 import *
from time import *
import sim
import calc
import eval
from IPython import embed

start_time = time()

print("initialization")
n = 100
dt = 0.01*ms
dur = 10*second
min_f = 100/second
max_f = 1000/second
delay_range = 10*ms
params = sim.realistic_params
steps = int(round(dur/dt))
delay = int(round(delay_range/dt))
min_p = int(round(1/max_f/dt))
max_p = int(round(1/min_f/dt))
print(str(round(time() - start_time, 2)) + " seconds")

print("preparation")
signal = sim.normal_noise(dt, steps, min_f, max_f)
neurons = sim.sensor_neurons(n, dt, params)
monitor = SpikeMonitor(neurons)
print(str(round(time() - start_time, 2)) + " seconds")

print("simulation")
run(dur, namespace = {'signal': signal})
spikes = monitor.spike_trains()
print(str(round(time() - start_time, 2)) + " seconds")

print("integration")
input = np.zeros((n, steps), dtype=int)
for i in range(n):
  times = np.around(spikes[i] / dt + delay * i / n).astype(int)
  times = times[:np.searchsorted(times, steps)]
  input[i][times] = 1
models = [{'label': "same delay",
           'color': "silver",
           'output': calc.same_delay(input, delay)},
          {'label': "narrow sum",
           'color': "gold",
           'output': calc.narrow_sum(input, delay, min_p)},
          {'label': "smart weights",
           'color': "blue",
           'output': calc.smart_weights(input, delay, min_p, max_p)},
          {'label': "dynamic state, fast",
           'color': "blueviolet",
           'output': calc.dynamic_state(input, delay, min_p, max_p, 0.02)},
          {'label': "dynamic state, slow",
           'color': "magenta",
           'output': calc.dynamic_state(input, delay, min_p, max_p, 0.005)}  
         ]
print(str(round(time() - start_time, 2)) + " seconds")

print("evaluation")
eval.spect_plot(dt, signal, models, min_f, max_f, 1*second, 5)
print(str(round(time() - start_time, 2)) + " seconds")