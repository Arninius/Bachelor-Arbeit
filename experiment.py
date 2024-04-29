from brian2 import *
import functions as fs
from scipy import signal as sig
from scipy.fft import rfft, rfftfreq

delta_t = 0.01*ms
defaultclock.dt = delta_t
sim_length = 1000*ms

num_sensors = 10
sensor_tau = 0.1*ms
grouper_tau = 1*ms
threshold = 0*mV #1
reset = 0*mV
refractory = 0*ms
noisiness = 100*mV/ms

sensor_equations = '''
dvoltage/dt = (signal(t)-voltage)/sensor_tau + xi_voltage*noisiness*sqrt(delta_t): volt (unless refractory)
'''
sensors = NeuronGroup(num_sensors, sensor_equations, threshold='voltage>threshold', reset='voltage=reset', refractory='refractory', method='euler')
groupers = NeuronGroup(1, 'dvoltage/dt = -voltage/grouper_tau: volt', method='euler')
# sensors.voltage = 'reset + rand() * (threshold-reset)'
synapses = Synapses(sensors, groupers, on_pre = 'voltage_post += 0.1*mV')
synapses.connect(j='0')
#store()

states = StateMonitor(groupers, 'voltage', record=0) # set record=True to record the states of all neurons
#spikes = SpikeMonitor(sensors)
pre_signal = fs.whitenoise(0, 0.1, delta_t/ms, (sim_length-delta_t)/ms)
signal = TimedArray(pre_signal*mV, delta_t)
run(sim_length)

#attractor_range = linspace(0, 2, 100)
#firing_rates = []
#for attractor in attractor_range:
    # pre_signal = fs.whitenoise(0, 1, delta_t/ms, (sim_length-delta_t)/ms)
    #pre_signal = [attractor]*int(sim_length/delta_t)
    #signal = TimedArray(pre_signal*mV, delta_t)
    #restore()
    #run(sim_length)
    #firing_rates.append(spikes.num_spikes/second)x^

input = signal(states.t)
output = states.voltage[0]
output /= std(output) / std(input)
output -= mean(output) - mean(input)

fig, axs = subplots(2)
axs[0].plot(states.t/ms, input/mV)
axs[0].plot(states.t/ms, output/mV)
in_freqs, in_spect = sig.welch(input, fs = 1/delta_t)
out_freqs, out_spect = sig.welch(output, fs = 1/delta_t)
axs[1].plot(in_freqs, in_spect)
axs[1].plot(out_freqs, out_spect)
#cohere(input, output)
#y = rfft(input)
#x = rfftfreq(input)
#plot(x, y)
#plot(sig.welch(input))
#eventplot(spikes.t/ms, lineoffsets=0, linelengths=0.1, colors='black')
#xlabel('Frequency')
#ylabel('Coherence')

#plot(attractor_range, firing_rates)
show()