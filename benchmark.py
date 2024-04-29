from brian2 import *
import functions as fs
from scipy import signal as sig
from IPython import embed

delta_t = 0.01*ms
defaultclock.dt = delta_t
sim_length = 10*second
num_sensors = 10
sensor_tau = 40*ms
threshold = 1*mV
reset = 0*mV
refractory = 0.1*ms
noisiness = 1*mV/ms

sensor_equations = 'dvoltage/dt = (signal(t)-voltage)/sensor_tau + xi_voltage*noisiness*sqrt(delta_t): volt (unless refractory)'
sensors = NeuronGroup(num_sensors, sensor_equations, threshold='voltage>threshold', reset='voltage=reset', refractory='refractory', method='euler')
pre_signal = fs.whitenoise(0, 0.2, delta_t/ms, (sim_length-delta_t)/ms) * 3 + 10
signal = TimedArray(pre_signal*mV, delta_t)
spikes = SpikeMonitor(sensors)
run(sim_length)

output = zeros(len(pre_signal))
output[int(spikes.t/delta_t)] += ms/delta_t/num_sensors
#embed()

fig, axs = subplots(3)
#axs[0].plot(states.t/ms, output/mV)
axs[0].plot(states.t/ms, output*num_sensors*delta_t/ms)
axs[0].plot(states.t/ms, input/mV)
axs[0].set_xlim(100, 200)
nfft = 2**15
in_freqs, in_spect = sig.welch(input, fs = 1/delta_t, nperseg = nfft)
out_freqs, out_spect = sig.welch(output, fs = 1/delta_t, nperseg = nfft)
coh_freqs, coh_spect = sig.coherence(input, output, fs=1/delta_t, nperseg = nfft)
#axs[1].plot(out_freqs*ms, out_spect/mV)
axs[1].plot(out_freqs*ms, out_spect*sensor_tau/ms*200)
axs[1].plot(in_freqs*ms, in_spect/mV*10000)
axs[1].set_xlim(0, 0.3)
axs[1].set_ylim(0, 1)
axs[2].plot(coh_freqs*ms, coh_spect)
axs[2].set_xlim(0, 0.3)
axs[2].set_ylim(0, 1)
#cohere(input, output)
#y = rfft(input)
#x = rfftfreq(input)
#plot(x, y)
#plot(sig.welch(input))
#axs[0].eventplot(spikes.t/ms, lineoffsets=0, linelengths=0.1, colors='black')
#xlabel('Frequency')
#ylabel('Coherence')

#plot(attractor_range, firing_rates)
show()