from brian2 import *
import functions as fs
from scipy import signal as sg
from IPython import embed

delta_t = 0.1*ms
defaultclock.dt = delta_t
sim_length = 5*second
num_sensors = 50
sensor_tau = 20*ms
threshold = -50*mV
reset = -70*mV
refractory = 1*ms
noisiness = 20*mV/ms
signal_maxfreq = ms/sensor_tau # per ms
signal_amplitude = 20*mV # SD around 0mV
freq_resolution = 10 # steps until maxfreq
def eval(coherences): return mean(coherences)
#def delay(i): return sqrt(i) * 5 * ms
def delay(i): return i * 1 * ms

sensor_equations = 'dvoltage/dt = (signal(t)-voltage)/sensor_tau + xi_voltage*noisiness*sqrt(delta_t): volt (unless refractory)'
sensors = NeuronGroup(num_sensors, sensor_equations, threshold='voltage>threshold', reset='voltage=reset', refractory='refractory', method='euler')
sensors.voltage = reset
pre_signal = fs.whitenoise(0, signal_maxfreq, delta_t/ms, (sim_length-delta_t)/ms)
signal = TimedArray(signal_amplitude*pre_signal, delta_t)
spikes = SpikeMonitor(sensors)
run(sim_length)

input = signal(arange(0, sim_length, delta_t))
performances = [zeros(freq_resolution)]
for n in range(1, num_sensors + 1):
    output = zeros(len(input))
    for s in range(spikes.num_spikes):
        if spikes.i[s] < n:
            t = spikes.t[s] + delay(spikes.i[s])
            if t < sim_length:
                output[int(t/delta_t)] += 1 # += ms/delta_t/n
    eval_delay = int(delay(n-1) / delta_t)
    _, coh_spect = sg.coherence(input[eval_delay:], output[eval_delay:], nperseg = freq_resolution/signal_maxfreq/(delta_t/ms))
    performances.append(coh_spect[1:freq_resolution])

plot(list(map(eval, performances)), color = "black", linewidth = 2, label = "overall")
freq_perfs = list(map(list, zip(*performances)))
for f in range(1, freq_resolution):
    gradient = f/freq_resolution
    plot(freq_perfs[f-1], color = (gradient, 1-gradient, 0.5), label = str(int(gradient*signal_maxfreq*1000)) + "Hz")
legend()
ylim(0, 1)
show()

#embed()