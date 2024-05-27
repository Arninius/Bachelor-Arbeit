from brian2 import *
import numpy as np
from IPython import embed

def setup_neurons(n):
    diff_eqs = 'dvoltage/dt = (signal(t) - voltage) / tau + xi_voltage * noisiness: volt (unless refractory)'
    return NeuronGroup(n, diff_eqs,
                       threshold = 'voltage>threshold',
                       reset = 'voltage=reset',
                       refractory = 'refractory',
                       method = 'euler')

def get_realistic_params():
    return {'tau': 1*ms,
            'threshold': -50*mV,
            'reset': -70*mV,
            'refractory': 0*ms,
            'noisiness': 10*mV/sqrt(ms)}

def gen_freq_signal(len, dt, freq = 100):
    n = int(len/dt)
    signal = [np.sin(2*pi*freq*i/n) 
              * np.sin(2*pi*1*i/n)
              for i in range(n)]
    plot(signal)
    show()
    return TimedArray(30*mV * signal, dt)

def gen_pink_signal(len, dt):
    n = int(len/dt)
    white_noise = np.fft.rfft(np.random.randn(n))
    filter = np.concatenate(([1], 1 / np.sqrt(np.fft.rfftfreq(n)[1:])))
    filter /= np.sqrt(np.mean(filter**2))
    pink_noise = white_noise * filter
    #plot(np.abs(pink_noise))
    return TimedArray(20*mV * np.fft.irfft(pink_noise), dt)

def gen_bounded_signal(len, dt, min, max):
    n = int(len/dt)
    white_noise = np.fft.rfft(np.random.randn(n))
    window = np.empty(1+max-min); window.fill(1/np.sqrt((1+max-min)/(n/2+1)))
    filter = np.concatenate((np.zeros(min), window, np.zeros(int(n/2)-max)))
    bounded_noise = white_noise * filter
    return TimedArray(10*mV * np.fft.irfft(bounded_noise), dt)

def simulate(neurons, params, signal):
    defaultclock.dt = signal.dt*second
    neurons.voltage = params['reset']
    spike_monitor = SpikeMonitor(neurons)
    params['signal'] = signal
    run(len(signal.values)*defaultclock.dt, namespace = params)
    return spike_monitor.spike_trains()