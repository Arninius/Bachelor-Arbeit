from brian2 import *
import numpy as np

def setup_neurons(n):
    diff_eqs = 'dvoltage/dt = (signal(t) - voltage) / tau + xi_voltage * noisiness: volt (unless refractory)'
    return NeuronGroup(n, diff_eqs,
                       threshold = 'voltage>threshold',
                       reset = 'voltage=reset',
                       refractory = 'refractory',
                       method = 'euler')

def get_realistic_params():
    return {'tau': 20*ms,
            'threshold': -50*mV,
            'reset': -70*mV,
            'refractory': 1*ms,
            'noisiness': 10*mV/sqrt(ms)}

def gen_realistic_signal(len, dt):
    n = int(len/dt)
    white_noise = np.fft.rfft(np.random.randn(n))
    filter = np.concatenate(([1], 1 / np.sqrt(np.fft.rfftfreq(n)[1:])))
    filter /= np.sqrt(np.mean(filter**2))
    pink_noise = white_noise * filter
    #plot(np.abs(pink_noise))
    return TimedArray(30*mV * np.fft.irfft(pink_noise), dt)

#def gen_precise_signal(len, dt):

def simulate(neurons, params, signal):
    defaultclock.dt = signal.dt*second
    neurons.voltage = params['reset']
    spike_monitor = SpikeMonitor(neurons)
    params['signal'] = signal
    run(len(signal.values)*defaultclock.dt, namespace = params)
    return spike_monitor.spike_trains()