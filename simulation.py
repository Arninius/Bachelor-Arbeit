from brian2 import *
import numpy as np
from scipy import signal as sg
from IPython import embed

realistic_params = {'tau': 1*ms, 'threshold': -50*mV, 'reset': -70*mV, 'refractory': 0*ms, 'noisiness': 10*mV/sqrt(ms)}

def sensor_neurons(n, dt, params):
    diff_eqs = 'dvoltage/dt = (signal(t) - voltage) / tau + xi_voltage * noisiness: volt (unless refractory)'
    neurons = NeuronGroup(n, diff_eqs, 'euler', None, 'voltage > threshold', 'voltage = reset', 'refractory',
                          namespace = params, dt = dt)
    neurons.voltage = params['reset']
    return neurons

def normal_noise(dt, steps, min_f = 0/second, max_f = np.inf): # Low performance
    white_noise = np.fft.rfft(np.random.randn(steps))
    freqs = np.fft.rfftfreq(steps)
    filter = np.empty(freqs.size)
    overgrow = 0.5*min_f
    min = np.searchsorted(freqs, (min_f - overgrow) * dt, side = 'left')
    max = np.searchsorted(freqs, (max_f + overgrow) * dt, side = 'right')
    if min == 0: filter[0] = 1; min = 1
    filter[min:max] = 1 #/ np.sqrt(freqs[min:max])
    filter /= np.sqrt(np.mean(filter**2))
    final_noise = np.fft.irfft(white_noise * filter, steps)
    return TimedArray(10*mV * final_noise, dt)