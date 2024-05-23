from brian2 import *
import numpy as np
from scipy import signal as sg
from IPython import embed

res = 500

def info(freqs, spect):
    return -np.trapz(np.log2(1-spect), freqs)

def coherence(freq):
    return lambda freqs, spect: spect[freqs == freq]

def smart_weights(n):
    w = np.ones(n)
    for p in range(1, 2**5 + 1):
        w += [np.cos(p*2*np.pi*i/n)/p**2 for i in range(n)]
    return w

def calc_overall(signal, spikes, weights, delay, func = info):
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
    #embed()
    freqs, spect = sg.coherence(signal.values[i*delay:], output[i*delay:], nperseg = res, fs = 1/signal.dt)
    return func(freqs, spect)

def calc_stepwise(signal, spikes, weights, delay, func = info):
    results = np.zeros(len(spikes)+1)
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
        freqs, spect = sg.coherence(signal.values[i*delay:], output[i*delay:], nperseg = res, fs = 1/signal.dt)
        results[i+1] = func(freqs, spect)
    return results