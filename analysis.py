from brian2 import *
import numpy as np
from scipy import signal as sg
from IPython import embed

res = 10000

def info(min, max):
    return lambda freqs, spect: -np.trapz(np.log2(1-spect[min:max]), freqs[min:max])

def coherence(freq):
    return lambda freqs, spect: spect[freqs == freq]

def smart_weights(n, max = 10.1):
    #w = np.ones(n)
    w = np.zeros(n)
    for p in np.arange(1, max, 0.1):
        for i in range(n):
            if i < (n/p) * np.floor(p):
                w[i] += np.cos(p*2*np.pi*i/n)
    #plot(w)
    return w

def calc_overall(signal, spikes, weights, delay, func = info(10, 20)):
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
    freqs, spect = sg.coherence(signal.values[delay:], output[delay:], nperseg = res, fs = 1/signal.dt)
    plot(freqs, spect, label = str(delay) + str(mean(weights)))
    #show()
    return func(freqs, spect)

def calc_stepwise(signal, spikes, weights, delay, func = info(10, 20)):
    results = np.zeros(len(spikes)+1)
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
        freqs, spect = sg.coherence(signal.values[delay:], output[delay:], nperseg = res, fs = 1/signal.dt)
        results[i+1] = func(freqs, spect)
    return results