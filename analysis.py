from brian2 import *
import numpy as np
from scipy import signal as sg
from IPython import embed

res = 10000

def infos():
    return lambda freqs, spect: -np.trapz(np.log2(1-spect), freqs)

def info(min, max):
    return lambda freqs, spect: -np.trapz(np.log2(1-spect[min:max+1]), freqs[min:max+1])

def info_at(freq):
    return lambda freqs, spect: -np.log2(1-spect[freq])

def coherence(freq): # deprecated
    return lambda freqs, spect: spect[freqs == freq]

def curvy_weights(n, freq):
    w = np.zeros(n)
    for i in range(n):
        w[i] = np.cos(freq*2*np.pi*i/n)
       # if w[i] < 0: w[i] = 0
    #plot(w)
    return w

def calc_overall(signal, spikes, weights, delay): #, func = info_at(1)):
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
    freqs, spect = sg.coherence(signal.values[i*delay:], output[i*delay:], nperseg = res, fs = 1/signal.dt)
    infos = [-np.log2(1-spect[f]) for f in range(len(freqs))]
    plot(infos, label = str(delay) + str(mean(weights)))
    #return func(freqs, spect)

def calc_stepwise(signal, spikes, weights, delay, func):
    results = np.zeros(len(spikes)+1)
    output = np.zeros(len(signal.values))
    for i in range(len(spikes)):
        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
        output[del_spikes[del_spikes < len(signal.values)]] += weights[i]
        freqs, spect = sg.coherence(signal.values[i*delay:], output[i*delay:], nperseg = res, fs = 1/signal.dt)
        results[i+1] = func(freqs, spect)
    return results

#def calc_stepwise_fitting(signal, spikes, w_func, delay, func, min, max):
#    results = np.zeros(len(spikes)+1)
#    output = np.zeros((max+1-min, len(signal.values)))
#    for i in range(len(spikes)):
#        del_spikes = np.around(spikes[i]/signal.dt/second).astype(int) + i * delay
#        output[:, del_spikes[del_spikes < len(signal.values)]] += w_func(i, np.arange(min, max+1))
#        freqs, spect = sg.coherence(signal.values[i*delay:], output[i*delay:], nperseg = res, fs = 1/signal.dt)
#        results[i+1] = func(freqs, spect)
#    return results