from brian2 import *
import numpy as np
import scipy as sc

def information(freqs, spect, df):
    if freqs.size == 0: return 0
    elif (freqs.size - 1) / df < 1: return -np.log2(1-spect[0])
    else: return -np.trapz(np.log2(1-spect[::df]), freqs[::df])

def coherences(dt, signal, output, min_f, max_f, window):
    freqs, spect = sc.signal.coherence(signal.values, output,
                                       fs = round(second/dt),
                                       nperseg = round(window/dt))
    min = np.searchsorted(np.around(freqs), min_f, side = 'left')
    max = np.searchsorted(np.around(freqs), max_f, side = 'right')
    return freqs[min:max], spect[min:max]

def spect_plot(dt, signal, models, min_f, max_f, window, res):
    for m in models:
        freqs, spect = coherences(dt, signal, m['output'], min_f, max_f, window)
        filter_size = round(spect.size / (res * max_f / min_f))
        spect = sc.ndimage.uniform_filter1d(spect, filter_size)
        plot(freqs, spect, label = m['label'], color = m['color'])
        print(m['label'] + ": " + str(round(mean(spect), 2)))
    xlabel("frequency in Hz")
    ylabel("coherence")
    legend()
    show()
