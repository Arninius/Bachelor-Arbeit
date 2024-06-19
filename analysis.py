from brian2 import *
import numpy as np
from scipy import signal as sg
from IPython import embed

def add_spikes(output, dt, steps, spikes, delay, weight):
    spike_steps = np.around((spikes + delay) / dt).astype(int)
    output[spike_steps[spike_steps < steps]] += weight

def const_delay_sum(n, dt, steps, spikes, delay):
    output = np.zeros(steps)
    for i in range(n):
        add_spikes(output, dt, steps, spikes[i], delay, 1)
    return output

def ideal_field_sum(dt, steps, signal, spikes, delays,
                    window, min_f, max_f):
    output = np.zeros(steps)
    ideal_eval = 0
    for i in range(0, np.searchsorted(delays, 1 / max_f)):
        add_spikes(output, dt, steps, spikes[i], delays[i], 1)
        freqs, spect = coherences(dt, signal, output, min_f, max_f, round(window/dt))
        eval = utility(freqs, spect)#information(freqs, spect, 1)
        if eval > ideal_eval:
            ideal_eval = eval
            ideal_output = np.copy(output)
    return ideal_output

def optimal_weight(delay, freq, decay = 0.8):
    return np.cos(delay*freq*2*np.pi) #* decay ** (delay*freq)

def freqs_delay_sum(n, dt, steps, spikes, delays, min_f, max_f):
    output = np.zeros(steps)
    for f in np.linspace(min_f, max_f, round(max_f/min_f)):
        for i in range(n):
            period = 0.5/f
            delay = np.ceil(delays[i] / period) * period
            #print(delay)
            add_spikes(output, dt, steps, spikes[i], delay,
                       weight = optimal_weight(delay, f))
    return output

def freq_weight_sum(n, dt, steps, spikes, delays, min_f, max_f):
    output = np.zeros(steps)
    for f in np.linspace(min_f, max_f, round(max_f/min_f)):
        for i in range(n):
            add_spikes(output, dt, steps, spikes[i], delays[i],
                       weight = optimal_weight(delays[i], f))
    return output

def laggy_sum(n, dt, signal, spikes, delays, freq, lag):
    length = len(signal.values)
    output = np.zeros(length)
    period = np.arange(0, round(lag/dt)) * dt
    for i in range(n):
        effect = np.cos((period + delays[i]) * freq * 2*np.pi)
        i_spikes = np.around((spikes[i]+delays[i])/dt).astype(int)
        i_output = np.zeros(length)
        i_output[i_spikes[i_spikes < length]] = 1
        output += np.convolve(i_output, effect, 'same')
    return output

def fourier_combine(n, dt, dur, steps, spikes, delays, min_f, max_f, fdt, alpha):
    #test = np.zeros(round(dur/fdt))
    output = np.zeros(steps)
    #f_steps = round(max_f/min_f)
    indices = np.zeros(n, dtype = uint)
    fourier = np.zeros(11, dtype = complex)#f_steps)
    #f_freqs = np.arange(1, 11)
    for t in np.linspace(0, dur, round(dur/fdt), endpoint = False):
        print(t)
        num_s = np.array([np.searchsorted(spikes[i][indices[i]:], t - delays[i]) for i in range(n)], dtype=uint)
        indices += num_s
        local_f = np.fft.rfft(num_s)[:11]
        #fourier[0] = local_f[0]
        fourier[1:11] *= np.exp(fdt * min_f * -2j * np.pi * np.arange(1, 11))
        fourier[1:11] += alpha * (local_f[1:11] - fourier[1:11])
        fourier[0] = 0
        #fourier = local_f
        #plot(np.fft.irfft(fourier))
        #show()
        #test[round(t/fdt)] = np.abs(local_f[1])
        #plot([mean(num_s[i:i+5]) for i in range(0, 100, 5)])
        #plot(np.fft.irfft(fourier))
        #show()
        output[round(t/dt)] = np.fft.irfft(fourier)[0]
    #plot(test)
    #show()
    return output

def utility(freqs, spect): return mean(spect)

def information(freqs, spect, df):
    #print(freqs[::df])
    if freqs.size == 0: return 0
    elif (freqs.size - 1) / df < 1: return -np.log2(1-spect[0])
    else: return -np.trapz(np.log2(1-spect[::df]), freqs[::df])

def coherences(dt, signal, output, min_f, max_f, nperseg, nfft = None):
    freqs, spect = sg.coherence(signal.values, output, fs = round(second/dt),
                                nperseg = nperseg, nfft = nfft)
    
    #embed()
    min = np.searchsorted(np.around(freqs), min_f, side = 'left')
    max = np.searchsorted(np.around(freqs), max_f, side = 'right')
    return freqs[min:max], spect[min:max]

def window_evaluate(dt, signal, output, min_f, max_f, windows):
    evals = np.empty(windows.size)
    for w in range(windows.size):
        freqs, spect = coherences(dt, signal, output, min_f, max_f,
                                  nperseg = round(windows[w]/dt))
        #embed()
        evals[w] = utility(freqs, spect) #information(freqs, spect, 1)
    #plot(freqs, spect)
    #show()
    return evals

def bi_res_evaluate(dt, signal, output, temp_res, freq_res, min_f, max_f):
    evals = np.empty((temp_res.size, freq_res.size))
    min_nfft = round(np.max(freq_res) / min_f / dt)
    #print(min_nfft)
    for x in range(temp_res.size):
        nperseg = round(second / temp_res[x] / dt)
        #print(nperseg)
        nfft = np.ceil(nperseg/min_nfft) * min_nfft
        #print(nfft)
        freqs, spect = coherences(dt, signal, output, min_f, max_f,
                                  nperseg, nfft)
        plot(freqs, spect)
        show()
        for y in range(freq_res.size):
            #print(int(ceil(freqs.size / freq_res[y] / 10)))
            evals[x][y] = information(freqs, spect, int(ceil(freqs.size / freq_res[y] / 10)))
    return evals