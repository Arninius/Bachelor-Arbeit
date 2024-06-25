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

def fourier_combine(n, dt, dur, steps, spikes, delays, min_f, max_f, res, alpha):
    span = 1/max_f/res
    output = np.zeros(steps)
    num_freqs = round(max_f/min_f)
    indices = np.zeros(n, dtype = uint)
    state = np.zeros(num_freqs, dtype = complex)
    for t in np.arange(dur/span) * span:
        #print(t)
        #start_time = tm.time()
        clusters = np.array([np.searchsorted(spikes[i][indices[i]:], t - delays[i])
                             for i in range(n)], dtype=uint)
        indices += clusters
        #print(tm.time()-start_time)
        local_state = np.fft.rfft(clusters)[1:num_freqs+1]
        #print(tm.time()-start_time)
        state *= np.exp(-2j * np.pi * span * min_f * np.arange(1, num_freqs+1))
        state += alpha * (local_state - state)
        #print(tm.time()-start_time)
        output[round(t/dt)] = np.sum(np.real(state)) # np.fft.irfft(fourier)[0]
        #print(tm.time()-start_time)
    return output

def new_fourier_combine(n, dt, dur, steps, spikes, delays, min_f, max_f, res, alpha):
    output = np.zeros(steps)
    num_freqs = round(max_f/min_f)
    for f in (np.arange(num_freqs) + 1) * min_f:
        print(f)
        span = 1/f/res
        num_clusters = int(ceil(delays[-1] / span))
        cluster_sizes = np.zeros(num_clusters)
        index = 0
        for c in range(num_clusters):
            cluster_sizes[c] = np.searchsorted(delays[int(index):], span * (c + 1))
            index += cluster_sizes[c]
        state = 0 + 0j
        indices = np.zeros(n, dtype = uint)
        for t in np.arange(dur/span) * span:
            clusters = np.zeros(num_clusters)
            for i in range(n):
                num_spikes = np.searchsorted(spikes[i][indices[i]:], t - delays[i])
                indices[i] += num_spikes
                clusters[int(delays[i] / span)] += num_spikes
            clusters /= cluster_sizes
            local_state = np.sum(clusters * np.exp(-2j * np.pi * f / min_f * np.arange(num_clusters)))
            state *= np.exp(-2j * np.pi * span * min_f * np.arange(1, num_freqs+1))
            state += alpha * (local_state - state)
            output[round(t/dt)] += np.sum(np.real(state)) * span / second
    return output

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


def freqs_delay_sum(n, dt, steps, spikes, delays, min_f, max_f):
    output = np.zeros(steps)
    for f in np.linspace(min_f, max_f, round(max_f/min_f)): # anpassen auf wie bei fourier_combine
        for i in range(n):
            period = 0.5/f
            delay = np.ceil(delays[i] / period) * period
            #print(delay)
            add_spikes(output, dt, steps, spikes[i], delay,
                       weight = optimal_weight(delay, f))
    return output

def bi_res_heatmap(dt, signal, models, min_f, max_f):
    temp_res = np.arange(1, 100, 10)
    freq_res = np.arange(1, 11)
    for m in models:
        print(m['label'])
        result = bi_res_evaluate(dt, signal, m['output'], temp_res, freq_res, min_f, max_f)
        if(m['label'] == "min delay"):
            max_info = result
        else:
            performance = result / max_info
            #print(performance)
            imshow(performance, cmap = 'Greys', vmin = 0, vmax = 1)
            show()

def window_line_plot(dt, signal, models, min_f, max_f):
    windows = np.around(10 ** np.linspace(0, 2, 10)) * 10*ms
    for m in models:
        evals = window_evaluate(dt, signal, m['output'], min_f, max_f, windows)
        plot(windows/ms, evals, label = m['label'], color = m['color'])
    xlabel("evaluation window in ms")
    xscale("log")
    xlim(10, 1000)
    ylabel("mean coherence")
    legend()
    show()