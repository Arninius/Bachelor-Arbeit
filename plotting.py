from brian2 import *
import numpy as np
import analysis as nl

def spect_plot(dt, signal, models, min_f, max_f, window):
    for m in models:
        freqs, spect = nl.coherences(dt, signal, m['output'], 
                                     min_f, max_f, round(window/dt))
        plot(freqs, spect, label = m['label'], color = m['color'])
        print(m['label'] + ": " + str(round(nl.utility(freqs, spect), 2)))
    xlabel("frequency in Hz")
    ylabel("coherence")
    legend()
    show()

def window_line_plot(dt, signal, models, min_f, max_f):
    windows = np.around(10 ** np.linspace(0, 2, 10)) * 10*ms
    for m in models:
        evals = nl.window_evaluate(dt, signal, m['output'], min_f, max_f, windows)
        plot(windows/ms, evals, label = m['label'], color = m['color'])
    xlabel("evaluation window in ms")
    xscale("log")
    xlim(10, 1000)
    ylabel("mean coherence")
    legend()
    show()

def bi_res_heatmap(dt, signal, models, min_f, max_f):
    temp_res = np.arange(1, 100, 10)
    freq_res = np.arange(1, 11)
    for m in models:
        print(m['label'])
        result = nl.bi_res_evaluate(dt, signal, m['output'], temp_res, freq_res, min_f, max_f)
        if(m['label'] == "min delay"):
            max_info = result
        else:
            performance = result / max_info
            #print(performance)
            imshow(performance, cmap = 'Greys', vmin = 0, vmax = 1)
            show()