import numpy as np
from IPython import embed

def same_delay(input, delay):
    level = np.zeros(input.shape)
    for i in range(input.shape[0]):
        shift = round(delay * (1 - i / input.shape[0]))
        level[i, shift:] = input[i, :-shift]
    return np.average(level, 0)

def narrow_sum(input, delay, min_p):
    num = round(input.shape[0] * 0.5 * min_p / delay)
    return np.average(input[:num], 0)

def smart_weights(input, delay, min_p, max_p):
    num_f = round(max_p/min_p)
    weights = [np.sum(np.cos(2*np.pi*np.arange(1, num_f+1)*i*delay/max_p))
               for i in np.arange(0, 1, 1 / input.shape[0])]
    return np.average(input, 0, weights)

def dynamic_state(input, delay, min_p, max_p, alpha):
    output = np.zeros(input.shape[1])
    num_f = round(max_p/min_p)
    state = np.zeros(num_f, dtype = complex)
    scope = input.shape[0] * max_p / delay
    input_freqs = np.fft.rfft(input, scope, 0)[1:num_f+1]
    phase_shift = np.exp(-2j * np.pi * np.arange(1, num_f+1) / max_p)
    for s in range(output.size):
        state *= phase_shift
        state += alpha * (input_freqs[:, s] - state)
        output[s] = np.sum(np.real(state))
    return output