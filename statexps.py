from brian2 import *
from scipy import signal as sig
from functions import whitenoise
from IPython import embed
import numpy as np

frequency = 20
delta_t = 0.01
stimulus = whitenoise(0, 20, dt=delta_t, duration=1000-delta_t)
#response = whitenoise(0, 20, dt=delta_t, duration=1000-delta_t)
response = stimulus + 1*np.random.randn(len(stimulus))
stimulus_freqs, stimulus_spectrum = sig.welch(stimulus, fs = 1/delta_t, nperseg = 1000)
#plot(stimulus_freqs, stimulus_spectrum)
response_freqs, response_spectrum = sig.welch(response, fs = 1/delta_t, nperseg = 1000)
#plot(response_freqs, response__spectrum)
cross_freqs, cross_spectrum = sig.csd(stimulus, response, fs = 1/delta_t, nperseg = 1000)
#plot(cross_freqs, abs(cross_spectrum))
#plot(cross_freqs, angle(cross_spectrum))
transfer_function = cross_spectrum / stimulus_spectrum
coherence_freqs, coherence_spectrum = sig.coherence(stimulus, response, fs = 1/delta_t, nperseg = 1000)
#embed()
fig, axs = subplots(6, 2)
axs[0,0].plot(arange(0, 1000, delta_t), stimulus)
axs[0,1].plot(arange(0, 1000, delta_t), response)
axs[1,0].plot(abs(rfft(stimulus)))
axs[1,1].plot(abs(rfft(response)))
axs[2,0].plot(stimulus_freqs, stimulus_spectrum)
axs[2,1].plot(response_freqs, response_spectrum)
axs[3,0].plot(cross_freqs, abs(cross_spectrum))
axs[3,1].plot(cross_freqs, angle(cross_spectrum))
axs[4,0].plot(cross_freqs, abs(transfer_function))
axs[4,1].plot(cross_freqs, angle(transfer_function))
axs[5,0].plot(coherence_freqs, coherence_spectrum)
#fig.savefig('spectra.pdf')
show()