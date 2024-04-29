from brian2 import *

def whitenoise(cflow, cfup, dt, duration, rng=numpy.random):
    # number of elements needed for the noise stimulus:
    n = int(ceil((duration+0.5*dt)/dt))
    # next power of two:
    nn = int(2**(ceil(log2(n))))
    # indices of frequencies with `cflow` and `cfup`:
    inx0 = int(round(dt*nn*cflow))
    inx1 = int(round(dt*nn*cfup))
    if inx0 < 0:
        inx0 = 0
    if inx1 >= nn/2:
        inx1 = nn/2
    # draw random numbers in Fourier domain:
    whitef = zeros((nn//2+1), dtype=complex)
    # zero and nyquist frequency must be real:
    if inx0 == 0:
        whitef[0] = 0
        inx0 = 1
    if inx1 >= nn//2:
        whitef[nn//2] = 1
        inx1 = nn//2-1
    phases = 2*pi*rng.rand(inx1 - inx0 + 1)
    whitef[inx0:inx1+1] = cos(phases) + 1j*np.sin(phases)
    # inverse FFT:
    noise = np.real(numpy.fft.irfft(whitef))
    # scaling factor to ensure standard deviation of one:
    sigma = nn / sqrt(2*float(inx1 - inx0))
    return noise[:n]*sigma