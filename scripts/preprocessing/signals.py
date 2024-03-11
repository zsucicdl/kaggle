from scipy import signal


def lpf(wave, fs=12 * 60 * 24, fe=60, n=3):
    nyq = fs / 2.0
    b, a = signal.butter(1, fe / nyq, btype='low')
    for i in range(0, n):
        wave = signal.filtfilt(b, a, wave)
    return wave
