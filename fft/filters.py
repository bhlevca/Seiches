import scipy as sp
import numpy as np

def butterworth_bad(data, btype, passband, stopband, debug = False):
    '''
     This function does butterworth filtering.

    Args:
       data (array):  the timeseries to be filtered.
       passband (float): the value from which we start filtering.
       stopband (float): the value from which we start filtering.
       btype  (string):  * 'low'  for low pass
                         * 'high' for high pass
                         * 'band' for band pass
       debug (Boolean): True/False - optional

    Kwargs:


    Returns:
       y (ndarray) : the filtered timeseries
       w (ndarray) : The frequencies at which h was computed.
       h (ndarray) : The frequency response.


    Raises:


    '''
    # design filter
    '''
    Lowpass    Wp < Ws, both scalars                          (Ws,1)                   (0,Wp)
    Highpass   Wp > Ws, both scalars                          (0,Ws)                   (Wp,1)
    Bandpass   The interval specified by Ws contains
               the one specified by Wp                        (0,Ws(1)) and (Ws(2),1)  (Wp(1),Wp(2))
               (Ws(1) < Wp(1) < Wp(2) < Ws(2)).

    Bandstop   The interval specified by Wp contains          (Ws(1),Ws(2))            (0,Wp(1)) and (Wp(2),1)
               the one specified by Ws
               (Wp(1) < Ws(1) < Ws(2) < Wp(2)).

    '''
    (N, Wn) = sp.signal.buttord(wp = passband, ws = stopband, gpass = 3, gstop = 60, analog = 0)
    (b, a) = sp.signal.butter(N, Wn, btype = btype, analog = 0, output = 'ba')
    # b *= 1e3
    if debug == True:
        print("b=" + str(b) + ", a=" + str(a))

    # filter frequency response
    # w is in rad/sample
    (w, h) = sp.signal.freqz(b, a)

    # filtered output
    # zi = signal.lfiltic(b, a, x[0:5], x[0:5])
    # (y, zi) = signal.lfilter(b, a, x, zi=zi)
    y = sp.signal.lfilter(b, a, data)
    return (y, w, h, N)


def butter_bandpass(lowcut, highcut, fs, order = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sp.signal.butter(order, [low, high], btype = 'band', analog = 0, output = 'ba')
    return b, a

def butter_highpass(highcut, fs, order = 5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = sp.signal.butter(order, high, btype = 'high', analog = 0, output = 'ba')
    return b, a

def butter_lowpass(lowcut, fs, order = 5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = sp.signal.butter(order, low, btype = 'low', analog = 0, output = 'ba')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order = 5):
    b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    y = sp.signal.lfilter(b, a, data)
    return y

def butterworth(data, btype = 'band', lowcut = None, highcut = None, fs = None, order = 5, worN = 2000):
    if btype == 'band':
        b, a = butter_bandpass(lowcut, highcut, fs, order = order)
    elif btype == 'high':
        b, a = butter_highpass(highcut, fs, order = order)
    elif btype == 'low':
        b, a = butter_lowpass(lowcut, fs, order = order)

    w, h = sp.signal.freqz(b, a, worN = worN)
    y = sp.signal.lfilter(b, a, data)

    return [y, w, h, b, a]


def fft_bandpassfilter(data, fs, lowcut, highcut):
    fft = np.fft.fft(data)
    n = len(data)
    timestep = 1.0 / fs
    freq = np.fft.fftfreq(n, d = timestep)
    bp = fft[:]
    for i in range(len(bp)):
        if freq[i] >= highcut or freq[i] < lowcut:
            bp[i] = 0
        #    print "Not Passed"
        # else :
        #    print "Passed"

    # must multipy by 2 to get the correct amplitude  due to FFT symetry
    ibp = 2 * sp.ifft(bp)
    return ibp
