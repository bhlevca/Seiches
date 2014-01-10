'''
Created on Jun 12, 2012

@author: bogdan
'''
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from matplotlib.dates import seconds
from matplotlib.dates import date2num, num2date
from matplotlib.dates import MONDAY, SATURDAY
import matplotlib.mlab
import matplotlib.dates
import numpy as np
import scipy as sp
import scipy.signal
from scipy import  stats
import math
import smooth
import time, os
import csv
import chi2
import filters
import pylab

years = matplotlib.dates.YearLocator()  # every year
months = matplotlib.dates.MonthLocator()  # every month
yearsFmt = matplotlib.dates.DateFormatter('%Y')
# every monday
mondays = matplotlib.dates.WeekdayLocator(MONDAY)


def timestamp2doy(dateTime):
    '''
    Converts from date time in seconds to day of the year
    '''
    dofy = np.zeros(len(dateTime))
    for j in range(0, len(dateTime)) :
        d = num2date(dateTime[j])
        dofy[j] = d.timetuple().tm_yday + d.timetuple().tm_hour / 24. + d.timetuple().tm_min / (24. * 60) + d.timetuple().tm_sec / (24. * 3600)

    return dofy

def drange(start, stop, step):
    '''
    Example:
    i0=drange(0.0, 1.0, 0.1)
    >>>["%g" % x for x in i0]
    ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    '''

    r = start
    while r < stop:
        yield r
        r += step
    # end while
# end drange


def readFile(path_in, fname):
    # read Lake data
    filename = path_in + '/' + fname

    ifile = open(filename, 'rb')
    reader = csv.reader(ifile, delimiter = ',', quotechar = '"')
    rownum = 0
    SensorDepth = []
    Time = []
    printHeaderVal = False
    for row in reader:
        try:
            SensorDepth.append(float(row[1]))
            Time.append(float(row[0]))
        except:
            pass
    # end for

    return [Time, SensorDepth]
# end readFile

def moving_average(x, n, type = 'simple'):
    """
    compute an n period moving average.
    type is 'simple' | 'exponential'
    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()
    a = np.convolve(x, weresultsights, mode = 'full')[:len(x)]
    a[:n] = a[n]
    return a


def nextpow2(i):
    """
    Find the next power of two

    >>> nextpow2(5)
    8
    >>> nextpow2(250)
    256
    """
    # do not use numpy here, math is much faster for single values
    buf = math.ceil(math.log(i) / math.log(2))
    return int(math.pow(2, buf))
# end nextpow2

def smoothSeries(data, span):
    '''
    '''
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    results_temp = smooth.smoothfit(data, span, windows[2])

    return results_temp['smoothed']
# end smoothSeries


# another moving average smoothing similar to self.moving _average()
def smoothSeriesWindow(data, WINDOW = 10):

    extended_data = np.hstack([[data[0]] * (WINDOW - 1), data])
    weightings = np.repeat(1.0, WINDOW) / WINDOW
    smoothed = np.convolve(extended_data, weightings)[WINDOW - 1:-(WINDOW - 1)]
    return smoothed

def smoothListGaussian(list, strippedXs = False, degree = 100):



    window = degree * 2 - 1

    weight = np.array([1.0] * window)

    weightGauss = []

    for i in range(window):

        i = i - degree + 1

        frac = i / float(window)

        gauss = 1 / (np.exp((4 * (frac)) ** 2))

        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight

    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):

        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return smoothed

# This is a moving average detrend
def detrend(data, degree = 10):
        detrended = [None] * degree
        for i in range(degree, len(data) - degree):
                chunk = data[i - degree:i + degree]
                chunk = sum(chunk) / len(chunk)
                detrended.append(data[i] - chunk)
        return detrended + [None] * degree
# end detrend

# This is a similar to sp.signal.detrend
def detrend_separate(y, order = 0):
    '''detrend multivariate series by series specific trends

    Paramters
    ---------
    y : ndarray
       data, can be 1d or nd. if ndim is greater then 1, then observations
       are along zero axis
    order : int
       degree of polynomial trend, 1 is linear, 0 is constant

    Returns
    -------
    y_detrended : ndarray
       detrended data in same shape as original

    '''
    nobs = y.shape[0]
    shape = y.shape
    y_ = y.reshape(nobs, -1)
    kvars_ = len(y_)
    t = np.arange(nobs)
    exog = np.vander(t, order + 1)
    params = np.linalg.lstsq(exog, y_)[0]
    fittedvalues = np.dot(exog, params)
    resid = (y_ - fittedvalues).reshape(*shape)
    return resid, params

def plotArray(title, xlabel, ylabel, x, y, legend = None, linewidth = 0.6, plottitle = False):

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        ax = fig.add_subplot(111)

        ax.plot(x, y, linewidth = 0.6)

        ax.xaxis.grid(True, 'minor')
        ax.grid(True)
        title = title

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if plottitle:
            plt.title(title)
        if legend != None:
            plt.legend(legend);

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        plt.draw()
        plt.show()
    # end

def plot_n_Array(title, xlabel, ylabel, x_arr, y_arr, legend = None, linewidth = 0.6, ymax_lim = None, log = False, \
                 plottitle = False, grid = False, fontsize = 18, noshow = False):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    ax = fig.add_subplot(111)

    i = 0;
    ls = ['-', '--', ':', '-.', '-', '--', ':', '-.']
    for a in x_arr:
        x = x_arr[i]
        y = y_arr[i]
        if log:
            ax.loglog(x, y, linestyle = ls[i], linewidth = 1.2 + 0.4 * i, basex = 10)
        else:
            ax.plot(x, y, linestyle = ls[i], linewidth = 1.2 + 0.4 * i)
        i += 1
    # end for

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(grid)
    plt.ylabel(ylabel).set_fontsize(fontsize)
    plt.xlabel(xlabel).set_fontsize(fontsize)
    if plottitle:
        plt.title(title).set_fontsize(fontsize + 2)

    if legend != None:
        plt.legend(legend);
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room fornumpy smoothing filter them
    if ymax_lim != None:
        plt.ylim(ymax = ymax_lim)
    if not noshow:
        plt.show()
    return ax
# end

def errorbar(ax, x0, y0, ci, color):
    ax.loglog([x0, x0], [y0 * ci[0], y0 * ci[1]], color = color)
    ax.loglog(x0, y0, 'bo')
    ax.loglog(x0, y0 * ci[0], 'b_')
    ax.loglog(x0, y0 * ci[1], 'b_')

def plot_n_Array_with_CI(title, xlabel, ylabel, x_arr, y_arr, ci05, ci95, legend = None, linewidth = 0.8, ymax_lim = None, log = False, \
                         fontsize = 18, plottitle = False, grid = False):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    ax = fig.add_subplot(111)

    i = 0
    lst = ['-', '--', '-.', ':', '-', '--', ':', '-.']
    colors = ['b', 'y', 'r', 'g', 'c', 'm', 'k', 'aqua']
    for a in x_arr:
        x = x_arr[i][3:]
        y = y_arr[i][3:]
        if len(x_arr) < 5:
            lwt = 3.5
        else:
            lwt = 1 + i * 0.6

        if log:
            ax.loglog(x, y, linestyle = lst[i], linewidth = lwt, basex = 10, color = colors[i])
        else:
            ax.plot(x, y, linestyle = lst[i], linewidth = lwt, color = colors[i])
        i += 1
    # end for

    # plot the confidence intervals
    i = 0
    Ymin = 10000000
    Ymax = 0
    for a in ci05:  # x_arr:
        if log:
            x = a[1:]
            y1 = ci05[0]
            y2 = ci95[0]
            ymx = max(y_arr[0][3:])
            ymin = min(y_arr[0][3:])
            y0 = ymx * 0.65
            # Choose a locatioon for the CI bar
            ax.set_yscale('log')
            # yerr = (y2 - y1) / 2.0
            # ax.errorbar(a[150], y0, xerr = None, yerr = yerr)
            errorbar(ax, a[int((len(a) - 1))] * 1.2, y0, [y1, y2], color = 'b')
            ax.annotate("95%", (a[int((len(a) - 1))] * 1.3, y0), ha = 'left', va = 'center', bbox = dict(fc = 'white', ec = 'none'))
            Ymin = min(Ymin, ymin)
            Ymax = max(Ymax, ymx)
        else:
            y1 = ci05[i][3:]
            y2 = ci95[i][3:]
            sd = 0.65 - i * 0.15

            ax.plot(x, y1, x, y2, color = [sd, sd, sd], alpha = 0.0)
            ax.fill_between(x, y1, y2, where = y2 > y1, facecolor = [sd, sd, sd], alpha = 1, interpolate = True, linewidth = 0.0)
        i += 1

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(grid, 'minor')
    ax.grid(grid)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.xlabel(xlabel, fontsize = fontsize)
    if plottitle:
        plt.title(title, fontsize = fontsize)

    if legend != None:
        plt.legend(legend);
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room fornumpy smoothing filter them
    if ymax_lim != None:
        plt.ylim(ymax = ymax_lim)
    if log:
        plt.ylim(ymin = Ymin * 0.85, ymax = Ymax * 1.15)


    plt.show()
# end


def plotTimeSeries(title, xlabel, ylabel, x, y, legend = None, linewidth = 0.6, plottitle = False, \
                    doy = False, grid = False):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')
    ax = fig.add_subplot(111)

    if doy:
        dofy = timestamp2doy(x)
        ax.plot(dofy, y, linewidth = 0.6)
    else:
        ax.plot(x, y, linewidth = 0.6)

        # format the ticks
        formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(mondays)
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

    ax.xaxis.grid(grid, 'minor')
    ax.grid(grid)
    title = title

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if plottitle:
        plt.title(title)
    if legend != None:
        plt.legend(legend);

    plt.draw()
    plt.show()
# end

def plot_n_TimeSeries(title, xlabel, ylabel, x_arr, y_arr, legend = None, linewidth = 0.8, plottitle = False, fontsize = 18, \
                       doy = False, minmax = None, grid = False, show = True):

    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    ax = fig.add_subplot(111)

    i = 0;
    ls = ['-b', '--y', ':m', '-.r', '-c', '--g', ':k', '-.aqua']
    for a in x_arr:
        x = x_arr[i]
        y = y_arr[i]
        if doy:
            dofy = timestamp2doy(x)
            ax.plot(dofy, y, ls[i])
        else:
            ax.plot(x, y, ls[i])
        i += 1
    # end for
    if not doy:
        # format the ticks
        formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(mondays)
        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        fig.autofmt_xdate()

    # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(grid, 'minor')
    ax.grid(grid)
    plt.ylabel(ylabel).set_fontsize(fontsize + 2)
    plt.xlabel(xlabel).set_fontsize(fontsize + 2)
    if plottitle:
        plt.title(title).set_fontsize(fontsize + 2)

    if legend != None:
        plt.legend(legend);
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    plt.xticks(fontsize = fontsize - 4)
    plt.yticks(fontsize = fontsize - 4)
    if minmax != None:
        plt.ylim(ymin = minmax[0], ymax = minmax[1])
    if show:
        plt.show()
# end

def flattop(N):
    a0 = 0.2810639
    a1 = 0.5208972
    a2 = 0.1980399
    w = np.zeros(N)
    for i in range(0, N):
        w[i] = a0 - a1 * math.cos(2 * np.pi * i / N) + a2 * math.cos(4 * np.pi * i / N)
    # end for
    return w
# end flattop

def findPeaks(self, f, mx):
    '''
    find max peaks
    '''
    # a = np.array([10.3, 2, 0.9, 4, 5, 6, 7, 34, 2, 5, 25, 3, -26, -20, -29], dtype = np.float)

    gradients = np.diff(mx)
    print gradients


    maxima_num = 0
    minima_num = 0
    max_locations = []
    min_locations = []
    count = 0
    for i in gradients[:-1]:
        count += 1

        if ((cmp(i, 0) > 0) & (cmp(gradients[count], 0) < 0) & (i != gradients[count])):
            maxima_num += 1
            max_locations.append(count)

        if ((cmp(i, 0) < 0) & (cmp(gradients[count], 0) > 0) & (i != gradients[count])):
            minima_num += 1
            min_locations.append(count)
    # end for

    turning_points = {'maxima_number':maxima_num, 'minima_number':minima_num, 'maxima_locations':max_locations, 'minima_locations':min_locations}

    pass
    return turning_points
    # print turning_points

    # plt.plot(a)
    # plt.show()

def bandSumCentredOnPeaks(f, mx, band):
    '''
    find max peaks and band around it 'n' bins
    '''
    # turning_points = fft_utils.findPeaks(f, mx)
    # locations = turning_points['maxima_locations']
    # number = turning_points['maxima_number']

    # this works better
    if 1 == 0:
        xs = np.arange(0, np.pi, 0.05)
        data = np.sin(xs)
        peakind = sp.signal.find_peaks_cwt(data, np.arange(1, 10),
                                           wavelet = ricker , max_distances = None,
                                           gap_thresh = None, min_length = None,
                                           min_snr = 2.0, noise_perc = 10)
        print peakind, xs[peakind], data[peakind]
        plt.plot(xs, data)
    # endif

    locations = sp.signal.find_peaks_cwt(mx, np.arange(1, 2),
                                         wavelet = scipy.signal.wavelets.ricker, max_distances = None,
                                         gap_thresh = 1, min_length = 1,
                                         min_snr = 1.0, noise_perc = 25)

    pd3 = peakdek.peakdet(mx, 1)


    for loc in locations:
        for b in range(-band / 2 + 1 , band / 2 + 1):
            if b != 0:
                mx[loc] += mx[loc + b].copy()
                mx[loc + b] = 0

    return mx






def RMS_values(data, nwaves):
    # detrend
    # demean

    rms = np.sqrt(1 / nwaves)

def autocorrelation(data, dt):
    '''This function calculates the autocorrelation function for the
       data set "origdata" for a lag timescale of 0 to "endlag" and outputs
       the autocorrelation function in to "a".
       function a = auto( origdata, endlag);
       (c) Dennis L. Hartmann
    '''

    N = len(data)
    a = np.zeros(N)
    endlag = N
    # now solve for autocorrelation for time lags from zero to endlag
    for lag in range(0, endlag - 1):
        data1 = data[0:N - lag]
        data1 = data1 - np.mean(data1)
        data2 = data[lag:N]
        data2 = data2 - np.mean(data2)
        print lag
        a[lag] = np.sum(data1 * data2) / np.sqrt(np.sum(data1 ** 2) * np.sum(data2 ** 2))
    # end for
    return a


def edof_stat(data, dt):
    # autocorrelation function
    a = autocorrelation(data, dt)
    n = len(data)
    Te = -dt / a
    tau = n * dt
    rt = math.exp(-tau / Te)
    dof = n * (-0.5 * math.log(rt))
    return dof


def dof(freq):
    '''
    NOTE: the confidence interval is calcuate for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================
    For non segmented data has 2 degrees of freedom for each datapoint Sj (Percival 1993 pag 20-21)
        Since Aj and Bj rv's are mutually uncorrelated on the assumption of Gaussianity for Aj and Bj

        and
         Data Analysis Methods in Physical Oceanography ( Emery 2001 ) p 424


        dof Sj is only a two degrees of freedom estimate of Sigma, there should be considerable variability Sj function
        '''
    dof = len(freq) - 2  # was = 2 - for each point of the FFT
    return dof


def edof(data, N, M, window_type):
    '''
    NOTE: the confidence interval is calcuate for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================

    Formulas from Prisley 1981:
    window_Type can be one of:
    ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'flattop']

    if window_type == 'flat':  # rectangular/Daniell
        dof = 1.0 * N / M
    if window_type == 'hanning':
        dof = (8.0 * N) / (3.0 * M)
    elif window_type == 'hamming':
        dof = 2.5164 * N / M
    elif window_type == 'bartlett':
        dof = 3.0 * N / M
    elif window_type == 'blackman':  # Not so sure about this one
        dof = 2.0 * N / M
    elif window_type == 'flattop':
        # ageneric formula or 50% ovelapping windows from  Data Analysis Methods in Physical Oceanography
        dof = 4.0 * N / M  #
    else:
         raise ValueError, "%f window not supported" % window_type
    '''

    '''
        Method from Spectral Analysis for Physical Applications  Percival 1993 p294
        Can be applied for 50% overlapping with Hanning windowing
        Other windows can be supported and other percents of overlapping
    '''
    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'flattop']
    # number of blocks for a 50% overlap  N/M is the number of distinct segments
    nb = 2 * N / M - 1

    if not window_type in windows:
        raise "Window %s not supported" % window_type
    dof = 36 * nb * nb / (19 * nb - 1)


    return dof

def confidence_interval(data, dof, p_val, log = False):
    '''
    NOTE: the confidence interval is calculated for EACH frequency of the spectrum in the frequency domain
    =====================================================================================================

    a() and b() are the 0.025 and 0.975% points of the X^2 distribution with v
    equivalent degrees of freedom [Priestley,1981, pp. 467-468].

    @param data : a time series or a FFT transform that is going to be evaluated.
                It can be a Power Density Spectrum ( module of Amplitude spectrum)
    @param dof  :Degrees of Freedom , usualy dof = n - scale for wavelets or n-2 for FFT

    @param p_val : the desired significance  p=0.05 for 95% confidence interval

    @return:  tuple of array with the interval  limits (a, b)
    '''

    p = 1 - p_val
    p_val = p / 2.

    chia = (stats.chi2.ppf(p_val, dof))  # typically 0.025
    a = dof / chia

    chib = (stats.chi2.ppf(1 - p_val, dof))  # typically 0.975
    b = dof / chib

    if not log:
        a *= data
        b *= data

    # alternate calculations
    # ci = 1. / [(1 - 2. / (9 * dof) + 1.96 * math.sqrt(2. / (9 * dof))) ** 3, (1 - 2. / (9 * dof) - 1.96 * math.sqrt(2. / (9 * dof))) ** 3]

    return (b, a)



if __name__ == '__main__':

    '''
    N = 100
    x = np.linspace(0, N, N)
    w = flattop(N)
    w2 = sp.signal.get_window("flattop", N, fftbins = 1)
    plt.plot(x, w, x, w2)
    plt.show()
    '''


    '''
    ########################################################
    # #butterworth - not working properly with scipy butord
    #######################################################

    # some constants
    samp_rate = 200
    sim_time = 5
    cuttoff_freq = 15.
    freqs = [1., 5., 10., 40.]

    low_cutoff = 11.0
    high_cuttoff = 45.0
    # samp_rate = 20
    # sim_time = 60
    # cuttoff_freq = 0.5
    # freqs = [0.1, 0.5, 1., 4.]

    nsamps = samp_rate * sim_time



    fig = plt.figure()

    # generate input signal
    t = np.linspace(0, sim_time, nsamps)

    x = 0
    for i in range(len(freqs)):
        x += np.cos(2 * math.pi * freqs[i] * t)
    time_dom = fig.add_subplot(232)
    plt.plot(t, x)
    plt.title('Filter Input - Time Domain')
    plt.grid(True)

    # input signal spectrum
    xfreq = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(nsamps, d = 1. / samp_rate)
    fig.add_subplot(233)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(xfreq)[0:nsamps / 2])
    plt.title('Filter Input - Frequency Domain')
    plt.text(0.03, 0.01, "freqs: " + str(freqs) + " Hz")
    plt.grid(True)

    # filter frequency response
    btype = 'bandpass'

    # norm_factor = math.pi / samp_rate
    norm_factor = 1.0 / samp_rate

    if btype == "low" :
        # norm_stop > norm_pass
        # norm_pass = 2 * math.pi * cuttoff_freq / (samp_rate / 2)
        norm_pass = cuttoff_freq * norm_factor
        norm_stop = cuttoff_freq * norm_factor * 2.0
    elif btype == 'high':
        # norm_pass > norm_stop
        norm_pass = cuttoff_freq * norm_factor
        norm_stop = cuttoff_freq * norm_factor / 2.0
    elif btype == 'bandpass':
        norm_pass = [2.0 * low_cutoff * norm_factor, high_cuttoff * norm_factor]
        norm_stop = [low_cutoff * norm_factor, 2.0 * high_cuttoff * norm_factor]

    # N the order of butterworth filter
    (y, w, h, N) = butterworth(x, norm_pass, norm_stop, btype, debug = True)
    # filtered output
    fig.add_subplot(235)
    plt.plot(t, y)
    plt.title('Filter output - Time Domain')
    plt.grid(True)

    # output spectrum
    yfreq = np.fft.fft(y)
    fig.add_subplot(236)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(yfreq)[0:nsamps / 2])
    # plt.plot(fft_freqs[0:nsamps / 2], np.abs(yfreq)[0:nsamps / 2])
    plt.title('Filter Output - Frequency Domain')
    plt.grid(True)

    debug = True
    if debug == True:
        fig_B = plt.figure()
        # plt.loglog(w * samp_rate / math.pi, np.abs(h))
        # w is in rad/sample therefore we ndde to multiply by sample and divide by pi
        plt.plot(w * samp_rate / (2 * math.pi), np.abs(h))
        plt.title('Filter Frequency Response')
        plt.text(2e-3, 1e-5, str(N) + "-th order " + btype + "pass Butterworth filter")
        plt.grid(True)

    plt.show()
    '''
    ############################
    # filtering
    ###########################
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    # Plot the frequency response for a few different orders.
    plt.figure(1)
    plt.clf()
    for order in [3, 6, 9]:
        b, a = filters.butter_bandpass(lowcut, highcut, fs, order = order)
        w, h = sp.signal.freqz(b, a, worN = 2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label = "order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label = 'sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc = 'best')

    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint = False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label = 'Noisy signal')

    y, w, h, N, delay = filters.butterworth(x, 'band', lowcut, highcut, fs, order = 6)
    if len(y) != len(t):
        t = scipy.signal.resample(t, len(y))
    plt.plot(t, y, label = 'Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles = '--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc = 'upper left')

    plt.show()

    ###################################
    # known signals on fft
    ###################################
    fig = plt.figure(2)
    samp_rate = 200
    sim_time = 5
    freqs = [3., 5., 10., 40.]
    lowcut = 2.1
    highcut = 4.0

    nsamps = samp_rate * sim_time

    # generate input signal
    t = np.linspace(0, sim_time, nsamps)

    x = 0
    for i in range(len(freqs)):
        x += np.cos(2 * math.pi * freqs[i] * t)
    time_dom = fig.add_subplot(232)
    plt.plot(t, x)
    plt.title('Filter Input - Time Domain')
    plt.grid(True)

     # input signal spectrum
    xfreq = np.fft.fft(x)
    fft_freqs = np.fft.fftfreq(nsamps, d = 1. / samp_rate)
    fig.add_subplot(233)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(xfreq)[0:nsamps / 2])
    plt.title('Filter Input - Frequency Domain')
    plt.text(0.03, 0.01, "freqs: " + str(freqs) + " Hz")
    plt.grid(True)
    N = 5
    # b, a = filters.butter_bandpass(lowcut, highcut, samp_rate, order = N)
    # w, h = sp.signal.freqz(b, a, worN = 2000)
    # y = filters.butter_bandpass_filter(x, lowcut, highcut, samp_rate, order = N)
    btype = 'band'
    # btype = 'high'
    # btype = 'low'

    y, w, h, b, a = filters.butterworth(x, btype, lowcut, highcut, samp_rate)
    # y1 = filters.fft_bandpassfilter(x, samp_rate, lowcut, highcut)

    # filtered output
    fig.add_subplot(235)
    # pylab.plot(t, y1)
    # plt.plot(t, y, t, abs(y1))
    plt.title('Filter output - Time Domain')
    plt.grid(True)

    # output spectrum
    yfreq = np.fft.fft(y)
    fig.add_subplot(236)
    plt.loglog(fft_freqs[0:nsamps / 2], np.abs(yfreq)[0:nsamps / 2])
    plt.title('Filter Output - Frequency Domain')
    plt.grid(True)

    debug = True
    if debug == True:
        fig_B = plt.figure()
        # plt.loglog(w * samp_rate / math.pi, np.abs(h))
        # w is in rad/sample therefore we ndde to multiply by sample and divide by pi
        plt.plot(w * samp_rate / (2 * math.pi), np.abs(h))
        plt.title('Filter Frequency Response')
        plt.text(2e-3, 1e-5, str(N) + "-th order " + 'band' + "pass Butterworth filter")
        plt.grid(True)

    plt.show()
