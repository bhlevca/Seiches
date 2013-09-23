'''
Created on Sept 19, 2012

@author: Bogdan Hlevca
@email: bogdan@hlevca.com
@copyright:
    This module is based on Sebastian Krieger's kPyWavelet  email: sebastian@nublia.com

    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/
    and on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.
@version: 0.1
@note: [1] Mallat, Stephane G. (1999). A wavelet tour of signal processing
       [2] Addison, Paul S. The illustrated wavelet transform handbook
       [3] Torrence, Christopher and Compo, Gilbert P. (1998). A Practical
           Guide to Wavelet Analysis

'''

import fft.fft_utils as fft_utils
import kPyWavelet.wavelet as wavelet
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter

from datetime import datetime
from datetime import timedelta
from matplotlib.dates import seconds
from matplotlib.dates import date2num, num2date
from matplotlib.dates import MONDAY, SATURDAY
import matplotlib.dates

years = matplotlib.dates.YearLocator()  # every year
months = matplotlib.dates.MonthLocator()  # every month
yearsFmt = matplotlib.dates.DateFormatter('%Y')
# every monday
mondays = matplotlib.dates.WeekdayLocator(MONDAY)

class kCwt(object):
    '''
    Python class for wavelet analysis and the statistical approach
    suggested by Torrence and Compo (1998) using the wavelet module. To run
    this script successfully, the matplotlib module has to be installed

    '''


    def __init__(self, path = None, file = None, tunits = "day", time = None, var = None):
        '''
        Constructor

        @param path: path to the data file
        @param file: file name of the data file
        @param tunits:"day", "hour", "sec" the unit of the time interval in the timeseries
        @param time: the time array, used for passing data directly, when not reading from a file
        @param var: the timeseries data, used for passing data directly, when not reading from a file

        '''
        # Data members
        self.mother = None
        self.dj = 0.25  # Four sub-octaves per octaves
        self.s0 = -1  # 2 * dt                      # Starting scale, here 6 months
        self.J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
        self.alpha = 0.0  # Lag-1 autocorrelation for white noise
        self.std = None  # Standard deviation
        self.variance = None  # Variance
        self.N = None  # timseries length
        self.freq = None
        self.period = None
        self.dt = None
        self.coi = None
        self.power = None  # Normalized wavelet power spectrum
        self.iwave = None  # inverse wavelet
        self.fft_power = None  # FFT power spectrum
        self.amplitude = None
        self.phase = None
        self.eps = None
        self.SensorDepth = None  # original time seriets
        self.signal = None  # detrended timeseries
        self.glbl_power = None
        self.glbl_signif = None
        self.units = None



        if path != None and file != None:
            self.filename = file
            # read Lake data
            [self.Time, self.SensorDepth] = fft_utils.readFile(path, file)
            self.eps = (self.Time[1] - self.Time[0]) / 100
            self.SensorDepth = np.array(self.SensorDepth)
            self.Time = np.array(self.Time)
            if self.Time[0] < 695056:
                self.Time += 695056

        if var != None and time != None :
            self.SensorDepth = np.array(var)
            self.Time = np.array(time)

        if tunits == 'day':
            self.tfactor = 86400
        elif tunits == 'hour':
            self.tfactor = 3600
        elif tunits == 'sec':
            self.tfactor = 1
        else:
            print "Wrong time units!"
            raise Exception('Error', 'Wrong time units!')
        # change to seconds after calculating tfactor
        self.tunits = 'sec'


    def doSpectralAnalysis(self, title, motherW = 'morlet', slevel = None, avg1 = None, avg2 = None, \
                           dj = None, s0 = None, J = None, alpha = None) :
        '''
        Calls the required function in sequence to perform a wavelet and Fourier Analysis

        @param title: Title of the analysis
        @param motherW: the mother wavelet name
        @param slevel: the significan level abaove which the value can't be considered random. default:0.95
        @param avg1: First value in a range of Y axis type  to plot in scalogram (ex periods)
        @param avg2: Last in a range of Y axis type  to plot in scalogram (ex periods)

        @param dt: float Sample spacing.
        @param dj: (float, optional) : Spacing between discrete scales. Default value is 0.25.
                   Smaller values will result in better scale resolution, but
                   slower calculation and plot.
        @param s0: (float, optional) : Smallest scale of the wavelet. Default value is 2*dt.
        @param J: (float, optional) : Number of scales less one. Scales range from s0 up to
                   s0 * 2**(J * dj), which gives a total of (J + 1) scales.
                   Default is J = (log2(N*dt/so))/dj.

        @return: None

        '''
        ':param: test'
        self.dj = dj
        self.s0 = s0
        self.J = J
        self.alpha = alpha
        self.title = title
        self.avg1 = avg1
        self.avg2 = avg2
        if motherW == 'dog':
            self.mother = wavelet.DOG()
        elif motherW == 'morlet':
            self.mother = wavelet.Morlet(6.)
        # [self.Time, SensorDepth1, X1, scales1, freq1, corr1 ]
        a = self._doSpectralAnalysisOnSeries()
        if slevel != None:
            self.get95Significance(slevel)
            self.getGlobalSpectrum(slevel)
        else:
            print "Call get95Significance(slevel) & getGlobalSpectrum(slevel) manually"
        if avg1 != None and avg2 != None:
            self.getScaleAverageSignificance(slevel, avg1, avg2)
        else:
            print "Call getScaleAverageSignificance() manually"

        return a

    def _doSpectralAnalysisOnSeries(self):
        ''' private workhorse
        '''

        if self.dj == None:
            self.dj = 0.25  # Four sub-octaves per octaves

        if self.s0 == None:
            self.s0 = -1  # 2 * dt                      # Starting scale, here 6 months

        if self.J == None:
            self.J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves

        if self.alpha == None:
            self.alpha = 0.0  # Lag-1 autocorrelation for white noise

        # 'dt' is a time step in the time series
        self.SensorDepth = mlab.detrend_linear(self.SensorDepth)

        self.dt = (self.Time[1] - self.Time[0]) * self.tfactor  # time is in days , convert to hours
        self.std = np.std(self.SensorDepth)  # Standard deviation
        self.variance = self.std ** 2  # Variance

        # normalize by standard deviation (not necessary, but makes it easier
        # to compare with plot on Interactive Wavelet page, at
        # "http://paos.colorado.edu/research/wavelets/plot/"
        self.signal = (self.SensorDepth - self.SensorDepth.mean()) / self.std  # Calculating anomaly and normalizing

        # The following routines perform the wavelet transform and siginificance
        # analysis for the chosen data set.
        wave, self.scales, self.freq, self.coi, self.fft, self.fftfreqs = \
            wavelet.cwt(self.signal, self.dt, self.dj, self.s0, self.J, self.mother)

        # this should reconstruct the initial signal
        self.iwave = wavelet.icwt(wave, self.scales, self.dt, self.dj, self.mother)
        self.N = self.SensorDepth.shape[0]

        # calculate power and amplitude spectrogram
        self.power = (np.abs(wave)) ** 2  # Normalized wavelet power spectrum
        self.fft_power = self.variance * np.abs(self.fft) ** 2  # FFT power spectrum
        self.amplitude = self.std * np.abs(wave) / 2.  # we use only half of the symmetrical
                                                                 # spectrum therefore divide by 2
        self.phase = np.angle(wave)

        self.period = 1. / self.freq

        return [wave, self.scales, self.freq, self.coi, self.fft, self.fftfreqs,
                self.iwave, self.power, self.fft_power, self.amplitude, self.phase]


    def get95Significance(self, slevel):
        signif, fft_theor = wavelet.significance(1.0, self.dt, self.scales, 0, self.alpha, \
                                                 significance_level = slevel, wavelet = self.mother)

        sig95 = (signif * np.ones((self.N, 1))).transpose()
        self.sig95 = self.power / sig95  # Where ratio > 1, power is significant
        return self.sig95

    def getGlobalSpectrum(self, slevel):
        self.glbl_power = self.variance * self.power.mean(axis = 1)
        dof = self.N - self.scales  # Correction for padding at edges
        self.glbl_signif, tmp = wavelet.significance(self.variance, self.dt, self.scales, 1, self.alpha, \
                                                significance_level = slevel, dof = dof, wavelet = self.mother)

        return self.glbl_signif

    def scinot(self, x, pos = None):
        '''
        Function to be used in the FuncFormatter to format scientific notation
        '''
        if x == 0:
            s = '0'
        else:
            xp = int(np.floor(np.log10(np.abs(x))))

            mn = x / 10.**xp
            # Here we truncate to 3 significant digits -- may not be enough
            # in all cases
            s = '$' + str('%.2f' % mn) + '\\times 10^{' + str(xp) + '}$'
            return s

    def getScaleAverageSignificance(self, slevel, avg1, avg2):
        # Scale average between avg1 and avg2 periods and significance level
        sel = pylab.find((self.period >= avg1) & (self.period < avg2))
        Cdelta = self.mother.cdelta

        # ones: Return a new array of given shape and type, filled with ones.
        scale_avg = (self.scales * np.ones((self.N, 1))).transpose()  # expand scale --> (J+1)x(N) array
        scale_avg = self.power / scale_avg  # [Eqn(24) Torrence & Compo (1998)

        # Cdelta = shape factor depeding on the wavelet used.
        #
        # To examine fluctuations in power over a range of scales/frequencies one can define a
        # scale averaged wavelet power" as a weighted sum of power spectrum over scales s1 to s2
        # here defined by the selected between avg1 and avg2
        # ]
        # By comparing [24] with [Eq 14] it can be shown that self.scale_avg is the average variance in a certain band
        # Here W[n]^2/s[j] = self.power / scale_avg , when n is the time index
        # sum(axis=0)  = sum over the scales
        #
        # Note: This can be used to examine the modulation of one time series by another or modulation of one frequency
        #       by another within the same timeseries (pag 73 Terrence & Compo (1998))
        self.scale_avg = self.variance * self.dj * self.dt / Cdelta * scale_avg[sel, :].sum(axis = 0)  # [Eqn(24)]

        # calculate the significant level for the averaged scales to represent the 95% (slevel) confidence interval
        self.scale_avg_signif, tmp = wavelet.significance(self.variance, self.dt, self.scales, 2, self.alpha,
                            significance_level = slevel, dof = [self.scales[sel[0]],
                            self.scales[sel[-1]]], wavelet = self.mother)
        return self.scale_avg_signif


    def plotAmplitudeSpectrogram(self, ylabel_ts, units_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2):
        '''
         The following routines plot the results in three different figures:
         - the global wavelet spectrum,
         - the wavelet amplitude spectrum,
         - the wavelet phase spectrum
         and Fourier spectra and finally the range averaged wavelet spectrum. In all
         sub-plots the significance levels are either includesuggested by Torrence and Compo (1998) using the wavelet module.
         To run this script successfully, the matplotlib module has to be installedd as dotted lines or as
         filled contour lines.

         @param ylablel_ts: label on the y axis on the data plot a) - string
         @param units_ts: units name for Y axis
         @param xlabel_sc: label to be placed on the X axis om the scalogram b) - string
         @param ylabel_sc: label to be placed on the Y axis om the scalogram b) - string
         @param sx_type: 'period' or 'freq' - creates the y axis on scalogram as scales/period or frequency
         @param x_type: 'date' will format the X axis as a date, 'time' will use regular numbers for time sequence
         @param val1: Range of sc_type (ex periods) to plot in scalogram
         @param val2: Range of sc_type (ex periods) to plot in scalogram

         @return: None
        '''
        fig1 = plt.figure()
        ax1 = fig1.add_axes([0.1, 0.1, 0.7, 0.60])
        ax3 = fig1.add_axes([0.83, 0.1, 0.03, 0.6])
        ax1.set_yscale('log')
        im1 = ax1.pcolormesh(self.Time, self.freq, self.amplitude)

        fig2 = plt.figure()
        ax2 = fig2.add_axes([0.1, 0.1, 0.7, 0.60])
        ax4 = fig2.add_axes([0.83, 0.1, 0.03, 0.6])
        ax2.set_yscale('log')
        im2 = ax2.pcolormesh(self.Time, self.freq, self.phase)

        # set correct way of axis, whitespace before and after with window
        # length
        ax1.axis('tight')
        # ax.set_xlim(0, end)
        ax1.grid(False)
        ax1.set_xlabel(xlabel_sc)
        ax1.set_ylabel(ylabel_sc)
        ax1.set_title('Amplitude ' + self.title)
        fig1.colorbar(im1, cax = ax3)

        if x_type == 'date':
            formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(formatter)
            ax1.xaxis.set_minor_locator(mondays)
            ax1.xaxis.grid(True, 'minor')
            fig1.autofmt_xdate()

        # set correct way of axis, whitespace before and after with window
        # length
        ax2.axis('tight')
        # ax.set_xlim(0, end)
        ax2.grid(False)
        ax2.set_xlabel(xlabel_sc)
        ax2.set_ylabel(ylabel_sc)
        ax2.set_title('Phase - ' + self.title)
        fig2.colorbar(im2, cax = ax4)

        fig3 = plt.figure()
        plt.title('Global Wavelet Spectrum Amplitude')

        A = np.sqrt(self.glbl_power) / 2.
        plt.plot(self.freq, A)
        plt.grid(True)
        plt.show()


    def plotSpectrogram(self, ylabel_ts, units_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2, raw = False):
        '''
         The following routines plot the results in four different subplots containing:
         - the original series,
         - the wavelet power spectrum,
         - the global wavelet and Fourier spectra
         - the range averaged wavelet spectrum.
         In all sub-plots the significance levels are either includesuggested by Torrence and Compo (1998)
         using the wavelet module.

          @param ylablel_ts: label on the y axis on the data plot a) - string
          @param units_ts: units name for Y axis
          @param xlabel_sc: label to be placed on the X axis om the scalogram b) - string
          @param ylabel_sc: label to be placed on the Y axis om the scalogram b) - string
          @param sx_type: 'period' or 'freq' - creates the y axis on scalogram as scales/period or frequency
          @param x_type: 'date' will format the X axis as a date, 'time' will use regular numbers for time sequence
          @param val1: Range of sc_type (ex periods) to plot in scalogram
          @param val2: Range of sc_type (ex periods) to plot in scalogram

          @return: None

        '''
        fontsize = 17
        pylab.close('all')
        # fontsize = 'medium'
        params = {'text.fontsize': fontsize,
                  'xtick.labelsize': fontsize,
                  'ytick.labelsize': fontsize,
                  'axes.titlesize': fontsize,
                  'axes.labelsize': fontsize,
                  'text.usetex': True
                 }
        pylab.rcParams.update(params)  # Plot parameters
        figprops = dict(figsize = (11, 8), dpi = 96)
        fig = plt.figure(**figprops)
        self.units = units_ts
        # First sub-plot, the original time series anomaly.



        if raw :
            ax = fig.add_axes([0.1, 0.75, 0.64, 0.2])
            # Plot the reconstructed signal. They are close to the original in case of simple signals.
            # The longer and more complex the signal is the more difficult is to cecomstruct.
            # The reconstructed signal is usually symmetrical
            ax.plot(self.Time, self.iwave, '-', linewidth = 1, color = [0.5, 0.5, 0.5])

            # Plot the original signal
            ax.plot(self.Time, self.SensorDepth, 'k', linewidth = 1.5)

            ax.set_title('a) %s' % (self.title,))
            if self.units != '':
              ax.set_ylabel(r'%s [$%s$]' % (ylabel_ts, self.units,))
            else:
              ax.set_ylabel(r'%s' % (ylabel_ts,))

        # Second sub-plot, the normalized wavelet power spectrum and significance level
        # contour lines and cone of influece hatched area.
        if raw:
            bx = fig.add_axes([0.1, 0.37, 0.64, 0.28], sharex = ax)
        else :
            bx = fig.add_axes([0.1, 0.55, 0.64, 0.38])

        if x_type == 'date':
            formatter = matplotlib.dates.DateFormatter('%Y-%m-%d')
            if raw:
                axis = ax
            else :
                axis = bx

            axis.xaxis.set_major_formatter(formatter)
            axis.xaxis.set_minor_locator(mondays)
            axis.xaxis.grid(True, 'minor')

            fig.autofmt_xdate()



        # levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
        if sc_type == 'freq':
            y_scales = self.freq
        else:
            y_scales = self.period
            # levels = np.arange(np.log2(self.period.min()), np.log2(self.period.max()), (np.log2(self.period.max()) - np.log2(self.period.min())) / 10)



        sel = pylab.find((y_scales >= val1) & (y_scales < val2))  # indices of selected data sales or freq
        y_scales = y_scales[sel[0]:sel[len(sel) - 1] + 1]
        power = self.power[sel[0]:sel[len(sel) - 1] + 1]
        sig95 = self.sig95 [sel[0]:sel[len(sel) - 1] + 1]
        glbl_signif = self.glbl_signif[sel[0]:sel[len(sel) - 1] + 1]
        glbl_power = self.glbl_power[sel[0]:sel[len(sel) - 1] + 1]

        levels = np.arange(power.min(), power.max() + power.min(), (power.max() - power.min()) / 32)

        # im = bx.contourf(self.Time, np.log2(y_scales), np.log2(power), np.log2(levels), cmap = cm.jet, extend = 'both')
        im = bx.contourf(self.Time, np.log2(y_scales), np.log2(power), 32, cmap = cm.jet, extend = 'both')

        # For out of levels representation enable the following two lines.
        # However, the above lines need to define the required levels
        # im.cmap.set_under('yellow')
        # im.cmap.set_over('cyan')

        bx.contour(self.Time, np.log2(y_scales), sig95, [-99, 1], colors = 'k', linewidths = 2.1)

        bx.fill(np.concatenate([self.Time[:1] - self.dt, self.Time, self.Time[-1:] + self.dt, \
                                self.Time[-1:] + self.dt, self.Time[:1] - self.dt, self.Time[:1] - self.dt]), \
                                np.log2(np.concatenate([[1e-9], self.coi, [1e-9], y_scales[-1:], y_scales[-1:], [1e-9]]))\
                                , 'k', alpha = 0.3, hatch = 'x')



        # for testing only
        # fig.colorbar(im) - if present it will shift the scales
        if raw:
            bx.set_title('b) Wavelet Power Spectrum (%s)' % (self.mother.name))
        else:
            bx.set_title('a) Wavelet Power Spectrum (%s) - %s' % (self.mother.name, self.title))
        bx.set_ylabel(ylabel_sc)
        Yticks = np.arange(y_scales.min(), y_scales.max(), (y_scales.max() - y_scales.min()) / 16)


        # formatter = FormatStrFormatter('%2.4f')
        # bx.yaxis.set_major_formatter(formatter)
        Yticks = 2 ** np.arange(np.ceil(np.log2(y_scales.min())),
                           np.ceil(np.log2(y_scales.max())))
        bx.set_yticks(np.log2(Yticks))
        bx.set_yticklabels(Yticks)
        # formatter = FuncFormatter(self.scinot)
        # bx.yaxis.set_major_formatter(formatter)
        bx.invert_yaxis()
        if x_type == 'date':
            bx.xaxis.set_major_formatter(formatter)
            bx.xaxis.set_minor_locator(mondays)
            bx.xaxis.grid(True, 'minor')
            fig.autofmt_xdate()

        # Third sub-plot, the global wavelet and Fourier power spectra and  kwavelet.tunitstheoretical
        # noise spectra.

        if raw:
            cx = fig.add_axes([0.78, 0.37, 0.19, 0.28], sharey = bx)
        else :
            cx = fig.add_axes([0.78, 0.55, 0.19, 0.38], sharey = bx)

        # plot the Fourier power spectrum first
        cx.plot(self.fft_power, np.log2(1. / self.fftfreqs), '-', color = [0.6, 0.6, 0.6], linewidth = 1.)

        # plot the wavelet global ower
        cx.plot(glbl_power, np.log2(y_scales), 'k-', linewidth = 1.5)

        # the line of chosen significance, ususaly 95%
        cx.plot(glbl_signif, np.log2(y_scales), 'k-.')

        if raw:
            cx.set_title('c) Global Wavelet Spectrum')
        else:
            cx.set_title('b) Global Wavelet Spectrum')
        if self.units != '':
          cx.set_xlabel(r'Power [$%s^2$]' % (self.units,))
        else:
          cx.set_xlabel(r'Power')

        cx.set_xlim([0, glbl_power.max() + self.variance])
        cx.set_ylim(np.log2([y_scales.min(), y_scales.max()]))
        cx.set_yticks(np.log2(Yticks))
        cx.set_yticklabels(Yticks)

        # cx.yaxis.set_major_formatter(formatter)

        pylab.setp(cx.get_yticklabels(), visible = False)
        cx.invert_yaxis()

        # Fourth sub-plot, the scale averaged wavelet spectrum as determined by the
        # avg1 and avg2 parameters

        if raw :
            dx = fig.add_axes([0.1, 0.07, 0.64, 0.2], sharex = ax)
        else :
            dx = fig.add_axes([0.1, 0.07, 0.64, 0.3], sharex = bx)
        dx.axhline(self.scale_avg_signif, color = 'k', linestyle = '--', linewidth = 1.)

        # plot the scale average for each time point.
        dx.plot(self.Time, self.scale_avg, 'k-', linewidth = 1.5)
        if raw:
            dx.set_title('d) Scale-averaged power  [$%.4f$-$%.4f$] (%s)' % (self.avg1, self.avg2, self.tunits))
        else:
            dx.set_title('c) Scale-averaged power  [$%.4f$-$%.4f$] (%s)' % (self.avg1, self.avg2, self.tunits))
        xlabel = 'Time (%s)' % self.tunits
        dx.set_xlabel(xlabel)
        if self.units != '':
          dx.set_ylabel(r'Average variance [$%s$]' % (self.units,))
        else:
          dx.set_ylabel(r'Average variance')
        #
        if x_type == 'date':
            dx.xaxis.set_major_formatter(formatter)
            dx.xaxis.set_minor_locator(mondays)
            dx.xaxis.grid(True, 'minor')
            fig.autofmt_xdate()

            axis.set_xlim([self.Time.min(), self.Time.max()])
        #

        # pylab.draw()
        pylab.show()


# test class
if __name__ == '__main__':
    '''
    Testing ground for local functions
    '''

    # 1) Test true amplitude
    Fs = 1000.0  # Sampling frequency
    T = 1.0 / Fs  # Sample time
    L = 1024  # Length of signal
    t = np.array(range(0, L)) * T  # Time vector
    # Sum of a 50 Hz sinusoid and a 120 Hz sinusoid
    x = np.array([])
    x1 = 0.7 * np.sin(2 * np.pi * 40 * t)
    def fun(t):
        x = np.zeros(len(t))
        for i in range(0, len(t)):
            if i < len(t) / 2:
                x[i] = 0.0 * np.sin(2 * np.pi * 120 * t[i])
            else:
                x[i] = 4.0 * np.sin(2 * np.pi * 120 * t[i])
        return x
    # x2 = 4.0 * np.sin(2 * np.pi * 120 * t)
    x2 = fun(t)
    x3 = 8.0 * np.sin(2 * np.pi * 200 * t)
    x4 = 6.0 * np.sin(2 * np.pi * 400 * t)
    title = 'Signal Corrupted with Zero-Mean Random Noise'
    xlabel = 'time (milliseconds)'
    x = (x1 + x2 + x3 + x4)

    avg1, avg2 = (0.001, 0.03)  # Range of periods to average
    slevel = 0.95  # Significance level
    tunits = 'sec'
    # tunits = '^{\circ}C'
    kwavelet = kCwt(None, None, tunits, time = t, var = x)

    dj = 0.025  # Four sub-octaves per octaves
    s0 = -1  # 2 * dt                      # Starting scale, here 6 months
    J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
    alpha = 0.0  # Lag-1 autocorrelation for white noise
    kwavelet.doSpectralAnalysis(title, "morlet", slevel, avg1, avg2, dj, s0, J, alpha)
    ylabel_ts = "amplitude"
    yunits_ts = 'mm'
    xlabel_sc = ""
    ylabel_sc = 'Period (%s)' % kwavelet.tunits
    # ylabel_sc = 'Freq (Hz)'
    sc_type = "period"
    # sc_type = "freq"
    val1, val2 = (0.001, 0.02)  # Range of sc_type (ex periods) to plot in spectogram
    x_type = 'time'
    kwavelet.plotSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)
    kwavelet.plotAmplitudeSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)




