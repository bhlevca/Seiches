'''
Created on Jun 12, 2012

@author: bogdan
'''
import numpy as np
import scipy as sp

import fft_utils
import FFTSpectralAnalysis
import matplotlib.mlab as mlab

class FFTGraphs(object):
    '''
    classdocs
    '''

    def __init__(self, path, file1, file2, show, tunits):
        '''
        Constructor
        '''
        self.path_in = path
        self.filename = file1
        self.filename1 = file2
        self.show = show
        self.fftsa = FFTSpectralAnalysis.FFTSpectralAnalysis(path, file1, file2, tunits)
        self.mx = None  # Amplitude values from spectral analysis Lake
        self.mx1 = None  # Amplitude values from spectral analysis Lake
        self.y = None  # Lake levels
        self.y1 = None  # Bay levels
        self.NumUniquePts = None  # Lake levels unique points
        self.NumUniquePts1 = None  # Bay levels unique points
        self.f = None  # Lake levels freq
        self.f1 = None  # Bay leevel frew
        self.Time = None  # Lake levels time
        self.Time1 = None  # Bay levels time
        self.fftx = None  # fourier transform lake levels
        self.fftx1 = None  # fourier transform bay levels
        self.power = None  # power spectrum lake levels
        self.power1 = None  # power spectrum bay levels
        self.x05 = None  # 5% conf level lake
        self.x95 = None  # 95% conf level lake
        self.x05_1 = None  # 5% conf level bay
        self.x95_1 = None  # 95% conf level bay
        self.num_segments = 1
    # end

    def doSpectralAnalysis(self, showOrig, tunits = 'sec', window = 'hanning', num_segments = 1, filter = None):

        self.num_segments = num_segments

        [self.y, self.Time, self.fftx, self.NumUniquePts, self.mx, self.f, self.power, self.x05, self.x95] = self.fftsa.FourierAnalysis(self.filename, showOrig, tunits, window, num_segments, filter)
        if self.filename1 != None:
            [self.y1, self.Time1, self.fftx1, self.NumUniquePts1, self.mx1, self.f1, self.power1, self.x05_1, self.x95_1] = self.fftsa.FourierAnalysis(self.filename1, showOrig, tunits, window, num_segments, filter)
            eps = (self.Time[1] - self.Time[0]) / 100

            # resample to be the same as first only if needed
            if (self.Time[1] - self.Time[0]) - (self.Time1[1] - self.Time1[0]) > eps:
                SensorDepth = sp.signal.resample(self.y1, len(self.Time))

                # redo the analysis withthe resampled data  #filter must be None here to prevent another filtering
                if num_segments == 1:
                    [self.y1, self.Time1, self.fftx1, self.NumUniquePts1, self.mx1, self.f1, self.power1, self.x05_1, self.x95_1] = \
                    self.fftsa.fourierTSAnalysis(self.Time, SensorDepth, self.show, self.fftsa.tunits, window, num_segments, None)
                else:
                    [f1, avg_fftx, avg_amplit, avg_power, x05, x95] = \
                      fftsa.WelchFourierAnalysis_overlap50pct(Time, SensorDepth, draw, self.tunits, window, num_segments, filter)
                    self.fftx1 = avg_fftx
                    self.mx1 = avg_amplit
                    self.f1 = f1
                    self.power1 = avg_power
                    self.x05_1 = x05
                    self.x95_1 = x95

            else:
                SensorDepth = self.y1
            # end if
        # end if
        return [self.Time, self.y, self.x05, self.x95]
    # end doSpectralAnalysis

    def plotLakeLevels(self, lake_name, bay_name, detrend = False):
        if self.show :
            # plot the original Lake oscillation input
            L = len(self.Time)
            xlabel = 'Time [days]'
            ylabel = 'Detrended Z(t) [m]'
            if self.filename1 != None:
                xa = np.array([self.Time, self.Time])
                if detrend:
                    self.y = fft_utils.detrend(self.y, 1)
                    self.y1 = fft_utils.detrend(self.y1, 1)

                    # These do not work properly for my purpose
                    # self.y, tmp = fft_utils.detrend_separate(self.y)
                    # self.y1, tmp = fft_utils.detrend_separate(self.y1)
                    # self.y = mlab.detrend_linear(self.y)
                    # self.y1 = mlab.detrend_linear(self.y1)

                ya = np.array([self.y, self.y1])

                legend = [lake_name, bay_name]


            else:
                xa = np.array([self.Time])
                ya = np.array([self.y])
                legend = ['lake']


            # end

            fft_utils.plot_n_TimeSeries("Detrended Lake Levels", xlabel, ylabel, xa, ya, legend)

        # end if
    # end plotLakeLevels



    def plotSingleSideAplitudeSpectrumFreq(self, lake_name, bay_name, funits = "Hz"):

        # smooth only if not segmented
        if self.num_segments == 1:
            sSeries = fft_utils.smoothSeries(self.mx, 5)
        else:
            sSeries = self.mx

        if self.filename1 != None and self.num_segments == 1:
            sSeries1 = fft_utils.smoothSeries(self.mx1, 5)
        else :
            sSeries1 = self.mx1
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Single-Sided Amplitude spectrum vs freq'
            if funits == 'Hz':
                xlabel = 'Frequency (Hz)'
                f = self.f
            elif funits == 'cph':
                xlabel = 'Frequency (cph)'
                f = self.f * 3600
            # end if

            ylabel = '|Z(t)| [m]'

            if self.filename1 != None:
                xa = np.array([f, f])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
                if self.num_segments != 1:
                    ci05 = [self.x05, self.x05_1]
                    ci95 = [self.x95, self.x95_1]
            else:
                xa = np.array([f])
                ya = np.array([sSeries])
                legend = [lake_name]
                if self.num_segments != 1:
                    ci05 = [self.x05]
                    ci95 = [self.x95]
            # end
            if self.num_segments == 1:
                fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend)
            else:
                fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel, xa, ya, ci05, ci95, legend)

    # end plotSingleSideAplitudeSpectrumFreq

    def plotSingleSideAplitudeSpectrumTime(self, lake_name, bay_name):
        sSeries = fft_utils.smoothSeries(self.mx, 5)
        if self.filename1 != None:
            sSeries1 = fft_utils.smoothSeries(self.mx1, 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Single-Sided Amplitude spectrum vs time [h]'
            xlabel = 'Time (h)'
            ylabel = '|Z(t)| [m]'
            if self.filename1 != None:
                tph = (1 / self.f) / 3600
                tph1 = (1 / self.f1) / 3600
                xa = np.array([tph, tph])
                ya = np.array([sSeries, sSeries1])
                legend = [lake_name, bay_name]
            else:
                tph = (1 / self.f) / 3600
                xa = np.array([tph])
                ya = np.array([sSeries])
                legend = ['lake']
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend, ymax_lim = 0.04)
    # end plotSingleSideAplitudeSpectrumTime

    def plotZoomedSingleSideAplitudeSpectrumFreq(self):
        sSeries = fft_utils.smoothSeries(self.mx[100:-1], 5)
        if self.filename1 != None:
            sSeries1 = fft_utils.smoothSeries(self.mx1[100:-1], 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Zoomed Single-Sided Amplitude spectrum vs freq'
            xlabel = 'Frequency (Hz)'
            ylabel = '|Z(t)| [m]'
            if self.filename1 != None:
                xa = np.array([self.f[100:-1], self.f[100:-1]])
                ya = np.array([sSeries, sSeries1])
                legend = ['lake', 'bay']
            else:
                xa = np.array([self.f[100:-1]])
                ya = np.array([sSeries])
                legend = ['lake']
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend)
    # end plotZoomedSingleSideAplitudeSpectrumFreq

    def plotZoomedSingleSideAplitudeSpectrumTime(self):
        zsSeries = fft_utils.smoothSeries(self.mx[100:-1], 5)
        if self.filename1 != None:
            zsSeries1 = fft_utils.smoothSeries(self.mx1[100:-1], 5);
        # end

        if self.show:
            # Plot single - sided amplitude spectrum.
            title = 'Zoomed Single-Sided Amplitude spectrum vs time [h]'
            xlabel = 'Time (h)'
            ylabel = '|Z(t)| [m]'
            if self.filename1 != None:
                tph = (1 / self.f[100:-1]) / 3600
                tph1 = (1 / self.f1[100:-1]) / 3600
                xa = np.array([tph, tph])
                ya = np.array([zsSeries, zsSeries1])
                legend = ['lake', 'bay']
            else:
                tph = (1 / self.f[100:-1]) / 3600
                xa = np.array([tph])
                ya = np.array([zsSeries])
                legend = ['lake']
            # end
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend)
    # end plotZoomedSingleSideAplitudeSpectrumTime

    def plotCospectralDensity(self):
        if self.show:
            # plot the power of the cospectral density F_in(w) * F_out(w)
            #
            zsSeries = fft_utils.smoothSeries(self.mx[100:-1], 5)
            if self.filename1 != None:
                zsSeries1 = fft_utils.smoothSeries(self.mx1[100:-1], 5);
                tph = (1 / self.f[100:-1]) / 3600
                convolution = zsSeries * zsSeries1
                xa = np.array([tph])
                ya = np.array([ convolution])
                tph = (1 / self.f[100:-1]) / 3600
                legend = ['Cospectral density']
                xlabel = 'T (h)'
                ylabel = 'Y^2(t) [m^2]'
                title = 'Zoomed in Power cospectral density'
                fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, legend)
            # end
        # end
    # end plotCospectralDensity

    def plotPhase(self):
        if self.show:
            # phase = np.unwrap(np.angle(self.fftx[0:self.NumUniquePts]))
            phase = np.unwrap(np.angle(self.fftx[0:len(self.fftx) / 2 + 1]))
            tph = (1.0 / self.f) / 3600
            xlabel = 'Time period (h)'
            ylabel = 'Phase (Degrees)'
            title = 'Phase delay'
            legend = [ 'phase']
            # avoind plotting the inf value of tph and start from index 1.
            fft_utils.plot_n_Array(title, xlabel, ylabel, [tph[1:]], [phase[1:] * 180 / np.pi], legend)
        # end
    # end plotPhase

