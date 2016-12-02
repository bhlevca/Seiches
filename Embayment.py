'''
Created on Jun 11, 2012

@author: bogdan
'''
import ufft.FFTGraphs as FFTGraphs
import ufft.fft_utils as fft_utils
#import ufft.Filter as Filter
import wavelets.kCwt
import scipy as sp
import numpy as np
import math
#import matplotlib.mlab as mlab
import EmbaymentPlot
import EmbaymentNonlinear
from optparse import OptionParser

path = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/LakeOntario-data'
path1 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/FMB'
path2 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/LOntario'
path3 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Toronto_Harbour'
path4 = '/home/bogdan/Documents/UofT/PhD/Data_Files/2010/Toberymory_tides'
path5 = '/home/bogdan/Documents/UofT/PhD/Data_Files/2013/Station-13320-Apr-09-2013/csv_processed'
path6 = '/home/bogdan/Documents/UofT/PhD/Data_Files/2013/Hobo-Apr-Nov-2013/WL/csv_processed/'

embayments = {
              'FMB' : {'A':850000., 'B':25. , 'H':1., 'L':130., 'LL':1600, 'h':2.5, 'BB':30,
                        'Period':[12.4, 5.2, 1.28, 0.8, 0.5, 0.36] ,  # h
                        'Amplitude':[0.034, 0.022, 0.017, 0.023, 0.021, 0.022],  # m
                        'Amplitude_bay':[0.024, 0.02, 0.014, 0.012, 0.0045, 0.002],  # m
                        'Phase':[5, 22, -15, 39, -4.6, -3.0],  # rad
                        'CD':0.0032,
                        'filename':path + '/Inner_Harbour_July_processed.csv'},
              #'Emb-Ah' : {'A':70000., 'B':120. , 'H':6, 'L':30., 'LL':365, 'h':2.5, 'BB':225, #L was 83
              #'Emb-Al' : {'A':70000., 'B':20. , 'H':1, 'L':190., 'LL':365, 'h':2.5, 'BB':225, #L was 83
              'Emb-A' : {'A':70000., 'B':75. , 'H':4, 'L':120., 'LL':365, 'h':2.5, 'BB':225, #L was 83
                        'Period':[12.4, 5.2, 1.28, 0.8, 0.5, 0.36, 0.2] ,  # h
                        'Amplitude':[0.034, 0.022, 0.017, 0.023, 0.021, 0.022, 0.005],  # m
                        'Amplitude_bay':[0.024, 0.02, 0.024, 0.012, 0.0045, 0.002, 0.021],  # m
                        'Phase':[5, 22, -15, 39, -4.6, -3.0, 39],  # rad
                        'CD':0.0032,
                        'filename':path6 + '/10279443_corr.csv'} ,#'/Inner_Harbour_July_processed.csv'},
              'Tob-IBP-ex' : {'A':150000., 'B':140. , 'H':2.143, 'L':570., 'LL':1000, 'h':1.5, 'BB':100,
                       'Period':[16.8 / 60, 15.8 / 60, 12.0 / 60, 8.0 / 60, 5.35 / 60, 4.5 / 60] ,  # h
                       'Amplitude':[0.02, 0.02, 0.018, 0.016, 0.015, 0.018],  # m
                       'Phase':[0, 0, 0, 0, 0, 0],  # rad
                       'CD':0.0032,
                       'filename':path4 + '/LL1.csv'},
              'Tob-IBP' : {'A':145000., 'B':140. , 'H':2.143, 'L':570., 'LL':1000, 'h':1.5, 'BB':100,
                       'Period':[16.8 / 60, 15.8 / 60, 12.0 / 60, 8.0 / 60] ,  # h
                       'Amplitude':[0.02, 0.02, 0.018, 0.016],  # m
                       'Amplitude_bay':[0.09, 0.11, 0.043, 0.058],  # m
                       'Phase':[0, 0, 0, 0],  # rad13320-07-APR-2013_slev.csv
                       'CD':0.0032,
                       'filename':path4 + '/LL1.csv'},
              'Tob-CIH' : {'A':64000., 'B':56. , 'H':1.9, 'L':175., 'LL':490, 'h':1.5, 'BB':50,
                       'Period':[16.8 / 60, 12.0 / 60, 9.2 / 60] ,  # h
                       'Amplitude':[0.02, 0.018, 0.014],  # m
                       'Amplitude_bay':[0.025, 0.078, 0.037],  # m
                       'Phase':[0, 0, 0],  # rad
                       'CD':0.0032,
                       'filename':path4 + '/LL4.csv'},
              'L-SUP' : {'A':180000., 'B':30. , 'H':1., 'L':2000., 'LL':1400, 'h':1.5, 'BB':100,
                       'Period':[2.9, 2.9] ,  # h
                       'Amplitude':[0.1, 0.1],  # m
                       'Phase':[0, 0],  # rad
                       'CD':0.0032,
                       'filename':None},
              'Tor_Harb' : {'A':None, 'B':None , 'H':None, 'L':None, 'LL':None, 'h':None, 'BB':None,
                       'Period':[16.8 / 60, 12.0 / 60, 9.2 / 60] ,  # h
                       'Amplitude':[0.02, 0.018, 0.014],  # m
                       'Amplitude_bay':[0.025, 0.078, 0.037],  # m
                       'Phase':[0, 0, 0],  # rad
                       'CD':0.0032,
                       'filename':path5 + '/10279444_corr.csv'}, #'/13320-07-APR-2013_slev.csv'},

              }


class Embayment(object):
    '''
    classdocs
    '''
    printtitle = False


    def __init__(self, name):
        '''
        Constructor
        '''
        # name can be: 'FMB', 'Emb_A', 'Tob-CIH', 'Tob-IBP'
        self.name = name
        dict = embayments[name]
        self.A = dict['A']
        self.B = dict['B']
        self.H = dict['H']
        self.L = dict['L']
        self.LL = dict['LL']
        self.h = dict['h']
        self.Period = dict['Period']
        self.Amplitude = dict['Amplitude']
        self.Phase = dict['Phase']
        self.Cd = dict['CD']
        self.filename = dict['filename']



    @staticmethod
    def set_PrintTitle(flag):
        Embayment.printtitle = flag


    @staticmethod
    def plotMultipleTimeseries(path_in, filenames, names, detrend = False, filtered = False, lowcut = None, highcut = None, \
                                tunits = 'sec', printtitle = False, minmax = None, grid = False, show = False, doy = True):
        # plot the original Lake oscillation input
        ts = []
        i = 0
        time = []
        for filename in filenames:
            [Time, SensorDepth] = fft_utils.readFile(path_in, filename)
            # import matplotlib.pyplot as plt
            # plt.plot(Time, SensorDepth)
            # plt.show()

            # must detrend from subtracting the median so all have a zero median
            if filtered:
                # filtered timeseries
                N = 5
                if tunits == 'day':
                    factor = 86400
                elif tunits == 'hour':
                    factor = 3600
                else:
                    factor = 1
                dt_s = (Time[2] - Time[1]) * factor  # Sampling period [s]
                samp_rate = 1.0 / dt_s
                btype = 'band'
                # y = fft_utils.filters.fft_bandpassfilter(SensorDepth, samp_rate, lowcut, highcut)
                y, w, h, N, delay = fft_utils.filters.butterworth(SensorDepth, btype, lowcut, highcut, samp_rate, order = 5)
                SensorDepth = y
            # end if filtered

            if detrend:
                ts.append(sp.signal.detrend(SensorDepth))
            else:
                ts.append(SensorDepth)
            time.append(Time)
            i += 1
        series = np.array(ts)


        L = len(Time)
        if doy:
            xlabel = 'Day of year'
        else:
            xlabel = 'Time (days)'
        ylabel = 'Detrended Z (m)'
        xa = np.array(time)

        # This is a moving average detrend
        if detrend:
            series2 = []
            # time2 = []
            for i in range(0, len(series)):
                x = fft_utils.detrend(series[i])
                # x = fft_utils.smoothSeriesWindow(series[i], 100)
                series2.append(x)
            series = np.array(series2)
            # xa = np.array(time2)


        legend = names
        # end

        fft_utils.plot_n_TimeSeries("Detrended Lake and Bay Levels", xlabel, ylabel, xa, \
                                    series, legend, plottitle = Embayment.printtitle, fontsize = 20, doy = doy, minmax = minmax, grid = grid, show = show)


    # end plotLakeLevels

    @staticmethod
    def plotSingleSideAplitudeSpectrumFreqAnalytic(graphobj, num_segments, lake_name, bay_name, funits = "Hz", y_label = None, title = None,
                                            log = False, fontsize = 20, tunits = None, plottitle = False, grid = False, ymax = None, \
                                            LL = None, B = None, h = None, a0 = None, bay = None):

        f = graphobj.plotSingleSideAplitudeSpectrumFreq

        if num_segments == 1:
            [title, xlabel, ylabel, xa, ya, legend, log, plottitle, ymax_lim] = \
             f(lake_name, bay_name, funits, y_label, title, log, fontsize, tunits, plottitle, grid, ymax, graph = False)
        else:
            [title, xlabel, ylabel, xa, ya, ci05, ci95, legend, log, fontsize, plottitle, ymax] = \
             f(lake_name, bay_name, funits, y_label, title, log, fontsize, tunits, plottitle, grid, ymax, graph = False)

        # add the theory curves
        if LL != None:
            embg = EmbaymentNonlinear.BayGeometry(LL, B, h)
            embNon = EmbaymentNonlinear.EmbaymentNonlinear(embg)
            # convert cph to rad/sec
            om = 2 * np.pi * xa[0] / 3600
            amp = embNon.calculateResponseVsAngularFreqSlow(a0, om, False)
            # amp = embNon.calculateResponseVsFrequency(a0, om, False)
            xa = np.append(xa, [xa[0]], axis = 0)
            ya = np.append(ya, [amp], axis = 0)
            ld = legend.append('Nonlinear analytical solution')

        if bay != None and LL != None:
            bay = Embayment(bay)
            embPlot = EmbaymentPlot.EmbaymentPlot(bay)
            amp_helm = embPlot.calculateResponseVsAngularFreqSlow(0.015, om, False)
            xa = np.append(xa, [xa[0]], axis = 0)
            ya = np.append(ya, [amp_helm], axis = 0)
            ld = legend.append('Helmoltz resonator solution')

        if num_segments == 1:
            fft_utils.plot_n_Array(title, xlabel, ylabel, xa, ya, ld, legend, plottitle, ymax_lim = ymax)
        else:
            fft_utils.plot_n_Array_with_CI(title, xlabel, ylabel, xa, ya, ci05, ci95, legend = legend, \
                                                log = log, fontsize = fontsize, plottitle = plottitle, grid = grid, ymax_lim = ymax)

    # end plotSingleSideAplitudeSpectrumFreq

    @staticmethod
    def SpectralAnalysis(bay, filenames, names, b_wavelets = False, window = "hanning", num_segments = None, tunits = 'day', \
                         funits = "Hz", filter = None, log = False, doy = False, grid = False, fname = None, domodel = False):

        # show extended calculation of spectrum analysis
        show = True

        bay_names = []
        lake_name = ""
        bay_name = ""
        tunits = "day"
        if bay == 'FMB':
            # Frenchman's bay data
            fftsa = FFTGraphs.FFTGraphs(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv', show, tunits)
            lake_name = "Lake Ontario"
            bay_name = "Frenchman's bay"
        elif bay == 'BUR':
            # Burlington data    43.333333N , 79.766667W (placed in the lake not the sheltered bay)
            fftsa = FFTGraphs.FFTGraphs(path, 'LO_Burlington-JAN-DEC-2011_date.csv', None, show, tunits)
            # fftsa = FFTGraphs.FFTGraphs(path, 'LO_Burlington-Apr26-Apr28-2011.csv', None, show, tunits)
            lake_name = "Lake  Ontario"
            bay_name = ""
        # Fathom Five National Park Outer Boat Passage
        elif bay == 'Tob-OBP':
            # NOTE:lake amplit is lower so switch places
            # fftsa = FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL2.csv', show, tunits)
            fftsa = FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL2-28jul2010.csv', show , tunits)
            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Outer Boat Passage"
        # Fathom Five National Park Inner Boat Passage
        elif bay == 'Tob-IBP':
            # NOTE:lake amplit is lower so switch places
            # fftsa = FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL1.csv', show, tunits)
            fftsa = FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL1-28jul2010.csv', show, tunits)

            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Inner Boat Passage"
        # Fathom Five National Park Cove Island Harbour
        elif bay == 'Tob-CIH':
            # NOTE:lake amplit is13320-07-APR-2013_slev.csv lower so switch places
            # fftsa = FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL4.csv', show, tunits)
            fftsa = FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL4-28jul2010.csv', show, tunits)
            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Cove Island Harbour"  #
        elif bay == 'Tob-HI':
            # NOTE:lake amplit is lower so switch places
            # fftsa = FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL3.csv', show, tunits)
            fftsa = FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL3-28jul2010.csv', show, tunits)
            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Harbour Island - lake"  #
        # Embayment A Tommy Thomson Park
        elif bay == 'Emb-A' or bay == 'Emb-B' or bay == 'Emb-C' or bay == 'Cell-1' or bay == 'Cell-2' or bay == 'Cell-3':
            # fftsa = FFTGraphs.FFTGraphs(path6 + bay, 'Stn_18_10279444.csv', 'Emb_A_10279443.csv', show, tunits)
            fftsa = FFTGraphs.FFTGraphs(path6 + bay, filenames[1], filenames[0], show, tunits)
            lake_name = names[1]
            bay_name = names[0]
        elif bay == 'Tob_All':
            fftsa1 = FFTGraphs.FFTGraphs(path4, 'LL4-28jul2010.csv', None, show, tunits)
            bay_names.append("Cove Island Harbour")
            fftsa2 = FFTGraphs.FFTGraphs(path4, 'LL1-28jul2010.csv', None, show, tunits)
            bay_names.append("Inner Boat Passage")
            fftsa3 = FFTGraphs.FFTGraphs(path4, 'LL2-28jul2010.csv', None, show , tunits)
            bay_names.append("Outer Boat Passage")
            fftsa4 = FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', None, show , tunits)
            bay_names.append("Harbour Island - lake")
        elif bay == 'Tor_Harb':
            fftsa = FFTGraphs.FFTGraphs(path5, fname, fname, show, tunits)
            lake_name = "Tor_Harb"
            bay_name = ""
        else:
            print "Unknown embayment"
            exit(1)
        # end

        if bay == 'Tob_All':
            showLevels = False
            detrend = False
            draw = False
            fftsa1.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)
            fftsa2.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)
            fftsa3.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)
            fftsa4.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)

            data = [fftsa1.mx, fftsa2.mx, fftsa3.mx, fftsa4.mx]
            ci05 = [fftsa1.x05, fftsa2.x05, fftsa3.x05, fftsa4.x05]
            ci95 = [fftsa1.x95, fftsa2.x95, fftsa3.x95, fftsa4.x95]
            freq = [fftsa1.f, fftsa2.f, fftsa3.f, fftsa4.f]
            FFTGraphs.plotSingleSideAplitudeSpectrumFreqMultiple(lake_name, bay_names, data, freq, [ci05, ci95], \
                                                                 num_segments, funits, y_label = None, title = None, \
                                                                 log = log, fontsize = 20, tunits = tunits, plottitle = Embayment.printtitle, grid = grid)

        else:
            showLevels = False
            detrend = False
            draw = False
            [Time, y, x05, x95, fftx, freq, mx] = fftsa.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)
            phase = np.zeros(len(fftx), dtype = np.float)
            deg = True
            phase = np.angle(fftx, deg)
            print "*****************************"
            print " PHASEs"
            for i in range(0, len(fftx)):
                print "Period %f  phase:%f  amplit:%f" % (1. / freq[i] / 3600, phase[i], mx[i])
            print "*****************************"

            fftsa.plotLakeLevels(lake_name, bay_name, detrend, y_label=None, title=None, plottitle=Embayment.printtitle, doy = doy, grid = grid)
                               
            if bay == 'Tob-OBP' :  # to have the same scale as IBP
                ymax = 0.14
            else:
                ymax = None




            if domodel:
                dict = embayments[bay]
                B = dict['BB']
                LL = dict['LL']
                h = dict['h']
                a0 = 0.14
                Embayment.plotSingleSideAplitudeSpectrumFreqAnalytic(fftsa, num_segments, lake_name, bay_name, funits, y_label = None, title = None, log = log, \
                                                         fontsize = 20, tunits = tunits, plottitle = Embayment.printtitle, grid = grid, ymax = ymax, \
                                                         LL = LL, B = B, h = h, a0 = a0, bay = bay)

            else:
                fftsa.plotSingleSideAplitudeSpectrumFreq(lake_name, bay_name, funits, y_label = None, title = None, log = log, \
                                                         fontsize = 20, tunits = tunits, plottitle = Embayment.printtitle, grid = grid, ymax = ymax)
            grid = False

            fftsa.plotPowerDensitySpectrumFreq(lake_name, bay_name, funits, plottitle = Embayment.printtitle, grid = grid)
            fftsa.plotSingleSideAplitudeSpectrumTime(lake_name, bay_name, plottitle = Embayment.printtitle, grid = grid)

            # fftsa.plotZoomedSingleSideAplitudeSpectrumFreq()
            # fftsa.plotZoomedSingleSideAplitudeSpectrumTime()
            fftsa.plotCospectralDensity(log = log)
            # fftsa.plotPhase()

#===============================================================================
#             if b_wavelets:
#                 # Wavelet Spectral analysis
#                 if bay == 'FMB':
#                     graph = wavelets.Graphs.Graphs(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv', show)
#                 elif bay == 'BUR':
#                     graph = wavelets.Graphs.Graphs(path, 'LO_Burlington-JAN-DEC-2011_date.csv', None, show)
#                     # graph = wavelets.Graphs.Graphs(path, 'LO_Burlington-Apr26-Apr28-2011.csv', None, show)
#                 elif bay == 'Tob-OBP':
#                     graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL2.csv', show)
#                     # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL2-28jul2010.csv', show)
#                 elif bay == 'Tob-IBP':
#                     graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL1.csv', show)
#                     # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL1-28jul2010.csv', show)
#                 # Fathom Five National Park Cove Island Harbour
#                 elif bay == 'Tob-CIH':
#                     graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL4.csv', show)
#                     # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL4-28jul2010.csv', show)
#                 elif bay == 'Tob-HI':
#                     graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL3.csv', show)
#                     # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL3-28jul2010.csv', show)
#                 # Embayment A Tommy Thomson Park
#                 elif bay == 'Emb-A' or bay == 'Emb-B' or bay == 'Emb-C' or bay == 'Cell-1' or bay == 'Cell-2' or bay == 'Cell-3':
#                     # graph = wavelets.Graphs.Graphs(path3, 'Stn_18_10279444.csv', 'Emb_A_10279443.csv', show)
#                     graph = wavelets.Graphs.Graphs(path3, filenames[1], filenames[0], show)
#                 else:
#                     print "Unknown embayment"
#                     exit(1)
# 
#                 graph.doSpectralAnalysis()
#                 graph.plotDateScalogram(scaleType = 'log', plotFreq = True, printtitle = Embayment.printtitle)
#                 graph.plotSingleSideAplitudeSpectrumTime(printtitle = Embayment.printtitle)
#                 graph.plotSingleSideAplitudeSpectrumFreq(printtitle = Embayment.printtitle)
#                 graph.showGraph()
#             # nd if b_wavelets
#===============================================================================
    # end SpectralAnalysis

    @staticmethod
    def HarmonicAnalysis(filename, path, freq_hours):

        [Time, SensorDepth] = fft_utils.readFile(path, filename)
        y = sp.signal.detrend(SensorDepth)
        miu = np.mean(y)
        T = freq_hours * 3600
        om = 2 * np.pi / T
        tunits = "day"
        if tunits == 'day':
            factor = 86400
        elif tunits == 'hour':
            factor = 3600
        else:
            factor = 1

        dt_s = (Time[2] - Time[1]) * factor  # Sampling period [s]
        A = 0.0
        B = 0.0
        for i in range(0, len(y)) :
            t = i * dt_s
            A += (y[i] - miu) * math.cos(om * t)
            B += (y[i] - miu) * math.sin(om * t)
        A = 2.0 / len(y) * A
        B = 2.0 / len(y) * B
        R = np.sqrt(A ** 2 + B ** 2)
        PHI = np.arctan(-B / A)
        PHI2 = np.arctan2(B, A)
        print ("period (h):%f  amplitude (m): %f - phase (deg): %f  (rad):%f  (rad2):%f") % (freq_hours, R, PHI * 180 / np.pi, PHI, PHI2)

    # end HarmonicAnalysis

    @staticmethod
    def waveletAnalysis(bay, title, tunits, slevel, avg1, avg2, val1, val2, \
                        dj = None, s0 = None, J = None, alpha = None, debug = False):

        ppath = path
        if bay == 'FMB':
            # Frenchman's bay data
            ppath = path
            file = 'Lake_Ontario_1115682_processed.csv'
        elif bay == 'BUR':
            # Burlington data    43.333333N , 79.766667W (placed in the lake not ht sheltered bay)
            ppath = path
            file = 'LO_Burlington-Apr26-Apr28-2011.csv'
        # Fathom Five National Park Outer Boat Passage
        elif bay == 'Tob-OBP':
            ppath = path4
            file = 'LL2.csv'
            # file = 'LL1-28jul2010.csv'
        elif bay == 'Tob-IBP':
            ppath = path4
            file = 'LL1.csv'
        # Fathom Five National Park Cove Island Harbour
        elif bay == 'Tob-CIH':
            ppath = path4
            file = 'LL4.csv'
            # file = 'LL4-28jul2010.csv'
        # Embayment A Tommy Thomson Park
        elif bay == 'Emb-A' or bay == 'Emb-B' or bay == 'Emb-C' or bay == 'Cell-1' or bay == 'Cell-2' or bay == 'Cell-3':
            # ppath = path3
            # file = '1115865-Station16-Gate-date.csv'
            ppath = path6 + bay
            file = filenames[1]
        else:
            print "Unknown embayment"
            exit(1)
        kwavelet = wavelets.kCwt.kCwt(ppath, file, tunits)
        dj = 0.05  # Four sub-octaves per octaves
        s0 = -1  # 2 * dt                      # Starting scale, here 6 months
        J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
        alpha = 0.5  # Lag-1 autocorrelation for white noise

        # [wave, scales, freq, coi, fft, fftfreqs, iwave, power, fft_power, amplitude, phase] = \
        kwavelet.doSpectralAnalysis(title, "morlet", slevel, avg1, avg2, dj, s0, J, alpha)
        if debug:
            print "fftfreq=", fftfreqs
            print "amplit=", amplitude
            print "phase=", phase


        ylabel_ts = "amplitude"
        yunits_ts = 'm'
        xlabel_sc = ""
        ylabel_sc = 'Period (%s)' % kwavelet.wpar1.tunits
        # ylabel_sc = 'Freq (Hz)'
        sc_type = "period"
        # sc_type = "freq"
        # x_type = 'date'
        x_type = 'dayofyear'
        kwavelet.plotSpectrogram(kwavelet.wpar1, ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)
        ylabel_sc = 'Frequency ($s^{-1}$)'
        # kwavelet.plotAmplitudeSpectrogram(kwavelet.wpar1, ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)

    def EmbaymentFlow(self, A, HbVect, dt):
        '''
        calculates flow in the channel based on water level fluctuations in the
        embayment
        '''
        Q = np.zeros(len(HbVect))
        for i in range(1, len(HbVect)):
            Q[i] = A * (HbVect[i - 1] - HbVect[i]) / dt
        # end
        return Q
    # end EmbaymentFlow

    def CalculateFlow(self, days):
        '''
         Water exchange
         Calculate the Flow in the Embayment
        '''
        embPlot = EmbaymentPlot.EmbaymentPlot(self)
        [t, X, c, k, w, x0, v0, R] = embPlot.Response(days)
        #
        # embPlot.plotForcingResponse(t, printtitle = Embayment.printtitle)    # too simple, unattractive
        embPlot.plotRespVsOmegaVarAmplit(printtitle = Embayment.printtitle)  # uses the spring equation, not necessary for the paper
        # embPlot.plotPhaseVsOmega(printtitle = Embayment.printtitle)          #not necessary for the paper
        #
        embPlot.plotRespVsOmegaVarFric(printtitle = Embayment.printtitle)

        embPlot.plotRespVsOmegaVarArea(printtitle = Embayment.printtitle)
        embPlot.plotRespVsOmegaVarMouth(printtitle = Embayment.printtitle)

        embPlot.plotRespVsOmegaVarMouthCurves(printtitle = Embayment.printtitle)  # trebitz
        embPlot.plotDimensionlessResponse(printtitle = Embayment.printtitle)
        embPlot.show()


        # calculate flushing time
        [Time, SensorDepth] = fft_utils.readFile("", self.filename)

        # Limit the time interval to the same number of days: days assuming that measuread days are more
        meas_days = int (Time[len(Time) - 1] - Time[1])
        interv = len(Time) * days / meas_days
        Qm = self.EmbaymentFlow(self.A, SensorDepth[:interv], (Time[2] - Time[1]) * 86400)

        Vm = 0
        summeas = 0
        for i in range(0, len(Qm) - 1):
            summeas = summeas + 0.5 * np.abs(SensorDepth[i] - SensorDepth[i - 1])
            if ((Qm[i] - Qm[i - 1]) > 0) and (Qm[i] > 0) :
                Vm = Vm + (Qm[i] + Qm[i - 1]) / 2 * self.B * self.H
            # end
        # end
        QWL=0
        dt=(Time[2] - Time[1])* 86400
        for i in range(0, len(Qm) - 1):
            QWL = QWL + self.A * 0.5*(SensorDepth[i] - SensorDepth[i - 1]) / dt
        QWL=abs(QWL)
        
        print "V meas=%f Sum meas=%f QWL=%f" % (Vm, summeas, QWL)
        Qp = self.EmbaymentFlow(self.A, R, (t[2] - t[1]))

        Vp = 0
        sumpred = 0
        for i in range(0, len(Qp) - 1):
            sumpred = sumpred + 0.5 * np.abs(R[i] - R[i - 1])
            if ((Qp[i] - Qp[i - 1]) > 0) and (Qp[i] > 0) :
                Vp = Vp + (Qp[i] + Qp[i - 1]) / 2 * self.B * self.H
            # end
        # end
        QWL=0
        for i in range(0, len(Qp) - 1):
            QWL = QWL + self.A * 0.5*(R[i] - R[i - 1]) / dt
        QWL=abs(QWL)

        print "V pred=%f, Sum pred=%f QWL=%f" % (Vp, sumpred, QWL)

        if self.name == 'Tob-IBP':
            Vm_max = np.max(Qm) / (self.B * self.H / 2)
            Vp_max = np.max(Qp) / (self.B * self.H / 2)
        else:
            Vm_max = np.max(Qm) / (self.B * self.H)
            Vp_max = np.max(Qp) / (self.B * self.H)
        # endif
        print "Bay=%s  Vm=%f m/s, Vp=%f m/s" % (self.name , Vm_max, Vp_max)
        print "Bay=%s  Qm=%f m^3/s, Qp=%f m^3/s" % (self.name, np.max(Qm), np.max(Qp))
    # end CalculateFlow

    @staticmethod
    def CalculateSpectral(bay, domodel = False, numseg = 1):
        showLevels = False
        detrend = False
        detrend = True
        num_segments = int(numseg)
        doSpectral = True
        dowavelets = False  # Scipy
        doWavelet =  False  #True  # Terrence & Compo
        doHarmonic = False
        doFiltering = False
        tunits = 'day'  # can be 'sec', 'hour'
        funits = "cph"
        window = 'hanning'
        log = False
        doy = True  # display time in day of the year instead of a timestamp
        grid = False

        if bay == 'Tor_Harb':
            filenames = ['13320-01-MAY-2013_slev.csv']
            filenames = ['13320-07-APR-2013_slev.csv']
            names = [ "Toronto Harbour"]
             # set to True for Butterworth filtering - just for testing.
            lowcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
            highcutoff = 0.00834  # Hz => 30 cph, or T=2 min
            minmax = None  # [-0.4, 0.4]
            Embayment.plotMultipleTimeseries(path5, filenames, names, detrend, doFiltering , lowcutoff, highcutoff, tunits,
                                             minmax = minmax, show = True, grid = False, doy = False)

            btype = 'band'
            if btype == 'low':  # pass freq < lowcutoff
                highcutoff = None
                lowcutoff = 0.00834 * 2  # Hz => 30 cph, or T=2 min
            elif btype == 'high':  # pass freq > highcutoff
                highcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                lowcutoff = None
            elif btype == 'band':  # pass highcutoff > freq > lowcutoff
                lowcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                highcutoff = 0.00834  # Hz => 30 cph, or T=2 min


            ftype = 'fft'
            # ftype = 'butter' THIS DOES NOT WORK PROPERLY for the random signal we have here
            if doFiltering:
                filter = [lowcutoff, highcutoff]  # Filter.Filter(doFiltering, lowcutoff, highcutoff, btype)
            else:
                filter = None

            if doSpectral:
                Embayment.SpectralAnalysis(bay, filenames, names, dowavelets, window, num_segments, \
                                           tunits = tunits, funits = funits, filter = filter, log = log, doy = doy, grid = grid, fname = None, domodel = domodel)



        elif bay == 'Tob-OBP' or bay == 'Tob-IBP' or  bay == 'Tob-CIH' or  bay == 'Tob_All' or bay == 'Tob-HI':
            filenames = ['LL1.csv', 'LL4.csv', 'LL2.csv', 'LL3.csv']
            # filenames = ['LL3.csv', '11690-01-JUL-2010_out.csv']
            # filenames = ['LL1-28jul2010.csv', 'LL4-28jul2010.csv', 'LL2-28jul2010.csv', 'LL3-28jul2010.csv']
            # filenames = ['Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv']

            names = [ "Inner Boat Passage" , "Cove Island Harbour", "Outer Boat Passage", "Lake Huron"]
            # names = ["Harbour Island - lake", "Station 11960"]
            # names = [ "Lake Ontario", "Frenchman's Bay"]

            # set to True for Butterworth filtering - just for testing.
            lowcutoff = 1.157407e-5
            highcutoff = 0.00834
            minmax = [-0.4, 0.4]
            Embayment.plotMultipleTimeseries(path4, filenames, names, detrend, doFiltering , lowcutoff, highcutoff, tunits, minmax = minmax)

            btype = 'band'
            if btype == 'low':  # pass freq < lowcutoff
                highcutoff = None
                lowcutoff = 0.00834 * 2  # Hz => 30 cph, or T=2 min
            elif btype == 'high':  # pass freq > highcutoff
                highcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                lowcutoff = None
            elif btype == 'band':  # pass highcutoff > freq > lowcutoff
                lowcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                highcutoff = 0.00834  # Hz => 30 cph, or T=2 min

            ftype = 'fft'
            # ftype = 'butter' THIS DOES NOT WORK PROPERLY for the random signal we have here
            if doFiltering:
                filter = [lowcutoff, highcutoff]  # Filter.Filter(doFiltering, lowcutoff, highcutoff, btype)
            else:
                filter = None

            if doSpectral:
                Embayment.SpectralAnalysis(bay, filenames, names, dowavelets, window, num_segments, \
                                           tunits = tunits, funits = funits, filter = filter, log = log, doy = doy, grid = grid, fname = None, domodel = domodel)

            tunits = 'day'
            slevel = 0.95
            # range 0-65000 is good to catch the high frequencies
            #       0-600000 if you need to catch internal waves with much lower frequencies and large periods
            val1, val2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram
            avg1, avg2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram

            title = bay + ""
            if doWavelet :
                debug = False
                Embayment.waveletAnalysis(bay, title, tunits, slevel, avg1, avg2, val1, val2, debug = debug)

            if doHarmonic:
                if bay == "":
                    freq_hours = 4.5
                    Embayment.HarmonicAnalysis('Lake_Ontario_1115682_processed.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-Mar23-Apr23-2011.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-JAN-DEC-2011_date.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-Apr26-Apr28-2011.csv', path, freq_hours)
                if bay == "Tob-IBP":
                    freq_hours = 0.266688
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
                    freq_hours = 0.120010
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
                    freq_hours = 0.07742
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
        # end if 'Tor_Harb'
        elif bay == 'Emb-A' or bay == 'Emb-B' or bay == 'Emb-C' or bay == 'Cell-1' or \
            bay == 'Cell-2' or bay == 'Cell-3' or bay == 'FMB':

            ppath = path6 + bay

            if bay == 'Emb-A':
                filenames = ['Emb_A_10279443.csv', 'Stn_18_10279444.csv']
                names = [ "Emb-A" , "Lake Ontario"]
            elif bay == 'Emb-B':
                filenames = ['Emb_B_1115681.csv', 'Stn_18_10279444.csv']
                names = [ "Emb-B" , "Lake Ontario"]
            elif bay == 'Emb-C':
                filenames = ['EmbC_10238147.csv', 'Stn_18_10279444.csv']
                names = [ "Emb-C" , "Lake Ontario"]
            elif bay == 'Cell-1':
                filenames = ['Cell1_10279696.csv', 'Cell2_10279693.csv']
                names = [ "Cell-1" , "Cell-2"]
            elif bay == 'Cell-2':
                filenames = ['Cell2_10279693.csv', 'Cell3_10279699.csv']
                names = [ "Cell-2" , "Cell-3"]
            elif bay == 'Cell-3':
                filenames = ['Cell3_10279699.csv', 'EmbC_10238147.csv']
                names = [ "Cell-3" , "Emb-C"]
            elif bay == 'FMB':
                filenames = ['Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv']
                names = [ "Lake Ontario" , "FMB"]
                ppath = path

            # set to True for Butterworth filtering - just for testing.
            lowcutoff = 1.157407e-5
            highcutoff = 0.00834
            minmax = [-0.4, 0.4]
            show = True
            Embayment.plotMultipleTimeseries(ppath, filenames, names, detrend, doFiltering , lowcutoff, highcutoff, tunits, minmax = minmax, show = show)

            btype = 'band'
            if btype == 'low':  # pass freq < lowcutoff
                highcutoff = None
                lowcutoff = 0.00834 * 2  # Hz => 30 cph, or T=2 min
            elif btype == 'high':  # pass freq > highcutoff
                highcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                lowcutoff = None
            elif btype == 'band':  # pass highcutoff > freq > lowcutoff
                lowcutoff = 1.157407e-5  # Hz => 0.0417 cph, or T=24 h
                highcutoff = 0.00834  # Hz => 30 cph, or T=2 min


            ftype = 'fft'
            # ftype = 'butter' THIS DOES NOT WORK PROPERLY for the random signal we have here
            if doFiltering:
                filter = [lowcutoff, highcutoff]  # Filter.Filter(doFiltering, lowcutoff, highcutoff, btype)
            else:
                filter = None

            if doSpectral:
                Embayment.SpectralAnalysis(bay, filenames, names, b_wavelets = dowavelets, window = window, num_segments = num_segments, \
                                           tunits = tunits, funits = funits, filter = filter, log = log, doy = doy, grid = grid, fname = None, domodel = domodel)

            slevel = 0.95
            # range 0-65000 is good to catch the high frequencies
            #       0-600000 if you need to catch internal waves with much lower frequencies and large periods
            val1, val2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram
            avg1, avg2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram

            title = bay + ""
            if doWavelet :
                debug = False
                Embayment.waveletAnalysis(bay, title, tunits, slevel, avg1, avg2, val1, val2, debug = debug)

            if doHarmonic:
                if bay == "":
                    freq_hours = 4.5
                    Embayment.HarmonicAnalysis('Lake_Ontario_1115682_processed.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-Mar23-Apr23-2011.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-JAN-DEC-2011_date.csv', path, freq_hours)
                    Embayment.HarmonicAnalysis('LO_Burlington-Apr26-Apr28-2011.csv', path, freq_hours)
                if bay == "Tob-IBP":
                    freq_hours = 0.266688
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
                    freq_hours = 0.120010
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
                    freq_hours = 0.07742
                    Embayment.HarmonicAnalysis('LL1-28jul2010.csv', path4, freq_hours)
        # end if 'Tor_Harb'

    # end CalculateSpectral

# end Embayment


if __name__ == '__main__':
    "options -n 4 -m -f"

    bay = 'Emb-A'
    #bay = 'Emb-B'
    #bay = 'Emb-C'
    #bay = 'Cell-1'
    # bay = 'Cell-2'
    # bay = 'Cell-3'
    #bay = 'FMB'
    # bay = 'BUR'
    # bay = 'Tob-OBP'
    #bay = 'Tob-IBP'
    #bay = 'Tob-CIH'
    # bay = 'Tob_All'
    # bay = 'Tob-HI'
    # bay = 'L-SUP'
    # bay = 'Tor_Harb'
    days = 10

    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-s", "--spectral", dest = "sp", action = "store_true", default = False, help = "Spectral analysis")
    parser.add_option("-m", "--model", dest = "mo", action = "store_true", default = False, help = "Spectral analysis with model simulation display")
    parser.add_option("-n", "--nsegments", dest = "ns", action = "store", default = 1, help = "Number of (Welch) segments for the spectral analysis")
    parser.add_option("-f", "--flushing", dest = "fl", action = "store_true", default = False, help = "Flusing timescales")
    parser.add_option("-t", "--title", dest = "ti", action = "store_true", default = False, help = "Print graph titles")

    (options, args) = parser.parse_args()
    if options.ti:
        Embayment.set_PrintTitle(True)
    if options.sp:
        model = options.mo
        print "* Calculate Spectral *"
        Embayment.CalculateSpectral(bay, model, options.ns)
    else:
        print ">> Do NOT Calculate Spectral <<"
    if options.fl:
        print "* Calculate Flow *"
        emb = Embayment(bay)
        emb.CalculateFlow(days)

    else:
        print ">> Do NOT Calculate Flow <<"
    print "Done."
