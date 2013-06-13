'''
Created on Jun 11, 2012

@author: bogdan
'''
import fft.FFTGraphs
import fft.fft_utils as fft_utils
import fft.Filter as Filter
import wavelets.Graphs
import wavelets.kCwt
import scipy as sp
import numpy as np
import math
import matplotlib.mlab as mlab

path = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/LakeOntario-data'
path1 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/FMB'
path2 = 'software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Data-long/LOntario'
path3 = '/software/software/scientific/Matlab_files/Helmoltz/Embayments-Exact/Toronto_Harbour'
path4 = '/home/bogdan/Documents/UofT/PhD/Data_Files/Toberymory_tides'

class Embayment(object):
    '''
    classdocs
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        # name can be: 'FMB', 'Emb_A', 'Tob_CIH', 'Tob_OBP'
        self.name = name

    def plotMultipleTimeseries(self, path_in, filenames, names, detrend = False, filtered = False, lowcut = None, highcut = None, tunits = 'sec'):
        # plot the original Lake oscillation input
        ts = []
        i = 0
        time = []
        for filename in filenames:
            [Time, SensorDepth] = fft_utils.readFile(path_in, filename)
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

            ts.append(sp.signal.detrend(SensorDepth))
            time.append(Time)
            i += 1
        series = np.array(ts)


        L = len(Time)
        xlabel = 'Time [days]'
        ylabel = 'Detrended Z(t) [m]'
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
        fft_utils.plot_n_TimeSeries("Detrended Lake and Bay Levels", xlabel, ylabel, xa, series, legend)



    # end plotLakeLevels

    def SpectralAnalysis(self, b_wavelets = False, window = "hanning", num_segments = None, tunits = 'day', funits = "Hz", filter = None):

        # show extended calculation of spectrum analysis
        show = True
        bay = self.name
        lake_name = ""
        bay_name = ""
        tunits = "day"
        if bay == 'FMB':
            # Frenchman's bay data
            fftsa = fft.FFTGraphs.FFTGraphs(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv', show, tunits)
            lake_name = "Lake Ontario"
            bay_name = "Frenchman's bay"
        elif bay == 'BUR':
            # Burlington data    43.333333N , 79.766667W (placed in the lake not the sheltered bay)
            fftsa = fft.FFTGraphs.FFTGraphs(path, 'LO_Burlington-JAN-DEC-2011_date.csv', None, show, tunits)
            # fftsa = fft.FFTGraphs.FFTGraphs(path, 'LO_Burlington-Apr26-Apr28-2011.csv', None, show, tunits)
            lake_name = "Lake  Ontario"
            bay_name = ""
        # Fathom Five National Park Outer Boat Passage
        elif bay == 'Tob-OBP':
            # NOTE:lake amplit is lower so switch places
            fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL2.csv', show, tunits)
            # fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL2-28jul2010.csv', show , tunits)
            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Outer Boat Passage"
        # Fathom Five National Park Inner Boat Passage
        elif bay == 'Tob-IBP':
            # NOTE:lake amplit is lower so switch places
            fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL1.csv', show, tunits)
            # fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL1-28jul2010.csv', show, tunits)

            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Inner Boat Passage"
        # Fathom Five National Park Cove Island Harbour
        elif bay == 'Tob-CIH':
            # NOTE:lake amplit is lower so switch places
            # fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3.csv', 'LL4.csv', show, tunits)
            fftsa = fft.FFTGraphs.FFTGraphs(path4, 'LL3-28jul2010.csv', 'LL4-28jul2010.csv', show, tunits)
            lake_name = "Harbour Island - lake"  # LL3.csv is actually the lake
            bay_name = "Cove Island Harbour"  #
        # Embayment A Tommy Thomson Park
        elif bay == 'Emb-A':
            fftsa = fft.FFTGraphs.FFTGraphs(path3, '1115865-Station16-Gate-date.csv', '1115861-Station14-EmbaymentA-date.csv', show, tunits)
            lake_name = "Lake Ontario"
            bay_name = " Emabayment A"
        else:
            print "Unknown embayment"
            exit(1)
        # end
        showLevels = False
        detrend = False
        draw = False
        log = False
        fftsa.doSpectralAnalysis(showLevels, draw, tunits, window, num_segments, filter, log)
        fftsa.plotLakeLevels(lake_name, bay_name, detrend)
        fftsa.plotSingleSideAplitudeSpectrumFreq(lake_name, bay_name, funits)
        fftsa.plotSingleSideAplitudeSpectrumTime(lake_name, bay_name)

        # fftsa.plotZoomedSingleSideAplitudeSpectrumFreq()
        # fftsa.plotZoomedSingleSideAplitudeSpectrumTime()
        # fftsa.plotCospectralDensity()
        fftsa.plotPhase()

        if b_wavelets:
            # Wavelet Spectral analysis
            if bay == 'FMB':
                graph = wavelets.Graphs.Graphs(path, 'Lake_Ontario_1115682_processed.csv', 'Inner_Harbour_July_processed.csv', show)
            elif bay == 'BUR':
                graph = wavelets.Graphs.Graphs(path, 'LO_Burlington-JAN-DEC-2011_date.csv', None, show)
                # graph = wavelets.Graphs.Graphs(path, 'LO_Burlington-Apr26-Apr28-2011.csv', None, show)
            elif bay == 'Tob-OBP':
                graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL2.csv', show)
                # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL2-28jul2010.csv', show)
            elif bay == 'Tob-IBP':
                graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL1.csv', show)
                # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL1-28jul2010.csv', show)
            # Fathom Five National Park Cove Island Harbour
            elif bay == 'Tob-CIH':
                graph = wavelets.Graphs.Graphs(path4, 'LL3.csv', 'LL4.csv', show)
                # graph = wavelets.Graphs.Graphs(path4, 'LL3-28jul2010.csv', 'LL4-28jul2010.csv', show)
            # Embayment A Tommy Thomson Park
            elif bay == 'Emb-A':
                graph = wavelets.Graphs.Graphs(path3, '1115865-Station16-Gate-date.csv', '1115861-Station14-EmbaymentA-date.csv', show)
            else:
                print "Unknown embayment"
                exit(1)

            graph.doSpectralAnalysis()
            graph.plotDateScalogram(scaleType = 'log', plotFreq = True)
            graph.plotSingleSideAplitudeSpectrumTime()
            graph.plotSingleSideAplitudeSpectrumFreq()
            graph.showGraph()
        # nd if b_wavelets
    # end SpectralAnalysis

    def HarmonicAnalysis(self, filename, freq_hours):

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
        print ("amplitude (m): %f") % R
    # end HarmonicAnalysis

    def waveletAnalysis(self, title, tunits, slevel, avg1, avg2, val1, val2, \
                        dj = None, s0 = None, J = None, alpha = None):

        ppath = path
        if self.name == 'FMB':
            # Frenchman's bay data
            ppath = path
            file = 'Lake_Ontario_1115682_processed.csv'
        elif self.name == 'BUR':
            # Burlington data    43.333333N , 79.766667W (placed in the lake not ht sheltered bay)
            ppath = path
            file = 'LO_Burlington-Apr26-Apr28-2011.csv'
        # Fathom Five National Park Outer Boat Passage
        elif self.name == 'Tob-OBP':
            ppath = path4
            file = 'LL2.csv'
            # file = 'LL1-28jul2010.csv'
        elif bay == 'Tob-IBP':
            ppath = path4
            file = 'LL1.csv'
        # Fathom Five National Park Cove Island Harbour
        elif self.name == 'Tob-CIH':
            ppath = path4
            file = 'LL4.csv'
            # file = 'LL4-28jul2010.csv'
        # Embayment A Tommy Thomson Park
        elif self.name == 'Emb-A':
            ppath = path3
            file = '1115865-Station16-Gate-date.csv'
        else:
            print "Unknown embayment"
            exit(1)
        kwavelet = wavelets.kCwt.kCwt(ppath, file, tunits)
        dj = 0.05  # Four sub-octaves per octaves
        s0 = -1  # 2 * dt                      # Starting scale, here 6 months
        J = -1  # 7 / dj                      # Seven powers of two with dj sub-octaves
        alpha = 0.5  # Lag-1 autocorrelation for white noise

        kwavelet.doSpectralAnalysis(title, "morlet", slevel, avg1, avg2, dj, s0, J, alpha)
        ylabel_ts = "amplitude"
        yunits_ts = 'm'
        xlabel_sc = ""
        ylabel_sc = 'Period (%s)' % kwavelet.tunits
        # ylabel_sc = 'Freq (Hz)'
        sc_type = "period"
        # sc_type = "freq"
        x_type = 'date'
        kwavelet.plotSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)
        ylabel_sc = 'Frequency ($s^{-1}$)'
        kwavelet.plotAmplitudeSpectrogram(ylabel_ts, yunits_ts, xlabel_sc, ylabel_sc, sc_type, x_type, val1, val2)

    def CalculateFlow(self, bay):
        # Water exchange
        if bay == 'FMB':
            A = 850000
            B = 25
            H = 1
            L = 130
            # Amplitudes in [m]

            # Hamblin

            a = 0.0365
            b = 0.023
            c = 0.015
            d = 0.018
            e = 0.016

            # FFT
            a = 0.0
            a = 0.0043
            b = 0.0026
            c = 0.0020
            d = 0.0011
            e = 0.0016

            Amplitude = [a, b, c, d, e]

            # periods in [h]
            T = 12.2;
            T1 = 5.065;
            T2 = 1.38;
            T3 = 0.81;
            T4 = 0.48;
            Period = [T, T1, T2, T3, T4]

            # phases in [rad]
            phi = 21.5
            phi1 = 22
            phi2 = -15
            phi3 = 39
            phi4 = -4.6
            Phase = [phi, phi1, phi2, phi3, phi4]

        elif bay == 'Emb-A':
            # Embaymeant A
            A = 70000
            B = 79.8
            H = 2
            L = 83.8
            # FFT

            a = 0.024
            b = 0.0235
            c = 0.019
            d = 0.004
            e = 0.004
            Amplitude = [a, b, c, d, e]

            # periods in [h]
            T1 = 24.8
            T2 = 12.4
            T3 = 4.9
            T4 = 2.26
            T5 = 1.07
            Period = [T1, T2, T3, T4, T5]
            # phases in [rad]
            phi1 = 21.5
            phi2 = 22
            phi3 = -15
            phi4 = 39
            phi5 = 22
            Phase = [phi1, phi2, phi3, phi4, phi5]

        # end

        Cd_const = True
        Cd = 0.003  # < ==  This has to change according to Mullarnery to 0.01

        # plot additional graphs
        doplot = True

        # [t, X, m, c, k, w, x0, v0, R] = ODEplot(bay, 'B', 1, A, B, H, L, Period, Amplitude, Phase, length(Period), Cd_const, doplot, Cd);
        # ODEplotVer( 'FrenchmanBay', 'B', 1, A,B,H,L, Period, Amplitude, Phase, length(Period),Cd_const,doplot,Cd);
        # ODEplot('Lake_Ontario_1115682_processed.csv', A,B,H,L)



        #
        # Calculate the Flow in the Embayment
        #
        Cd_const = False

        filename1 = 'Inner_Harbour_July_processed.csv'
        # [Time1,SensorDepth1] = textread(filename1, '%f %f' ,-1,'delimiter', ',');
        # resample to be the same as first
        '''
        ts=timeseries(SensorDepth1,Time1,'Name','level');
        ts=transpose(ts);
        res_ts=resample(ts,Time);
        SensorDepth2=res_ts.data(:); %get(res_ts,'Data');
        % predicted
        Q=embaymentFlow(A,B,H,R,t(2)-t(1));

        y=detrend(SensorDepth2);
        %use smoothed time-series to eliminate oscillations, which do not
        %contribute to flushing.
        y2=smoothSeries(SensorDepth2,8); %30 minutes
        % measured
        Q2=embaymentFlow(A,B,H,SensorDepth2,(Time(2)-Time(1))*86400);
        figure;
        [titl, errmsg] = sprintf('FM Bay Flow vs.  Time Diagram\n Cd=%4.3f[m] H=%4.1f B=%4.1f[m] L=%4.1f[m]',Cd,H,B,L);
        %plot(t(10:length(Q)),Q(10:length(Q)),'b'),xlabel('time [s]'), ylabel('Q [m^3/s]'), title(titl), grid on, hold  on
        tt=Time'; % transpose
        plot(tt(1:length(Q)),Q,'b'),xlabel('time [s]'), ylabel('Q [m^3/s]'), title(titl), grid on, hold  on

        plot(tt(1:length(Q)),Q2(1:length(Q)),'-.r');
        h = legend('predicted','measured',2, 'location', 'NorthEast');

        %plot the original Lake oscillation input
        L = length(Q);
        xlim = [tt(1),tt(L)];  % plotting range
        Xticks = tt(1):(tt(length(Q))-tt(1))/3:tt(length(Q));
        Xtickslabel={};
        for i=1:1:length(Xticks)
            Xtickslabel{i}=datestr(Xticks(i),' mmm dd HH:MM');
        end
        set(gca,'XLim',xlim(:),    'XTick',Xticks(:),'XTickLabel',Xtickslabel(:));


        summeas=0;
        sumpred=0;
        Vp=0;s
        for i=2:length(Q)
            %{
            if (R(i)-R(i-1) > 0) && (R(i) >0)
                sumpred=sumpred+0.5*(R(i)-R(i-1));
            end;

            %}
            sumpred=sumpred+0.5*abs(R(i)-R(i-1));
           if ((Q(i) - Q(i-1)) >0) && (Q(i) > 0 ) %&& (Q(i-1) >0)
                Vp=Vp+(Q(i)+Q(i-1))/2*300;
            end
        end

        Vm=0;
        for i=2:length(Q)
            %{
            if (SensorDepth2(i)-SensorDepth2(i-1) > 0) && (SensorDepth2(i) >0)
                summeas=summeas+0.5*(SensorDepth2(i)-SensorDepth2(i-1));
            end
            %}
            %summeas=summeas+0.5*abs(SensorDepth2(i)-SensorDepth2(i-1));
            summeas=summeas+0.5*abs(y2(i)-y2(i-1));
            if ((Q2(i) - Q2(i-1)) >0) && (Q2(i) > 0 ) %&& (Q2(i-1) >0)
                Vm=Vm+(Q2(i)+Q2(i-1))/2*300;
            end
        end



        display('Sum measured'), display(summeas);
        display('Sum predicted'), display(sumpred);
        display('Vol meas'), display(Vm);
        display('Vol pred'), display(Vp);
        set(h,'Interpreter','none')
        '''
    # end CalculateFlow

if __name__ == '__main__':
    bay = 'Emb-A'
    bay = 'FMB'
    # bay = 'BUR'
    bay = 'Tob-OBP'
    bay = 'Tob-IBP'
    bay = 'Tob-CIH'

    emb = Embayment(bay)

    showLevels = False
    detrend = False
    # detrend = True

    filenames = ['LL1.csv', 'LL4.csv', 'LL2.csv', 'LL3.csv']
    # filenames = ['LL1-28jul2010.csv', 'LL4-28jul2010.csv', 'LL2-28jul2010.csv', 'LL3-28jul2010.csv']
    names = [ "Inner Boat Passage" , "Cove Island Harbour", "Outer Boat Passage", "Harbour Island - lake"]

    # set to True for Butterworth filtering - just for testing.
    doFiltering = False
    lowcutoff = 1.157407e-5
    highcutoff = 0.00834
    tunits = 'day'  # can be 'sec', 'hour'
    funits = "cph"
    emb.plotMultipleTimeseries(path4, filenames, names, detrend, doFiltering , lowcutoff, highcutoff, tunits)


    doSpectral = True
    dowavelets = True
    doWavelet = True
    doHarmonic = False
    doFiltering = False
    tunits = 'day'  # can be 'sec', 'hour'
    window = 'hanning'
    num_segments = 10

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
        emb.SpectralAnalysis(dowavelets, window, num_segments, tunits, funits, filter)

    tunits = 'day'
    slevel = 0.95
    # range 0-65000 is good to catch the high frequencies
    #       0-600000 if you need to catch internal waves with much lower frequencies and large periods
    val1, val2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram
    avg1, avg2 = (0, 65000)  # Range of sc_type (ex periods) to plot in spectogram

    title = bay + ""
    if doWavelet :
        emb.waveletAnalysis(title, tunits, slevel, avg1, avg2, val1, val2)

    if doHarmonic:
        freq_hours = 4.5
        emb.HarmonicAnalysis('Lake_Ontario_1115682_processed.csv', freq_hours)
        emb.HarmonicAnalysis('LO_Burlington-Mar23-Apr23-2011.csv', freq_hours)
        emb.HarmonicAnalysis('LO_Burlington-JAN-DEC-2011_date.csv', freq_hours)
        emb.HarmonicAnalysis('LO_Burlington-Apr26-Apr28-2011.csv', freq_hours)

    # emb.CalculateFlow(bay)
    print "Done."
