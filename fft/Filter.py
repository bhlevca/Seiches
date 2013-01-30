'''
Created on Jan 18, 2013

@author: bogdan
'''
import filters

class Filter(object):

    def __init__(self, doFiltering = None, ftype = 'fft', lowcutoff = None, highcutoff = None, btype = 'band', order = 5):
        '''
        Constructor
        '''
        self.doFilter = doFiltering
        self.lowcutoff = lowcutoff
        self.highcutoff = highcutoff
        self.btype = btype
        self.order = order
        self.ftype = ftype

    def butter_bandpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return filters.butter_bandpass(self.lowcutoff, self.highcutoff, fs, ord)

    def butter_highpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return  filters.butter_highpass(self.highcutoff, fs, ord)

    def butter_lowpass(self, fs, order = 5):
        if order != None:
            ord = order
        else:
            ord = self.order
        return filters.butter_lowpass(self.lowcutoff, fs, ord)

    def butterworth(self, data, fs, order = 5, worN = 2000):
        # returns [y, w, h, b, a]
        if order != None:
            ord = order
        else:
            ord = self.order
        if self.doFilter:
            if self.ftype == 'fft':
                y = filters.fft_bandpassfilter(data, fs, self.lowcutoff, self.highcutoff)
                return [y, None, None, None, None]
            else:
                return filters.butterworth(data, self.btype, lowcut = self.lowcutoff, highcut = self.highcutoff, fs = fs, order = ord, worN = worN)

        else:
            return [data, None, None, None, None]
