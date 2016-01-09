'''
Created on Jun 11, 2012

@author: bogdan
'''
import ufft.FFTGraphs as FFTGraphs
import ufft.fft_utils as fft_utils
import ufft.Filter as Filter
import wavelets.kCwt
import scipy as sp
import numpy as np
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


class BayGeometry(object):

    def __init__(self, L, B, h):
        self.L = L  # Length
        self.B = B  # Width
        self.h = h  # average depth
        self.W = self.B / 2.  # half basin width (m) B=2*W


# end class BayGeometry


class EmbaymentNonlinear(object):



    def __init__(self, bay):
        # constants
        self.L = bay.L  # Length
        self.B = bay.B  # Width
        self.h = bay.h  # average depth
        self.W = bay.B / 2.  # half basin width (m) B=2*W
        self.gam = np.exp(0.5772157)  # Or gam = 1.781072481 - Euler constant
        self.g = 9.81  # gravitational acceleration

    def makex1(self, eps):
        def x1(x):
            return eps * x
        return x1

    def makeL1(self, eps):
        def L1(L):
            return eps * L
        return L1

    def makeW1(self, eps):
        def W1(W):
            return eps * W
        return W1

    def maket1(self, eps):
        def t1(W):
            return eps * t
        return t1

    def makeOM(self, eps):
        def OM(om):
            return om / 2. / eps
        return OM



    def initialiseScaledDimsFunction(self, eps):
        self.x1f = self.makex1(eps)
        self.L1f = self.makeL1(eps)
        self.W1f = self.makeW1(eps)
        self.t1f = self.maket1(eps)
        self.OMf = self.makeOM(eps)

    def initialiseScaledDims(self, om, t, x):
        L1 = self.L1f(self.L)
        OM = self.OMf(om)
        t1 = self.t1f(t)
        x1 = self.x1f(x)
        W1 = self.W1f(self.W)
        return [t1, OM, x1, L1, W1]

    def initialiseScaledDimsFunctionSlow(self, eps):
        self.L1fs = self.makeL1(eps)
        self.W1fs = self.makeW1(eps)
        self.OMfs = self.makeOM(eps)

    def initialiseScaledDimsSlow(self, om):
        L1 = self.L1fs(self.L)
        OM = self.OMfs(om)
        W1 = self.W1fs(self.W)
        return [OM, L1, W1]

    def calculateResponse(self, t, a0, freq, x):

        # calculate independent variables
        T = 1. / freq
        C = np.sqrt(self.g * self.h)  # phase velocity

        # WRONG om = 2 * np.pi * freq
        # WRONG k = om / C  # wavenumber
        # WRONG lamd = C / freq
        # WRONG k = 2 * np.pi / lamd
        # WRONG om = np.sqrt(self.g * k * np.tanh(k * self.h))  # dispersion relation

        om = 2 * np.pi * freq
        k = self.dispersion_w(om, self.h)

        eps = k * a0  # or k*a0/np.pi ?       # the wave steepness/slope

        # deal with the closures
        self.initialiseScaledDimsFunction(eps)
        [t1, OM, x1, L1, W1] = self.initialiseScaledDims(om, t, x)

        Cg = C / 2.*(1 + 2 * k * self.h / (2 * np.sinh(2 * k * self.h)))  # Group velocity from Mei book "Theory and Applications of Ocean Surface Waves" pp 18
        Kf = 2 * OM / np.sqrt(self.g * self.h)
        Kg = 2 * OM / Cg
        Gamma = 1 - np.exp(2j * Kg * L1)  # or the same thing =>Gamma = -2j * np.sin(Kg * L1) * np.exp(1j * Kg * L1)
        Q = -self.g / (4 * om ** 2) * Cg ** 2 / (self.g * self.h - Cg ** 2) * (2 * om * k / Cg + k ** 2 - om ** 4 / self.g ** 2)
        Z = np.cos(Kf * L1) + 2 * Kf * W1 / np.pi * np.sin(Kf * L1) * np.log(2 * self.gam * Kf * W1 / np.pi / np.e) - 1j * Kf * W1 * np.sin(Kf * L1)

        # The response

        # 1) Surface
        zf = np.zeros((len(x1), len(t1)))
        for i in range(0, len(x1)):
            zf[i] = 0.5 * Q * a0 ** 2 * np.cos(Kf * (x1[i] + L1)) * np.real(Gamma / Z * np.exp(-2j * om * t1))


#===============================================================================
#         legend = ["wl0", "wl300", "wlend"]
#         fft_utils.plot_n_Array("Bay Water Levels in time", "time (s)", "water level (m)", [t1, t1, t1], [zf[0], zf[300], zf[len(x1) - 1]], legend, plottitle = "title", fontsize = 20)
#
#         legend = ["wlend", "w300", "w700", "w600", "w227"]
#         fft_utils.plot_n_Array("Bay Water Levels in space", "x (m)", "water level (m)", [x1, x1, x1, x1, x1], [zf[:, 0], zf[:, 300], zf[:, 700], zf[:, 600], zf[:, 227]], legend, plottitle = "title", fontsize = 20)
#===============================================================================



        # 2) amplitude
        A = Q * a0 ** 2 * Gamma / 2. / np.abs(Z)
        AL = Q * a0 ** 2 / 2. / np.abs(Z)
        legend = ["kh=", "kh=", "kh"]

        print "A=%f ; AL=%f amplif_Factor=%f" % (np.abs(A), np.abs(AL), np.abs(AL) / (a0 ** 2 * k))
        print "Kf*L1=%f, n=%f" % (Kf * L1, Kf * L1 / np.pi)

        kl = np.linspace(0, 40, 600)
        j = 0
        AG = np.zeros(len(kl))
        AA = np.zeros(len(kl))
        i = 0
        for KL in kl:
            z = np.cos(KL) + 2 * Kf * W1 / np.pi * np.sin(KL) * np.log(2 * self.gam * Kf * W1 / np.pi / np.e) - 1j * Kf * W1 * np.sin(KL)
            Gamma2 = 1 - np.exp(2j * KL / Cg / C)
            AG[i] = -Q * a0 ** 2 * Gamma2 / np.abs(z) / 2
            AA[i] = -Q * a0 ** 2 / np.abs(z) / 2
            i += 1
        legend = ["A Gamma", "A no SW reflection"]
        ax = fft_utils.plot_n_Array("Amplification factor for a long bay", "$(K_1*L_1)$", "water level (m)", [kl, kl], [AG, AA], \
                                     legend, plottitle = "title", fontsize = 20, noshow = True)
        ax.plot(L1 * Kf, A, 'bd')
        plt.show()


        # plot |A|/ka0 vs Kf*W1

        # take the fourier analysis - to be done


    def dispersion_w(self, w, depth, gravity = 9.8066):
        '''
        // GNU General Public License Agreement
        // Copyright (C) 2004-2010 CodeCogs, Zyba Ltd, Broadwood, Holford, TA5 1DU, England.
        //
        // This program is free software; you can redistribute it and/or modify it under
        // the terms of the GNU General Public License as published by CodeCogs.
        // You must retain a copy of this licence in all copies.
        //
        // This program is distributed in the hope that it will be useful, but WITHOUT ANY
        // WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
        // PARTICULAR PURPOSE. See the GNU General Public License for more details.
        // ---------------------------------------------------------------------------------

        '''

        if w == 0: return 0
        else:
            k_deep = w * w / gravity  # k in infinitely deep water (this also always forms a lower bound).

            if depth <= 0.0: return k_deep

            # We know this error function is well behaved, k_deep is also always a lower bound.
            k = k_deep  # k_deep is a good first estimate
            e = 1
            while(np.abs(e) > 0.00000001):
                kd = k * depth
                coshkd = np.cosh(kd)
                tanhkd = np.tanh(kd)
                e = k_deep - k * tanhkd  # error
                dedk = tanhkd + kd / (coshkd * coshkd)  # rate of change of error wrt to k (+ve)

                # use only half the error correction
                k += e / dedk;
            # end

            return k
    # end dispersion_w


    def calculateResponseVsFrequency(self, a0, om, graph = False):

        # calculate independent variables
        C = np.sqrt(self.g * self.h)  # phase velocity

        A = np.zeros(len(om))
        i = 0
        for w in om:
            k = self.dispersion_w(w, self.h)
            Cg = C / 2.*(1 + 2 * k * self.h / (2 * np.sinh(2 * k * self.h)))  # Group velocity from Mei book "Theory and Applications of Ocean Surface Waves" pp 18
            Z = np.cos(k * self.L) + 2 * k * self.W / np.pi * np.sin(k * self.L) * np.log(2 * self.gam * k * self.W / np.pi / np.e) - 1j * k * self.W * np.sin(k * self.L)

            # The response

            # 1) amplitude
            A[i] = a0 / np.abs(Z)
            i += 1
        # end for

        if graph:
            legend = ["kh"]
            fft_utils.plot_n_Array("Amplification factor for a long bay", "w", "amplif", [om], [A], legend, plottitle = "title", fontsize = 20)
        else:
            return A

    def calculateResponseVsAngularFreqSlow(self, a0, om, graph = False):

        # calculate independent variables
        C = np.sqrt(self.g * self.h)  # phase velocity
        A = np.zeros(len(om))
        i = 0
        for w in om:
            k = self.dispersion_w(w, self.h)
            eps = k * a0  # or k*a0/np.pi ?       # the wave steepness/slope

            # deal with the closures
            self.initialiseScaledDimsFunctionSlow(eps)
            [OM, L1, W1] = self.initialiseScaledDimsSlow(w)

            Cg = C / 2.*(1 + 2 * k * self.h / (2 * np.sinh(2 * k * self.h)))  # Group velocity from Mei book "Theory and Applications of Ocean Surface Waves" pp 18
            Kf = 2 * OM / C
            Kg = 2 * OM / Cg
            Gamma = 1 - np.exp(2j * Kg * L1)  # or the same thing =>Gamma = -2j * np.sin(Kg * L1) * np.exp(1j * Kg * L1)
            Q = -self.g / (4 * w ** 2) * Cg ** 2 / (self.g * self.h - Cg ** 2) * (2 * w * k / Cg + k ** 2 - w ** 4 / self.g ** 2)
            Z = np.cos(Kf * L1) + 2 * Kf * W1 / np.pi * np.sin(Kf * L1) * np.log(2 * self.gam * Kf * W1 / np.pi / np.e) - 1j * Kf * W1 * np.sin(Kf * L1)

            # 1) amplitude
            A[i] = Q * a0 * a0 * Gamma / np.abs(Z) / 2.
            # print "w=%f,  A[%d]=%f" % (w, i, A[i])

            i += 1
        # end for
        if graph:
            print "om=", om
            legend = ["kh"]
            fft_utils.plot_n_Array("Amplification factor for a long bay", "w", "amplif", [om[1:]], np.abs([A[1:]]), legend, plottitle = "title", fontsize = 20)
        return np.abs(A)






# end class EmbaymentNonlinear


if __name__ == '__main__':

    # input values
    # 0 the embayment
    L = 1000  # basin length (m)
    B = 140  # Basin Width
    h = 1.5  # average depth of the basin
    bay = BayGeometry(L, B, h)

    # 1) wind waves parameters
    a0 = 0.1  # max amplitude
    T = 5  # sec
    freq = 1. / T

    # 2)  time one hour with 5 sec step
    t = np.linspace(0, 3600, 720)

    # 3)  space 0 to L, 100 steps
    x = np.linspace(0, -L, 1400)

    embNon = EmbaymentNonlinear(bay)
    embNon.calculateResponse(t, a0, freq, x)

    # 4)  omega 0 to 15, 2000 steps
    om = np.linspace(0.00001, 0.1, 200)
    embNon.calculateResponseVsFrequency(a0, om, True)
    embNon.calculateResponseVsAngularFreqSlow(a0, om, True)



    print "Embayment Nonlinear Done!"
