import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#############
# constants
#############
# gravity [m/s^2]
g = 9.81;

# head loss due to flow separation
f = 1.55;

omega_char = unichr(0x3c9).encode('utf-8')

class EmbaymentPlot(object):

    def __init__(self, bay):
        self.location_name = bay.name
        self.A = bay.A
        self.B = bay.B
        self.H = bay.H
        self.L = bay.L
        self.Period = bay.Period
        self.Amplitude = bay.Amplitude
        self.Phase = bay.Phase
        self.Cd = bay.Cd
        self.w0 = None


        # local arrays
        self.w = np.zeros(len(self.Amplitude), dtype = np.ndarray)
        self.X = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # Bay oscillations
        self.fwave = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # forcing oscillations (lake)
        self.c = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # damping effect
        self.k = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # elastic constant
        self.Fa = np.zeros(len(self.Amplitude), dtype = np.ndarray)
        self.phy = np.zeros(len(self.Amplitude), dtype = np.ndarray)
        self.tsup = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # suplementary period due to phase lag
        self.G = np.zeros(len(self.Amplitude), dtype = np.ndarray)  # hyptetic response of an oscillator

    def show(self):
        plt.show()

    def amplitudef(self, amplitude_e, w, w0, n0):
        '''
        This is eq (3) from Terra et al. (2005) , the bay (resulting) amplitude
        '''
        ampl = np.absolute(amplitude_e) * \
                np.sqrt((np.sqrt((1 - (w / w0) ** 2) ** 4 + 4 * n0 ** 2 * (w / w0) ** 4 * (np.absolute(amplitude_e)) ** 2) - \
                      (1 - (w / w0) ** 2) ** 2) / (2 * n0 ** 2 * (w / w0) ** 4 * (np.absolute(amplitude_e)) ** 2))
        return ampl
        # end amplitudef


    # THIS NEEDS TO BE REMOVED
    def ampratiof(self, amplitude_e, w, w0, n0):
        amplr = np.sqrt((np.sqrt((1 - (w / w0) ** 2) ** 4 + 4 * n0 ** 2 * (w / w0) ** 4 * (np.absolute(amplitude_e)) ** 2) - (1 - (w / w0) ** 2) ** 2)\
                     / (2 * n0 ** 2 * (w / w0) ** 4 * (np.absolute(amplitude_e)) ** 2))
        return amplr
        # end ampratiof

    def fourierODE(self, m, c, k, Fa, w):
        '''
        % [x,v]=solveODE(m,c,k,f,w,x0,v0,t)
        % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        % This function solves the differential equation
        % a*x''(t)+b*x'(t)+c*x(t)=f(t)
        % with x(0)=x0 and x'(0)=v0
        %
        % m,c,k  - mass, damping and stiffness coefficients
        % f1     - the forcing function
        % w      - frequency of the forcing function
        % t      - vector of times to evaluate the solution
        % x,v    - computed position and velocity vectors
        % wn     - w0 eigenfrequency
        '''
        F = np.zeros(len(w))

        ccrit = 2 * np.sqrt(m * k)
        wn = np.sqrt(k / m)

        # If the system is undamped and resonance will
        # occur, add a little damping
        if c == 0 and w == wn:
            c = ccrit / 1e6
        # end if

        # If damping is critical, modify the damping
        # very slightly to avoid repeated roots
        if c == ccrit:
            c = c * (1 + 1e-6)
        # end if

        # Forced response particular solution

        for i in range(0, len(w)):
            F[i] = Fa / (k - m * w[i] ** 2 + 1j * c * w[i])
        # end for

        return F
    # fourierODE

    def phaseODE(self, n0, w0, amplitude_e, om):

        for j in range(0, len(om)):
            amplitude[j] = abs(amplitude_e) * np.sqrt((np.sqrt((1 - (om[j] / w0) ** 2) ** 4 + 4 * n0 ** 2 * (om[j] / w0) ** 4 * (abs(amplitude_e)) ** 2)\
                 - (1 - (om[j] / w0) ** 2) ** 2) / (2 * n0 ** 2 * (om[j] / w0) ** 4 * (abs(amplitude_e)) ** 2))

            PHI[j] = np.arccos((1 - (om[j] / w0) ** 2) * amplitude[j] / amplitude_e)
        # end for
        return PHI


    def  Response(self, days):
            '''
            % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            % This function plots the response and animates the
            % motion of a damped linear harmonic embayment oscillator
            % characterized by the differential equation
            % z''+n0*w*|alpha|*z'-w0^2*z=w0^2*ze(t)
            % with initial conditions x(0)=x0, x'(0)=v0.
            % The animation depicts forced motion of a block
            % attached to a wall by a spring. The block
            % slides on a horizontal plane which provides
            % viscous damping.

            % example - Omit this parameter for interactive input.
            %           Use smdplot(1) to run a sample problem.
            % t,X     - time vector and displacement response
            % m,c,k   - mass, damping coefficient,  replaced with a,b,c
            %           spring stiffness constant
            % w0      - eigen frequency
            % A       - area of the embayment
            % L       - length of the channel
            % B       - channel width
            % H       - channel depth
            % fm      - head loss coefficient ( fm/L )
            % O       - B*H  - channel cross section
            % n0      - 8*fm*A/(3*pi*O*L)
            % g       - gravitational acceleration 9.81 m/s^2
            %
            % f1,f2,w - force components and forcing frequency
            % x0,v0   - initial position and velocity
            %
            % User m functions called: spring smdsolve inputv
            % -----------------------------------------------
            '''
            #===============================================================================
            print ('SOLUTION FOR z\"+n*w*a*z\'-w0^2*z=w0^2*ze')
            print('Width: %f') % self.B

            tmax = days * 86400;  # 5days
            nt = 2880;  # 24h 12*24 = 12 intervals of 5 mins * 24 h
            x0 = 0.0; v0 = 0;

            O = self.B * self.H;
            # head loss coeff. includes flow separation and bottom friction
            fm = self.L * (f / self.L + self.Cd / self.H);

            # linearized loss term coefficient
            n0 = 8 * fm * self.A / (3 * np.pi * O * self.L);

            # eigenfrequency
            self.w0 = np.sqrt(g * O / self.L / self.A);
            print 'embayment eigen angular frequency %s=%f (rad)' % (omega_char, self.w0)

            freq0 = self.w0 / (2 * np.pi)
            print 'embayment eigen frequency f=%f (Hz)' % freq0

            T0 = 1 / freq0  # sec
            print 'embayment eigen period T=%f (hours) = %f (min)' % (T0 / 3600, T0 / 60)

            t = np.linspace(0, tmax, nt);

            Fin = 0  # forcing
            R = 0  # Response
            for i in range(0, len(self.Amplitude)):
                freq = 1 / (self.Period[i] * 3600)
                Fin += self.Amplitude[i] * np.sin(2 * np.pi * freq * t + self.Phase[i])

                self.w[i] = 2 * np.pi * freq
                bay_ampl = self.amplitudef(self.Amplitude[i], self.w[i], self.w0, n0)

                self.X[i] = np.real(bay_ampl * np.exp(1j * self.w[i] * t))  # bay response oscillation function
                self.fwave[i] = np.real(self.Amplitude[i] * np.exp(1j * (self.w[i] * t)))  # forcing oscillation function
                self.c[i] = n0 * self.w[i] * bay_ampl  # damping effect
                self.k[i] = self.w0 ** 2 ;  # elastic constant
                self.Fa[i] = self.w0 ** 2 * self.Amplitude[i]
                self.phy[i] = np.arccos((1 - (self.w[i] / self.w0) ** 2) * bay_ampl / self.Amplitude[i])
                self.tsup[i] = self.phy[i] / self.w[i]  # suplementary period due to phase lag
                print "*********************************"
                print 'embayment amplitude=', bay_ampl
                print 'phase =%f' % self.phy[i]
                print 'response frequency=%f (Hz), angular freq = %f (rad) period= %f (min)' % (freq, self.w[i], 1 / freq / 60)
                print ''
                R += self.X[i]
            # end for
            return [t, self.X, self.c, self.k, self.w, x0, v0, R]
        # end Response

    def plotForcingResponse(self, t):
        '''
        Plot individual frequency responses
        '''
        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        ff, ax = plt.subplots(len(self.Amplitude))
        plt.subplots_adjust(hspace = 0.8)
        yFormatter = FormatStrFormatter('%.3f')

        for i in range(0, len(self.Amplitude)):
            nPoints = 300
            ax[i].plot((t[0:nPoints] + self.tsup[i]) / 3600, self.X[i][0:nPoints])
            ax[i].set_xlabel('Time (h)')
            ax[i].plot(t[0:nPoints] / 3600, self.fwave[i][0:nPoints], '-.r')

            ax[i].legend(['bay', 'lake'])
            title = 'Response embayment: %s - Forcing: a=%5.3f (m), T=%5.2f (h)' % (self.location_name, self.Amplitude[i], self.Period[i])
            ax[i].set_title(title)
            ax[i].set_ylabel('Displ. (m)')
            ax[i].grid(True)
            mn1 = np.min(self.X[i][0:nPoints])
            mn2 = np.min(self.fwave[i][0:nPoints])
            ma1 = np.max(self.X[i][0:nPoints])
            ma2 = np.max(self.fwave[i][0:nPoints])
            mn = min(mn1, mn2)
            ma = max(ma1, ma2)
            step = (ma - mn) / 3
            ax[i].yaxis.set_major_formatter(yFormatter)
            ax[i].set_yticks(np.arange(mn, ma + ma / 10., step))
        # end for


    def plotRespVsOmegaVarAmplit(self):
        # Plot the response |G(w)| versus frequency (omega)

        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        steps = 1000
        om = np.linspace(0.0001, self.w0 * 5, steps)

        m = 1  # m = mass
        ctr = 0
        Fa_sum = 0.
        # variable amplitude
        for i in range(0, len(self.Fa)):
            self.G[i] = self.fourierODE(m, self.c[i], self.k[i], self.Fa[i], om)

            # fric term like parallel circuits
            ctr += 1 / self.c[i]
            Fa_sum += self.Fa[i]
        # end for

        # ct = 1 / ctr
        # GT = fourierODE(m, ct, k, Fa_sum, om);

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        legend = []
        title = 'Hypothetical Response at different friction values - Embayment: %s' % self.location_name
        plt.title(title)
        plt.ylabel('|G(w)|')
        plt.xlabel('w/w0')
        plt.grid(True)

        for i in range(0, len(self.Amplitude)):
            plt.plot(om / self.w0, abs(self.G[i]))
            # ax[i].set_legend('bay', 'lake')
            lgnd = 'T=%f   a=%f' % (self.Period[i], self.Amplitude[i])
            legend.append(lgnd)
        # end for
        # plt.plot(om / w0, abs(GT))
        plt.legend(legend)


    def plotRespVsOmegaVarFric(self):
        '''Plot the response |G(w)| versus frequency (omega)
        '''

        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        steps = 1000;
        om = np.linspace(0.0001, w0 * 5, steps);

        m = 1  # m = mass
        ctr = 0

        # variable amplitude
        for i in range(0, len(self.Fa)):
            self.G[i] = self.fourierODE(m, self.c[0] * i, self.k[i], self.Fa[0], om)
            # fric term like parallel circuits
            ctr += 1 / self.c[i]
            Fa_sum += self.Fa[i]
        # end for

        # ct = 1 / ctr
        # GT = fourierODE(m, ct, k, Fa_sum, om);

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        legend = []
        title = 'Hypothetical Response at different friction values - Embayment: %s' % location_name
        plt.title(title)
        plt.ylabel('|G(w)|')
        plt.xlabel('w/w0')
        plt.grid(True)

        for i in range(0, len(self.Amplitude)):
            plt.plot(om / self.w0, abs(self.G[i]))
            lgnd = 'fric=c*%d' % i
            legend.append(lgnd)

        # end for
        # plt.plot(om / w0, abs(GT))
        plt.legend(legend)


    def plotPhaseVsOmega(self):
        '''Plot the Phase diagram
            variable amplitude
        '''

        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        for i in range(0, len(self.Fa)):
            PHI[i] = phaseODE(n0 * i, w0, self.Amplitude[0], om)
        # end for

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        legend = []
        T0 = 2 * np.pi / self.w0 / 3600
        T = 2 * np.pi / om / 3600

        title = 'Phase lag - Embayment: %s' % location_name
        plt.title(title)
        plt.ylabel('Phase (rad)')
        plt.xlabel('w/w0')
        plt.grid(True)

        for i in range(0, len(self.Amplitude)):
            plt.plot(T0 / T, PHI[i])
            lgnd = 'fric=c*%d' % i
            legend.append(lgnd)
        # end for
        plt.plot(om / w0, abs(GT))
        plt.legend(legend)


    def plotRespVsOmegaVarArea(self):
        '''Plot the response |G(w)| versus frequency (omega)
        '''
        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        steps = 1000;
        om = np.linspace(0.0001, w0 * 5, steps);

        m = 1  # m = mass
        ctr = 0

        # variable amplitude
        for i in range(0, len(self.Fa)):
            self.G[i] = self.fourierODE(m, c[0] * i, self.k[i], self.Fa[0], om)
            # fric term like parallel circuits
            ctr += 1 / self.c[i]
            Fa_sum += self.Fa[i]
        # end for

        # ct = 1 / ctr
        # GT = fourierODE(m, ct, k, Fa_sum, om);

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        legend = []
        title = 'Hypothetical Response at different bay areas - Embayment: %s' % location_name
        plt.title(title)
        plt.ylabel('|G(w)|')
        plt.xlabel('w/w0')
        plt.grid(True)

        for i in range(0, len(Amplitude)):
            plt.plot(om / self.w0, abs(self.G[i]))
            lgnd = 'fric=c*%d' % i
            legend.append(lgnd)
        # end for
        # plt.plot(om / w0, abs(GT))
        plt.legend(legend)

    def plotRespVsOmegaVarMouth(self):
        '''Plot the response |G(w)| versus frequency (omega)
        '''

        if self.w0 == None:
            print "Error! Response not calculated yet."
            exit(0)

        steps = 1000;
        om = np.linspace(0.0001, w0 * 5, steps);

        m = 1  # m = mass
        ctr = 0

        # variable amplitude
        for i in range(0, len(self.Fa)):
            self.G[i] = self.fourierODE(m, c[0] * i, self.k[i], self.Fa[0], om)
            # fric term like parallel circuits
            ctr += 1 / self.c[i]
            Fa_sum += self.Fa[i]
        # end for

        # ct = 1 / ctr
        # GT = fourierODE(m, ct, k, Fa_sum, om);

        fig = plt.figure(facecolor = 'w', edgecolor = 'k')
        legend = []
        title = 'Hypothetical Response at different mouth sizes - Embayment: %s' % location_name
        plt.title(title)
        plt.ylabel('|G(w)|')
        plt.xlabel('w/w0')
        plt.grid(True)

        for i in range(0, len(Amplitude)):
            plt.plot(om / w0, abs(self.G[i]))
            lgnd = 'fric=c*%d' % i
            legend.append(lgnd)
        # end for
        # plt.plot(om / w0, abs(GT))
        plt.legend(legend)
