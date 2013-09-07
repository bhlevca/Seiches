# # Shallow Water Chapter Recap
# This is an executable program that illustrates the statements
# introduced in the Shallow Water Chapter of "Experiments in MATLAB".
# You can access it with
#
# water_recap
# edit water_recap
# publish water_recap
#
# Related EXM programs
#
# waterwave
# # Finite Differences
# A simple example of the grid operations in waterwave.5
# # Create a two dimensional grid.

import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import animation

class ShallowWater(object):

    def __init__(self):
        m = 21;
        a = np.linspace(-1, 1, m)
        b = np.linspace(-1, 1, m)
        xx = np.array(a)
        yy = np.array(b)
        self.x, self.y = np.meshgrid(xx, yy)

        # # Indices in the four compass directions.
        a1 = np.array(range(1, m))
        a2 = np.array([m - 1])
        self.n = np.concatenate((a1, a2), axis = 1)
        self.e = self.n
        self.s = np.concatenate((np.array([0]), np.array(range(0, m - 1))), axis = 1)
        self.w = self.s;
        # # A relation parameter. Try other values.
        # Experiment with omega slightly greater than one.
        self.omega = 1

        self.fig = plt.figure()  # figsize = plt.figaspect(2.))
        self.ax = self.fig.gca(projection = '3d')

        # # The water drop function from waterwave.
        self.U = np.exp(-5 * (self.x ** 2 + self.y ** 2));
        self.maxU = np.max(self.U)

    def drawNow(self, heightR):
            self.surf.remove()
            self.surf = self.ax.plot_surface(
                self.x, self.y, heightR, rstride = 1, cstride = 1,
                cmap = cm.jet, linewidth = 0, antialiased = False)
            self.ax.set_zlim3d(0, self.maxU)
            plt.draw()  # redraw the canvas
            # time.sleep(1)

    def calculate(self):
        m = 21;
        # [x,y] = ndgrid(-1: 2/(m-1): 1);
        a = np.linspace(-1, 1, m)
        b = np.linspace(-1, 1, m)
        xx = np.array(a)
        yy = np.array(b)
        self.x, self.y = np.meshgrid(xx, yy)


        # # The water drop function from waterwave.
        U = np.exp(-5 * (self.x ** 2 + self.y ** 2));
        # # Surf plot of the function
        # h = surf(x,y,U);
        # axis off
        # ax = axis;

        plt.ion()

        fig = plt.figure()  # (figsize = plt.figaspect(2.))
        self.ax = fig.gca(projection = '3d')

        self.maxU = np.max(U)
        self.ax.set_zlim3d(0, self.maxU)
        self.surf = self.ax.plot_surface(self.x, self.y, U)

        plt.draw()
        self.ax.grid(b = True, which = 'both')

        # interactive mode


        # # Colormap
        # c = (37:100)';
        # cyan = [0*c c c]/100;
        # colormap(cyan)
        # pause(1)

        # # Indices in the four compass directions.
        a1 = np.array(range(1, m))
        a2 = np.array([m - 1])
        n = np.concatenate((a1, a2), axis = 1)
        e = n
        s = np.concatenate((np.array([0]), np.array(range(0, m - 1))), axis = 1)
        w = s;
        # # A relation parameter. Try other values.
        # Experiment with omega slightly greater than one.
        omega = 1

        # # Relax.
        # Repeatedly replace grid values by relaxed average of four neighbors.
        tfinal = 500
        for t in range(0, tfinal):  # t = 1:tfinal
            # print t
            U = (1 - omega) * U + omega * (U[n, :] + U[:, e] + U[s, :] + U[:, w]) / 4;
            # plt.set()  # set(h,'zdata',U);
            # axis(ax)
            # drawnow
            self.ax.view_init(15, 0.1 * t)
            self.drawNow(U)

        plt.show()


    def init(self):

        zero = np.zeros(self.U.shape)
        zero.fill(-1)
        # self.ax.grid(b = True, which = 'both')
        self.surf = self.ax.plot_surface(self.x, self.y, zero)
        self.ax.set_zlim((0, self.maxU))
        self.ax.set_zlim3d((0, self.maxU))
        self.fig.canvas.draw()
        return [self.surf]

    def animate(self, i):
        self.U = (1 - self.omega) * self.U + self.omega * (self.U[self.n, :] + self.U[:, self.e] + self.U[self.s, :] + self.U[:, self.w]) / 4;

        self.surf.remove()

        # self.ax.cla()
        self.ax.set_zlim3d(0, self.maxU)
        self.ax.set_xlim((-1, 1))
        self.ax.set_ylim((-1, 1))
        self.ax.set_zlim((0, self.maxU))
        self.surf = self.ax.plot_surface(
                self.x, self.y, self.U, rstride = 1, cstride = 1,
                cmap = cm.jet, linewidth = 0, antialiased = False)

        self.ax.view_init(15, 0.1 * i)

        self.fig.canvas.draw()
        return [self.surf]

    def calculate2(self):

        # instantiate the animator.
        anim = animation.FuncAnimation(self.fig, self.animate, init_func = self.init,
                               frames = 500, interval = 2, blit = True)


        plt.show()


if __name__ == '__main__':
    sw = ShallowWater()
    # sw.calculate()
    sw.calculate2()
