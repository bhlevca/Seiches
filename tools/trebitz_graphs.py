'''
Created on Aug 25, 2013

@author: bogdan
'''
import numpy as np
import scipy as sp
import matplotlib.ticker as ticker
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import csv

def readFile(path_in, fname):
    # read Lake data
    filename = path_in + '/' + fname

    ifile = open(filename, 'rb')
    reader = csv.reader(ifile, delimiter = ',', quotechar = '"')
    rownum = 0
    # 0.0545, 0.1297,LF ,0.8
    MouthArea = []
    RelativeAmplit = []
    Name = []
    Area = []
    printHeaderVal = False
    for row in reader:
        try:
            MouthArea.append(float(row[0]))
            RelativeAmplit.append(float(row[1]))
            Name.append(str(row[2]))
            Area.append(float(row[3]))

        except:
            pass
    # end for

    return [MouthArea, RelativeAmplit, Name, Area]
# end readFile


def plot_point_Array(title, xlabel, ylabel, x_arr, y_arr, p_arr, legend = None, linewidth = 0.6, \
                      ymax_lim = None, ymin_lim = None, xmax_lim = None, xmin_lim = None,
                      arrowprops = None, annotate = False, log = 'lin', hline = None):
    fig = plt.figure(facecolor = 'w', edgecolor = 'k')

    ax = fig.add_subplot(111)

    xmax = np.max(x_arr)
    ymax = np.max(y_arr)
    xmin = np.min(x_arr)
    ymin = np.min(y_arr)

    ticks = np.arange(ymin_lim, ymax_lim, (ymax_lim - ymin_lim) / 4.)
    tickslab = np.array(ticks);
    for i in range(0, len(ticks)):
        tickslab[i] = str(ticks[i])

    ls = ['bo', 'g^', 'rs']
    if log == 'loglog':
        ax.loglog(x_arr, y_arr, ls[0], basex = 10)
        ax.set_yscale('log')
        ax.set_yticks(ticks)
        # ax.set_yticklabels(tickslab)
        ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
    elif log == 'logx':
        ax.plot(x_arr, y_arr, ls[0])
        ax.set_xscale('log')
        ax.set_yticks(ticks)

    else:
        ax.plot(x_arr, y_arr, ls[0])
        ax.set_yticks(ticks)


    if hline != None and xmin_lim != None and xmax_lim != None:
        # horizontal line from (70,100) to (70, 250)
        # ax.hlines([hline], xmin_lim, xmax_lim, lw = 2)
        eps = 0.000001
        t = np.linspace(xmin_lim + eps, xmax_lim, 100)
        xl = np.array(t);
        for i in range(0, len(xl)):
            xl[i] = hline
        ax.plot(t, xl)

    if annotate:
        for i in range(0, len(p_arr)):
            if arrowprops:
                ax.annotate(p_arr[i], (x_arr[i], y_arr[i]), xytext = (xmax / 8., ymax / 8.), textcoords = 'offset points', \
                            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0.5', color = 'red'), \
                            ha = 'left', va = 'center', bbox = dict(fc = 'white', ec = 'none'))
            else:
                ax.annotate(p_arr[i], (x_arr[i], y_arr[i]), xytext = (xmax / 8., ymax / 8.), textcoords = 'offset points', \
                            ha = 'left', va = 'center', bbox = dict(fc = 'white', ec = 'none'))

        # ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(True, 'minor')
    ax.grid(True)
    plt.ylabel(ylabel).set_fontsize(16)
    plt.xlabel(xlabel).set_fontsize(16)
    plt.title(title).set_fontsize(20)

    if legend != None:
        plt.legend(legend, loc = 2, numpoints = 1);
    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room fornumpy smoothing filter them
    if ymax_lim != None:
        plt.ylim(ymax = ymax_lim)
    else:
        plt.ylim(ymax = ymax + ymax / 8.)
    if ymin_lim != None:
        plt.ylim(ymin = ymin_lim)

    if xmax_lim != None:
        plt.xlim(xmax = xmax_lim)
    # else:
    #    plt.xlim(xmax = xmax + xmax / 8.)


    if xmin_lim != None:
        plt.xlim(xmin = xmin - xmin / 4.)  # xmin_lim)
    else:
        plt.xlim(xmin = xmin - xmin / 4.)
    plt.draw()
    plt.show()
# end



if __name__ == '__main__':
    path = "/home/bogdan/Documents/UofT/PhD/docear/projects/Papers-Written/Environmental_Fluid_Mechanics/support_data"
    filename = "trebitz_resp.txt"
    [MouthArea, RelativeAmplit, Name, Area] = readFile(path, filename)
    title = "Lake Superior embayments. Relative amplification vs. mouth area"
    ylabel = "Relative Amplitude (bay/lake)"
    xlabel = "Mouth Area m$^2$"
    plot_point_Array(title, xlabel, ylabel, MouthArea, RelativeAmplit, Area, legend = ["L. Sup. Emb."], linewidth = 0.6, ymax_lim = 2.0, ymin_lim = 0,
                     annotate = True, log = 'logx' , hline = 1, xmax_lim = 80, xmin_lim = 0)
