import scipy.io
import numpy as np
from matplotlib.mlab import *
import matplotlib.pyplot as plt
from skimage.draw import line, polygon, circle, ellipse


def plot_masked(data, vertices , offset):
    """Plots the image masked outside of a circle using masked arrays"""

    # Mask portions of the data array outside of the circle
    # fill polygon
    poly = np.array(vertices)
    img = np.zeros(data.shape, 'uint8')

    rr, cc = polygon(poly[:, 0], poly[:, 1], data.shape)
    img[rr, cc] = data[rr, cc] #- offset

    # Plot
    plt.figure()
    im = plt.imshow(img, interpolation = 'bilinear')
    plt.title('Fathom Five Marine National Park')
    plt.colorbar(im)

    plt.show()






def main():
    mat = scipy.io.loadmat('/home/bogdan/Documents/UofT/PhD/Data_Files/Toberymory_tides/DEM/Fathom_Five_Bathy.mat')
    data = mat['FF']
    fig = plt.figure()
    #extent = (-middle, middle, -middle, middle)
    #plt.imshow(data - 180, interpolation = 'nearest', extent = extent, origin = 'lower')
    offset = 179
    im = plt.imshow(data - offset, interpolation = 'spline16')
    #im = plt.imshow(data, interpolation = 'bilinear')

    fig.colorbar(im)
    plt.clim(-25, 25)


    levels = [176, 178, 180, 181 ]
    colors = ['k', 'r', 'b', 'g']
    linewidths = [1, 1, 1, 1]
    levels = [offset]
    colors = ['g']
    linewidths = [1]



    plt.contour(data, levels, colors = colors, linewidths = linewidths)
    #plt.contour(data - 1 , colors = ['b'], linewidths = [1])
    #plt.contour(data + 1 , colors = ['g'], linewidths = [1])

    plt.axis('off')



    x = [1619, 2198, 690, 305, 661, 1619]
    y = [1162, 429, 276, 505, 1605, 1162]
    plt. plot(x, y, linewidth = 2, color = 'r')
    plt.title('Elevation within Fathom Five relative to Lake Level', fontsize = 18)
    #axis off
    vertices = [(y[0], x[0]), (y[1], x[1]), (y[2], x[2]), (y[3], x[3]), (y[4], x[4])]
    plot_masked(data, vertices, offset)
    fig2 = plt.figure()

    #mask = plt. poly2mask(x1, y1, 1801, 2801);
    #imshow(mask)
    #plot(x1, y1, 'LineWidth', 3, 'Color', 'r');

    #Inside = FF.*mask;
    #imagesc(Inside)
    #colorbar

    #Inside2 = reshape(Inside, 1, 1801 * 2801);

    #sizeFF = sum(sum(mask)) % = 1481393

    #H = histc(Inside2, [50 156 166 175 250]);

    '''
    figure

    bar(H(1:4)/sizeFF)

    ylabel('Percentage of total area','Fontsize',18)
    %xlabel('Depth range','Fontsize',18)
    set(gca,'Fontsize',18)
    set(gca,'XTick',1:1:4)
    set(gca,'XTickLabel',{'>20m depth','10-20 m depth','0-10 depth','land'})
    title('Depth ranges as a fraction of total marine park','Fontsize',18)


    figure

    bar(H(1:3)/sum(H(1:3)))

    ylabel('Percentage of total marine area','Fontsize',18)
    %xlabel('Depth range','Fontsize',18)
    set(gca,'Fontsize',18)
    set(gca,'XTick',1:1:3)
    set(gca,'XTickLabel',{'>20m depth','10-20 m depth','0-10 depth'})
    title('Depth ranges as a fraction of marine area of FF park','Fontsize',18)

    '''

main()
