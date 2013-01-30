from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# setup Lambert Conformal basemap.
# set resolution=None to skip processing of boundary datasets.
m = Basemap(width = 2000000, height = 1900000, projection = 'lcc',
            resolution = 'f', lat_1 = 43., lat_2 = 46, lat_0 = 44, lon_0 = -81.)
m.etopo()
plt.show()
