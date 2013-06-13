from numpy import clip, log10, array, kaiser
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import numpy

#frequency respose of a kaiser window
window = kaiser(51, 14)
plt.plot(window)
plt.title("Kaiser window")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()
A = fft(window, 2048) / 25.5
mag = abs(fftshift(A))
freq = numpy.linspace(-0.5, 0.5, len(A))
response = 20 * log10(mag)
response = clip(response, -100, 100)
plt.plot(freq, response)
plt.title("Frequency response of Kaiser window")
plt.ylabel("Magnitude [dB]")
plt.xlabel("Normalized frequency [cycles per sample]")
plt.axis('tight'); plt.show()

WINDOW = 5
data = [1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
extended_data = numpy.hstack([[data[0]] * (WINDOW - 1), data])
weightings = numpy.repeat(1.0, WINDOW) / WINDOW
smoothed = numpy.convolve(extended_data, weightings)[WINDOW - 1:-(WINDOW - 1)]
time = range(0, len(data))

plt.plot(time, data)
plt.plot(time, smoothed)
plt.show()
