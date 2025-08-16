import random

import numpy as np
from scipy.stats import qmc
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq
import matplotlib.pyplot as plt


def generate_coloured_noise(shape, spectrum_func):
	(ny, nx) = shape

	# Create frequency grid
	fx = fftfreq(nx).reshape(1, nx)
	fy = fftfreq(ny).reshape(ny, 1)
	f = np.sqrt(fx ** 2 + fy ** 2)
	f[0, 0] = 1e-6  # avoid division by zero

	# Generate random complex amplitudes (this is key!)
	rand_real = np.random.randn(ny, nx)
	rand_imag = np.random.randn(ny, nx)
	rand_complex = rand_real + 1j * rand_imag

	# Apply spectrum shaping
	spectrum = spectrum_func(f) * rand_complex

	# Convert back to spatial domain
	noise = np.real(ifft2(spectrum))

	return noise

def generate_phase_only_noise(shape, spectrum_func):
	(ny, nx) = shape

	fx = fftfreq(nx).reshape(1, nx)
	fy = fftfreq(ny).reshape(ny, 1)
	f = np.sqrt(fx ** 2 + fy ** 2)
	f[0, 0] = 1e-6

	# Only phase (your original approach)
	phase = np.random.rand(ny, nx) * 2 * np.pi
	spectrum = spectrum_func(f) * np.exp(1j * phase)

	noise = np.real(ifft2(spectrum))
	return noise

def analyze_power_spectrum(noise):
	fft_noise = fft2(noise)
	power = np.abs(fft_noise) ** 2

	# Create radial frequency bins
	ny, nx = noise.shape
	fx = fftfreq(nx).reshape(1, nx)
	fy = fftfreq(ny).reshape(ny, 1)
	f = np.sqrt(fx ** 2 + fy ** 2)

	# Bin the power spectrum radially
	f_flat = f.flatten()
	power_flat = power.flatten()

	# Remove DC component
	mask = f_flat > 0
	f_flat = f_flat[mask]
	power_flat = power_flat[mask]

	# Sort by frequency
	sort_idx = np.argsort(f_flat)
	f_sorted = f_flat[sort_idx]
	power_sorted = power_flat[sort_idx]

	return f_sorted, power_sorted


def measure_high_frequency_power(noise, cutoff = 0.3):
	fft_noise = np.fft.fft2(noise)
	power = np.abs(fft_noise)**2

	# Create frequency grid
	ny, nx = noise.shape
	fx = np.fft.fftfreq(nx).reshape(1, nx)
	fy = np.fft.fftfreq(ny).reshape(ny, 1)
	f = np.sqrt(fx**2 + fy**2)

	# Calculate power in high frequencies
	high_freq_mask = f > cutoff
	high_freq_power = np.sum(power[high_freq_mask])
	total_power = np.sum(power)

	return high_freq_power / total_power


def plot_2d_spectrum(noise, title=""):
	# Get the 2D FFT
	fft_noise = np.fft.fft2(noise)
	power_spectrum = np.abs(fft_noise)**2

	# Shift zero frequency to center for better visualization
	power_spectrum_shifted = np.fft.fftshift(power_spectrum)

	# Use log scale to see details
	log_power = np.log10(power_spectrum_shifted + 1e-10)

	plt.figure(figsize=(8, 6))
	plt.imshow(log_power, cmap='gray', origin='lower')
	plt.colorbar(label='Log Power')
	plt.title(f'2D Power Spectrum: {title}')
	plt.xlabel('Frequency X')
	plt.ylabel('Frequency Y')
	plt.show()



def plot_radial_spectrum(noise, title=""):
	# Your existing analyze_power_spectrum function works here
	f_sorted, power_sorted = analyze_power_spectrum(noise)

	plt.figure(figsize=(8, 6))
	plt.loglog(f_sorted, power_sorted, 'o-', alpha=0.7)
	plt.xlabel('Frequency')
	plt.ylabel('Power')
	plt.title(f'Radial Power Spectrum: {title}')
	plt.grid(True, alpha=0.3)
	plt.show()


def normalize_to_unit_range(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return (img - img_min) / (img_max - img_min)




def create_grayscale_ramp(shape):
	(height, width) = shape
	linear_space = np.linspace(0.0, 1.0, width, dtype = np.float32)
	ramp = np.tile(linear_space, (height, 1))
	return ramp


def create_gray_image(shape, brightness):
	image = np.full(shape, brightness)
	return image

def dither_with_noise(input_image, noise):
	output_image = (input_image > noise)
	return output_image

def dither_with_noise2(gray_image, noise_image):
	assert gray_image.shape == noise_image.shape, "Images must have same dimensions"

	mask = gray_image > noise_image
	result = np.where(mask, 1.0, 0.0)

	return result


def plot_two_ramps_vertical(topramp, bottomramp, toplabel, bottomlabel):
	plt.figure(figsize=(10, 4))
	plt.subplot(2, 1, 1)
	plt.imshow(topramp, cmap='gray', vmin=0.0, vmax=1.0)
	plt.title(toplabel)
	plt.axis('off')

	# Bottom subplot: the dithered image
	plt.subplot(2, 1, 2)
	plt.imshow(bottomramp, cmap='gray', vmin=0, vmax=1)
	plt.title(bottomlabel)
	plt.axis('off')