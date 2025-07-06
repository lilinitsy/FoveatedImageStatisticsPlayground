import random

import numpy as np
from scipy.stats import qmc
from scipy.fftpack import fft2, ifft2, fftshift, fftfreq


'''
def generate_coloured_noise(shape, rand_input, spectrum_func):
    """Generate 2D colored noise using a frequency-domain filter."""
    (ny, nx) = shape
    # Create frequency grid
    fx = np.fft.fftfreq(nx).reshape(1, nx)
    fy = np.fft.fftfreq(ny).reshape(ny, 1)
    f = np.sqrt(fx ** 2 + fy ** 2)
    f[0, 0] = 1e-6  # avoid division by zero

    # Create noise in frequency domain
    spectrum = spectrum_func(f)
    spectrum = spectrum * np.exp(1j * rand_input)

    # Convert back to spatial domain
    noise = np.real(ifft2(spectrum))
    
    # Normalize to 0-1 range
    noise -= noise.min()
    noise /= noise.max()
    
    return noise


# Basically a wrapper for consistency purposes
def generate_white_noise(rand_input):
    return rand_input
'''


def generate_coloured_noise(shape, spectrum_func):
    """Generate 2D colored noise using proper complex random amplitudes."""
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
    """Your original approach - phase only."""
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
    """Analyze the radial power spectrum of 2D noise."""
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