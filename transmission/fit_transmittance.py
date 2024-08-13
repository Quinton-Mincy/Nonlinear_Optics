import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the nk data
nkAlq = np.loadtxt('../Thin-Film-Interference-measurements/nk_Alq3.txt', skiprows=0)
nptcda1 = np.loadtxt('../Thin-Film-Interference-measurements/n_PTCDA_Niranjala (2 par and 2 per).txt', skiprows=1)
nptcda2 = np.loadtxt('../Thin-Film-Interference-measurements/PTCDA (FDTD) nkparSi.txt', skiprows=0)
nptcda3 = np.loadtxt('../Thin-Film-Interference-measurements/PTCDA (FDTD).txt', skiprows=0)
nptcda4 = np.loadtxt('../Thin-Film-Interference-measurements/PTCDAInterpolatedn.txt', skiprows=0)
nbk7 = np.loadtxt('../Thin-Film-Interference-measurements/N-BK7.txt', skiprows=1)

# Initial guess for the parameters
lower_bound = 400
initial_guess = [2050, 1.6]
bound = ([1950,1],[2100,1.7])
# Path to your measurement file
measurement_file = 'csv_dir/Sample_5/Measurement_average.csv'  # Update this with the actual path to your measurement file

#oldest sample: initial_guess = [90, 1.5]
# Optimal d1g: 62.0280559817507, Optimal a: 1.6447441706168087
# Optimal d1g: 64.6252947994685, Optimal a: 1.4964734475231438
# Optimal d1g: 65.84947492428728, Optimal a: 1.8951255657282549
# Optimal d1g: 66.34402827453877, Optimal a: 1.6171933851368532
# Optimal d1g: 89.5142020383705, Optimal a: 1.4465415952821978

#Second oldest sample: initial_guess = [100, 1.5]
# Optimal d1g: 85.26870670911174, Optimal a: 2.3828410185761766
# Optimal d1g: 85.21524923570655, Optimal a: 2.5176604029697893
# Optimal d1g: 85.27987216175751, Optimal a: 2.427054573584272
# Optimal d1g: 85.23089383536339, Optimal a: 2.4020685559680537
# Optimal d1g: 85.24790533681104, Optimal a: 2.4572718863934906


#Newest Sample:  initial_guess = [1100, 1.5]
# Optimal d1g: 1060.2408209589973, Optimal a: 1.496054751274879
# Optimal d1g: 1070.4483279682481, Optimal a: 1.4076380464356844
# Optimal d1g: 1067.9782005219288, Optimal a: 1.3688968462043607
# Optimal d1g: 1070.557819891184, Optimal a: 1.4496589462261493
# Optimal d1g: 1073.1324449959277, Optimal a: 1.3591713245162795


# Interpolation function
def nkinterp(ndata, kdata, wldata, wl, wlkdata=np.zeros(1)):
    if len(wlkdata) == 1:
        wlkdata = wldata
    ntil = np.interp(wl, wldata, ndata) + 1.0j * np.interp(wl, wlkdata, kdata)
    return ntil

# Define functions for Fresnel coefficients and M matrices
def rij(ni, nj):
    return (ni - nj) / (ni + nj)

def tij(ni, nj):
    return 2 * ni / (ni + nj)

def delta(ni, di, wl):
    return np.pi * 2 * ni * di / wl

def dot(A, B):
    C = np.zeros_like(A)
    for i in range(A.shape[2]):
        C[:, :, i] = A[:, :, i].dot(B[:, :, i])
    return C

def Mij(ni, nj, dj, wl):
    return 1 / tij(ni, nj) * np.asarray([
        [np.exp(-1j * delta(nj, dj, wl)), rij(ni, nj) * np.exp(1j * delta(nj, dj, wl))],
        [rij(ni, nj) * np.exp(-1j * delta(nj, dj, wl)), np.exp(1j * delta(nj, dj, wl))]
    ])

def M(d, n, wl, layers):
    Mtot = np.zeros((2, 2, len(wl)))
    for i in range(len(layers) - 2):
        if i == 0:
            Mtot = Mij(n[layers[i]], n[layers[i + 1]], d[i + 1], wl)
        else:
            Mtot = dot(Mtot, Mij(n[layers[i]], n[layers[i + 1]], d[i + 1], wl))
    rback = rij(n[layers[-2]], n[layers[-1]])
    rbmat = np.asarray([[np.ones_like(rback), rback],
                        [rback, np.ones_like(rback)]])
    Mtot = dot(Mtot, rbmat)
    return 1 / tij(n[layers[-2]], n[layers[-1]]) * Mtot

# Define the theoretical model for transmittance
def theoretical_transmission(wl, d1g, a, n0, n1, n2, n3, layers):
    d = [0, d1g, 0]
    Mfb = M(d, [n0, n1, n2, n3], wl, layers)
    transmittance = np.abs(1 / Mfb[0, 0, :]) ** 2 / a
    return transmittance

# Load measurement data
def process_file(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    wavelengths = data[:, 0]
    intensities = data[:, 1]
    mask = wavelengths >= lower_bound  # Only include wavelengths >= 400 nm
    wavelengths = wavelengths[mask]
    intensities = intensities[mask]
    return wavelengths, intensities


measure_wavelengths, measure_intensities = process_file(measurement_file)

# Interpolate your measurement data to match the wl range
wl = np.linspace(lower_bound, 900, 3000)
measure_intensities_interp = np.interp(wl, measure_wavelengths, measure_intensities)

# Define layers and refractive indices
n0 = 1.0004 * wl / wl
n1 = nkinterp(nptcda4[:, 1], nptcda4[:, 2], nptcda4[:, 0], wl)
n2 = nkinterp(nkAlq[:, 1], nkAlq[:, 2], nkAlq[:, 0], wl)
n3 = nkinterp(nbk7[0:101, 1], nbk7[102:125, 1], nbk7[0:101, 0] * 1000, wl, nbk7[102:125, 0] * 1000)
layers = [3, 2, 0]  # The material type of each sequential layer

# Define the fitting function
def fitting_function(wl, d1g, a):
    return theoretical_transmission(wl, d1g, a, n0, n1, n2, n3, layers)

# Optimize the parameters
optimal_params, covariance = curve_fit(fitting_function, wl, measure_intensities_interp, p0=initial_guess,bounds=bound)

d1g_opt, a_opt = optimal_params
print(f"Optimal d1g: {d1g_opt}, Optimal a: {a_opt}")

# Calculate the transmittance with the optimized parameters
theor_transmittance_opt = fitting_function(wl, d1g_opt, a_opt)

# Plot the results
fig, ax = plt.subplots()

# Plot the measured data (noisy data)
ax.plot(measure_wavelengths, measure_intensities, 'b.', label='Measured Data')


# Plot the fitted transmittance (fitted model)
ax.plot(wl, theor_transmittance_opt, 'r-', label=f'Fitted Model\n(d1g={d1g_opt:.2f}, a={a_opt:.2f})')

# plot settings
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Transmittance')
ax.set_ylim(-.1, 1.2)
ax.legend()
plt.show()