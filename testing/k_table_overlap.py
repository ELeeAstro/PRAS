import numpy as np
import matplotlib.pylab as plt
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import Chebyshev
from itertools import product
from scipy.interpolate import make_lsq_spline, BSpline

# Avogradro's number
Avo = 6.02214076e23

# Rescale legendre coefficents
def rescale(x0,x1,gx,gw):
  gx = (x1 - x0)/2.0 * gx + (x1 + x0)/2.0
  gw = gw * (x1 - x0) / 2.0
  return gx, gw

# Species names
sp = ['H2O','CH4']
nsp = len(sp)

# Species molecular weights
mw = [18.01528,16.0425]

# Species VMR
VMR = [4.8e-3,4.1e-3]

# Name of files
fnames = ['cs/Out_00000_42000_01000_p000.bin','cs/Out_00000_12000_01000_p000.bin']

# File wavenumber end and start
wne = [42000.0,12000.0]
wns = 0.0
dwn = 0.01 # spacing in cm-1

# Get wn grid for each species cross section table
nwn = (wne[0] - wns)/dwn
wn_1 = np.arange(wns,wne[0],dwn)
nwn = (wne[1] - wns)/dwn
wn_2 = np.arange(wns,wne[1],dwn)

data_1 = np.zeros(len(wn_1))
data_2 = np.zeros(len(wn_2))

# Read cross section data
data_1 = np.fromfile(fnames[0],dtype=np.float32, count=-1)  * mw[0] / Avo
data_2 = np.fromfile(fnames[1],dtype=np.float32, count=-1)  * mw[1] / Avo

# Generate legendre coefficents
ng = 16
gx = np.zeros(ng)
gw = np.zeros(ng)
gx_scal = np.zeros(ng)
gw_scal = np.zeros(ng)

gx[:], gw[:] = leggauss(ng)

g_split_point = 0.90
g_split = False
# Rescale between 0 and 1
if (g_split == False):
  gx_scal[:], gw_scal[:] = rescale(0.0,1.0,gx[:],gw[:])
else:
  ngh = int(ng/2)
  gx[0:ngh], gw[0:ngh] = leggauss(ngh)
  gx_scal[0:ngh], gw_scal[0:ngh] = rescale(0.0,g_split_point,gx[0:ngh],gw[0:ngh])
  gx[ngh:], gw[ngh:] = leggauss(ngh)
  gx_scal[ngh:], gw_scal[ngh:] = rescale(g_split_point,1.0,gx[ngh:],gw[ngh:])

# Wavelength start and end of bin (um)
wl_s = 8.70
wl_e = 20.0

# Wavenumber start and end of bin
wn_s = 1.0/(wl_e * 1e-4)
wn_e = 1.0/(wl_s * 1e-4)

# Get index of bin start and end in cross section file
idx_1 = np.searchsorted(wn_1,wn_s)
idx_2 = np.searchsorted(wn_1,wn_e)
i_1 = np.arange(0,idx_2-idx_1,1)
cs_1 = data_1[idx_1:idx_2]

idx_1 = np.searchsorted(wn_2,wn_s)
idx_2 = np.searchsorted(wn_2,wn_e)
i_2 = np.arange(0,idx_2-idx_1,1)
cs_2 = data_2[idx_1:idx_2]

# Sort cross sections
cs_1_sort = np.maximum(np.sort(cs_1),1e-45)
cs_2_sort = np.maximum(np.sort(cs_2),1e-45)

# Get k table for each species
x_1 = np.linspace(0.0,1.0,len(cs_1_sort),endpoint=True)
k_1 = 10.0**np.interp(gx_scal,x_1,np.log10(cs_1_sort))

x_2 = np.linspace(0.0,1.0,len(cs_2_sort),endpoint=True)
k_2 = 10.0**np.interp(gx_scal,x_2,np.log10(cs_2_sort))

# Make combined cross section tables for end checking
cs_comb = VMR[0] * cs_1  + VMR[1] * cs_2
cs_comb_sort = np.maximum(np.sort(cs_comb),1e-45)
i_comb = np.arange(0,len(cs_comb_sort),1)
x_comb = np.linspace(0.0,1.0,len(cs_comb_sort),endpoint=True)
k_comb = 10.0**np.interp(gx_scal,x_comb,np.log10(cs_comb_sort))

# Usual ROM method

chi1, chi2 = VMR
ng = len(k_1)

# Build the ROM tensor product
k_rom_matrix = np.zeros((ng, ng))
w_rom_matrix = np.zeros((ng, ng))

for i in range(ng):
    for j in range(ng):
        k_rom_matrix[i, j] = chi1 * k_1[i] + chi2 * k_2[j]
        w_rom_matrix[i, j] = gw_scal[i] * gw_scal[j]

# Flatten into 1D array
k_rom_flat = k_rom_matrix.ravel()
w_rom_flat = w_rom_matrix.ravel()

# Sort k-values and associated weights
sort_idx = np.argsort(k_rom_flat)
k_rom_sorted = np.maximum(k_rom_flat[sort_idx], 1e-45)  # prevent log(0)
w_rom_sorted = w_rom_flat[sort_idx]

# Compute cumulative g = sum of weights
g_rom = np.cumsum(w_rom_sorted)
g_rom /= g_rom[-1]  # Normalize to [0, 1]

# Interpolate to your standard gx_scal points
k_rom_combined = 10**np.interp(gx_scal, g_rom, np.log10(k_rom_sorted))

# Below section for attempt to mix species from k-table alone

from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.legendre import Legendre

# Fit log-cross sections (as before)
deg = 24
g = np.linspace(0,1,len(cs_1_sort))
p1 = Chebyshev.fit(g, np.log10(cs_1_sort), deg, domain=[0, 1]).convert()
#p1 = Legendre.fit(g, np.log10(cs_1_sort), deg, domain=[0, 1]).convert()
g = np.linspace(0,1,len(cs_2_sort))
p2 = Chebyshev.fit(g, np.log10(cs_2_sort), deg, domain=[0, 1]).convert()
#p2 = Legendre.fit(g, np.log10(cs_2_sort), deg, domain=[0, 1]).convert()

# Fit polynomial to data
# g1 = np.linspace(0.0,1.0,len(cs_1_sort))
# k = 3
# n_internal_knots = 64
# t_internal = np.linspace(g1[1], g1[-2], n_internal_knots)
# t1 = np.concatenate(([0] * (k + 1), t_internal, [1] * (k + 1)))

# g2 = np.linspace(0.0,1.0,len(cs_2_sort))
# k = 3
# n_internal_knots = 64
# t_internal = np.linspace(g2[1], g2[-2], n_internal_knots)
# t2 = np.concatenate(([0] * (k + 1), t_internal, [1.1] * (k + 1)))

# # Build spline
# spl_1 = make_lsq_spline(g1, np.log10(cs_1_sort), t1, k=k)
# spl_2 = make_lsq_spline(g2, np.log10(cs_2_sort), t2, k=k)

# c1 = spl_1.c[:]
# c2 = spl_2.c[:]

# spline_1 = BSpline(t1, c1, 3)
# spline_2 = BSpline(t2, c2, 3)

def k1(g):
    return 10**p1(g)

def k2(g):
    return 10**p2(g)

ngrid = 1000
g1 = np.linspace(0, 1, ngrid)
g2 = np.linspace(0, 1, ngrid)
G1, G2 = np.meshgrid(g1, g2)

print(len(G1), np.shape(G1), G1)

K1 = k1(G1)
K2 = k2(G2)

#K1 = 10.0**spline_1(G1)
#K2 = 10.0**spline_2(G2)

Ktot = VMR[0] * K1 + VMR[1] * K2

print(len(Ktot), np.shape(Ktot))

# Flatten and sort
Ktot_flat = np.sort(Ktot.ravel())
Ktot_flat = np.maximum(Ktot_flat, 1e-45)  # avoid log(0)

# Get cumulative distribution
g_comb = np.linspace(0, 1, len(Ktot_flat))

# Interpolate log k to Gauss points
k_polyconv = 10.0**np.interp(gx_scal, g_comb, np.log10(Ktot_flat))

# Plot all cross section and k-table results to check method works

fig = plt.figure()

plt.plot(wn_1,data_1)
plt.plot(wn_2,data_2)

plt.yscale('log')

fig = plt.figure()

plt.plot(i_1,cs_1)
plt.plot(i_2,cs_2)
plt.plot(i_comb,cs_comb)

plt.yscale('log')

fig = plt.figure()

plt.plot(i_1,cs_1_sort)
plt.plot(i_2,cs_2_sort)
plt.plot(i_comb,cs_comb_sort)

plt.yscale('log')

fig = plt.figure()

plt.scatter(gx_scal,k_1, marker='s')
plt.scatter(gx_scal,k_2, marker='s')
plt.scatter(gx_scal,k_comb, marker='s')
plt.scatter(gx_scal, k_polyconv, label='Fitted Poly Random Overlap')
plt.scatter(gx_scal,k_rom_combined,label='ROM method',marker='d')

plt.legend()

plt.yscale('log')

fig = plt.figure()

plt.scatter(gx_scal,k_comb/k_polyconv, marker='s')
plt.scatter(gx_scal,k_comb/k_rom_combined, marker='d')

plt.legend()

plt.show()
