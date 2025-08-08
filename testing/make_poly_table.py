import numpy as np
import matplotlib.pylab as plt
import yaml
from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import make_lsq_spline

# Avogradro's number
Avo = 6.02214076e23


# Get parameters from yaml file
with open('input.yaml', 'r') as file:
  param = yaml.safe_load(file)['k_table_mix']

# Extract appropriate parameters
nsp = param['nsp']
sp = param['sp']
mw = param['mw']
lbl_files = param['lbl_files']
wl_file = param['wl_file'] 
wn_f = param['wn_f']
p_deg =  param['p_deg']

# Read in wavelength file and get wavenumber edges and reverse
data = np.loadtxt(wl_file)
wl_e = data[:,1]
wl_e = wl_e[::-1]
wn_e = 1.0/(wl_e*1e-4)

dwn = 0.01

nb = int(len(wn_e) - 1)

### Start creating k tables for each species ###

for n in range(nsp):

  # Read in lbl data
  cs = np.fromfile('../cs/'+sp[n]+'/'+lbl_files[n],dtype=np.float32, count=-1) * mw[n] / Avo
  wn_l = np.arange(0.0,wn_f[n],dwn)

  # Create a new file for output
  fout = open('p_tables/'+sp[n]+'_'+str(nb)+'_poly_table.txt','w')

  fout.write(str(nb) + ' ' + str(p_deg) + '\n')

  # Place internal knots safely inside (g[k], g[-k-1])
  # Degree of spline
  k = 3
  t_internal_1 = np.linspace(0.001, 0.1, int(p_deg/4))
  t_internal_2 = np.linspace(0.2, 0.8, int(p_deg/4))
  t_internal_3 = np.linspace(0.90, 0.999, int(p_deg/2))
  t_internal = np.concatenate((t_internal_1, t_internal_2,  t_internal_3))
  t = np.concatenate(([0] * (k + 1), t_internal, [1] * (k + 1)))

  fout.write(" ".join(str(g) for g in t[:]) + '\n')

  for b in range(nb):

    # Find index for bins
    idx_l = np.searchsorted(wn_l,wn_e[b])
    idx_h = np.searchsorted(wn_l,wn_e[b+1])

    print(n, b, idx_l, idx_h)

    # If outside file range wavenumber range we have to make some adjustments
    if (wn_e[b+1] > wn_f[n]):
      # Find number of wavenumbers that are missing from band
      nwn = len(np.arange(wn_f[n],wn_e[b+1],dwn))
      # Make array of small numbers
      cs_pad = np.zeros(nwn)
      cs_pad[:] = 1e-44

      # Extract cs in bin range
      cs_b_1 = cs[idx_l:idx_h]

      # Append padded values to array
      cs_b = np.concatenate((cs_pad, cs_b_1), axis=0)

    else:
      # Extract cs in bin range as normal
      cs_b = cs[idx_l:idx_h]

    cs_b = np.array(cs_b)
    ncs_b = len(cs_b)

    # Sort cs data in bin
    cs_b_sort = np.maximum(np.sort(cs_b),1e-44)
    cs_b_sort = np.log10(cs_b_sort)

    # Fit polynomial to data
    g = np.linspace(1e-9, 1.0 - 1e-9, ncs_b)

    # Add repeat points to attempt flattening
    g_ext = np.concatenate([[0.0]*3, g, [1.0]*3])
    cs_b_ext = np.concatenate([[ cs_b_sort[0]]*3,  cs_b_sort, [cs_b_sort[-1]]*3])

    #p = Chebyshev.fit(g_ext, cs_b_ext, p_deg, domain=[0, 1]).convert()

    # Build spline
    spl = make_lsq_spline(g_ext, cs_b_ext, t, k=k, method='qr', check_finite=False)

    # Output the polynomial-coefficients line to file
    fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in spl.c[:]) + '\n')
    #fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in p[:]) + '\n')


