import numpy as np
import matplotlib.pylab as plt
import yaml
from numpy.polynomial.legendre import leggauss

from aux import rescale

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
ng = param['ng']
split = param['split']
g_split = param['g_split']


# Get legendre coefficients 
gx = np.zeros(ng)
gw = np.zeros(ng)
gx_scal = np.zeros(ng)
gw_scal = np.zeros(ng)

gx[:], gw[:] = leggauss(ng)

# Rescale between 0 and 1
if (split == False):
  gx_scal[:], gw_scal[:] = rescale(0.0,1.0,gx[:],gw[:])
else:
  ngh = int(ng/2)
  gx[0:ngh], gw[0:ngh] = leggauss(ngh)
  gx_scal[0:ngh], gw_scal[0:ngh] = rescale(0.0,g_split,gx[0:ngh],gw[0:ngh])
  gx[ngh:], gw[ngh:] = leggauss(ngh)
  gx_scal[ngh:], gw_scal[ngh:] = rescale(g_split,1.0,gx[ngh:],gw[ngh:])


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
  fout = open('k_tables/'+sp[n]+'_'+str(nb)+'_k_table.txt','w')

  fout.write(str(nb) + ' ' + str(ng) + '\n')

  fout.write(" ".join(str(g) for g in gx_scal[:]) + '\n')
  fout.write(" ".join(str(g) for g in gw_scal[:]) + '\n')

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

    # Get k coefficients for species
    x = np.linspace(0.0,1.0,ncs_b,endpoint=True)
    k = np.interp(gx_scal,x,cs_b_sort)

    # Output the k-coefficient line to file
    fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in k[:]) + '\n')

  fout.close()

