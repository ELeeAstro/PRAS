### Perform the RORR method ###
import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
from numpy.polynomial.legendre import leggauss
import time

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
VMR = param['VMR']

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

wn = np.arange(0.0,wn_e[-1],dwn)
nwn = len(wn)

nb = int(len(wn_e) - 1)

cs_comb = np.zeros(nwn)

# Add all lbl files weighted by VMR

# Read in lbl data
for n in range(nsp):

  # Read in lbl data
  cs = np.fromfile('../cs/'+sp[n]+'/'+lbl_files[n],dtype=np.float32, count=-1) * mw[n] / Avo

  ncs = len(cs)

  if (ncs < nwn):
    cs_comb[:ncs] = cs_comb[:ncs] + cs[:] * VMR[n]
  else:
    cs_comb[:] = cs_comb[:] + cs[:nwn] * VMR[n]
#
# Find band indexes and sort

k = np.zeros((nb,ng))

# Create a new file for output
fout = open('k_tables/'+'comb'+'_'+str(nb)+'_cs_table.txt','w')

fout.write(str(nb) + ' ' + str(ng) + '\n')

fout.write(" ".join(str(g) for g in gx_scal[:]) + '\n')
fout.write(" ".join(str(g) for g in gw_scal[:]) + '\n')

start = time.time()

for b in range(nb):

  # Find index for bins
  idx_l = np.searchsorted(wn,wn_e[b])
  idx_h = np.searchsorted(wn,wn_e[b+1])

  # Extract cs in bin range as normal
  cs_b = cs_comb[idx_l:idx_h]

  cs_b = np.array(cs_b)
  ncs_b = len(cs_b)

  # Sort cs data in bin
  cs_b_sort = np.maximum(np.sort(cs_b),1e-40)
  cs_b_sort = np.log10(cs_b_sort)

  x = np.linspace(0.0,1.0,len(cs_b_sort),endpoint=True)
  k[b,:] = 10.0**np.interp(gx_scal,x,cs_b_sort)

  fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in k[b,:]) + '\n')

end = time.time()

print(end - start)

fout.close()


fig = plt.figure()

plt.plot(wn[:],cs_comb[:])

plt.yscale('log')


fig = plt.figure()

for b in range(nb):
  plt.scatter(gx_scal[:],k[b,:])

plt.yscale('log')

#plt.savefig('cs_comb.png',dpi=300,bbox_inches='tight')


plt.show()