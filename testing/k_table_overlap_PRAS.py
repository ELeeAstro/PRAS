### Perform the RORR method ###
import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
from scipy.interpolate import BSpline, make_lsq_spline
from numpy.polynomial.chebyshev import chebval, Chebyshev
from numpy.polynomial.legendre import leggauss
import time

from scipy.stats import qmc

from aux import rescale

# Get parameters from yaml file
with open('input.yaml', 'r') as file:
  param = yaml.safe_load(file)['k_table_mix']

nsp = param['nsp']
sp = param['sp']
wl_file = param['wl_file']
p_deg = param['p_deg']
n_samp = param['n_samp']
ng = param['ng']
VMR = param['VMR']
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

# Create arrays for poly tables
nt = p_deg + 4
t = np.zeros((nsp, nt))
k_pc = np.zeros((nsp, nb, p_deg+4))

# Read in poly_table data
for n in range(nsp):
  file = 'p_tables/'+sp[n]+'_'+str(nb)+'_poly_table.txt'

  with open(file, "r") as f:
    dum, dum = f.readline().split()
    t = f.readline().split()

  data = np.loadtxt(file,skiprows=2)
  k = 3
  k_pc[n,:,:] = data[:,2:]

### Start Polynomial reconstruction method ###

k_g = np.zeros((nb,ng))

rng1 = np.random.default_rng(seed=123)  # Gas 1

k_samp = np.zeros((nsp,n_samp))

splines = [[None for b in range(nb)] for n in range(nsp)]
for n in range(nsp):
  for b in range(nb):
    splines[n][b] = BSpline(t[:], k_pc[n,b,:], k)


Ktot = np.zeros(n_samp)
Ktot_flat = np.zeros(n_samp)
g = np.linspace(0.0, 1.0, n_samp)

sampler = qmc.LatinHypercube(d=1, scramble=True, strength=1, rng=rng1)
#sampler = qmc.Halton(d=1, scramble=True, rng=rng1)
#sampler = qmc.Sobol(d=1, scramble=True, rng=rng1)

start = time.time()

for b in range(nb):

  for n in range(nsp):
    #g1 = rng1.random(n_samp)
    g1 = sampler.random(n=n_samp)[:,0]

    k_samp[n,:] = VMR[n] * 10.0**splines[n][b](g1[:])
  
  Ktot[:] = np.maximum(np.sum(k_samp[:,:],axis=0),1e-44)
  Ktot_flat[:] = np.sort(Ktot[:])
  k_g[b,:] = 10.0**np.interp(gx_scal, g, np.log10(Ktot_flat[:]))

end = time.time()

print(end - start)

# Create a new file for output
fout = open('p_tables/'+'comb'+'_'+str(nb)+'_PRAS_table.txt','w')

fout.write(str(nb) + ' ' + str(ng) + '\n')

fout.write(" ".join(str(g) for g in gx_scal[:]) + '\n')
fout.write(" ".join(str(g) for g in gw_scal[:]) + '\n')

# Output the k-coefficient line to file
for b in range(nb):
  fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in k_g[b,:]) + '\n')

fout.close()

fig = plt.figure()

g = np.linspace(0.0,1.0,n_samp)

for b in range(nb):
  #spline_comb = BSpline(t, k_pc_comb[b,:], k)
  #plt.plot(g[:],10.0**spline_comb(g))
  #plt.plot(g, 10.0**chebval(g, k_pc_comb[b,:]))
  plt.scatter(gx_scal[:],k_g[b,:])

plt.yscale('log')

#plt.savefig('poly_comb.png',dpi=300,bbox_inches='tight')

plt.show()