import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import BSpline

# Get parameters from yaml file
with open('input.yaml', 'r') as file:
  param = yaml.safe_load(file)['k_table_mix']

nsp = param['nsp']
sp = param['sp']
wl_file = param['wl_file']
ng = param['ng']
p_deg =  param['p_deg']

# Read in wavelength file and get wavenumber edges and reverse
data = np.loadtxt(wl_file)
wl_e = data[:,1]
wl_e = wl_e[::-1]
wn_e = 1.0/(wl_e*1e-4)

dwn = 0.01

nb = int(len(wn_e) - 1)

# Create arrays for k tables
g_w = np.zeros((nsp, ng))
g_x = np.zeros((nsp, ng))
k_g = np.zeros((nsp, nb, ng))

# Read in k_table data
for n in range(nsp):
  file = 'k_table/'+sp[n]+'_'+str(nb)+'_k_table.txt'

  with open(file, "r") as f:
    dum, dum = f.readline().split()
    g_x[n,:] = f.readline().split()
    g_w[n,:] = f.readline().split()

  data = np.loadtxt(file,skiprows=3)
  k_g[n,:,:] = 10.0**data[:,2:]

  #print(n, k_g)

# Create array for polynomial data
n_samp = 1000
p_samp = np.linspace(0,1,n_samp)

#p_c = np.zeros((nsp, nb, p_deg+4))
p_c = np.zeros((nsp, nb, p_deg+1))
k_p = np.zeros((nsp, nb, n_samp))

# Read in polynomial data and sample
for n in range(nsp):
  file = 'poly_table/'+sp[n]+'_'+str(nb)+'_poly_table.txt'

  with open(file, "r") as f:
    dum, dum = f.readline().split()
    t = f.readline().split()

  data = np.loadtxt(file,skiprows=2)

  k = 3

  p_c[n,:,:] = data[:,2:]

  for b in range(nb):
    k_p[n,b,:] = 10.0**chebval(p_samp, p_c[n,b,:])
    #spline = BSpline(t, p_c[n,b], k)
    #k_p[n,b,:] = 10.0**spline(p_samp)
    


for n in range(nsp):
  fig = plt.figure()

  for b in range(nb):
    plt.scatter(g_x[n,:],k_g[n,b,:])
    plt.plot(p_samp[:],k_p[n,b,:])

  plt.yscale('log')

plt.show()