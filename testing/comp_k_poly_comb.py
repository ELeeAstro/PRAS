import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
from numpy.polynomial.chebyshev import chebval
from scipy.interpolate import BSpline

# Get parameters from yaml file
with open('input.yaml', 'r') as file:
  param = yaml.safe_load(file)['k_table_mix']

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

# Create arrays for lbl data
k_lbl = np.zeros((nb, ng))

# Read in lbl data
file = 'k_tables/'+'comb'+'_'+str(nb)+'_cs_table.txt'

with open(file, "r") as f:
  dum, dum = f.readline().split()
  dum = f.readline().split()
  dum = f.readline().split()

data = np.loadtxt(file,skiprows=3)
k_lbl[:,:] = data[:,2:]

# Create arrays for k tables
g_w_R = np.zeros(ng)
g_x_R = np.zeros(ng)
k_g_R = np.zeros((nb, ng))

# Read in RORR k_table data
file = 'k_tables/'+'comb'+'_'+str(nb)+'_k_table_RORR.txt'

with open(file, "r") as f:
  dum, dum = f.readline().split()
  g_x_R[:] = f.readline().split()
  g_w_R[:] = f.readline().split()

data = np.loadtxt(file,skiprows=3)
k_g_R[:,:] = data[:,2:]

# Create arrays for k tables
g_w_A = np.zeros(ng)
g_x_A = np.zeros(ng)
k_g_A = np.zeros((nb, ng))

# Read in AEE k_table data
file = 'k_tables/'+'comb'+'_'+str(nb)+'_k_table_EE.txt'

with open(file, "r") as f:
  dum, dum = f.readline().split()
  g_x_A[:] = f.readline().split()
  g_w_A[:] = f.readline().split()

data = np.loadtxt(file,skiprows=3)
k_g_A[:,:] = data[:,2:]

# Create array for polynomial data
n_samp = 1000
p_samp = np.linspace(0,1,n_samp)

p_c = np.zeros((nb, p_deg+1))
k_p = np.zeros((nb, n_samp))
k_p_g = np.zeros((nb, ng))

# Read in polynomial data and sample
file = 'p_tables/'+'comb'+'_'+str(nb)+'_PRAS_table.txt'

# Create arrays for k tables
g_w_p = np.zeros(ng)
g_x_p = np.zeros(ng)
k_g_p = np.zeros((nb, ng))

with open(file, "r") as f:
  dum, dum = f.readline().split()
  g_x_p[:] = f.readline().split()
  g_w_p[:] = f.readline().split()

data = np.loadtxt(file,skiprows=3)
k_g_p[:,:] = data[:,2:]

fig = plt.figure()

col = sns.color_palette("husl", nb)

for b in range(nb):
  plt.scatter(g_x_R[:],k_lbl[b,:],marker='x',color=col[b])
  plt.scatter(g_x_R[:],k_g_R[b,:],color=col[b])
  plt.scatter(g_x_A[:],k_g_A[b,:],color=col[b],alpha=0.6,marker='s')
  plt.plot(g_x_p[:],k_g_p[b,:],color=col[b])
  #plt.plot(p_samp[:],k_p[b,:],c=col[b])

plt.ylabel(r'$\sigma$ [cm$^{2}$ molecule$^{-1}$]',fontsize=16)
plt.xlabel(r'$g$',fontsize=16)

plt.yscale('log')

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('2_cs_compare.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

col = sns.color_palette("husl", nb)

for b in range(nb):
  plt.scatter(g_x_A[:],k_g_A[b,:]/k_lbl[b,:],color=col[b],marker='s')


plt.hlines(1.0,0.0,1.0,color='black')

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.ylabel(r'$\sigma_{\rm EE}$/$\sigma_{\rm Ref}$',fontsize=16)
plt.xlabel(r'$g$',fontsize=16)

plt.yscale('log')
plt.ylim(0.1,10)

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('2_EE_compare.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

col = sns.color_palette("husl", nb)

for b in range(nb):
  plt.scatter(g_x_R[:],k_g_R[b,:]/k_lbl[b,:],color=col[b],rasterized=True)

plt.hlines(1.0,0.0,1.0,color='black')

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.ylabel(r'$\sigma_{\rm RORR}$/$\sigma_{\rm Ref}$',fontsize=16)
plt.xlabel(r'$g$',fontsize=16)

plt.yscale('log')
plt.ylim(0.1,10)

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('2_RORR_compare.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

col = sns.color_palette("husl", nb)

for b in range(nb):
  plt.scatter(g_x_p[:],k_g_p[b,:]/k_lbl[b,:],marker='d',color=col[b],rasterized=True)

plt.hlines(1.0,0.0,1.0,color='black')

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.ylabel(r'$\sigma_{\rm PRAS}$/$\sigma_{\rm Ref}$',fontsize=16)
plt.xlabel(r'$g$',fontsize=16)

plt.yscale('log')
plt.ylim(0.1,10)

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('2_PRAS_compare.pdf',dpi=300,bbox_inches='tight')

plt.show()