import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.chebyshev import Chebyshev, chebval
from scipy.interpolate import make_lsq_spline, BSpline

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
ng = 16
split = True
g_split = 0.9
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

n = 0

# Read in lbl data
lbl_files[0] = 'Out_00000_42000_01000_n800.bin'
cs = np.fromfile('../cs/H2O/'+lbl_files[n],dtype=np.float32, count=-1) * mw[n] / Avo

ncs = len(cs)

if (ncs < nwn):
  cs_comb[:ncs] = cs_comb[:ncs] + cs[:] 
else:
  cs_comb[:] = cs_comb[:] + cs[:nwn] 

k = np.zeros((ng))

b = 4

# Find index for bins
idx_l = np.searchsorted(wn,wn_e[b])
idx_h = np.searchsorted(wn,wn_e[b+1])

# Extract cs in bin range as normal
cs_b = cs_comb[idx_l:idx_h]

cs_b = np.array(cs_b)
ncs_b = len(cs_b)

# Sort cs data in bin
cs_b_sort = np.sort(cs_b)
cs_b_sort = np.log10(cs_b_sort)
cs_b_sort[:] = np.maximum(cs_b_sort[:],-44)

g = np.linspace(1e-12, 1.0 - 1e-12, len(cs_b_sort))
k[:] = 10.0**np.interp(gx_scal,g,cs_b_sort)

# Add repeat points to attempt flattening

g_ext = np.concatenate([[0.0]*3, g, [1.0]*3])
cs_b_ext = np.concatenate([[ cs_b_sort[0]]*3,  cs_b_sort, [cs_b_sort[-1]]*3])
p_deg = 20
p = Chebyshev.fit(g_ext, cs_b_ext, p_deg, domain=[0, 1]).convert().coef

kk = 3
p_deg = 16
t_internal_1 = np.linspace(0.001, 0.1, int(p_deg/4))
t_internal_2 = np.linspace(0.2, 0.8, int(p_deg/4))
t_internal_3 = np.linspace(0.90, 0.999, int(p_deg/2))
t_internal = np.concatenate((t_internal_1, t_internal_2,  t_internal_3))
t = np.concatenate(([0] * (kk + 1), t_internal, [1] * (kk + 1)))

print(t)
spl = make_lsq_spline(g_ext, cs_b_ext, t, k=kk, method='qr', check_finite=False)

# Sample polynomial and spline - here not randomly, but just to get structure
n_s = 1000
g1 = np.linspace(0,1,n_s,endpoint=True)

spline = BSpline(t[:], spl.c[:], 3)

k_sort = 10.0**cs_b_sort
k_C = 10.0**chebval(g, p)
k_S = 10.0**spline(g)

res_k = np.zeros(len(gx_scal))
res_sort = (k_sort - k_sort)/k_sort * 100.0
res_C = (k_sort - k_C)/k_sort * 100.0
res_S = (k_sort - k_S)/k_sort * 100.0

col = sns.color_palette("colorblind")

fig, axs = plt.subplots(2)

axs[0].scatter(gx_scal,k,label='G-L quad. nodes',color='black')
axs[0].plot(g,k_sort,label='Sorted x-sec.',color=col[0])
axs[0].plot(g,k_C,label='Chebyshev poly.',color=col[1])
axs[0].plot(g,k_S,ls='dashed',label='LSQ spline',color=col[2])


axs[0].set_ylabel(r'$\sigma$ [cm$^{2}$ molecule$^{-1}$]',fontsize=14)

axs[0].set_yscale('log')
axs[0].legend()


axs[0].set_ylim(1e-27,1e-17)

axs[1].scatter(gx_scal,res_k,color='black')
axs[1].plot(g,res_sort,color=col[0])
axs[1].plot(g,res_C,color=col[1])
axs[1].plot(g,res_S,ls='dashed',color=col[2])

axs[1].set_ylabel(r'residuals [%]',fontsize=14)

axs[1].set_xlabel(r'$g$',fontsize=14)

axs[1].set_ylim(-10,10)

plt.savefig('1_fit_res_ex.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

plt.plot(wn[idx_l:idx_h][::100],cs_b[::100],color='black')

plt.ylabel(r'$\sigma$ [cm$^{2}$ molecule$^{-1}$]',fontsize=16)
plt.xlabel(r'$\nu$ [cm$^{-1}$]',fontsize=16)

plt.xlim(2800,4100)

plt.ylim(1e-26,1e-17)

plt.yscale('log')

plt.tick_params(axis='both',which='major',labelsize=12)
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('1_full_opac.pdf',dpi=300,bbox_inches='tight')

plt.show()
