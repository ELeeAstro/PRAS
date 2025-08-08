import numpy as np
import yaml
import seaborn as sns
import matplotlib.pylab as plt
from numpy.polynomial.legendre import leggauss

from RT_opac import read_k_table, interp_k_table, mix_k_table_EE, mix_k_table_RORR, read_p_table, interp_p_table, mix_p_table_PRAS
from RT_flux import shortwave, longwave

import time

# Rescale legendre coefficents
def rescale(x0,x1,gx,gw):
  gx = (x1 - x0)/2.0 * gx + (x1 + x0)/2.0
  gw = gw * (x1 - x0) / 2.0
  return gx, gw

# Physical constants
R = 8.31446261815324e7
kb = 1.380649e-16
amu = 1.66053906892e-24

### Generate T-p profile ###


# Open parameter YAML file and read parameters for RT calculation
with open('RT_input.yaml', 'r') as file:
  param = yaml.safe_load(file)['RT_input']

# Now extract the parameters from the YAML file into local variables

# Give number of layers in 1D atmosphere - number of levels is nlay + 1
nlay = param['nlay']
nlev = nlay + 1

sp = param['sp']
sp_r = [r'OH',r'H$_{2}$O',r'CO',r'CO$_{2}$',r'CH$_{4}$',r'C$_{2}$H$_{2}$',r'NH$_{3}$',r'HCN']
nsp = len(sp)
grav = param['grav']
cp = param['cp']
T_int = param['T_int']

# Get top and bottom pressure layers in bar - convert to dyne
ptop = param['ptop'] * 1e6
pbot = param['pbot'] * 1e6

# Get pressure at levels (edges) - assume log-spaced
pe = np.logspace(np.log10(ptop),np.log10(pbot),nlev)

# Get pressure at layers using level spacing
pl = np.zeros(nlay)
pl[:] = (pe[1:] - pe[0:-1])/np.log(pe[1:]/pe[0:-1])

# Get T-p-VMR profile from VULCAN file

T_p_VMR_file = param['T_p_VMR_file']
file = '../T_P_VMR_input/'+T_p_VMR_file

f = open(file,'r')
h1 = f.readline().split()
sp_f = f.readline().split()[4:]
f.close()

data = np.loadtxt(file,skiprows=2)

pe_f = data[:,0]
Te_f = data[:,1]
mu_f = data[:,3]
VMR_f = data[:,4:]
nsp_f = len(VMR_f[0,:])

Te = np.zeros(nlev)
Tl = np.zeros(nlay)
mu = np.zeros(nlay)
VMR = np.zeros((nlay, nsp))

VMR_idx = []
for s in range(nsp-2):
  VMR_idx.append(sp_f.index(sp[s]))
  #print(VMR_idx[s], sp_f[VMR_idx[s]])

# Interpolate to pressure layers
for k in range(nlev):
  Te[k] = np.interp(pe[k],pe_f[::-1],Te_f[::-1])
for k in range(nlay):
  Tl[k] = np.interp(pl[k],pe_f[::-1],Te_f[::-1])
  mu[k] = np.interp(pl[k],pe_f[::-1],mu_f[::-1])
  for s in range(nsp-2):
    VMR[k,s] = 10.0**np.interp(pl[k],pe_f[::-1],np.log10(VMR_f[::-1,VMR_idx[s]]))

VMR[:,-2] = 1e-5 # Add Na VMR to array
VMR[:,-1] = 1e-6 # Add K VMR to array

# Atmospheric mass density
rho = np.zeros(nlay)
rho[:] = (pl[:]*mu[:]*amu)/(kb * Tl[:])

# Atmospheric number density
nd = np.zeros(nlay)
nd[:] = (pl[:])/(kb * Tl[:])

# for k in range(nlay):
#   print(k, pl[k],Tl[k], mu[k], VMR[k,:])

# Read in wavelength grid and number of bands
nb = param['nb']
wl_file = param['wl_file']

# Read edge wavelength file and get bin wavenumbers
wl_file = param['wl_file']
data = np.loadtxt(wl_file)
wl_e = data[:,1]
wl_e = wl_e[::-1]
wn_e = 1.0/(wl_e*1e-4)
nb = int(len(wn_e) - 1)

### Plot T_p_VMR profile for paper ###

# fig = plt.figure() # Start figure
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()
# colour = sns.color_palette('colorblind') 
# Tp = ax2.plot(Tl,pl/1e6,c='black',label=r'T-p',ls='dotted')
# for s in range(nsp):
#   ax1.plot(VMR[:,s],pl/1e6,c=colour[s],ls='solid',lw=2,label=sp_r[s])
# ax1.set_xlabel(r'VMR',fontsize=16)
# ax2.set_xlabel(r'$T$ [K]',fontsize=16)
# ax1.set_ylabel(r'$p$ [bar]',fontsize=16)
# ax1.tick_params(axis='both',which='major',labelsize=14)
# ax2.tick_params(axis='both',which='major',labelsize=14)
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.set_xlim(1e-8,1e-1)
# ax1.set_ylim(1e-6,1e2)
# ax2.set_xlim(500,3500)
# ax2.set_zorder(1)
# ax1.legend()
# plt.gca().invert_yaxis()
# plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
# plt.savefig('T_p_VMR.pdf',dpi=300,bbox_inches='tight')
# plt.show()
# quit()

### Get opacity structure of atmosphere ###
ng = param['ng']
split = param['split']
g_split = param['g_split']

# Calculate required Legendre points
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

mode = param['mode']



if (mode == 'PRAS' or mode == 'PRAS_2' or mode == 'PRAS_3'):


  p_tab_ext = param['p_tab_ext']
  p_deg = param['p_deg']
  n_samp = param['n_samp']

  print('-- Read --')

  read_p_table(nsp,sp,p_tab_ext)

  print('-- Interpolate --')

  pc_int = interp_p_table(nsp,nlay,Tl,pl)

  print('-- Mixing --')

  start = time.time()

  for j in range(100):

    print(j)

    cs_mix = mix_p_table_PRAS(nsp,nlay,nb,ng,n_samp,pc_int,VMR,gx_scal)

else:

  k_tab_ext = param['k_tab_ext']

  print('-- Read --')

  read_k_table(nsp,sp,k_tab_ext)

  print('-- Interpolate --')

  kc_int = interp_k_table(nsp,nlay,Tl,pl)

  print('-- Mixing --')

  start = time.time()

  for j in range(100):

    print(j)

    if (mode == 'PM'):
      cs_mix = mix_k_table_PM()
    elif (mode == 'EE'):
      cs_mix = mix_k_table_EE(nsp,nlay,nb,ng,kc_int,VMR,gx_scal,gw_scal,pe,grav,nd,rho)
    elif (mode == 'RORR' or mode == 'RORR_2' or mode == 'RORR_3' or mode == 'RORR_4'):
      cs_mix = mix_k_table_RORR(nsp,nlay,nb,ng,kc_int,VMR,gx_scal,gw_scal)

end = time.time()

print(end - start)

quit()

# Output k-coefficients for comparisons
# file = 'results/'+mode+'_vert_k.txt'
# f = open(file,'w')
# f.write(str(nlay) + ' ' + str(nb) + ' ' + str(ng) + '\n')
# for k in range(nlay):
#   for b in range(nb):
#     f.write(" ".join(str(g) for g in cs_mix[k,b,:]) + '\n')
# f.close()

### Start flux calculation ###

print('-- Flux Calculation --')

# Convert mixed cross-sections [cm^2 molecule-1] to mass units [cm^2 g-1]
k_mix = np.zeros((nlay,nb,ng))
for k in range(nlay):
  for b in range(nb):
    k_mix[k,b,:] = cs_mix[k,b,:] * nd[k] / rho[k]

# Find optical depth of atmosphere in each band and g value
mu_z = param['mu_z']

tau = np.zeros((nlev, nb, ng))
for k in range(nlev-1):
  for b in range(nb):
    tau[k+1,b,:] = tau[k,b,:] + (k_mix[k,b,:] * (pe[k+1] - pe[k]))/grav

F_s_u, F_s_d, F_s_net = shortwave(nlev, nb, ng, tau, mu_z, gw_scal)
F_l_u, F_l_d, F_l_net, OLR = longwave(nlev, nb, ng, Te, T_int, tau, wn_e, gw_scal)

print('-- Output --')

# Output OLR  #
# file = 'results_OLR/'+mode+'_vert_OLR.txt'
# f = open(file,'w')
# f.write(str(nb) + '\n')
# for b in range(nb):
#   f.write(str(b) + ' ' + str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + str(OLR[b]) + '\n')
# f.close()
# quit()

# Output vertical fluxes
file = 'results/'+mode+'_vert_flux.txt'
f = open(file,'w')
f.write(str(nlev) + '\n')
for k in range(nlev):
  f.write(str(F_s_u[k]) + ' ' + str(F_s_d[k]) + '  ' + str(F_l_u[k]) + ' ' + str(F_l_d[k]) + '\n')
f.close()

# Output heating rates #
F_net = np.zeros(nlev)
F_net[:] = F_l_net[:] + F_s_net[:]
dT_rad_sw = np.zeros(nlay)
dT_rad_sw[:] = (grav/cp) * (F_s_net[1:]-F_s_net[0:-1])/(pe[1:]-pe[0:-1])
dT_rad_lw = np.zeros(nlay)
dT_rad_lw[:] = (grav/cp) * (F_l_net[1:]-F_l_net[0:-1])/(pe[1:]-pe[0:-1])

file = 'results/'+mode+'_vert_heating.txt'
f = open(file,'w')
f.write(str(nlay) + '\n')
for k in range(nlay):
  f.write(str(k) + ' ' + str(pl[k]/1e6) + ' ' + str(Tl[k]) + ' ' + str(dT_rad_sw[k]) +  ' ' + str(dT_rad_lw[k]) + '\n')
f.close()

