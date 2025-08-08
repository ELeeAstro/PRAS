import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.interpolate import BSpline, RectBivariateSpline
import matplotlib.pylab as plt
from scipy.stats import qmc

# Global list to hold each species' table
ptb = []
ktb = []

@dataclass
class p_tab:
  sp: str
  n_T: int
  n_p: int
  n_b: int
  n_kn: int
  p_deg: int

  T: List[float] = field(default_factory=list)
  p: List[float] = field(default_factory=list)

  wl_e: List[float] = field(default_factory=list)
  wn_e: List[float] = field(default_factory=list)

  kn: List[float] = field(default_factory=list)

  pc: Optional[np.ndarray] = None  # Shape: (n_T, n_p, n_b, n_kn)

  pc_interp: Optional[List[List[RectBivariateSpline]]] = None 

@dataclass
class k_tab:
  sp: str
  n_T: int
  n_p: int
  n_b: int
  n_g: int

  T: List[float] = field(default_factory=list)
  p: List[float] = field(default_factory=list)

  wl_e: List[float] = field(default_factory=list)
  wn_e: List[float] = field(default_factory=list)

  gx: List[float] = field(default_factory=list)
  gw: List[float] = field(default_factory=list)

  kc: Optional[np.ndarray] = None  # Shape: (n_T, n_p, n_b, n_kn)

  kc_interp: Optional[List[List[RectBivariateSpline]]] = None 


def read_p_table(nsp, sp, p_tab_ext):

  for n in range(nsp):
    fname = f'../kp_tables/{p_tab_ext}/{sp[n]}_{p_tab_ext}.txt'
    with open(fname, 'r') as f:

      line = f.readline().split()
      species_name = str(line[0])

      print(n, species_name)
      if species_name != sp[n]:
        print('Different species p-table read in order:', species_name, sp[n])

      line = f.readline().split()
      n_T = int(line[0])
      n_p = int(line[1])
      n_b = int(line[2])
      n_kn = int(line[3])
      p_deg = 3

      T = [float(i) for i in f.readline().split()]
      p_vals = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line
      wl_e = [float(i) for i in f.readline().split()]
      wn_e = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line
      kn = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line

      # Read the 4D pc array
      pc = np.zeros((n_T, n_p, n_b, n_kn-4))
      for i in range(n_T):
        for j in range(n_p):
          for b in range(n_b):
            line = f.readline().split()
            pc[i, j, b, :] = [float(m) for m in line]

      pc_interp = [[RectBivariateSpline(np.log10(T), np.log10(p_vals), pc[:, :, b, r])
        for r in range(n_kn - 4)]
          for b in range(n_b)]

    # Create the p_tab object and append to global list
    ptb.append(
      p_tab(
      sp=species_name,
      n_T=n_T,
      n_p=n_p,
      n_b=n_b,
      n_kn=n_kn,
      p_deg=p_deg,
      T=T,
      p=p_vals,
      wl_e=wl_e,
      wn_e=wn_e,
      kn=kn,
      pc=pc,
      pc_interp=pc_interp))

  return


def interp_p_table(nsp,nlay,Tl,pl):

  nb = ptb[0].n_b
  nkn = ptb[0].n_kn

  pc_int = np.zeros((nlay,nsp,nb,nkn-4))

  for k in range(nlay):
    for n in range(nsp):
      for b in range(nb):
        for r in range(nkn-4):
          pc_int[k, n, b, r] = ptb[n].pc_interp[b][r](np.log10(Tl[k]), np.log10(pl[k]/1e6))[0][0]

  ###  Check we can recreate spline for a layer ###
  # fig = plt.figure()
  # k = 0
  # ns = 1000
  # k_test = np.zeros((nsp,nb,ns))
  # splines = [[None for b in range(nb)] for n in range(nsp)]
  # for n in range(nsp):
  #   for b in range(nb):
  #     splines[n][b] = BSpline(ptb[n].kn[:], pc_int[k,n,b,:], 3)

  #     g = np.linspace(0.0, 1.0, ns)

  #     k_test[n,b,:] = 10.0**splines[n][b](g[:])
  #     print(n,b,k_test[n,b,:])
  #     plt.plot(g,k_test[n,b,:])
      
  #   plt.yscale('log')
  #   plt.show()
  
  return pc_int


def mix_p_table_PRAS(nsp,nlay,nb,ng,n_samp,pc_int,VMR,gx_scal):

  rng1 = np.random.default_rng(seed=123)  # Gas 1
  sampler = qmc.LatinHypercube(d=1, scramble=True, strength=1, rng=rng1)

  cs_mix = np.zeros((nlay,nb,ng))
  cs_samp = np.zeros((nsp,n_samp))
  cs_tot = np.zeros(n_samp)
  cs_tot_flat = np.zeros(n_samp)

  g = np.linspace(0.0, 1.0, n_samp)

  for k in range(nlay):
    for b in range(nb):
      splines = [BSpline(ptb[n].kn[:], pc_int[k,n,b,:], 3) for n in range(nsp)]
      for n in range(nsp):

        gs = sampler.random(n=n_samp)[:,0]

        cs_samp[n,:] = VMR[k,n] * 10.0**splines[n](gs[:])
      
      cs_tot[:] = np.maximum(np.sum(cs_samp[:,:],axis=0),1e-40)
      cs_tot_flat[:] = np.sort(np.log10(cs_tot[:]))
      cs_mix[k,b,:] = np.interp(gx_scal, g, cs_tot_flat[:])


  
  # Plot k coefficents for a layer
  # fig = plt.figure()
  # k = int(nlay/2)
  # for b in range(nb):
  #   plt.scatter(gx_scal,cs_mix[k,b,:])
      
  #   plt.yscale('log')
  # plt.show()
  

  return 10.0**cs_mix


def read_k_table(nsp,sp,k_tab_ext):


  for n in range(nsp):
    fname = f'../kp_tables/{k_tab_ext}/{sp[n]}_{k_tab_ext}.txt'
    with open(fname, 'r') as f:

      line = f.readline().split()
      species_name = str(line[0])

      print(n, species_name)
      if species_name != sp[n]:
        print('Different species k-table read in order:', species_name, sp[n])

      line = f.readline().split()
      n_T = int(line[0])
      n_p = int(line[1])
      n_b = int(line[2])
      n_g = int(line[3])

      T = [float(i) for i in f.readline().split()]
      p_vals = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line
      wl_e = [float(i) for i in f.readline().split()]
      wn_e = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line
      gx = [float(i) for i in f.readline().split()]
      gw = [float(i) for i in f.readline().split()]

      _ = f.readline()  # Skip line

      # Read the 4D pc array
      kc = np.zeros((n_T, n_p, n_b, n_g))
      for i in range(n_T):
        for j in range(n_p):
          for b in range(n_b):
            line = f.readline().split()
            kc[i, j, b, :] = [float(m) for m in line]

      kc_interp = [[RectBivariateSpline(np.log10(T), np.log10(p_vals), kc[:, :, b, r])
        for r in range(n_g)]
          for b in range(n_b)]

    # Create the p_tab object and append to global list
    ktb.append(
      k_tab(
      sp=species_name,
      n_T=n_T,
      n_p=n_p,
      n_b=n_b,
      n_g=n_g,
      T=T,
      p=p_vals,
      wl_e=wl_e,
      wn_e=wn_e,
      gx=gx,
      gw=gw,
      kc=kc,
      kc_interp=kc_interp))

  return

def interp_k_table(nsp,nlay,Tl,pl):


  nb = ktb[0].n_b
  ng = ktb[0].n_g

  kc_int = np.zeros((nlay,nsp,nb,ng))

  for k in range(nlay):
    for n in range(nsp):
      for b in range(nb):
        for r in range(ng):
          kc_int[k, n, b, r] = ktb[n].kc_interp[b][r](np.log10(Tl[k]), np.log10(pl[k]/1e6))[0][0]

  ###  Check we can recreate spline for a layer ###

  # fig = plt.figure()
  # k = int(nlay/2)
  # for n in range(nsp):
  #   for b in range(nb):
  #     plt.scatter(ktb[n].gx,kc_int[k,n,b,:])
      
  #   plt.yscale('log')
  #   plt.show()

  return kc_int


def mix_k_table_EE(nsp,nlay,nb,ng,kc_int,VMR,gx_scal,gw_scal,pe,grav,nd,rho):
  
  nlev = nlay + 1

  cs_mix = np.zeros((nlay,nb,ng))
  

  for b in range(nb):

    # Find grey opacity for each species in each layer
    k_grey = np.zeros((nlay,nsp))
    for k in range(nlay):
      for n in range(nsp):
        for g in range(ng):
          k_grey[k,n] += ktb[n].gw[g] * kc_int[k,n,b,g] 
        k_grey[k,n] = k_grey[k,n] * VMR[k,n] * nd[k]/rho[k]

    # Find the optical depth for each species and total optical depth
    tau_grey = np.zeros(nsp)
    tau_tot = 0.0
    for k in range(nlay):
      for n in range(nsp):
        tau_grey[n] += (k_grey[k,n] * (pe[k+1] - pe[k]))/grav
      tau_tot = np.sum(tau_grey[:])
      if (tau_tot >= 1.0):
        break

    i = np.argmax(tau_grey[:])

    cs_mix[k,b,:] = kc_int[k,i,b,:] * VMR[k,i]

    for n in range(nsp):
      if (i == n):
      # Don't mix dominant species
        continue
      # Add grey opacity to main k table
      cs_mix[k,b,:] += k_grey[k,n] / nd[k]*rho[k]

  return cs_mix

def mix_k_table_RORR(nsp,nlay,nb,ng,kc_int,VMR,gx_scal,gw_scal):

  cs_mix = np.zeros((nlay,nb,ng))


  for k in range(nlay):
    for b in range(nb):

      VMR_tot = VMR[k,0]
      cs_mix[k,b,:] = kc_int[k,0,b,:] * VMR_tot

      k_rom_matrix = np.zeros((ng,ng))
      w_rom_matrix = np.zeros((ng,ng))

      for n in range(1,nsp):

        # Add to VMR tot
        VMR_tot += VMR[k,n]

        for i in range(ng):
          for j in range(ng):
            k_rom_matrix[i,j] = ((cs_mix[k,b, i] + VMR[k,n] * kc_int[k, n, b, j]))/VMR_tot
            w_rom_matrix[i,j] = ktb[n-1].gw[i] * ktb[n].gw[j]
            
        # Flatten into 1D array
        k_rom_flat = k_rom_matrix.ravel()
        w_rom_flat = w_rom_matrix.ravel()

        # Sort k-values and associated weights
        sort_idx = np.argsort(k_rom_flat)
        k_rom_sorted = np.maximum(k_rom_flat[sort_idx], 1e-40)  # prevent log(0)
        w_rom_sorted = w_rom_flat[sort_idx]

        # Compute cumulative g = sum of weights
        g_rom = np.cumsum(w_rom_sorted)
        g_rom /= g_rom[-1]  # Normalize to [0, 1]

        # Interpolate to your standard gx_scal points
        cs_mix[k,b,:] = np.interp(gx_scal[:], g_rom, np.log10(k_rom_sorted)) * VMR_tot

  return 10.0**cs_mix
