import numpy as np

sb = 5.670374419e-8
hp = 6.62607015e-34
kb = 1.380649e-23
c_s = 2.99792458e8
c1 = (hp * c_s) / kb
c2 = c_s**2
n2 = 2.0 * hp * c2


nmu = 4
uarr = [0.0091177205, 0.1034869099, 0.4177464746, 0.8510589811]
w = [0.0005392947, 0.0388879085, 0.3574186924, 0.6031541043]

def integrate_planck(nb,wn_e,T):
  
  iB = np.zeros(nb+1)
  be = np.zeros(nb)

  # Code for integrating the blackbody function between two wavenumbers
  # This is a method that uses a sum convergence
  # Taken from: spectralcalc.com/blackbody/inband_radiance.html
  for ww in range(nb+1):

    x = (c1 * 100.0 * wn_e[ww])/T
    x2 = x**2
    x3 = x**3

    itera = np.minimum(int(2.0 + 20.0/x),150)

    summ = 0.0
    for j in range(1,itera+1):
      dn = 1.0/float(j)
      summ += np.exp(-np.minimum(float(j)*x,300.0)) * \
        (x3 + (3.0 * x2 + 6.0*(x+dn)*dn)*dn)*dn

    iB[ww] = n2 * (T/c1)**(4) * summ

  for ww in range(nb):
    be[ww] = np.maximum(iB[ww+1] - iB[ww],0.0)

  # Convert W m-2 sr-1 to erg s-1 cm-2 sr-1
  be[:] = be[:] * 1e3
  
  return be

def shortwave(nlev, nb, ng, tau, mu_z, gw_scal):

  
  F_s_u = np.zeros(nlev)
  F_s_d = np.zeros(nlev)
  F_s_net = np.zeros(nlev)


  F_s_d_b = np.zeros((nlev,nb))
  for b in range(nb):
    F_s_d_g = np.zeros((nlev,ng))
    for g in range(ng):
      F_s_d_g[:,g] = mu_z * np.exp(-tau[:,b,g]/mu_z)
      F_s_d_b[:,b] += F_s_d_g[:,g] * gw_scal[g]

    F_s_d[:] += F_s_d_b[:,b]

  F_s_net[:] = F_s_u[:] - F_s_d[:]

  return F_s_u, F_s_d, F_s_net


def short_char(nlev,tau,be,be_int):
    
  nlay = nlev - 1

  dtau = np.zeros(nlay)
  dtau[:] = tau[1:] - tau[0:-1]

  dell = np.zeros(nlay)
  edel = np.zeros(nlay)
  e0i = np.zeros(nlay)
  Am = np.zeros(nlay)
  Bm = np.zeros(nlay)
  Gp = np.zeros(nlay)
  Bp = np.zeros(nlay)
  e1i = np.zeros(nlay)
  e1i_del = np.zeros(nlay)

  lw_down_g = np.zeros(nlev)
  lw_up_g = np.zeros(nlev)
  flx_down = np.zeros(nlev)
  flx_up = np.zeros(nlev)

  # Start loops to integrate in mu space
  for m in range(nmu):

    dell[:] = dtau[:]/uarr[m]
    edel[:] = np.exp(-dell[:])
    e0i[:] = 1.0 - edel[:]

    # Prepare loop
    # Olson & Kunasz (1987) linear interpolant parameters
    for k in range(nlay):
      if (edel[k] > 0.999):
        # If we are in very low optical depth regime, then use an isothermal approximation
        Am[k] = (0.5*(be[k+1] + be[k]) * e0i[k])/be[k]
        Bm[k] = 0.0
        Gp[k] = 0.0
        Bp[k] = Am[k]
      else:
        # Use linear interpolants
        e1i[k] = dell[k] - e0i[k]
        e1i_del[k] = e1i[k]/dell[k] # The equivalent to the linear in tau term

        Am[k] = e0i[k] - e1i_del[k] # Am(k) = Gp(k), just indexed differently
        Bm[k] = e1i_del[k] # Bm(k) = Bp(k), just indexed differently
        Gp[k] = Am[k]
        Bp[k] = Bm[k]

    # Begin two-stream loops
    # Perform downward loop first
    # Top boundary condition - 0 flux downward from top boundary
    lw_down_g[0] = 0.0
    for k in range(nlay):
      lw_down_g[k+1] = lw_down_g[k]*edel[k] + Am[k]*be[k] + Bm[k]*be[k+1] # TS intensity

    # Perform upward loop
    # Lower boundary condition - internal heat definition Fint = F_down - F_up
    # here we use the same condition but use intensity units to be consistent
    lw_up_g[-1] = lw_down_g[-1] + be_int
    for k in range(nlay-1, -1, -1):
      lw_up_g[k] = lw_up_g[k+1]*edel[k] + Bp[k]*be[k] + Gp[k]*be[k+1] # TS intensity

    # Sum up flux arrays with Gaussian quadrature weights and points for this mu stream
    flx_down[:] = flx_down[:] + lw_down_g[:] * w[m]
    flx_up[:] = flx_up[:] + lw_up_g[:] * w[m]

    # The flux is the intensity * pi
  flx_down[:] = np.pi * flx_down[:]
  flx_up[:] = np.pi * flx_up[:]

  return flx_up, flx_down

def longwave(nlev,nb,ng,Te,T_int,tau,wn_e,gw_scal):

  # Fine temperature edges

  be = np.zeros((nlev,nb))
  for k in range(nlev):
    be[k,:] = integrate_planck(nb,wn_e[::-1],Te[k])
    be[k,:] = be[k,::-1]
  be_int = np.zeros(nb)
  be_int = integrate_planck(nb,wn_e[::-1],T_int)
  be_int[:] = be_int[::-1]

  F_l_u = np.zeros(nlev)
  F_l_d = np.zeros(nlev)
  F_l_net = np.zeros(nlev)

  F_l_u_b = np.zeros((nlev,nb))
  F_l_d_b = np.zeros((nlev,nb))
  for b in range(nb):
    F_l_u_g = np.zeros((nlev,ng))
    F_l_d_g = np.zeros((nlev,ng))
    for g in range(ng):
      F_l_u_g[:,g], F_l_d_g[:,g] = short_char(nlev,tau[:,b,g],be[:,b],be_int[b])
      F_l_u_b[:,b] += F_l_u_g[:,g] * gw_scal[g]
      F_l_d_b[:,b] += F_l_d_g[:,g] * gw_scal[g]

    F_l_u[:] += F_l_u_b[:,b]
    F_l_d[:] += F_l_d_b[:,b]

  F_l_net[:] = F_l_u[:] - F_l_d[:]

  OLR = np.zeros(nb)
  OLR[:] = F_l_u_b[0,:]

  return F_l_u, F_l_d, F_l_net, OLR
