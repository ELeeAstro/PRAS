### Perform the RORR method ###
import numpy as np
import matplotlib.pylab as plt
import yaml
import seaborn as sns
import time

# Get parameters from yaml file
with open('input.yaml', 'r') as file:
  param = yaml.safe_load(file)['k_table_mix']

nsp = param['nsp']
sp = param['sp']
wl_file = param['wl_file']
ng = param['ng']
VMR = param['VMR']

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
  file = 'k_tables/'+sp[n]+'_'+str(nb)+'_k_table.txt'

  with open(file, "r") as f:
    dum, dum = f.readline().split()
    g_x[n,:] = f.readline().split()
    g_w[n,:] = f.readline().split()

  data = np.loadtxt(file,skiprows=3)
  k_g[n,:,:] = 10.0**data[:,2:]

### Start RORR method ###

# Build the ROM tensor product
k_rom_matrix = np.zeros((ng, ng))
w_rom_matrix = np.zeros((ng, ng))

k_g_comb = np.zeros((nb, ng))

start = time.time()


for b in range(nb):

  VMR_tot = VMR[0]
  k_g_comb[b,:] = k_g[0,b,:] * VMR_tot

  for n in range(1,nsp):

    for i in range(ng):
      for j in range(ng):
        k_rom_matrix[i, j] = k_g_comb[b,i] + VMR[n] * k_g[n,b,j]
        w_rom_matrix[i, j] = g_w[n,i] * g_w[n,j]

    # Add to VMR tot
    VMR_tot += VMR[n]

    # Flatten into 1D array
    k_rom_flat = k_rom_matrix.ravel()
    w_rom_flat = w_rom_matrix.ravel()

    # Sort k-values and associated weights
    sort_idx = np.argsort(k_rom_flat)
    k_rom_sorted = np.maximum(k_rom_flat[sort_idx], 1e-45)  # prevent log(0)
    w_rom_sorted = w_rom_flat[sort_idx]

    # Compute cumulative g = sum of weights
    g_rom = np.cumsum(w_rom_sorted)
    g_rom /= g_rom[-1]  # Normalize to [0, 1]

    # Interpolate to your standard gx_scal points
    k_g_comb[b,:] = 10.0**np.interp(g_x[n,:], g_rom, np.log10(k_rom_sorted)) * VMR_tot

end = time.time()

print(end - start)

# Create a new file for output
fout = open('k_tables/'+'comb'+'_'+str(nb)+'_k_table_RORR.txt','w')

fout.write(str(nb) + ' ' + str(ng) + '\n')

fout.write(" ".join(str(g) for g in g_x[0,:]) + '\n')
fout.write(" ".join(str(g) for g in g_w[0,:]) + '\n')

# Output the k-coefficient line to file
for b in range(nb):
  fout.write(str(wl_e[b]) + ' ' + str(wl_e[b+1]) + ' ' + " ".join(str(g) for g in k_g_comb[b,:]) + '\n')

fout.close()

fig = plt.figure()

for b in range(nb):
  plt.scatter(g_x[0,:],k_g_comb[b,:])

plt.yscale('log')

#plt.savefig('RORR_comb.png',dpi=300,bbox_inches='tight')


plt.show()