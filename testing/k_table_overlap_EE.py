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

### Start AEE method ###

k_g_comb = np.zeros((nb, ng))
k_grey = np.zeros((nsp))

start = time.time()

for b in range(nb):

  # Find largest grey opacity in bin
  for n in range(nsp):
    k_grey[n] = np.sum(g_w[n,:] * k_g[n,b,:]) * VMR[n]

  i = np.argmax(k_grey[:])

  k_g_comb[b,:] = k_g[i,b,:] * VMR[i]

  for n in range(nsp):
    if (i == n):
      # Don't mix dominant species
      continue
    # Add grey opacity to main k table
    k_g_comb[b,:] += k_grey[n]

end = time.time()

print(end - start)

# Create a new file for output
fout = open('k_tables/'+'comb'+'_'+str(nb)+'_k_table_EE.txt','w')

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

#plt.savefig('EE_comb.png',dpi=300,bbox_inches='tight')


plt.show()