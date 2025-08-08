import numpy as np
import matplotlib.pylab as plt
import seaborn as sns


# read T-p profile at flux
f1 = 'results/EE_vert_heating.txt'
data= np.loadtxt(f1,skiprows=1)
p1 = data[:,1]
T1 = data[:,2]
h1 = data[:,4]

f1 = 'results/EE_vert_flux.txt'
data= np.loadtxt(f1,skiprows=1)
f_u = data[:,2]
f_d = data[:,3]
fs1_u = (f_u[1:] +  f_u[0:-1])/2.0
fs1_d = (f_d[1:] +  f_d[0:-1])/2.0
fs1 = fs1_u - fs1_d


# read T-p profile at flux
f2 = 'results/RORR_vert_heating.txt'
data= np.loadtxt(f2,skiprows=1)
p2 = data[:,1]
T2 = data[:,2]
h2 = data[:,4]

f2 = 'results/RORR_vert_flux.txt'
data= np.loadtxt(f2,skiprows=1)
f_u = data[:,2]
f_d = data[:,3]
fs2_u = (f_u[1:] +  f_u[0:-1])/2.0
fs2_d = (f_d[1:] +  f_d[0:-1])/2.0
fs2 = fs2_u - fs2_d

# read T-p profile at flux
f3 = 'results/RORR_2_vert_heating.txt'
data= np.loadtxt(f3,skiprows=1)
p3 = data[:,1]
T3 = data
h3 = data[:,4]

f3 = 'results/RORR_2_vert_flux.txt'
data= np.loadtxt(f3,skiprows=1)
f_u = data[:,2]
f_d = data[:,3]
fs3_u = (f_u[1:] +  f_u[0:-1])/2.0
fs3_d = (f_d[1:] +  f_d[0:-1])/2.0
fs3 = fs3_u - fs3_d

# read T-p profile at flux
f4 = 'results/RORR_3_vert_heating.txt'
data= np.loadtxt(f4,skiprows=1)
p4 = data[:,1]
T4 = data[:,2]
h4 = data[:,4]

f4 = 'results/RORR_3_vert_flux.txt'
data= np.loadtxt(f4,skiprows=1)
f = data[:,1]
f_u = data[:,2]
f_d = data[:,3]
fs4_u = (f_u[1:] +  f_u[0:-1])/2.0
fs4_d = (f_d[1:] +  f_d[0:-1])/2.0
fs4 = fs4_u - fs4_d

# read T-p profile at flux
f5 = 'results/PRAS_vert_heating.txt'
data= np.loadtxt(f5,skiprows=1)
p5 = data[:,1]
T5 = data[:,2]
h5 = data[:,4]

f5 = 'results/PRAS_vert_flux.txt'
data= np.loadtxt(f5,skiprows=1)
f = data[:,1]
f_u = data[:,2]
f_d = data[:,3]
fs5_u = (f_u[1:] +  f_u[0:-1])/2.0
fs5_d = (f_d[1:] +  f_d[0:-1])/2.0
fs5 = fs5_u - fs5_d


# read T-p profile at flux
f6 = 'results/PRAS_2_vert_heating.txt'
data= np.loadtxt(f6,skiprows=1)
p6 = data[:,1]
T6 = data[:,2]
h6 = data[:,4]

f6 = 'results/PRAS_2_vert_flux.txt'
data= np.loadtxt(f6,skiprows=1)
f = data[:,1]
f_u = data[:,2]
f_d = data[:,3]
fs6_u = (f_u[1:] +  f_u[0:-1])/2.0
fs6_d = (f_d[1:] +  f_d[0:-1])/2.0
fs6 = fs6_u - fs6_d

# read T-p profile at flux
f7 = 'results/PRAS_3_vert_heating.txt'
data= np.loadtxt(f7,skiprows=1)
p7 = data[:,1]
T7 = data[:,2]
h7 = data[:,4]

f7 = 'results/PRAS_3_vert_flux.txt'
data= np.loadtxt(f7,skiprows=1)
f = data[:,1]
f_u = data[:,2]
f_d = data[:,3]
fs7_u = (f_u[1:] +  f_u[0:-1])/2.0
fs7_d = (f_d[1:] +  f_d[0:-1])/2.0
fs7 = fs7_u - fs7_d


col = sns.color_palette('colorblind')

fig, ax = plt.subplots()

plt.plot(fs1,p1,label='16+16 EE',c=col[0],ls='dotted')
plt.plot(fs2,p2,label='4+4 RORR',c=col[1])
plt.plot(fs3,p3,label='8+8 RORR',c=col[2])
plt.plot(fs4,p4,label='16+16 RORR',c='black')
plt.plot(fs5,p5,label='16+16 PRAS (100)',ls='dashed',c=col[3])
plt.plot(fs6,p6,label='16+16 PRAS (1000)',ls='dashed',c=col[4])
plt.plot(fs7,p7,label='16+16 PRAS (10000)',ls='dashed',c=col[7])

x1, x2, y1, y2 = 2.8e5, 3.5e5, 1e1, 1e-1  # subregion of the original image
#axins = ax.inset_axes([0.70, 0.2, 0.25, 0.7], xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
#ind = ax.indicate_inset_zoom(axins, edgecolor="black")
#ind.connectors[0].set_visible(False)
#ind.connectors[1].set_visible(False)
#ind.connectors[2].set_visible(False)
#ind.connectors[3].set_visible(False)
#axins.set_yscale('log')
# axins.plot(fs1,p1,c=col[0],ls='dotted')
# axins.plot(fs2,p2,c=col[1])
# axins.plot(fs3,p3,c=col[2])
# axins.plot(fs4,p4,c='black')
# axins.plot(fs5,p5,ls='dashed',c=col[3])
# axins.plot(fs6,p6,ls='dashed',c=col[4])
# axins.plot(fs7,p7,ls='dashed',c=col[7])


plt.xlabel(r'$F_{\rm x}$ [erg s$^{-1}$ cm$^{-2}$]',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend(loc='upper left')

plt.yscale('log')
plt.xscale('log')

plt.ylim(1e-6,100)
plt.xlim(1.0e6,1.3e8)


plt.gca().invert_yaxis()

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('lw_flux.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

plt.plot(h1/h4-1,p1,label='16+16 EE',c=col[0],ls='dotted')
plt.plot(h2/h4-1,p2,label='4+4 RORR',c=col[1])
plt.plot(h3/h4-1,p3,label='8+8 RORR',c=col[2])
plt.plot(h4/h4-1,p4,label='16+16 RORR',c='black')
plt.plot(h5/h4-1,p5,label='16+16 PRAS (100)',ls='dashed',c=col[3],alpha=0.3)
plt.plot(h6/h4-1,p6,label='16+16 PRAS (1000)',ls='dashed',c=col[4])
plt.plot(h7/h4-1,p7,label='16+16 PRAS (10000)',ls='dashed',c=col[7])

print(h6/h4)

plt.xlabel(r'$d$$T_{\rm x}$/$d$$T_{(RORR_{16+16})}$ - 1',fontsize=16)
plt.ylabel(r'$p$ [bar]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend(loc='lower left')

plt.yscale('log')

plt.ylim(1e-6,100)
plt.xlim(0.6-1,1.4-1)

plt.gca().invert_yaxis()

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('lw_heat.pdf',dpi=300,bbox_inches='tight')


plt.show()

