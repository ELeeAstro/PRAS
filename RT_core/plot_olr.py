import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

f1 = 'results_OLR/RORR_vert_OLR.txt'
data = np.loadtxt(f1,skiprows=1)
wl1_e_a = data[:,1]
wl1_e_b = data[:,2]
wl1 = (wl1_e_a[:] + wl1_e_b[:])/2.0
olr_1 = data[:,3]

f2 = 'results_OLR/RORR_2_vert_OLR.txt'
data = np.loadtxt(f2,skiprows=1)
wl2_e_a = data[:,1]
wl2_e_b = data[:,2]
wl2 = (wl2_e_a[:] + wl2_e_b[:])/2.0
olr_2 = data[:,3]

f3 = 'results_OLR/RORR_3_vert_OLR.txt'
data = np.loadtxt(f3,skiprows=1)
wl3_e_a = data[:,1]
wl3_e_b = data[:,2]
wl3 = (wl3_e_a[:] + wl3_e_b[:])/2.0
olr_3 = data[:,3]

f4 = 'results_OLR/PRAS_vert_OLR.txt'
data = np.loadtxt(f4,skiprows=1)
wl4_e_a = data[:,1]
wl4_e_b = data[:,2]
wl4 = (wl4_e_a[:] + wl4_e_b[:])/2.0
olr_4 = data[:,3]

f5 = 'results_OLR/PRAS_2_vert_OLR.txt'
data = np.loadtxt(f5,skiprows=1)
wl5_e_a = data[:,1]
wl5_e_b = data[:,2]
wl5 = (wl5_e_a[:] + wl5_e_b[:])/2.0
olr_5 = data[:,3]

f6 = 'results_OLR/PRAS_3_vert_OLR.txt'
data = np.loadtxt(f6,skiprows=1)
wl6_e_a = data[:,1]
wl6_e_b = data[:,2]
wl6 = (wl6_e_a[:] + wl6_e_b[:])/2.0
olr_6 = data[:,3]

f7= 'results_OLR/RORR_4_vert_OLR.txt'
data = np.loadtxt(f7,skiprows=1)
wl7_e_a = data[:,1]
wl7_e_b = data[:,2]
wl7 = (wl7_e_a[:] + wl7_e_b[:])/2.0
olr_7 = data[:,3]

fig = plt.figure()

col = sns.color_palette('colorblind')

plt.plot(wl1,olr_1/(wl1*1e-4),label='4+4 RORR',c=col[0])
plt.plot(wl2,olr_2/(wl2*1e-4),label='8+8 RORR',c=col[1])
plt.plot(wl3,olr_3/(wl3*1e-4),label='16+16 RORR',c='black')
plt.plot(wl4,olr_4/(wl4*1e-4),label='16+16 PRAS (100)',c=col[3],ls='dashed')
plt.plot(wl5,olr_5/(wl5*1e-4),label='16+16 PRAS (1000)',c=col[4],ls='dashed')
plt.plot(wl6,olr_6/(wl6*1e-4),label='16+16 PRAS (10000)',c=col[7],ls='dashed')
#plt.plot(wl7,olr_7/(wl6*1e-4),label='20 RORR',c=col[9])


plt.xlabel(r'$\lambda$ [$\mu$m]',fontsize=16)
plt.ylabel(r'$F_{x}$ [erg s$^{-1}$ cm$^{-2}$ cm$^{-1}$]',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend(loc='lower right')

plt.yscale('log')
plt.xscale('log')

plt.ylim(1e3,1e10)
plt.xlim(0.2,30)

xticks = [0.2,0.3,0.5,1,2,3,4,5,10,20,30]
xticks_lab = ['0.2','0.3','0.5','1','2','3','4','5','10','20','30']

plt.xticks(xticks,xticks_lab)

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('OLR.pdf',dpi=300,bbox_inches='tight')

fig = plt.figure()

col = sns.color_palette('colorblind')

plt.plot(wl1,olr_1/olr_3 - 1.0,label='4+4 RORR',c=col[0])
plt.plot(wl2,olr_2/olr_3- 1.0,label='8+8 RORR',c=col[1])
plt.plot(wl3,olr_3/olr_3- 1.0,label='16+16 RORR',c='black')
plt.plot(wl4,olr_4/olr_3- 1.0,label='16+16 PRAS (100)',c=col[3],ls='dashed')
plt.plot(wl5,olr_5/olr_3- 1.0,label='16+16 PRAS (1000)',c=col[4],ls='dashed')
plt.plot(wl6,olr_6/olr_3- 1.0,label='16+16 PRAS (10000)',c=col[7],ls='dashed')
#plt.plot(wl7,olr_7/olr_3,label='20 RORR',c=col[9])


plt.xlabel(r'$\lambda$ [$\mu$m]',fontsize=16)
plt.ylabel(r'$F_{x}$/$F_{(RORR_{16+16})}$ - 1',fontsize=16)
plt.tick_params(axis='both',which='major',labelsize=14)
plt.legend(loc='upper left')

#plt.yscale('log')
plt.xscale('log')

plt.ylim(0.94-1,1.1-1)
plt.xlim(0.2,30)

xticks = [0.2,0.3,0.5,1,2,3,4,5,10,20,30]
xticks_lab = ['0.2','0.3','0.5','1','2','3','4','5','10','20','30']

plt.xticks(xticks,xticks_lab)

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)

plt.savefig('OLR_rel.pdf',dpi=300,bbox_inches='tight')


plt.show()
