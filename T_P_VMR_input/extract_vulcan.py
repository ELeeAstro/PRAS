import numpy as np
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

# Read VULCAN file
Teq = 'HD 189 - 1x'
fname= '../output/HD189_P_ph_1x.txt'
sav_f = 'HD189_P_ph_1x.pdf'

data = np.loadtxt(fname,skiprows=2)
plev = data[:,0]/1e6
Tlev = data[:,1]
Height = data[:,2]
mulev = data[:,3]
VMRlev = data[:,4:]
print(len(VMRlev[0,:]))

#sp = ['OH', 'H2', 'H2O', 'H', 'O', 'CH','C','CH2','CH3','CH4','C2', 'C2H2','C2H','C2H3','C2H4','C2H5','C2H6', \
#        'CO', 'CO2','CH2OH','H2CO','HCO','CH3O','CH3OH','CH3CO','O2','H2CCO','HCCO','N' ,'NH','CN','HCN','NO','NH2', \
#        'N2','NH3','N2H2','N2H','N2H3','N2H4','HNO','H2CN','HNCO','NO2','N2O','C4H2','CH2NH2','CH2NH','CH3NH2','CH3CHO', \
#        'HNO2', 'NCO','HO2','H2O2','HC3N','CH3CN','CH2CN', 'C2H3CN','SH','HSO','H2S','C3H3','C3H2','C3H4','C6H5','C6H6', \
#        'S','S2','SO','CS','OCS','CS2','NS','HS2','SO2','S4','S8','HCS','S3','C4H3','C4H5','S2O','CH3SH','CH3S','O_1','CH2_1', \
#        'N_2D','He','S8_l_s']

sp = ['O','OH','O2','H','H2','H2O','O_1','HO2','H2O2','HOPO2','PO3','PO','PO2','P','PH','HPO','HOPO','P2O3','PH3','PH2','P2O','P2','P2O2','H2POH','HPOH','He','P2H','P2H4', \
'P2H2','P4','P4O6','H3PO4']
nsp = len(sp)

sp_plt = ['PH3','PH2','PH','HOPO','HOPO2','H3PO4','PO','P4','P2','P','P4O6']
nsp_plt = len(sp_plt)

rsp_plt = [r'PH$_{3}$',r'PH$_{2}$',r'PH',r'HOPO',r'HOPO$_{2}$',r'H$_{3}$PO$_{4}$',r'PO',r'P$_{4}$',r'P$_{2}$',r'P',r'P$_{4}$O$_{6}$']

fig, ax = plt.subplots()

col = sns.color_palette("husl", nsp_plt)
#col = sns.color_palette('colorblind')

for i in range(nsp_plt):
  idx = sp.index(sp_plt[i])
  print(i,idx)
  plt.plot(VMRlev[:,idx],plev[:],c=col[i],label=rsp_plt[i])


plt.legend(ncol=1,loc='lower left',title=Teq)

plt.xscale('log')
plt.yscale('log')

plt.xlim(1e-20,1e-3)
#plt.ylim(1e-3,1e3)
plt.ylim(1e-8,1e3)

ticks = [1e-20,1e-15,1e-10,1e-5]
ticklab = [r'10$^{-20}$',r'10$^{-15}$',r'10$^{-10}$',r'10$^{-5}$']
plt.xticks(ticks=ticks,labels=ticklab)

ticks = [1e-19,1e-18,1e-17,1e-16,1e-14,1e-13,1e-12,1e-11,1e-9,1e-8,1e-7,1e-6]
plt.xticks(ticks=ticks,labels=None,minor=True)

ticks = [1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
ticklab = [r'1000',r'100',r'10',r'1',r'0.1',r'0.01',r'10$^{-3}$',r'10$^{-4}$',r'10$^{-5}$',r'10$^{-6}$',r'10$^{-7}$',r'10$^{-8}$']
plt.yticks(ticks=ticks,labels=ticklab)

ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.gca().invert_yaxis()

plt.ylabel(r'p$_{\rm gas}$ [bar]',fontsize=16)
plt.xlabel(r'VMR',fontsize=16)

plt.tick_params(axis='both',which='major',labelsize=14)

plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
plt.savefig(sav_f,dpi=144,bbox_inches='tight')

plt.show()
