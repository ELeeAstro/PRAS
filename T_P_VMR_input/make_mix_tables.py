import numpy as np
import pickle

plot_EQ = False
output = open('HD_189_VULCAN.txt', "w")


vul = 'PRAS.vul'
with open(vul, 'rb') as handle:
  vul = pickle.load(handle)

species = vul['variable']['species']
out_species = species

ost = '{:<8s}'.format('(dyn/cm2)')  + '{:>9s}'.format('(K)') + '{:>9s}'.format('(cm)') + '{:>10s}'.format('(g/mol)')+ '\n'
ost += '{:<8s}'.format('Pressure')  + '{:>9s}'.format('Temp')+ '{:>9s}'.format('Hight')+ '{:>9s}'.format('mu')
for sp in out_species: ost += '{:>10s}'.format(sp)

#ost += '{:>14s}'.format('J_H2O')

ost +='\n'

for n, p in enumerate(vul['atm']['pco']):
    ost += '{:<8.3E}'.format(p)  + '{:>8.1f}'.format(vul['atm']['Tco'][n])  + '{:>10.2E}'.format(vul['atm']['zco'][n]) + '{:>8.2f}'.format(vul['atm']['mu'][n])
    for sp in out_species:
        if plot_EQ == True:
            ost += '{:>12.4E}'.format(vul['variable']['y_ini'][n,species.index(sp)]/vul['atm']['n_0'][n])
        else:
            ost += '{:>12.4E}'.format(vul['variable']['ymix'][n,species.index(sp)])

    #ost += '{:>12.4E}'.format(vul['variable']['J_sp'][('H2O', 0)][n]) #*vul['variable']['y'][n,species.index('H2O')])

    ost += '\n'

ost = ost[:-1]
output.write(ost)
output.close()
