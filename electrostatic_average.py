from ase.io.cube import read_cube
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


filename = ['Strain1_Files/RuO2-110-0-/velectrostatic.cube', 'Strain1_Files/RuO2-110-0-/vreference.cube', 'Strain1_Files/RuO2-110-0-/dvtot.cube']
cube_data = {}
for fname in filename:
    with open(fname) as f:
        V_e = read_cube(f)
    cube_data[fname] = V_e['data']
    cube_data[fname] = V_e['data']
    cube_data[fname+'_average'] = np.mean(np.mean(cube_data[fname], axis=0), axis=0)

def moving_average(a, n=101):
    sums = np.cumsum(a, dtype=float)
    averages = (sums[n:]-sums[0:-n])/n
    return averages

#Plotting fonts
fig, ax = plt.subplots(figsize=(8,6))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

#Plot
n = 101
C = 58.562
c_step = C/(1125-1)
plt.plot(np.arange(0, C+0.01, c_step), cube_data['Strain1_Files/RuO2-110-0-/velectrostatic.cube_average'], color='black')
plt.plot(np.arange(c_step*n/2, C-c_step*n/2+0.01, c_step), 
         moving_average(cube_data['Strain1_Files/RuO2-110-0-/velectrostatic.cube_average']), color='darkgreen')
#plt.plot(np.linspace(0,58.562,1125), cube_data['vreference.cube_average'])
#plt.plot(np.linspace(0,58.562,1125), cube_data['dvtot.cube_average'])
plt.xlabel(r'$z$ ($\rm \AA$)')
plt.ylabel(r'Electrostatic potential (eV)')
plt.xlim(0,60)
plt.ylim(-2.5, 0.25)
plt.xticks(np.arange(plt.xlim()[0], plt.xlim()[1]+0.1, 5))
plt.yticks(np.arange(plt.ylim()[0], plt.ylim()[1]+0.1, 0.25))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
sns.despine(offset=10, trim=True)

plt.tight_layout()
#plt.show()
plt.savefig('V_electrostatic.png', dpi=300)