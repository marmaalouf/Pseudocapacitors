#!/usr/bin/env python

from __future__ import division
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random
import numpy as np
import numba
from numba import njit
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir",type=str,required=True)
parser.add_argument("--name",type=str,required=False)
parser.add_argument("--gamma",nargs='+',required=False)
parser.add_argument("--pH",nargs='+',required=False)
args = parser.parse_args()

from grepping_energies import E_Dict
from grepping_fermi import F_Dict


def main():
# Monte Carlo simulation for the RuO2(110) surface

    surface_capacitance = 10                         # Capacitance of the surface
    filling = 0.0                           # Initial starting coverage
    list_voltage = np.arange(4.4, 7.01,0.01) # Applied voltage on the sytem
    n1 = 20                                 # Number of sites along x (must be even)
    n2 = 20                                 # Number of sites along y (must be even)
    nstep = 7*20*n1*n2                      # Number of Monte Carlo steps
    nrun = 100                             # Number of Monte Carlo runs per voltage
    kB = 1.38E-23/(1.6E-19)                 # Voltzmann constant in eV
    T = 300.0                               # Temperature in Kelvin
    
    if args.rootdir[:args.rootdir.find('_')] == "Strain0":
        area = 6.40518915692 * 6.22414894911
    elif args.rootdir[:args.rootdir.find('_')] == "Strain2":
        area = 6.48066630851  * 6.07796336923
    elif args.rootdir[:args.rootdir.find('_')] == "StrainJin":
        area = 6.55250850753 * 5.95277605493
    elif args.rootdir[:args.rootdir.find('_')] == "Strain1":
        area = 6.55614346009 * 5.93177778936
    else:
        area = 6.2*6.3                          # Surface area/Cell in A^2
    capacitance = surface_capacitance/100/16*area   # x mF/cm^2 = x/100/16 e/A^2/facet
    phi_she = 4.4                           # Standard Hydrogen Electrode (SHE)
    EH2 = -13.605703976*2.33322171
    d_mu = [0]                              #Chemical potential shift values
    d_mu_H = 0
    dataname = args.name

    if vars(args)["pH"] != None:
        pH_L = []
        for pH in args.pH:
            pH_L.append(float(pH))
    else:
        pH_L = [0.3]
        
    if vars(args)["gamma"] != None:
        gamma = []
        for gam in args.gamma:
            gamma.append(float(gam))
    else:
        gamma = [1]
    
    # Initialize Adsorption Energies, Voltages and Number of Hydrogens
    energy, voltage, nhydrogen, name, occurrence = initialize_thermodynamic_data(area, gamma[0])

    list_nhydrogen_per_site = []
    list_charge_per_site = []
    nhydrogen_per_site = (nrun+1)*[0]
    charge_per_site = (nrun+1)*[0]
    energy_per_site = (nrun+1)*[0]

    # Perform the Monte Carlo simulation
    # This is used to vary the pH (chemical potential which includes potential and concentration effects)
    for pH in pH_L:
        
        SamplingTest=[]
        DictSamplingMult={}
        SampleNum={}
        
        # Iterate through the applied voltage
        for phi in list_voltage:
            
            # We average over 100 different runs to get an average coverage
            for irun in range(1,nrun+1):
                #Pregenerate all the random number changes
                random_site = np.random.randint(1,n1+1,size=(nstep+1,2))
                random_values = np.random.rand(nstep+1,4)
                random_sites = np.append(random_site,random_values,axis=1)
                
                # Initialize the state of the slab with the specified coverage
                sigma_old = np.zeros((n1+1,n2+1))
                sigma_old = init_surface(filling, sigma_old)
                sigma_test_old = sigma_old
            
                # Calculate the initial potential energy
                u_old, q_old = initial_potential_energy(sigma_old,energy,voltage,nhydrogen,phi,capacitance,phi_she,pH,kB,T,d_mu_H)

                # Perform the Metropolis loop
                # This allows each site on the surface to have the chance to change 140
                for i in range(1, nstep+1):
                    # Apply a random change to a supercell
                    sigma_new = change_random(sigma_old,random_sites[i],n1,n2)
                    # Calculate the new energy and charge
                    u_new, q_new = new_potential_energy(u_old,q_old,sigma_new,
                                                        energy,voltage,nhydrogen,
                                                        phi,capacitance,phi_she,
                                                        random_sites[i][0],
                                                        random_sites[i][1],pH,
                                                        kB,T,d_mu_H)
                    # Apply the Metropolis selection criterion
                    if random.random() < np.exp(-(np.sum(u_new)-np.sum(u_old))
                                                /(kB*T)):
                        #Copy the new lower energy surface to the old variables
                        sigma_old[:] = sigma_new
                        u_old[:] = u_new
                        q_old[:] = q_new
                        
                    # #Testing of Sampling
                    # sigma_test_new = change_random(sigma_test_old,random_sites[i],n1,n2)
                    # sigma_test_old = sigma_test_new
                    # # Store configurations in a list for counting
                    # name_new, O_new = Counting(sigma_test_new, name, occurrence, random_sites[i][0], random_sites[i][1])
                    # SamplingTest.append(name_new)
                    # DictSamplingMult.update({name_new: O_new})
                    
                #Average the hydrogen and charge per site
                nhydrogen_per_site[irun] = draw_state(sigma_old)/n1/n2
                charge_per_site[irun] = (np.sum(q_old)-draw_state(sigma_old))/n1/n2
                energy_per_site[irun] = (np.sum(u_old))/n1/n2
            #Average the different runs to get an overall average
            list_nhydrogen_per_site.append(np.sum(nhydrogen_per_site)/nrun)
            list_charge_per_site.append(np.sum(charge_per_site)/nrun)
            
            with open("MC_Data/%s_pH_%s_%sruns_gamma_%s_Vi_%sVf_%s_Area_%s_Cap_%s.txt"%(dataname,pH,nrun,gamma[0],list_voltage[0], round(list_voltage[-1]),area, capacitance),'a') as cap_file:
                cap_file.write("Energy: %.3f Voltage: %.3f Capacitance: %.3f Coverage: %.3f and Charge: %.3f\n"%(np.sum(energy_per_site)/nrun,phi,capacitance,np.sum(nhydrogen_per_site)/nrun,np.sum(charge_per_site)/nrun))
         
            
            
        # for key in sorted(DictSamplingMult):
        #     SampleNum.update({key: SamplingTest.count(key)/DictSamplingMult[key]})    
        # with open("SampleTest_gamma_%s_Vi_%sVf_%s_Cap_%s.py"%(gamma,list_voltage[0], round(list_voltage[-1]),surface_capacitance),'a') as cap_file:
        #     cap_file.write("SampleNums_pH%s=%s\n"%(round(pH), SampleNum))
        
        
def plot_histogram(dictionary, bar_width=0.2, tick_spacing=0.5):
    keys = list(dictionary.keys())
    values = list(dictionary.values())

    plt.bar(keys, values, width=bar_width)
    plt.xlabel('Configurations')
    plt.ylabel('Times Visited')
    plt.title('Histogram of Dictionary Elements')
    
    plt.show()

def init_surface(filling, sigma_old):
    # Extracts the size of the array
    n1, n2 = sigma_old.shape
    for i in range(1,n1):
        for j in range(1,n2):
            # Check to see if it is a bridge or top site
            if i % 2 == 1:
                sigma_old[i,j] = math.ceil(1+3*filling)
            else:
                sigma_old[i,j] = math.ceil(5+2*filling)
    return sigma_old


def draw_state(sigma):
    # Extracts the size of the array
    n1, n2 = sigma.shape
    state = np.zeros((n1,n2))
    nhydrogen_per_site = np.array([0, 0, 1, 1, 2, 0, 1, 1, 1, 2, 2, 1, 2, 2, 2, 3, 3])
    for i1 in range(1, n1):
        for i2 in range(1, n2):
            state[i1,i2] = nhydrogen_per_site[int(sigma[i1,i2])]
    # Finds the total number of hydrogen on the surface
    nhydrogen_total = np.sum(state)
    return nhydrogen_total
    
    
@njit(nogil=True, cache=True)
def change_random(sigma_old,random_nums,n1,n2):
# This function selects a site at random and changes the state
        
    sigma_new = sigma_old.copy()
    
    index = 1
    for j1 in [int(random_nums[0]), int(random_nums[0])+1]:
        for j2 in [int(random_nums[1]), int(random_nums[1])+1]:
            index = index + 1
            # This portion checks to see if it's at the edge of the grid and wraps around if needed
            k1 = j1
            if k1 > n1:
                k1 = 1
            k2 = j2
            if k2 > n2:
                k2 = 1
            if int(sigma_old[k1][k2]) <= 4:
                sigma_new[k1][k2] = int(math.ceil(random_nums[index]*4))
            else:
                sigma_new[k1][k2] = int(math.ceil(random_nums[index]*3+4))

    return sigma_new
            

def initial_potential_energy(sigma,energy,voltage,nhydrogen,phi,capacitance,phi_she,pH,kB,T,d_mu):
    n1, n2 = np.shape(sigma)
    q = np.zeros((n1,n2))
    u = np.zeros((n1,n2))
    for i1 in range(1, n1):
        for i2 in range(1, n2):
            i1p = i1 + 1
            if i1p > n1-1:
                i1p = 1
            i2p = i2 + 1
            if i2p > n2-1:
                i2p = 1
            e0 = energy[int(sigma[i1,i2]),int(sigma[i1p,i2]),
                        int(sigma[i1,i2p]),int(sigma[i1p,i2p])]
            v0 = voltage[int(sigma[i1,i2]),int(sigma[i1p,i2]),
                        int(sigma[i1,i2p]),int(sigma[i1p,i2p])]
            n0 = nhydrogen[int(sigma[i1,i2]),int(sigma[i1p,i2]),
                        int(sigma[i1,i2p]),int(sigma[i1p,i2p])]
            # The Hamiltonian used to calculate the energy and charge
            u[i1,i2] = 0.25*(e0 + v0*capacitance*(phi-v0) + n0*(phi - phi_she) + n0*(kB*T*math.log(10)*pH))# - capacitance*(phi-v0)*phi)# -(1/2)*n_new*EH2)
            q[i1,i2] = 0.25*(capacitance*(phi-v0))
    return u, q 
    
    
@njit(nogil=True, cache=True)
def new_potential_energy(u_old,q_old,sigma_new,energy,voltage,nhydrogen,phi,
                        capacitance,phi_she,p1,p2,pH,kB,T,d_mu):
    n1, n2 = sigma_new.shape
    n1 = n1 - 1 
    n2 = n2 - 1
    u_new = u_old.copy()
    q_new = q_old.copy()

    for i1 in range(int(p1)-1, int(p1)+2):
        for i2 in range(int(p2)-1, int(p2)+2):
            # i1 or i2 is equal to zero it will be out of the established array 
            # and should be wrapped around to the other side of the grid
            if i1 == 0:
                i1 = n1
            if i2 == 0:
                i2 == n2
            # Check to see if it is bigger than the grid to set it equal to 1
            if i1 > n1:
                diff = i1-n1
                i1 = diff
            if i2 > n2:
                diff = i2-n2
                i2 = diff
            i1p = i1 + 1
            i2p = i2 + 1
            if i1p > n1:
                diff = i1p-n1
                i1p = diff
            if i2p > n2:
                diff = i2p-n2
                i2p = diff
            e_new = energy[int(sigma_new[i1,i2]),int(sigma_new[i1p,i2]),
                        int(sigma_new[i1,i2p]),int(sigma_new[i1p,i2p])]           
            v_new = voltage[int(sigma_new[i1,i2]),int(sigma_new[i1p,i2]),
                        int(sigma_new[i1,i2p]),int(sigma_new[i1p,i2p])]            
            n_new = nhydrogen[int(sigma_new[i1,i2]),int(sigma_new[i1p,i2]),
                            int(sigma_new[i1,i2p]),int(sigma_new[i1p,i2p])]
            
            u_new[i1,i2] = 0.25*(e_new + v_new*capacitance*(phi-v_new) + n_new*(phi - phi_she) + n_new*(kB*T*math.log(10)*pH))# - capacitance*(phi-v_new)*phi)# -(1/2)*n_new*EH2)
            q_new[i1,i2] = 0.25*(capacitance*(phi-v_new))
                
    return u_new, q_new

def Counting(sigma_new, name, Oc, p1,p2):
    n1, n2 = sigma_new.shape
    n1 = n1 - 1 
    n2 = n2 - 1

    for i1 in range(int(p1)-1, int(p1)+2):
        for i2 in range(int(p2)-1, int(p2)+2):
            # i1 or i2 is equal to zero it will be out of the established array 
            # and should be wrapped around to the other side of the grid
            if i1 == 0:
                i1 = n1
            if i2 == 0:
                i2 == n2
            # Check to see if it is bigger than the grid to set it equal to 1
            if i1 > n1:
                diff = i1-n1
                i1 = diff
            if i2 > n2:
                diff = i2-n2
                i2 = diff
            i1p = i1 + 1
            i2p = i2 + 1
            if i1p > n1:
                diff = i1p-n1
                i1p = diff
            if i2p > n2:
                diff = i2p-n2
                i2p = diff
                
            name_new = str(int(name[int(sigma_new[i1,i2]),int(sigma_new[i1p,i2]),int(sigma_new[i1,i2p]),int(sigma_new[i1p,i2p])]))
            occurrence_new = Oc[int(sigma_new[i1,i2]),int(sigma_new[i1p,i2]),int(sigma_new[i1,i2p]),int(sigma_new[i1p,i2p])]
                
    return name_new, occurrence_new

def initialize_thermodynamic_data(area, gamma = 1):

    # Set an area of high energy values so that they will not be selected
    e = 100*np.ones((17,17,17,17))              
    v = np.zeros((17,17,17,17))
    n = np.zeros((17,17,17,17))
    c = np.zeros((17,17,17,17))
    occurrence = 10000*np.ones((17,17,17,17))
    name = 900*np.ones((17,17,17,17))
    
    #Non-subsurface Values
    
    #RuO2
    
    e[ 1, 5, 1, 5]= E_Dict['']      ;  v[ 1, 5, 1, 5]= F_Dict[''] ;  n[ 1, 5, 1, 5]=0 ; c[ 1, 5, 1, 5]=7.29 ; occurrence[ 1, 5, 1, 5]=1 ; name[ 1, 5, 1, 5]= 10 ; #-
    e[ 5, 1, 5, 1]= E_Dict['']      ;  v[ 5, 1, 5, 1]= F_Dict[''] ;  n[ 5, 1, 5, 1]=0 ; c[ 5, 1, 5, 1]=7.29 ; occurrence[ 5, 1, 5, 1]=1 ; name[ 5, 1, 5, 1]= 10 ;
    
    #H-RuO2
        
    e[ 2, 5, 1, 5]=E_Dict['2'] ;  v[ 2, 5, 1, 5]=F_Dict['2'] ;  n[ 2, 5, 1, 5]=1 ; c[ 2, 5, 1, 5]=9.49 ; occurrence[ 2, 5, 1, 5]=4 ; name[ 2, 5, 1, 5]=21 ; #2
    e[ 3, 5, 1, 5]=E_Dict['2'] ;  v[ 3, 5, 1, 5]=F_Dict['2'] ;  n[ 3, 5, 1, 5]=1 ; c[ 3, 5, 1, 5]=9.49 ; occurrence[ 3, 5, 1, 5]=4 ; name[ 3, 5, 1, 5]=21 ;
    e[ 1, 5, 2, 5]=E_Dict['2'] ;  v[ 1, 5, 2, 5]=F_Dict['2'] ;  n[ 1, 5, 2, 5]=1 ; c[ 1, 5, 2, 5]=9.49 ; occurrence[ 1, 5, 2, 5]=4 ; name[ 1, 5, 2, 5]=21 ; 
    e[ 1, 5, 3, 5]=E_Dict['2'] ;  v[ 1, 5, 3, 5]=F_Dict['2'] ;  n[ 1, 5, 3, 5]=1 ; c[ 1, 5, 3, 5]=9.49 ; occurrence[ 1, 5, 3, 5]=4 ; name[ 1, 5, 3, 5]=21 ; 
    e[ 5, 2, 5, 1]=E_Dict['2'] ;  v[ 5, 2, 5, 1]=F_Dict['2'] ;  n[ 5, 2, 5, 1]=1 ; c[ 5, 2, 5, 1]=9.49 ; occurrence[ 5, 2, 5, 1]=4 ; name[ 5, 2, 5, 1]=21 ; 
    e[ 5, 3, 5, 1]=E_Dict['2'] ;  v[ 5, 3, 5, 1]=F_Dict['2'] ;  n[ 5, 3, 5, 1]=1 ; c[ 5, 3, 5, 1]=9.49 ; occurrence[ 5, 3, 5, 1]=4 ; name[ 5, 3, 5, 1]=21 ; 
    e[ 5, 1, 5, 2]=E_Dict['2'] ;  v[ 5, 1, 5, 2]=F_Dict['2'] ;  n[ 5, 1, 5, 2]=1 ; c[ 5, 1, 5, 2]=9.49 ; occurrence[ 5, 1, 5, 2]=4 ; name[ 5, 1, 5, 2]=21 ; 
    e[ 5, 1, 5, 3]=E_Dict['2'] ;  v[ 5, 1, 5, 3]=F_Dict['2'] ;  n[ 5, 1, 5, 3]=1 ; c[ 5, 1, 5, 3]=9.49 ; occurrence[ 5, 1, 5, 3]=4 ; name[ 5, 1, 5, 3]=21 ; 
    
    e[ 1, 6, 1, 5]=E_Dict['0'] ;  v[ 1, 6, 1, 5]=F_Dict['0'] ;  n[ 1, 6, 1, 5]=1 ; c[ 1, 6, 1, 5]=9.84 ; occurrence[ 1, 6, 1, 5]=4 ; name[ 1, 6, 1, 5]=0 ; #0
    e[ 1, 7, 1, 5]=E_Dict['0'] ;  v[ 1, 7, 1, 5]=F_Dict['0'] ;  n[ 1, 7, 1, 5]=1 ; c[ 1, 7, 1, 5]=9.84 ; occurrence[ 1, 7, 1, 5]=4 ; name[ 1, 7, 1, 5]=0 ;  
    e[ 1, 5, 1, 6]=E_Dict['0'] ;  v[ 1, 5, 1, 6]=F_Dict['0'] ;  n[ 1, 5, 1, 6]=1 ; c[ 1, 5, 1, 6]=9.84 ; occurrence[ 1, 5, 1, 6]=4 ; name[ 1, 5, 1, 6]=0 ;
    e[ 1, 5, 1, 7]=E_Dict['0'] ;  v[ 1, 5, 1, 7]=F_Dict['0'] ;  n[ 1, 5, 1, 7]=1 ; c[ 1, 5, 1, 7]=9.84 ; occurrence[ 1, 5, 1, 7]=4 ; name[ 1, 5, 1, 7]=0 ;
    e[ 6, 1, 5, 1]=E_Dict['0'] ;  v[ 6, 1, 5, 1]=F_Dict['0'] ;  n[ 6, 1, 5, 1]=1 ; c[ 6, 1, 5, 1]=9.84 ; occurrence[ 6, 1, 5, 1]=4 ; name[ 6, 1, 5, 1]=0 ;
    e[ 7, 1, 5, 1]=E_Dict['0'] ;  v[ 7, 1, 5, 1]=F_Dict['0'] ;  n[ 7, 1, 5, 1]=1 ; c[ 7, 1, 5, 1]=9.84 ; occurrence[ 7, 1, 5, 1]=4 ; name[ 7, 1, 5, 1]=0 ;
    e[ 5, 1, 6, 1]=E_Dict['0'] ;  v[ 5, 1, 6, 1]=F_Dict['0'] ;  n[ 5, 1, 6, 1]=1 ; c[ 5, 1, 6, 1]=9.84 ; occurrence[ 5, 1, 6, 1]=4 ; name[ 5, 1, 6, 1]=0 ;
    e[ 5, 1, 7, 1]=E_Dict['0'] ;  v[ 5, 1, 7, 1]=F_Dict['0'] ;  n[ 5, 1, 7, 1]=1 ; c[ 5, 1, 7, 1]=9.84 ; occurrence[ 5, 1, 7, 1]=4 ; name[ 5, 1, 7, 1]=0 ;
    
    #2H-RuO2                        #24 missing: v=5.3251 c=8.75
        
    e[ 2, 5, 2, 5]=E_Dict['23'] ;  v[ 2, 5, 2, 5]=F_Dict['23'] ;  n[ 2, 5, 2, 5]=2 ; c[ 2, 5, 2, 5]=11.49 ; occurrence[ 2, 5, 2, 5]=2 ; name[ 2, 5, 2, 5]=231; #23
    e[ 3, 5, 3, 5]=E_Dict['23'] ;  v[ 3, 5, 3, 5]=F_Dict['23'] ;  n[ 3, 5, 3, 5]=2 ; c[ 3, 5, 3, 5]=11.49 ; occurrence[ 3, 5, 3, 5]=2 ; name[ 3, 5, 3, 5]=231;
    e[ 5, 2, 5, 2]=E_Dict['23'] ;  v[ 5, 2, 5, 2]=F_Dict['23'] ;  n[ 5, 2, 5, 2]=2 ; c[ 5, 2, 5, 2]=11.49 ; occurrence[ 5, 2, 5, 2]=2 ; name[ 5, 2, 5, 2]=231;
    e[ 5, 3, 5, 3]=E_Dict['23'] ;  v[ 5, 3, 5, 3]=F_Dict['23'] ;  n[ 5, 3, 5, 3]=2 ; c[ 5, 3, 5, 3]=11.49 ; occurrence[ 5, 3, 5, 3]=2 ; name[ 5, 3, 5, 3]=231;
    
    e[ 1, 6, 1, 6]=E_Dict['01'] ;  v[ 1, 6, 1, 6]=F_Dict['01'] ;  n[ 1, 6, 1, 6]=2 ; c[ 1, 6, 1, 6]=10.45 ; occurrence[ 1, 6, 1, 6]=2 ; name[ 1, 6, 1, 6]=1 ; #01 
    e[ 1, 7, 1, 7]=E_Dict['01'] ;  v[ 1, 7, 1, 7]=F_Dict['01'] ;  n[ 1, 7, 1, 7]=2 ; c[ 1, 7, 1, 7]=10.45 ; occurrence[ 1, 7, 1, 7]=2 ; name[ 1, 7, 1, 7]=1 ;     
    e[ 6, 1, 6, 1]=E_Dict['01'] ;  v[ 6, 1, 6, 1]=F_Dict['01'] ;  n[ 6, 1, 6, 1]=2 ; c[ 6, 1, 6, 1]=10.45 ; occurrence[ 6, 1, 6, 1]=2 ; name[ 6, 1, 6, 1]=1 ;
    e[ 7, 1, 7, 1]=E_Dict['01'] ;  v[ 7, 1, 7, 1]=F_Dict['01'] ;  n[ 7, 1, 7, 1]=2 ; c[ 7, 1, 7, 1]=10.45 ; occurrence[ 7, 1, 7, 1]=2 ; name[ 7, 1, 7, 1]=1 ;
    
    e[ 4, 5, 1, 5]=E_Dict['24'] ;  v[ 4, 5, 1, 5]=F_Dict['24'] ;  n[ 4, 5, 1, 5]=2 ; c[ 4, 5, 1, 5]=10.45 ; occurrence[ 4, 5, 1, 5]=2 ; name[ 4, 5, 1, 5]=241 ; #24
    e[ 1, 5, 4, 5]=E_Dict['24'] ;  v[ 1, 5, 4, 5]=F_Dict['24'] ;  n[ 1, 5, 4, 5]=2 ; c[ 1, 5, 4, 5]=10.45 ; occurrence[ 1, 5, 4, 5]=2 ; name[ 1, 5, 4, 5]=241 ;
    e[ 5, 4, 5, 1]=E_Dict['24'] ;  v[ 5, 4, 5, 1]=F_Dict['24'] ;  n[ 5, 4, 5, 1]=2 ; c[ 5, 4, 5, 1]=10.45 ; occurrence[ 5, 4, 5, 1]=2 ; name[ 5, 4, 5, 1]=241 ;
    e[ 5, 1, 5, 4]=E_Dict['24'] ;  v[ 5, 1, 5, 4]=F_Dict['24'] ;  n[ 5, 1, 5, 4]=2 ; c[ 5, 1, 5, 4]=10.45 ; occurrence[ 5, 1, 5, 4]=2 ; name[ 5, 1, 5, 4]=241 ;
    
    e[ 2, 5, 3, 5]=E_Dict['25'] ;  v[ 2, 5, 3, 5]=F_Dict['25'] ;  n[ 2, 5, 3, 5]=2 ; c[ 2, 5, 3, 5]=13.02 ; occurrence[ 2, 5, 3, 5]=2 ; name[ 2, 5, 3, 5]=251 ;  #25
    e[ 3, 5, 2, 5]=E_Dict['25'] ;  v[ 3, 5, 2, 5]=F_Dict['25'] ;  n[ 3, 5, 2, 5]=2 ; c[ 3, 5, 2, 5]=13.02 ; occurrence[ 3, 5, 2, 5]=2 ; name[ 3, 5, 2, 5]=251 ;
    e[ 5, 2, 5, 3]=E_Dict['25'] ;  v[ 5, 2, 5, 3]=F_Dict['25'] ;  n[ 5, 2, 5, 3]=2 ; c[ 5, 2, 5, 3]=13.02 ; occurrence[ 5, 2, 5, 3]=2 ; name[ 5, 2, 5, 3]=251 ;
    e[ 5, 3, 5, 2]=E_Dict['25'] ;  v[ 5, 3, 5, 2]=F_Dict['25'] ;  n[ 5, 3, 5, 2]=2 ; c[ 5, 3, 5, 2]=13.02 ; occurrence[ 5, 3, 5, 2]=2 ; name[ 5, 3, 5, 2]=251 ;
    
    e[ 2, 5, 1, 6]=E_Dict['03'] ;  v[ 2, 5, 1, 6]=F_Dict['03'] ;  n[ 2, 5, 1, 6]=2 ; c[ 2, 5, 1, 6]=8.22 ; occurrence[ 2, 5, 1, 6]=8 ; name[ 2, 5, 1, 6]=3 ; #03 
    e[ 3, 5, 1, 6]=E_Dict['03'] ;  v[ 3, 5, 1, 6]=F_Dict['03'] ;  n[ 3, 5, 1, 6]=2 ; c[ 3, 5, 1, 6]=8.22 ; occurrence[ 3, 5, 1, 6]=8 ; name[ 3, 5, 1, 6]=3 ;
    e[ 2, 5, 1, 7]=E_Dict['03'] ;  v[ 2, 5, 1, 7]=F_Dict['03'] ;  n[ 2, 5, 1, 7]=2 ; c[ 2, 5, 1, 7]=8.22 ; occurrence[ 2, 5, 1, 7]=8 ; name[ 2, 5, 1, 7]=3 ;
    e[ 3, 5, 1, 7]=E_Dict['03'] ;  v[ 3, 5, 1, 7]=F_Dict['03'] ;  n[ 3, 5, 1, 7]=2 ; c[ 3, 5, 1, 7]=8.22 ; occurrence[ 3, 5, 1, 7]=8 ; name[ 3, 5, 1, 7]=3 ;
    e[ 1, 6, 2, 5]=E_Dict['03'] ;  v[ 1, 6, 2, 5]=F_Dict['03'] ;  n[ 1, 6, 2, 5]=2 ; c[ 1, 6, 2, 5]=8.22 ; occurrence[ 1, 6, 2, 5]=8 ; name[ 1, 6, 2, 5]=3 ;
    e[ 1, 6, 3, 5]=E_Dict['03'] ;  v[ 1, 6, 3, 5]=F_Dict['03'] ;  n[ 1, 6, 3, 5]=2 ; c[ 1, 6, 3, 5]=8.22 ; occurrence[ 1, 6, 3, 5]=8 ; name[ 1, 6, 3, 5]=3 ;
    e[ 1, 7, 2, 5]=E_Dict['03'] ;  v[ 1, 7, 2, 5]=F_Dict['03'] ;  n[ 1, 7, 2, 5]=2 ; c[ 1, 7, 2, 5]=8.22 ; occurrence[ 1, 7, 2, 5]=8 ; name[ 1, 7, 2, 5]=3 ;
    e[ 1, 7, 3, 5]=E_Dict['03'] ;  v[ 1, 7, 3, 5]=F_Dict['03'] ;  n[ 1, 7, 3, 5]=2 ; c[ 1, 7, 3, 5]=8.22 ; occurrence[ 1, 7, 3, 5]=8 ; name[ 1, 7, 3, 5]=3 ;
    e[ 5, 2, 6, 1]=E_Dict['03'] ;  v[ 5, 2, 6, 1]=F_Dict['03'] ;  n[ 5, 2, 6, 1]=2 ; c[ 5, 2, 6, 1]=8.22 ; occurrence[ 5, 2, 6, 1]=8 ; name[ 5, 2, 6, 1]=3 ;
    e[ 5, 3, 6, 1]=E_Dict['03'] ;  v[ 5, 3, 6, 1]=F_Dict['03'] ;  n[ 5, 3, 6, 1]=2 ; c[ 5, 3, 6, 1]=8.22 ; occurrence[ 5, 3, 6, 1]=8 ; name[ 5, 3, 6, 1]=3 ;
    e[ 5, 2, 7, 1]=E_Dict['03'] ;  v[ 5, 2, 7, 1]=F_Dict['03'] ;  n[ 5, 2, 7, 1]=2 ; c[ 5, 2, 7, 1]=8.22 ; occurrence[ 5, 2, 7, 1]=8 ; name[ 5, 2, 7, 1]=3 ;
    e[ 5, 3, 7, 1]=E_Dict['03'] ;  v[ 5, 3, 7, 1]=F_Dict['03'] ;  n[ 5, 3, 7, 1]=2 ; c[ 5, 3, 7, 1]=8.22 ; occurrence[ 5, 3, 7, 1]=8 ; name[ 5, 3, 7, 1]=3 ;
    e[ 6, 1, 5, 2]=E_Dict['03'] ;  v[ 6, 1, 5, 2]=F_Dict['03'] ;  n[ 6, 1, 5, 2]=2 ; c[ 6, 1, 5, 2]=8.22 ; occurrence[ 6, 1, 5, 2]=8 ; name[ 6, 1, 5, 2]=3 ;
    e[ 6, 1, 5, 3]=E_Dict['03'] ;  v[ 6, 1, 5, 3]=F_Dict['03'] ;  n[ 6, 1, 5, 3]=2 ; c[ 6, 1, 5, 3]=8.22 ; occurrence[ 6, 1, 5, 3]=8 ; name[ 6, 1, 5, 3]=3 ;
    e[ 7, 1, 5, 2]=E_Dict['03'] ;  v[ 7, 1, 5, 2]=F_Dict['03'] ;  n[ 7, 1, 5, 2]=2 ; c[ 7, 1, 5, 2]=8.22 ; occurrence[ 7, 1, 5, 2]=8 ; name[ 7, 1, 5, 2]=3 ;
    e[ 7, 1, 5, 3]=E_Dict['03'] ;  v[ 7, 1, 5, 3]=F_Dict['03'] ;  n[ 7, 1, 5, 3]=2 ; c[ 7, 1, 5, 3]=8.22 ; occurrence[ 7, 1, 5, 3]=8 ; name[ 7, 1, 5, 3]=3 ;

    e[ 2, 6, 1, 5]=E_Dict['02'] ;  v[ 2, 6, 1, 5]=F_Dict['02'] ;  n[ 2, 6, 1, 5]=2 ; c[ 2, 6, 1, 5]=8.52 ; occurrence[ 2, 6, 1, 5]=8 ; name[ 2, 6, 1, 5]=2 ; #02
    e[ 3, 6, 1, 5]=E_Dict['02'] ;  v[ 3, 6, 1, 5]=F_Dict['02'] ;  n[ 3, 6, 1, 5]=2 ; c[ 3, 6, 1, 5]=8.52 ; occurrence[ 3, 6, 1, 5]=8 ; name[ 3, 6, 1, 5]=2 ;
    e[ 2, 7, 1, 5]=E_Dict['02'] ;  v[ 2, 7, 1, 5]=F_Dict['02'] ;  n[ 2, 7, 1, 5]=2 ; c[ 2, 7, 1, 5]=8.52 ; occurrence[ 2, 7, 1, 5]=8 ; name[ 2, 7, 1, 5]=2 ;
    e[ 3, 7, 1, 5]=E_Dict['02'] ;  v[ 3, 7, 1, 5]=F_Dict['02'] ;  n[ 3, 7, 1, 5]=2 ; c[ 3, 7, 1, 5]=8.52 ; occurrence[ 3, 7, 1, 5]=8 ; name[ 3, 7, 1, 5]=2 ;
    e[ 1, 5, 2, 6]=E_Dict['02'] ;  v[ 1, 5, 2, 6]=F_Dict['02'] ;  n[ 1, 5, 2, 6]=2 ; c[ 1, 5, 2, 6]=8.52 ; occurrence[ 1, 5, 2, 6]=8 ; name[ 1, 5, 2, 6]=2 ;
    e[ 1, 5, 3, 6]=E_Dict['02'] ;  v[ 1, 5, 3, 6]=F_Dict['02'] ;  n[ 1, 5, 3, 6]=2 ; c[ 1, 5, 3, 6]=8.52 ; occurrence[ 1, 5, 3, 6]=8 ; name[ 1, 5, 3, 6]=2 ;
    e[ 1, 5, 2, 7]=E_Dict['02'] ;  v[ 1, 5, 2, 7]=F_Dict['02'] ;  n[ 1, 5, 2, 7]=2 ; c[ 1, 5, 2, 7]=8.52 ; occurrence[ 1, 5, 2, 7]=8 ; name[ 1, 5, 2, 7]=2 ;
    e[ 1, 5, 3, 7]=E_Dict['02'] ;  v[ 1, 5, 3, 7]=F_Dict['02'] ;  n[ 1, 5, 3, 7]=2 ; c[ 1, 5, 3, 7]=8.52 ; occurrence[ 1, 5, 3, 7]=8 ; name[ 1, 5, 3, 7]=2 ;
    e[ 6, 2, 5, 1]=E_Dict['02'] ;  v[ 6, 2, 5, 1]=F_Dict['02'] ;  n[ 6, 2, 5, 1]=2 ; c[ 6, 2, 5, 1]=8.52 ; occurrence[ 6, 2, 5, 1]=8 ; name[ 6, 2, 5, 1]=2 ;
    e[ 6, 3, 5, 1]=E_Dict['02'] ;  v[ 6, 3, 5, 1]=F_Dict['02'] ;  n[ 6, 3, 5, 1]=2 ; c[ 6, 3, 5, 1]=8.52 ; occurrence[ 6, 3, 5, 1]=8 ; name[ 6, 3, 5, 1]=2 ;
    e[ 7, 2, 5, 1]=E_Dict['02'] ;  v[ 7, 2, 5, 1]=F_Dict['02'] ;  n[ 7, 2, 5, 1]=2 ; c[ 7, 2, 5, 1]=8.52 ; occurrence[ 7, 2, 5, 1]=8 ; name[ 7, 2, 5, 1]=2 ;
    e[ 7, 3, 5, 1]=E_Dict['02'] ;  v[ 7, 3, 5, 1]=F_Dict['02'] ;  n[ 7, 3, 5, 1]=2 ; c[ 7, 3, 5, 1]=8.52 ; occurrence[ 7, 3, 5, 1]=8 ; name[ 7, 3, 5, 1]=2 ;
    e[ 5, 1, 6, 2]=E_Dict['02'] ;  v[ 5, 1, 6, 2]=F_Dict['02'] ;  n[ 5, 1, 6, 2]=2 ; c[ 5, 1, 6, 2]=8.52 ; occurrence[ 5, 1, 6, 2]=8 ; name[ 5, 1, 6, 2]=2 ;
    e[ 5, 1, 6, 3]=E_Dict['02'] ;  v[ 5, 1, 6, 3]=F_Dict['02'] ;  n[ 5, 1, 6, 3]=2 ; c[ 5, 1, 6, 3]=8.52 ; occurrence[ 5, 1, 6, 3]=8 ; name[ 5, 1, 6, 3]=2 ;
    e[ 5, 1, 7, 2]=E_Dict['02'] ;  v[ 5, 1, 7, 2]=F_Dict['02'] ;  n[ 5, 1, 7, 2]=2 ; c[ 5, 1, 7, 2]=8.52 ; occurrence[ 5, 1, 7, 2]=8 ; name[ 5, 1, 7, 2]=2 ;
    e[ 5, 1, 7, 3]=E_Dict['02'] ;  v[ 5, 1, 7, 3]=F_Dict['02'] ;  n[ 5, 1, 7, 3]=2 ; c[ 5, 1, 7, 3]=8.52 ; occurrence[ 5, 1, 7, 3]=8 ; name[ 5, 1, 7, 3]=2 ;
    
    #3H-RuO2                        #234 missing: v=5.53 c=9.23
        
    e[ 1, 6, 2, 6]=E_Dict['012'] ;  v[ 1, 6, 2, 6]=F_Dict['012'] ;  n[ 1, 6, 2, 6]=3 ; c[ 1, 6, 2, 6]=7.98 ; occurrence[ 1, 6, 2, 6]=8 ; name[ 1, 6, 2, 6]=12 ; #012
    e[ 1, 7, 2, 7]=E_Dict['012'] ;  v[ 1, 7, 2, 7]=F_Dict['012'] ;  n[ 1, 7, 2, 7]=3 ; c[ 1, 7, 2, 7]=7.98 ; occurrence[ 1, 7, 2, 7]=8 ; name[ 1, 7, 2, 7]=12 ;
    e[ 2, 6, 1, 6]=E_Dict['012'] ;  v[ 2, 6, 1, 6]=F_Dict['012'] ;  n[ 2, 6, 1, 6]=3 ; c[ 2, 6, 1, 6]=7.98 ; occurrence[ 2, 6, 1, 6]=8 ; name[ 2, 6, 1, 6]=12 ;
    e[ 2, 7, 1, 7]=E_Dict['012'] ;  v[ 2, 7, 1, 7]=F_Dict['012'] ;  n[ 2, 7, 1, 7]=3 ; c[ 2, 7, 1, 7]=7.98 ; occurrence[ 2, 7, 1, 7]=8 ; name[ 2, 7, 1, 7]=12 ;
    e[ 1, 6, 3, 6]=E_Dict['012'] ;  v[ 1, 6, 3, 6]=F_Dict['012'] ;  n[ 1, 6, 3, 6]=3 ; c[ 1, 6, 3, 6]=7.98 ; occurrence[ 1, 6, 3, 6]=8 ; name[ 1, 6, 3, 6]=12 ;
    e[ 1, 7, 3, 7]=E_Dict['012'] ;  v[ 1, 7, 3, 7]=F_Dict['012'] ;  n[ 1, 7, 3, 7]=3 ; c[ 1, 7, 3, 7]=7.98 ; occurrence[ 1, 7, 3, 7]=8 ; name[ 1, 7, 3, 7]=12 ;
    e[ 3, 6, 1, 6]=E_Dict['012'] ;  v[ 3, 6, 1, 6]=F_Dict['012'] ;  n[ 3, 6, 1, 6]=3 ; c[ 3, 6, 1, 6]=7.98 ; occurrence[ 3, 6, 1, 6]=8 ; name[ 3, 6, 1, 6]=12 ;
    e[ 3, 7, 1, 7]=E_Dict['012'] ;  v[ 3, 7, 1, 7]=F_Dict['012'] ;  n[ 3, 7, 1, 7]=3 ; c[ 3, 7, 1, 7]=7.98 ; occurrence[ 3, 7, 1, 7]=8 ; name[ 3, 7, 1, 7]=12 ;
    e[ 6, 1, 6, 2]=E_Dict['012'] ;  v[ 6, 1, 6, 2]=F_Dict['012'] ;  n[ 6, 1, 6, 2]=3 ; c[ 6, 1, 6, 2]=7.98 ; occurrence[ 6, 1, 6, 2]=8 ; name[ 6, 1, 6, 2]=12 ;
    e[ 7, 1, 7, 2]=E_Dict['012'] ;  v[ 7, 1, 7, 2]=F_Dict['012'] ;  n[ 7, 1, 7, 2]=3 ; c[ 7, 1, 7, 2]=7.98 ; occurrence[ 7, 1, 7, 2]=8 ; name[ 7, 1, 7, 2]=12 ;
    e[ 6, 2, 6, 1]=E_Dict['012'] ;  v[ 6, 2, 6, 1]=F_Dict['012'] ;  n[ 6, 2, 6, 1]=3 ; c[ 6, 2, 6, 1]=7.98 ; occurrence[ 6, 2, 6, 1]=8 ; name[ 6, 2, 6, 1]=12 ;
    e[ 7, 2, 7, 1]=E_Dict['012'] ;  v[ 7, 2, 7, 1]=F_Dict['012'] ;  n[ 7, 2, 7, 1]=3 ; c[ 7, 2, 7, 1]=7.98 ; occurrence[ 7, 2, 7, 1]=8 ; name[ 7, 2, 7, 1]=12 ;
    e[ 6, 1, 6, 3]=E_Dict['012'] ;  v[ 6, 1, 6, 3]=F_Dict['012'] ;  n[ 6, 1, 6, 3]=3 ; c[ 6, 1, 6, 3]=7.98 ; occurrence[ 6, 1, 6, 3]=8 ; name[ 6, 1, 6, 3]=12 ;
    e[ 7, 1, 7, 3]=E_Dict['012'] ;  v[ 7, 1, 7, 3]=F_Dict['012'] ;  n[ 7, 1, 7, 3]=3 ; c[ 7, 1, 7, 3]=7.98 ; occurrence[ 7, 1, 7, 3]=8 ; name[ 7, 1, 7, 3]=12 ;
    e[ 6, 3, 6, 1]=E_Dict['012'] ;  v[ 6, 3, 6, 1]=F_Dict['012'] ;  n[ 6, 3, 6, 1]=3 ; c[ 6, 3, 6, 1]=7.98 ; occurrence[ 6, 3, 6, 1]=8 ; name[ 6, 3, 6, 1]=12 ;
    e[ 7, 3, 7, 1]=E_Dict['012'] ;  v[ 7, 3, 7, 1]=F_Dict['012'] ;  n[ 7, 3, 7, 1]=3 ; c[ 7, 3, 7, 1]=7.98 ; occurrence[ 7, 3, 7, 1]=8 ; name[ 7, 3, 7, 1]=12 ;
    
    e[ 2, 6, 2, 5]=E_Dict['023'] ;  v[ 2, 6, 2, 5]=F_Dict['023'] ;  n[ 2, 6, 2, 5]=3 ; c[ 2, 6, 2, 5]=9.17 ; occurrence[ 2, 6, 2, 5]=8 ; name[ 2, 6, 2, 5]=23 ; #023
    e[ 3, 6, 3, 5]=E_Dict['023'] ;  v[ 3, 6, 3, 5]=F_Dict['023'] ;  n[ 3, 6, 3, 5]=3 ; c[ 3, 6, 3, 5]=9.17 ; occurrence[ 3, 6, 3, 5]=8 ; name[ 3, 6, 3, 5]=23 ;
    e[ 2, 5, 2, 6]=E_Dict['023'] ;  v[ 2, 5, 2, 6]=F_Dict['023'] ;  n[ 2, 5, 2, 6]=3 ; c[ 2, 5, 2, 6]=9.17 ; occurrence[ 2, 5, 2, 6]=8 ; name[ 2, 5, 2, 6]=23 ;
    e[ 3, 5, 3, 6]=E_Dict['023'] ;  v[ 3, 5, 3, 6]=F_Dict['023'] ;  n[ 3, 5, 3, 6]=3 ; c[ 3, 5, 3, 6]=9.17 ; occurrence[ 3, 5, 3, 6]=8 ; name[ 3, 5, 3, 6]=23 ;
    e[ 2, 7, 2, 5]=E_Dict['023'] ;  v[ 2, 7, 2, 5]=F_Dict['023'] ;  n[ 2, 7, 2, 5]=3 ; c[ 2, 7, 2, 5]=9.17 ; occurrence[ 2, 7, 2, 5]=8 ; name[ 2, 7, 2, 5]=23 ;
    e[ 3, 7, 3, 5]=E_Dict['023'] ;  v[ 3, 7, 3, 5]=F_Dict['023'] ;  n[ 3, 7, 3, 5]=3 ; c[ 3, 7, 3, 5]=9.17 ; occurrence[ 3, 7, 3, 5]=8 ; name[ 3, 7, 3, 5]=23 ;
    e[ 2, 5, 2, 7]=E_Dict['023'] ;  v[ 2, 5, 2, 7]=F_Dict['023'] ;  n[ 2, 5, 2, 7]=3 ; c[ 2, 5, 2, 7]=9.17 ; occurrence[ 2, 5, 2, 7]=8 ; name[ 2, 5, 2, 7]=23 ;
    e[ 3, 5, 3, 7]=E_Dict['023'] ;  v[ 3, 5, 3, 7]=F_Dict['023'] ;  n[ 3, 5, 3, 7]=3 ; c[ 3, 5, 3, 7]=9.17 ; occurrence[ 3, 5, 3, 7]=8 ; name[ 3, 5, 3, 7]=23 ;
    e[ 6, 2, 5, 2]=E_Dict['023'] ;  v[ 6, 2, 5, 2]=F_Dict['023'] ;  n[ 6, 2, 5, 2]=3 ; c[ 6, 2, 5, 2]=9.17 ; occurrence[ 6, 2, 5, 2]=8 ; name[ 6, 2, 5, 2]=23 ;
    e[ 6, 3, 5, 3]=E_Dict['023'] ;  v[ 6, 3, 5, 3]=F_Dict['023'] ;  n[ 6, 3, 5, 3]=3 ; c[ 6, 3, 5, 3]=9.17 ; occurrence[ 6, 3, 5, 3]=8 ; name[ 6, 3, 5, 3]=23 ;
    e[ 5, 2, 6, 2]=E_Dict['023'] ;  v[ 5, 2, 6, 2]=F_Dict['023'] ;  n[ 5, 2, 6, 2]=3 ; c[ 5, 2, 6, 2]=9.17 ; occurrence[ 5, 2, 6, 2]=8 ; name[ 5, 2, 6, 2]=23 ;
    e[ 5, 3, 6, 3]=E_Dict['023'] ;  v[ 5, 3, 6, 3]=F_Dict['023'] ;  n[ 5, 3, 6, 3]=3 ; c[ 5, 3, 6, 3]=9.17 ; occurrence[ 5, 3, 6, 3]=8 ; name[ 5, 3, 6, 3]=23 ;
    e[ 7, 2, 5, 2]=E_Dict['023'] ;  v[ 7, 2, 5, 2]=F_Dict['023'] ;  n[ 7, 2, 5, 2]=3 ; c[ 7, 2, 5, 2]=9.17 ; occurrence[ 7, 2, 5, 2]=8 ; name[ 7, 2, 5, 2]=23 ;
    e[ 7, 3, 5, 3]=E_Dict['023'] ;  v[ 7, 3, 5, 3]=F_Dict['023'] ;  n[ 7, 3, 5, 3]=3 ; c[ 7, 3, 5, 3]=9.17 ; occurrence[ 7, 3, 5, 3]=8 ; name[ 7, 3, 5, 3]=23 ;
    e[ 5, 2, 7, 2]=E_Dict['023'] ;  v[ 5, 2, 7, 2]=F_Dict['023'] ;  n[ 5, 2, 7, 2]=3 ; c[ 5, 2, 7, 2]=9.17 ; occurrence[ 5, 2, 7, 2]=8 ; name[ 5, 2, 7, 2]=23 ;
    e[ 5, 3, 7, 3]=E_Dict['023'] ;  v[ 5, 3, 7, 3]=F_Dict['023'] ;  n[ 5, 3, 7, 3]=3 ; c[ 5, 3, 7, 3]=9.17 ; occurrence[ 5, 3, 7, 3]=8 ; name[ 5, 3, 7, 3]=23 ;
    
    e[ 4, 6, 1, 5]=E_Dict['024'] ;  v[ 4, 6, 1, 5]=F_Dict['024'] ;  n[ 4, 6, 1, 5]=3 ; c[ 4, 6, 1, 5]=8.08 ; occurrence[ 4, 6, 1, 5]=4 ; name[ 4, 6, 1, 5]=24 ; #024
    e[ 4, 7, 1, 5]=E_Dict['024'] ;  v[ 4, 7, 1, 5]=F_Dict['024'] ;  n[ 4, 7, 1, 5]=3 ; c[ 4, 7, 1, 5]=8.08 ; occurrence[ 4, 7, 1, 5]=4 ; name[ 4, 7, 1, 5]=24 ;
    e[ 1, 5, 4, 6]=E_Dict['024'] ;  v[ 1, 5, 4, 6]=F_Dict['024'] ;  n[ 1, 5, 4, 6]=3 ; c[ 1, 5, 4, 6]=8.08 ; occurrence[ 1, 5, 4, 6]=4 ; name[ 1, 5, 4, 6]=24 ;
    e[ 1, 5, 4, 7]=E_Dict['024'] ;  v[ 1, 5, 4, 7]=F_Dict['024'] ;  n[ 1, 5, 4, 7]=3 ; c[ 1, 5, 4, 7]=8.08 ; occurrence[ 1, 5, 4, 7]=4 ; name[ 1, 5, 4, 7]=24 ;
    e[ 6, 4, 5, 1]=E_Dict['024'] ;  v[ 6, 4, 5, 1]=F_Dict['024'] ;  n[ 6, 4, 5, 1]=3 ; c[ 6, 4, 5, 1]=8.08 ; occurrence[ 6, 4, 5, 1]=4 ; name[ 6, 4, 5, 1]=24 ;
    e[ 7, 4, 5, 1]=E_Dict['024'] ;  v[ 7, 4, 5, 1]=F_Dict['024'] ;  n[ 7, 4, 5, 1]=3 ; c[ 7, 4, 5, 1]=8.08 ; occurrence[ 7, 4, 5, 1]=4 ; name[ 7, 4, 5, 1]=24 ;
    e[ 5, 1, 6, 4]=E_Dict['024'] ;  v[ 5, 1, 6, 4]=F_Dict['024'] ;  n[ 5, 1, 6, 4]=3 ; c[ 5, 1, 6, 4]=8.08 ; occurrence[ 5, 1, 6, 4]=4 ; name[ 5, 1, 6, 4]=24 ;
    e[ 5, 1, 7, 4]=E_Dict['024'] ;  v[ 5, 1, 7, 4]=F_Dict['024'] ;  n[ 5, 1, 7, 4]=3 ; c[ 5, 1, 7, 4]=8.08 ; occurrence[ 5, 1, 7, 4]=4 ; name[ 5, 1, 7, 4]=24 ;
    
    e[ 2, 6, 3, 5]=E_Dict['025'] ;  v[ 2, 6, 3, 5]=F_Dict['025'] ;  n[ 2, 6, 3, 5]=3 ; c[ 2, 6, 3, 5]=9.04 ; occurrence[ 2, 6, 3, 5]=8 ; name[ 2, 6, 3, 5]=25 ; #025
    e[ 3, 6, 2, 5]=E_Dict['025'] ;  v[ 3, 6, 2, 5]=F_Dict['025'] ;  n[ 3, 6, 2, 5]=3 ; c[ 3, 6, 2, 5]=9.04 ; occurrence[ 3, 6, 2, 5]=8 ; name[ 3, 6, 2, 5]=25 ;
    e[ 2, 5, 3, 6]=E_Dict['025'] ;  v[ 2, 5, 3, 6]=F_Dict['025'] ;  n[ 2, 5, 3, 6]=3 ; c[ 2, 5, 3, 6]=9.04 ; occurrence[ 2, 5, 3, 6]=8 ; name[ 2, 5, 3, 6]=25 ;
    e[ 3, 5, 2, 6]=E_Dict['025'] ;  v[ 3, 5, 2, 6]=F_Dict['025'] ;  n[ 3, 5, 2, 6]=3 ; c[ 3, 5, 2, 6]=9.04 ; occurrence[ 3, 5, 2, 6]=8 ; name[ 3, 5, 2, 6]=25 ;
    e[ 2, 7, 3, 5]=E_Dict['025'] ;  v[ 2, 7, 3, 5]=F_Dict['025'] ;  n[ 2, 7, 3, 5]=3 ; c[ 2, 7, 3, 5]=9.04 ; occurrence[ 2, 7, 3, 5]=8 ; name[ 2, 7, 3, 5]=25 ;
    e[ 3, 7, 2, 5]=E_Dict['025'] ;  v[ 3, 7, 2, 5]=F_Dict['025'] ;  n[ 3, 7, 2, 5]=3 ; c[ 3, 7, 2, 5]=9.04 ; occurrence[ 3, 7, 2, 5]=8 ; name[ 3, 7, 2, 5]=25 ;
    e[ 2, 5, 3, 7]=E_Dict['025'] ;  v[ 2, 5, 3, 7]=F_Dict['025'] ;  n[ 2, 5, 3, 7]=3 ; c[ 2, 5, 3, 7]=9.04 ; occurrence[ 2, 5, 3, 7]=8 ; name[ 2, 5, 3, 7]=25 ;
    e[ 3, 5, 2, 7]=E_Dict['025'] ;  v[ 3, 5, 2, 7]=F_Dict['025'] ;  n[ 3, 5, 2, 7]=3 ; c[ 3, 5, 2, 7]=9.04 ; occurrence[ 3, 5, 2, 7]=8 ; name[ 3, 5, 2, 7]=25 ;
    e[ 6, 2, 5, 3]=E_Dict['025'] ;  v[ 6, 2, 5, 3]=F_Dict['025'] ;  n[ 6, 2, 5, 3]=3 ; c[ 6, 2, 5, 3]=9.04 ; occurrence[ 6, 2, 5, 3]=8 ; name[ 6, 2, 5, 3]=25 ;
    e[ 6, 3, 5, 2]=E_Dict['025'] ;  v[ 6, 3, 5, 2]=F_Dict['025'] ;  n[ 6, 3, 5, 2]=3 ; c[ 6, 3, 5, 2]=9.04 ; occurrence[ 6, 3, 5, 2]=8 ; name[ 6, 3, 5, 2]=25 ;
    e[ 5, 2, 6, 3]=E_Dict['025'] ;  v[ 5, 2, 6, 3]=F_Dict['025'] ;  n[ 5, 2, 6, 3]=3 ; c[ 5, 2, 6, 3]=9.04 ; occurrence[ 5, 2, 6, 3]=8 ; name[ 5, 2, 6, 3]=25 ;
    e[ 5, 3, 6, 2]=E_Dict['025'] ;  v[ 5, 3, 6, 2]=F_Dict['025'] ;  n[ 5, 3, 6, 2]=3 ; c[ 5, 3, 6, 2]=9.04 ; occurrence[ 5, 3, 6, 2]=8 ; name[ 5, 3, 6, 2]=25 ;
    e[ 7, 2, 5, 3]=E_Dict['025'] ;  v[ 7, 2, 5, 3]=F_Dict['025'] ;  n[ 7, 2, 5, 3]=3 ; c[ 7, 2, 5, 3]=9.04 ; occurrence[ 7, 2, 5, 3]=8 ; name[ 7, 2, 5, 3]=25 ;
    e[ 7, 3, 5, 2]=E_Dict['025'] ;  v[ 7, 3, 5, 2]=F_Dict['025'] ;  n[ 7, 3, 5, 2]=3 ; c[ 7, 3, 5, 2]=9.04 ; occurrence[ 7, 3, 5, 2]=8 ; name[ 7, 3, 5, 2]=25 ;
    e[ 5, 2, 7, 3]=E_Dict['025'] ;  v[ 5, 2, 7, 3]=F_Dict['025'] ;  n[ 5, 2, 7, 3]=3 ; c[ 5, 2, 7, 3]=9.04 ; occurrence[ 5, 2, 7, 3]=8 ; name[ 5, 2, 7, 3]=25 ;
    e[ 5, 3, 7, 2]=E_Dict['025'] ;  v[ 5, 3, 7, 2]=F_Dict['025'] ;  n[ 5, 3, 7, 2]=3 ; c[ 5, 3, 7, 2]=9.04 ; occurrence[ 5, 3, 7, 2]=8 ; name[ 5, 3, 7, 2]=25 ;
    
    e[ 4, 5, 1, 6]=E_Dict['035'] ;  v[ 4, 5, 1, 6]=F_Dict['035'] ;  n[ 4, 5, 1, 6]=3 ; c[ 4, 5, 1, 6]=8.09 ; occurrence[ 4, 5, 1, 6]=4 ; name[ 4, 5, 1, 6]=35 ; #035
    e[ 4, 5, 1, 7]=E_Dict['035'] ;  v[ 4, 5, 1, 7]=F_Dict['035'] ;  n[ 4, 5, 1, 7]=3 ; c[ 4, 5, 1, 7]=8.09 ; occurrence[ 4, 5, 1, 7]=4 ; name[ 4, 5, 1, 7]=35 ;
    e[ 1, 6, 4, 5]=E_Dict['035'] ;  v[ 1, 6, 4, 5]=F_Dict['035'] ;  n[ 1, 6, 4, 5]=3 ; c[ 1, 6, 4, 5]=8.09 ; occurrence[ 1, 6, 4, 5]=4 ; name[ 1, 6, 4, 5]=35 ;
    e[ 1, 7, 4, 5]=E_Dict['035'] ;  v[ 1, 7, 4, 5]=F_Dict['035'] ;  n[ 1, 7, 4, 5]=3 ; c[ 1, 7, 4, 5]=8.09 ; occurrence[ 1, 7, 4, 5]=4 ; name[ 1, 7, 4, 5]=35 ;
    e[ 5, 4, 6, 1]=E_Dict['035'] ;  v[ 5, 4, 6, 1]=F_Dict['035'] ;  n[ 5, 4, 6, 1]=3 ; c[ 5, 4, 6, 1]=8.09 ; occurrence[ 5, 4, 6, 1]=4 ; name[ 5, 4, 6, 1]=35 ;
    e[ 5, 4, 7, 1]=E_Dict['035'] ;  v[ 5, 4, 7, 1]=F_Dict['035'] ;  n[ 5, 4, 7, 1]=3 ; c[ 5, 4, 7, 1]=8.09 ; occurrence[ 5, 4, 7, 1]=4 ; name[ 5, 4, 7, 1]=35 ;
    e[ 6, 1, 5, 4]=E_Dict['035'] ;  v[ 6, 1, 5, 4]=F_Dict['035'] ;  n[ 6, 1, 5, 4]=3 ; c[ 6, 1, 5, 4]=8.09 ; occurrence[ 6, 1, 5, 4]=4 ; name[ 6, 1, 5, 4]=35 ;
    e[ 7, 1, 5, 4]=E_Dict['035'] ;  v[ 7, 1, 5, 4]=F_Dict['035'] ;  n[ 7, 1, 5, 4]=3 ; c[ 7, 1, 5, 4]=8.09 ; occurrence[ 7, 1, 5, 4]=4 ; name[ 7, 1, 5, 4]=35 ;
    
    e[ 5, 2, 5, 4]=E_Dict['234'] ;  v[ 5, 2, 5, 4]=F_Dict['234'] ;  n[ 5, 2, 5, 4]= 0 ; c[ 5, 2, 5, 4]= 0 ; occurrence[ 5, 2, 5, 4]=4 ; name[ 5, 2, 5, 4]=2341 ; #234
    e[ 2, 5, 4, 5]=E_Dict['234'] ;  v[ 2, 5, 4, 5]=F_Dict['234'] ;  n[ 2, 5, 4, 5]= 0 ; c[ 2, 5, 4, 5]= 0 ; occurrence[ 2, 5, 4, 5]=4 ; name[ 2, 5, 4, 5]=2341 ;
    e[ 5, 3, 5, 4]=E_Dict['234'] ;  v[ 5, 3, 5, 4]=F_Dict['234'] ;  n[ 5, 3, 5, 4]= 0 ; c[ 5, 3, 5, 4]= 0 ; occurrence[ 5, 3, 5, 4]=4 ; name[ 5, 3, 5, 4]=2341 ;
    e[ 3, 5, 4, 5]=E_Dict['234'] ;  v[ 3, 5, 4, 5]=F_Dict['234'] ;  n[ 3, 5, 4, 5]= 0 ; c[ 3, 5, 4, 5]= 0 ; occurrence[ 3, 5, 4, 5]=4 ; name[ 3, 5, 4, 5]=2341 ;
    e[ 5, 4, 5, 2]=E_Dict['234'] ;  v[ 5, 4, 5, 2]=F_Dict['234'] ;  n[ 5, 4, 5, 2]= 0 ; c[ 5, 4, 5, 2]= 0 ; occurrence[ 5, 4, 5, 2]=4 ; name[ 5, 4, 5, 2]=2341 ;
    e[ 4, 5, 2, 5]=E_Dict['234'] ;  v[ 4, 5, 2, 5]=F_Dict['234'] ;  n[ 4, 5, 2, 5]= 0 ; c[ 4, 5, 2, 5]= 0 ; occurrence[ 4, 5, 2, 5]=4 ; name[ 4, 5, 2, 5]=2341 ;
    e[ 5, 4, 5, 3]=E_Dict['234'] ;  v[ 5, 4, 5, 3]=F_Dict['234'] ;  n[ 5, 4, 5, 3]= 0 ; c[ 5, 4, 5, 3]= 0 ; occurrence[ 5, 4, 5, 3]=4 ; name[ 5, 4, 5, 3]=2341 ;
    e[ 4, 5, 3, 5]=E_Dict['234'] ;  v[ 4, 5, 3, 5]=F_Dict['234'] ;  n[ 4, 5, 3, 5]= 0 ; c[ 4, 5, 3, 5]= 0 ; occurrence[ 4, 5, 3, 5]=4 ; name[ 4, 5, 3, 5]=2341 ;
    
    #4H-RuO2                        #NO NEED for missing 0235 [v=5.56 c=8.43] because it is the same as 0125
        
    e[ 2, 6, 2, 6]=E_Dict['0123'] ;  v[ 2, 6, 2, 6]=F_Dict['0123'] ;  n[ 2, 6, 2, 6]=4 ; c[ 2, 6, 2, 6]=8.16 ; occurrence[ 2, 6, 2, 6]=4 ; name[ 2, 6, 2, 6]=123 ; #0123
    e[ 3, 6, 3, 6]=E_Dict['0123'] ;  v[ 3, 6, 3, 6]=F_Dict['0123'] ;  n[ 3, 6, 3, 6]=4 ; c[ 3, 6, 3, 6]=8.16 ; occurrence[ 3, 6, 3, 6]=4 ; name[ 3, 6, 3, 6]=123 ;
    e[ 2, 7, 2, 7]=E_Dict['0123'] ;  v[ 2, 7, 2, 7]=F_Dict['0123'] ;  n[ 2, 7, 2, 7]=4 ; c[ 2, 7, 2, 7]=8.16 ; occurrence[ 2, 7, 2, 7]=4 ; name[ 2, 7, 2, 7]=123 ;
    e[ 3, 7, 3, 7]=E_Dict['0123'] ;  v[ 3, 7, 3, 7]=F_Dict['0123'] ;  n[ 3, 7, 3, 7]=4 ; c[ 3, 7, 3, 7]=8.16 ; occurrence[ 3, 7, 3, 7]=4 ; name[ 3, 7, 3, 7]=123 ;
    e[ 6, 2, 6, 2]=E_Dict['0123'] ;  v[ 6, 2, 6, 2]=F_Dict['0123'] ;  n[ 6, 2, 6, 2]=4 ; c[ 6, 2, 6, 2]=8.16 ; occurrence[ 6, 2, 6, 2]=4 ; name[ 6, 2, 6, 2]=123 ;
    e[ 6, 3, 6, 3]=E_Dict['0123'] ;  v[ 6, 3, 6, 3]=F_Dict['0123'] ;  n[ 6, 3, 6, 3]=4 ; c[ 6, 3, 6, 3]=8.16 ; occurrence[ 6, 3, 6, 3]=4 ; name[ 6, 3, 6, 3]=123 ;
    e[ 7, 2, 7, 2]=E_Dict['0123'] ;  v[ 7, 2, 7, 2]=F_Dict['0123'] ;  n[ 7, 2, 7, 2]=4 ; c[ 7, 2, 7, 2]=8.16 ; occurrence[ 7, 2, 7, 2]=4 ; name[ 7, 2, 7, 2]=123 ;
    e[ 7, 3, 7, 3]=E_Dict['0123'] ;  v[ 7, 3, 7, 3]=F_Dict['0123'] ;  n[ 7, 3, 7, 3]=4 ; c[ 7, 3, 7, 3]=8.16 ; occurrence[ 7, 3, 7, 3]=4 ; name[ 7, 3, 7, 3]=123 ;
    
    e[ 4, 6, 1, 6]=E_Dict['0124'] ;  v[ 4, 6, 1, 6]=F_Dict['0124'] ;  n[ 4, 6, 1, 6]=4 ; c[ 4, 6, 1, 6]=8.01 ; occurrence[ 4, 6, 1, 6]=4 ; name[ 4, 6, 1, 6]=124 ; #0124
    e[ 4, 7, 1, 7]=E_Dict['0124'] ;  v[ 4, 7, 1, 7]=F_Dict['0124'] ;  n[ 4, 7, 1, 7]=4 ; c[ 4, 7, 1, 7]=8.01 ; occurrence[ 4, 7, 1, 7]=4 ; name[ 4, 7, 1, 7]=124 ;
    e[ 1, 6, 4, 6]=E_Dict['0124'] ;  v[ 1, 6, 4, 6]=F_Dict['0124'] ;  n[ 1, 6, 4, 6]=4 ; c[ 1, 6, 4, 6]=8.01 ; occurrence[ 1, 6, 4, 6]=4 ; name[ 1, 6, 4, 6]=124 ;
    e[ 1, 7, 4, 7]=E_Dict['0124'] ;  v[ 1, 7, 4, 7]=F_Dict['0124'] ;  n[ 1, 7, 4, 7]=4 ; c[ 1, 7, 4, 7]=8.01 ; occurrence[ 1, 7, 4, 7]=4 ; name[ 1, 7, 4, 7]=124 ;
    e[ 6, 4, 6, 1]=E_Dict['0124'] ;  v[ 6, 4, 6, 1]=F_Dict['0124'] ;  n[ 6, 4, 6, 1]=4 ; c[ 6, 4, 6, 1]=8.01 ; occurrence[ 6, 4, 6, 1]=4 ; name[ 6, 4, 6, 1]=124 ;
    e[ 7, 4, 7, 1]=E_Dict['0124'] ;  v[ 7, 4, 7, 1]=F_Dict['0124'] ;  n[ 7, 4, 7, 1]=4 ; c[ 7, 4, 7, 1]=8.01 ; occurrence[ 7, 4, 7, 1]=4 ; name[ 7, 4, 7, 1]=124 ;
    e[ 6, 1, 6, 4]=E_Dict['0124'] ;  v[ 6, 1, 6, 4]=F_Dict['0124'] ;  n[ 6, 1, 6, 4]=4 ; c[ 6, 1, 6, 4]=8.01 ; occurrence[ 6, 1, 6, 4]=4 ; name[ 6, 1, 6, 4]=124 ;
    e[ 7, 1, 7, 4]=E_Dict['0124'] ;  v[ 7, 1, 7, 4]=F_Dict['0124'] ;  n[ 7, 1, 7, 4]=4 ; c[ 7, 1, 7, 4]=8.01 ; occurrence[ 7, 1, 7, 4]=4 ; name[ 7, 1, 7, 4]=124 ;
    
    e[ 2, 6, 3, 6]=E_Dict['0125'] ;  v[ 2, 6, 3, 6]=F_Dict['0125'] ;  n[ 2, 6, 3, 6]=4 ; c[ 2, 6, 3, 6]=9.70 ; occurrence[ 2, 6, 3, 6]=4 ; name[ 2, 6, 3, 6]=125 ; #0125
    e[ 3, 6, 2, 6]=E_Dict['0125'] ;  v[ 3, 6, 2, 6]=F_Dict['0125'] ;  n[ 3, 6, 2, 6]=4 ; c[ 3, 6, 2, 6]=9.70 ; occurrence[ 3, 6, 2, 6]=4 ; name[ 3, 6, 2, 6]=125 ;
    e[ 2, 7, 3, 7]=E_Dict['0125'] ;  v[ 2, 7, 3, 7]=F_Dict['0125'] ;  n[ 2, 7, 3, 7]=4 ; c[ 2, 7, 3, 7]=9.70 ; occurrence[ 2, 7, 3, 7]=4 ; name[ 2, 7, 3, 7]=125 ;
    e[ 3, 7, 2, 7]=E_Dict['0125'] ;  v[ 3, 7, 2, 7]=F_Dict['0125'] ;  n[ 3, 7, 2, 7]=4 ; c[ 3, 7, 2, 7]=9.70 ; occurrence[ 3, 7, 2, 7]=4 ; name[ 3, 7, 2, 7]=125 ;
    e[ 6, 2, 6, 3]=E_Dict['0125'] ;  v[ 6, 2, 6, 3]=F_Dict['0125'] ;  n[ 6, 2, 6, 3]=4 ; c[ 6, 2, 6, 3]=9.70 ; occurrence[ 6, 2, 6, 3]=4 ; name[ 6, 2, 6, 3]=125 ;
    e[ 6, 3, 6, 2]=E_Dict['0125'] ;  v[ 6, 3, 6, 2]=F_Dict['0125'] ;  n[ 6, 3, 6, 2]=4 ; c[ 6, 3, 6, 2]=9.70 ; occurrence[ 6, 3, 6, 2]=4 ; name[ 6, 3, 6, 2]=125 ;
    e[ 7, 2, 7, 3]=E_Dict['0125'] ;  v[ 7, 2, 7, 3]=F_Dict['0125'] ;  n[ 7, 2, 7, 3]=4 ; c[ 7, 2, 7, 3]=9.70 ; occurrence[ 7, 2, 7, 3]=4 ; name[ 7, 2, 7, 3]=125 ; 
    e[ 7, 3, 7, 2]=E_Dict['0125'] ;  v[ 7, 3, 7, 2]=F_Dict['0125'] ;  n[ 7, 3, 7, 2]=4 ; c[ 7, 3, 7, 2]=9.70 ; occurrence[ 7, 3, 7, 2]=4 ; name[ 7, 3, 7, 2]=125 ;
    
    e[ 4, 5, 4, 5]=E_Dict['2345'] ;  v[ 4, 5, 4, 5]=F_Dict['2345'] ;  n[ 4, 5, 4, 5]=4 ; c[ 4, 5, 4, 5]=6.53 ; occurrence[ 4, 5, 4, 5]=1 ; name[ 4, 5, 4, 5]=23451 ; #2345
    e[ 5, 4, 5, 4]=E_Dict['2345'] ;  v[ 5, 4, 5, 4]=F_Dict['2345'] ;  n[ 5, 4, 5, 4]=4 ; c[ 5, 4, 5, 4]=6.53 ; occurrence[ 5, 4, 5, 4]=1 ; name[ 5, 4, 5, 4]=23451 ;
    
    e[ 4, 6, 2, 5]=E_Dict['0234'] ;  v[ 4, 6, 2, 5]=F_Dict['0234'] ;  n[ 4, 6, 2, 5]=4 ; c[ 4, 6, 2, 5]=7.63 ; occurrence[ 4, 6, 2, 5]=8 ; name[ 4, 6, 2, 5]=234 ; #0234
    e[ 4, 7, 2, 5]=E_Dict['0234'] ;  v[ 4, 7, 2, 5]=F_Dict['0234'] ;  n[ 4, 7, 2, 5]=4 ; c[ 4, 7, 2, 5]=7.63 ; occurrence[ 4, 7, 2, 5]=8 ; name[ 4, 7, 2, 5]=234 ;
    e[ 4, 6, 3, 5]=E_Dict['0234'] ;  v[ 4, 6, 3, 5]=F_Dict['0234'] ;  n[ 4, 6, 3, 5]=4 ; c[ 4, 6, 3, 5]=7.63 ; occurrence[ 4, 6, 3, 5]=8 ; name[ 4, 6, 3, 5]=234 ;
    e[ 4, 7, 3, 5]=E_Dict['0234'] ;  v[ 4, 7, 3, 5]=F_Dict['0234'] ;  n[ 4, 7, 3, 5]=4 ; c[ 4, 7, 3, 5]=7.63 ; occurrence[ 4, 7, 3, 5]=8 ; name[ 4, 7, 3, 5]=234 ;
    e[ 2, 5, 4, 6]=E_Dict['0234'] ;  v[ 2, 5, 4, 6]=F_Dict['0234'] ;  n[ 2, 5, 4, 6]=4 ; c[ 2, 5, 4, 6]=7.63 ; occurrence[ 2, 5, 4, 6]=8 ; name[ 2, 5, 4, 6]=234 ;
    e[ 2, 5, 4, 7]=E_Dict['0234'] ;  v[ 2, 5, 4, 7]=F_Dict['0234'] ;  n[ 2, 5, 4, 7]=4 ; c[ 2, 5, 4, 7]=7.63 ; occurrence[ 2, 5, 4, 7]=8 ; name[ 2, 5, 4, 7]=234 ;
    e[ 3, 5, 4, 6]=E_Dict['0234'] ;  v[ 3, 5, 4, 6]=F_Dict['0234'] ;  n[ 3, 5, 4, 6]=4 ; c[ 3, 5, 4, 6]=7.63 ; occurrence[ 3, 5, 4, 6]=8 ; name[ 3, 5, 4, 6]=234 ;
    e[ 3, 5, 4, 7]=E_Dict['0234'] ;  v[ 3, 5, 4, 7]=F_Dict['0234'] ;  n[ 3, 5, 4, 7]=4 ; c[ 3, 5, 4, 7]=7.63 ; occurrence[ 3, 5, 4, 7]=8 ; name[ 3, 5, 4, 7]=234 ;
    e[ 6, 4, 5, 2]=E_Dict['0234'] ;  v[ 6, 4, 5, 2]=F_Dict['0234'] ;  n[ 6, 4, 5, 2]=4 ; c[ 6, 4, 5, 2]=7.63 ; occurrence[ 6, 4, 5, 2]=8 ; name[ 6, 4, 5, 2]=234 ;
    e[ 7, 4, 5, 2]=E_Dict['0234'] ;  v[ 7, 4, 5, 2]=F_Dict['0234'] ;  n[ 7, 4, 5, 2]=4 ; c[ 7, 4, 5, 2]=7.63 ; occurrence[ 7, 4, 5, 2]=8 ; name[ 7, 4, 5, 2]=234 ;
    e[ 6, 4, 5, 3]=E_Dict['0234'] ;  v[ 6, 4, 5, 3]=F_Dict['0234'] ;  n[ 6, 4, 5, 3]=4 ; c[ 6, 4, 5, 3]=7.63 ; occurrence[ 6, 4, 5, 3]=8 ; name[ 6, 4, 5, 3]=234 ;
    e[ 7, 4, 5, 3]=E_Dict['0234'] ;  v[ 7, 4, 5, 3]=F_Dict['0234'] ;  n[ 7, 4, 5, 3]=4 ; c[ 7, 4, 5, 3]=7.63 ; occurrence[ 7, 4, 5, 3]=8 ; name[ 7, 4, 5, 3]=234 ;
    e[ 5, 2, 6, 4]=E_Dict['0234'] ;  v[ 5, 2, 6, 4]=F_Dict['0234'] ;  n[ 5, 2, 6, 4]=4 ; c[ 5, 2, 6, 4]=7.63 ; occurrence[ 5, 2, 6, 4]=8 ; name[ 5, 2, 6, 4]=234 ;
    e[ 5, 2, 7, 4]=E_Dict['0234'] ;  v[ 5, 2, 7, 4]=F_Dict['0234'] ;  n[ 5, 2, 7, 4]=4 ; c[ 5, 2, 7, 4]=7.63 ; occurrence[ 5, 2, 7, 4]=8 ; name[ 5, 2, 7, 4]=234 ;
    e[ 5, 3, 6, 4]=E_Dict['0234'] ;  v[ 5, 3, 6, 4]=F_Dict['0234'] ;  n[ 5, 3, 6, 4]=4 ; c[ 5, 3, 6, 4]=7.63 ; occurrence[ 5, 3, 6, 4]=8 ; name[ 5, 3, 6, 4]=234 ;
    e[ 5, 3, 7, 4]=E_Dict['0234'] ;  v[ 5, 3, 7, 4]=F_Dict['0234'] ;  n[ 5, 3, 7, 4]=4 ; c[ 5, 3, 7, 4]=7.63 ; occurrence[ 5, 3, 7, 4]=8 ; name[ 5, 3, 7, 4]=234 ;
    
    #5H-RuO2
        
    e[ 4, 5, 4, 6]=E_Dict['02345'] ;  v[ 4, 5, 4, 6]=F_Dict['02345'] ;  n[ 4, 5, 4, 6]=5 ; c[ 4, 5, 4, 6]=6.38 ; occurrence[ 4, 5, 4, 6]=4 ; name[ 4, 5, 4, 6]=2345 ; #02345
    e[ 4, 5, 4, 7]=E_Dict['02345'] ;  v[ 4, 5, 4, 7]=F_Dict['02345'] ;  n[ 4, 5, 4, 7]=5 ; c[ 4, 5, 4, 7]=6.38 ; occurrence[ 4, 5, 4, 7]=4 ; name[ 4, 5, 4, 7]=2345 ;
    e[ 4, 6, 4, 5]=E_Dict['02345'] ;  v[ 4, 6, 4, 5]=F_Dict['02345'] ;  n[ 4, 6, 4, 5]=5 ; c[ 4, 6, 4, 5]=6.38 ; occurrence[ 4, 6, 4, 5]=4 ; name[ 4, 6, 4, 5]=2345 ;
    e[ 4, 7, 4, 5]=E_Dict['02345'] ;  v[ 4, 7, 4, 5]=F_Dict['02345'] ;  n[ 4, 7, 4, 5]=5 ; c[ 4, 7, 4, 5]=6.38 ; occurrence[ 4, 7, 4, 5]=4 ; name[ 4, 7, 4, 5]=2345 ;
    e[ 5, 4, 6, 4]=E_Dict['02345'] ;  v[ 5, 4, 6, 4]=F_Dict['02345'] ;  n[ 5, 4, 6, 4]=5 ; c[ 5, 4, 6, 4]=6.38 ; occurrence[ 5, 4, 6, 4]=4 ; name[ 5, 4, 6, 4]=2345 ;
    e[ 5, 4, 7, 4]=E_Dict['02345'] ;  v[ 5, 4, 7, 4]=F_Dict['02345'] ;  n[ 5, 4, 7, 4]=5 ; c[ 5, 4, 7, 4]=6.38 ; occurrence[ 5, 4, 7, 4]=4 ; name[ 5, 4, 7, 4]=2345 ;
    e[ 6, 4, 5, 4]=E_Dict['02345'] ;  v[ 6, 4, 5, 4]=F_Dict['02345'] ;  n[ 6, 4, 5, 4]=5 ; c[ 6, 4, 5, 4]=6.38 ; occurrence[ 6, 4, 5, 4]=4 ; name[ 6, 4, 5, 4]=2345 ;
    e[ 7, 4, 5, 4]=E_Dict['02345'] ;  v[ 7, 4, 5, 4]=F_Dict['02345'] ;  n[ 7, 4, 5, 4]=5 ; c[ 7, 4, 5, 4]=6.38 ; occurrence[ 7, 4, 5, 4]=4 ; name[ 7, 4, 5, 4]=2345 ;
    
    e[ 4, 6, 3, 6]=E_Dict['01234'] ;  v[ 4, 6, 3, 6]=F_Dict['01234'] ;  n[ 4, 6, 3, 6]=5 ; c[ 4, 6, 3, 6]=6.60 ; occurrence[ 4, 6, 3, 6]=8 ; name[ 4, 6, 3, 6]=1234 ; #01234
    e[ 3, 6, 4, 6]=E_Dict['01234'] ;  v[ 3, 6, 4, 6]=F_Dict['01234'] ;  n[ 3, 6, 4, 6]=5 ; c[ 3, 6, 4, 6]=6.60 ; occurrence[ 3, 6, 4, 6]=8 ; name[ 3, 6, 4, 6]=1234 ;
    e[ 4, 7, 3, 7]=E_Dict['01234'] ;  v[ 4, 7, 3, 7]=F_Dict['01234'] ;  n[ 4, 7, 3, 7]=5 ; c[ 4, 7, 3, 7]=6.60 ; occurrence[ 4, 7, 3, 7]=8 ; name[ 4, 7, 3, 7]=1234 ;
    e[ 3, 7, 4, 7]=E_Dict['01234'] ;  v[ 3, 7, 4, 7]=F_Dict['01234'] ;  n[ 3, 7, 4, 7]=5 ; c[ 3, 7, 4, 7]=6.60 ; occurrence[ 3, 7, 4, 7]=8 ; name[ 3, 7, 4, 7]=1234 ;
    e[ 2, 6, 4, 6]=E_Dict['01234'] ;  v[ 2, 6, 4, 6]=F_Dict['01234'] ;  n[ 2, 6, 4, 6]=5 ; c[ 2, 6, 4, 6]=6.60 ; occurrence[ 2, 6, 4, 6]=8 ; name[ 2, 6, 4, 6]=1234 ;
    e[ 4, 6, 2, 6]=E_Dict['01234'] ;  v[ 4, 6, 2, 6]=F_Dict['01234'] ;  n[ 4, 6, 2, 6]=5 ; c[ 4, 6, 2, 6]=6.60 ; occurrence[ 4, 6, 2, 6]=8 ; name[ 4, 6, 2, 6]=1234 ;
    e[ 2, 7, 4, 7]=E_Dict['01234'] ;  v[ 2, 7, 4, 7]=F_Dict['01234'] ;  n[ 2, 7, 4, 7]=5 ; c[ 2, 7, 4, 7]=6.60 ; occurrence[ 2, 7, 4, 7]=8 ; name[ 2, 7, 4, 7]=1234 ;
    e[ 4, 7, 2, 7]=E_Dict['01234'] ;  v[ 4, 7, 2, 7]=F_Dict['01234'] ;  n[ 4, 7, 2, 7]=5 ; c[ 4, 7, 2, 7]=6.60 ; occurrence[ 4, 7, 2, 7]=8 ; name[ 4, 7, 2, 7]=1234 ;
    e[ 6, 4, 6, 3]=E_Dict['01234'] ;  v[ 6, 4, 6, 3]=F_Dict['01234'] ;  n[ 6, 4, 6, 3]=5 ; c[ 6, 4, 6, 3]=6.60 ; occurrence[ 6, 4, 6, 3]=8 ; name[ 6, 4, 6, 3]=1234 ;
    e[ 6, 3, 6, 4]=E_Dict['01234'] ;  v[ 6, 3, 6, 4]=F_Dict['01234'] ;  n[ 6, 3, 6, 4]=5 ; c[ 6, 3, 6, 4]=6.60 ; occurrence[ 6, 3, 6, 4]=8 ; name[ 6, 3, 6, 4]=1234 ;
    e[ 7, 4, 7, 3]=E_Dict['01234'] ;  v[ 7, 4, 7, 3]=F_Dict['01234'] ;  n[ 7, 4, 7, 3]=5 ; c[ 7, 4, 7, 3]=6.60 ; occurrence[ 7, 4, 7, 3]=8 ; name[ 7, 4, 7, 3]=1234 ;
    e[ 7, 3, 7, 4]=E_Dict['01234'] ;  v[ 7, 3, 7, 4]=F_Dict['01234'] ;  n[ 7, 3, 7, 4]=5 ; c[ 7, 3, 7, 4]=6.60 ; occurrence[ 7, 3, 7, 4]=8 ; name[ 7, 3, 7, 4]=1234 ;
    e[ 6, 2, 6, 4]=E_Dict['01234'] ;  v[ 6, 2, 6, 4]=F_Dict['01234'] ;  n[ 6, 2, 6, 4]=5 ; c[ 6, 2, 6, 4]=6.60 ; occurrence[ 6, 2, 6, 4]=8 ; name[ 6, 2, 6, 4]=1234 ;
    e[ 6, 4, 6, 2]=E_Dict['01234'] ;  v[ 6, 4, 6, 2]=F_Dict['01234'] ;  n[ 6, 4, 6, 2]=5 ; c[ 6, 4, 6, 2]=6.60 ; occurrence[ 6, 4, 6, 2]=8 ; name[ 6, 4, 6, 2]=1234 ;
    e[ 7, 2, 7, 4]=E_Dict['01234'] ;  v[ 7, 2, 7, 4]=F_Dict['01234'] ;  n[ 7, 2, 7, 4]=5 ; c[ 7, 2, 7, 4]=6.60 ; occurrence[ 7, 2, 7, 4]=8 ; name[ 7, 2, 7, 4]=1234 ;
    e[ 7, 4, 7, 2]=E_Dict['01234'] ;  v[ 7, 4, 7, 2]=F_Dict['01234'] ;  n[ 7, 4, 7, 2]=5 ; c[ 7, 4, 7, 2]=6.60 ; occurrence[ 7, 4, 7, 2]=8 ; name[ 7, 4, 7, 2]=1234 ;
    
    #6H-RuO2

    e[ 4, 6, 4, 6]=E_Dict['012345'] ;  v[ 4, 6, 4, 6]=F_Dict['012345'] ;  n[ 4, 6, 4, 6]=6 ; c[ 4, 6, 4, 6]=6.08 ; occurrence[ 4, 6, 4, 6]=2 ; name[ 4, 6, 4, 6]=12345 ; #012345
    e[ 4, 7, 4, 7]=E_Dict['012345'] ;  v[ 4, 7, 4, 7]=F_Dict['012345'] ;  n[ 4, 7, 4, 7]=6 ; c[ 4, 7, 4, 7]=6.08 ; occurrence[ 4, 7, 4, 7]=2 ; name[ 4, 7, 4, 7]=12345 ;
    e[ 6, 4, 6, 4]=E_Dict['012345'] ;  v[ 6, 4, 6, 4]=F_Dict['012345'] ;  n[ 6, 4, 6, 4]=6 ; c[ 6, 4, 6, 4]=6.08 ; occurrence[ 6, 4, 6, 4]=2 ; name[ 6, 4, 6, 4]=12345 ;
    e[ 7, 4, 7, 4]=E_Dict['012345'] ;  v[ 7, 4, 7, 4]=F_Dict['012345'] ;  n[ 7, 4, 7, 4]=6 ; c[ 7, 4, 7, 4]=6.08 ; occurrence[ 7, 4, 7, 4]=2 ; name[ 7, 4, 7, 4]=12345 ;

    # #Subsurface Values

    # #1H-RuO2

    # #Blank
    # e[ 1, 5, 1, 8]=-1.096 ; v[ 1, 5, 1, 8]=6.8796 ; n[ 1, 5, 1, 8]=1 ; c[ 1, 5, 1, 8]=9.31 ;
    # e[ 5, 1, 8, 1]=-1.096 ; v[ 5, 1, 8, 1]=6.8796 ; n[ 5, 1, 8, 1]=1 ; c[ 5, 1, 8, 1]=9.31 ;
    # e[ 1, 8, 1, 5]=-1.096 ; v[ 1, 8, 1, 5]=6.8796 ; n[ 1, 8, 1, 5]=1 ; c[ 1, 8, 1, 5]=9.31 ;
    # e[ 8, 1, 5, 1]=-1.096 ; v[ 8, 1, 5, 1]=6.8796 ; n[ 8, 1, 5, 1]=1 ; c[ 8, 1, 5, 1]=9.31 ;
    # e[ 5, 1, 11, 1]=-1.096 ; v[ 5, 1, 11, 1]=6.8796 ; n[ 5, 1, 11, 1]=1 ; c[ 5, 1, 11, 1]=9.31 ;
    # e[ 1, 5, 1, 11]=-1.096 ; v[ 1, 5, 1, 11]=6.8796 ; n[ 1, 5, 1, 11]=1 ; c[ 1, 5, 1, 11]=9.31 ;
    # e[ 11, 1, 5, 1]=-1.096 ; v[ 11, 1, 5, 1]=6.8796 ; n[ 11, 1, 5, 1]=1 ; c[ 11, 1, 5, 1]=9.31 ;
    # e[ 1, 11, 1, 5]=-1.096 ; v[ 1, 11, 1, 5]=6.8796 ; n[ 1, 11, 1, 5]=1 ; c[ 1, 11, 1, 5]=9.31 ;
    
    # #0
    # e[ 1, 5, 1, 10]=-3.013 ; v[ 1, 5, 1, 10]=6.7257 ; n[ 1, 5, 1, 10]=2 ; c[ 1, 5, 1, 10]=8.48 ;
    # e[ 5, 1, 10, 1]=-3.013 ; v[ 5, 1, 10, 1]=6.7257 ; n[ 5, 1, 10, 1]=2 ; c[ 5, 1, 10, 1]=8.48 ;
    # e[ 1, 10, 1, 5]=-3.013 ; v[ 1, 10, 1, 5]=6.7257 ; n[ 1, 10, 1, 5]=2 ; c[ 1, 10, 1, 5]=8.48 ;
    # e[ 10, 1, 5, 1]=-3.013 ; v[ 10, 1, 5, 1]=6.7257 ; n[ 10, 1, 5, 1]=2 ; c[ 10, 1, 5, 1]=8.48 ;
    # e[ 6, 1, 11, 1]=-3.013 ; v[ 6, 1, 11, 1]=6.7257 ; n[ 6, 1, 11, 1]=2 ; c[ 6, 1, 11, 1]=8.48 ;
    # e[ 1, 6, 1, 11]=-3.013 ; v[ 1, 6, 1, 11]=6.7257 ; n[ 1, 6, 1, 11]=2 ; c[ 1, 6, 1, 11]=8.48 ;
    # e[ 11, 1, 6, 1]=-3.013 ; v[ 11, 1, 6, 1]=6.7257 ; n[ 11, 1, 6, 1]=2 ; c[ 11, 1, 6, 1]=8.48 ;
    # e[ 1, 11, 1, 6]=-3.013 ; v[ 1, 11, 1, 6]=6.7257 ; n[ 1, 11, 1, 6]=2 ; c[ 1, 11, 1, 6]=8.48 ;
    
    # #01
    # e[ 1, 7, 1, 10]=-5.080 ; v[ 1, 7, 1, 10]=6.2161 ; n[ 1, 7, 1, 10]=3 ; c[ 1, 7, 1, 10]=8.11 ;
    # e[ 7, 1, 10, 1]=-5.080 ; v[ 7, 1, 10, 1]=6.2161 ; n[ 7, 1, 10, 1]=3 ; c[ 7, 1, 10, 1]=8.11 ;
    # e[ 1, 10, 1, 7]=-5.080 ; v[ 1, 10, 1, 7]=6.2161 ; n[ 1, 10, 1, 7]=3 ; c[ 1, 10, 1, 7]=8.11 ;
    # e[ 10, 1, 7, 1]=-5.080 ; v[ 10, 1, 7, 1]=6.2161 ; n[ 10, 1, 7, 1]=3 ; c[ 10, 1, 7, 1]=8.11 ;
    # e[ 6, 1, 12, 1]=-5.080 ; v[ 6, 1, 12, 1]=6.2161 ; n[ 6, 1, 12, 1]=3 ; c[ 6, 1, 12, 1]=8.11 ;
    # e[ 1, 6, 1, 12]=-5.080 ; v[ 1, 6, 1, 12]=6.2161 ; n[ 1, 6, 1, 12]=3 ; c[ 1, 6, 1, 12]=8.11 ;
    # e[ 12, 1, 6, 1]=-5.080 ; v[ 12, 1, 6, 1]=6.2161 ; n[ 12, 1, 6, 1]=3 ; c[ 12, 1, 6, 1]=8.11 ;
    # e[ 1, 12, 1, 6]=-5.080 ; v[ 1, 12, 1, 6]=6.2161 ; n[ 1, 12, 1, 6]=3 ; c[ 1, 12, 1, 6]=8.11 ;
    
    # #012
    # e[ 1, 7, 2, 10]=-6.924 ; v[ 1, 7, 2, 10]=5.9541 ; n[ 1, 7, 2, 10]=4 ; c[ 1, 7, 2, 10]=7.10 ;
    # e[ 7, 1, 10, 2]=-6.924 ; v[ 7, 1, 10, 2]=5.9541 ; n[ 7, 1, 10, 2]=4 ; c[ 7, 1, 10, 2]=7.10 ;
    # e[ 2, 10, 1, 7]=-6.924 ; v[ 2, 10, 1, 7]=5.9541 ; n[ 2, 10, 1, 7]=4 ; c[ 2, 10, 1, 7]=7.10 ;
    # e[ 10, 2, 7, 1]=-6.924 ; v[ 10, 2, 7, 1]=5.9541 ; n[ 10, 2, 7, 1]=4 ; c[ 10, 2, 7, 1]=7.10 ;
    # e[ 6, 3, 12, 1]=-6.924 ; v[ 6, 3, 12, 1]=5.9541 ; n[ 6, 3, 12, 1]=4 ; c[ 6, 3, 12, 1]=7.10 ;
    # e[ 3, 6, 1, 12]=-6.924 ; v[ 3, 6, 1, 12]=5.9541 ; n[ 3, 6, 1, 12]=4 ; c[ 3, 6, 1, 12]=7.10 ;
    # e[ 12, 1, 6, 3]=-6.924 ; v[ 12, 1, 6, 3]=5.9541 ; n[ 12, 1, 6, 3]=4 ; c[ 12, 1, 6, 3]=7.10 ;
    # e[ 1, 12, 3, 6]=-6.924 ; v[ 1, 12, 3, 6]=5.9541 ; n[ 1, 12, 3, 6]=4 ; c[ 1, 12, 3, 6]=7.10 ;
    
    # #0123
    # e[ 2, 7, 2, 10]=-8.526 ; v[ 2, 7, 2, 10]=5.6643 ; n[ 2, 7, 2, 10]=5 ; c[ 2, 7, 2, 10]=7.37 ;
    # e[ 7, 2, 10, 2]=-8.526 ; v[ 7, 2, 10, 2]=5.6643 ; n[ 7, 2, 10, 2]=5 ; c[ 7, 2, 10, 2]=7.37 ;
    # e[ 2, 10, 2, 7]=-8.526 ; v[ 2, 10, 2, 7]=5.6643 ; n[ 2, 10, 2, 7]=5 ; c[ 2, 10, 2, 7]=7.37 ;
    # e[ 10, 2, 7, 2]=-8.526 ; v[ 10, 2, 7, 2]=5.6643 ; n[ 10, 2, 7, 2]=5 ; c[ 10, 2, 7, 2]=7.37 ;
    # e[ 6, 3, 12, 3]=-8.526 ; v[ 6, 3, 12, 3]=5.6643 ; n[ 6, 3, 12, 3]=5 ; c[ 6, 3, 12, 3]=7.37 ;
    # e[ 3, 6, 3, 12]=-8.526 ; v[ 3, 6, 3, 12]=5.6643 ; n[ 3, 6, 3, 12]=5 ; c[ 3, 6, 3, 12]=7.37 ;
    # e[ 12, 3, 6, 3]=-8.526 ; v[ 12, 3, 6, 3]=5.6643 ; n[ 12, 3, 6, 3]=5 ; c[ 12, 3, 6, 3]=7.37 ;
    # e[ 3, 12, 3, 6]=-8.526 ; v[ 3, 12, 3, 6]=5.6643 ; n[ 3, 12, 3, 6]=5 ; c[ 3, 12, 3, 6]=7.37 ;
    
    # #01234
    # e[ 2, 7, 4, 10]=-10.102 ; v[ 2, 7, 4, 10]=4.5451 ; n[ 2, 7, 4, 10]=6 ; c[ 2, 7, 4, 10]=6.22 ;
    # e[ 7, 2, 10, 4]=-10.102 ; v[ 7, 2, 10, 4]=4.5451 ; n[ 7, 2, 10, 4]=6 ; c[ 7, 2, 10, 4]=6.22 ;
    # e[ 4, 10, 2, 7]=-10.102 ; v[ 4, 10, 2, 7]=4.5451 ; n[ 4, 10, 2, 7]=6 ; c[ 4, 10, 2, 7]=6.22 ;
    # e[ 10, 4, 7, 2]=-10.102 ; v[ 10, 4, 7, 2]=4.5451 ; n[ 10, 4, 7, 2]=6 ; c[ 10, 4, 7, 2]=6.22 ;
    # e[ 6, 4, 12, 3]=-10.102 ; v[ 6, 4, 12, 3]=4.5451 ; n[ 6, 4, 12, 3]=6 ; c[ 6, 4, 12, 3]=6.22 ;
    # e[ 4, 6, 3, 12]=-10.102 ; v[ 4, 6, 3, 12]=4.5451 ; n[ 4, 6, 3, 12]=6 ; c[ 4, 6, 3, 12]=6.22 ;
    # e[ 12, 3, 6, 4]=-10.102 ; v[ 12, 3, 6, 4]=4.5451 ; n[ 12, 3, 6, 4]=6 ; c[ 12, 3, 6, 4]=6.22 ;
    # e[ 3, 12, 4, 6]=-10.102 ; v[ 3, 12, 4, 6]=4.5451 ; n[ 3, 12, 4, 6]=6 ; c[ 3, 12, 4, 6]=6.22 ;
    
    # #012345
    # e[ 4, 7, 4, 10]=-10.869 ; v[ 4, 7, 4, 10]=3.0613 ; n[ 4, 7, 4, 10]=7 ; c[ 4, 7, 4, 10]=5.18 ;
    # e[ 7, 4, 10, 4]=-10.869 ; v[ 7, 4, 10, 4]=3.0613 ; n[ 7, 4, 10, 4]=7 ; c[ 7, 4, 10, 4]=5.18 ;
    # e[ 4, 10, 4, 7]=-10.869 ; v[ 4, 10, 4, 7]=3.0613 ; n[ 4, 10, 4, 7]=7 ; c[ 4, 10, 4, 7]=5.18 ;
    # e[ 10, 4, 7, 4]=-10.869 ; v[ 10, 4, 7, 4]=3.0613 ; n[ 10, 4, 7, 4]=7 ; c[ 10, 4, 7, 4]=5.18 ;
    # e[ 6, 4, 12, 4]=-10.869 ; v[ 6, 4, 12, 4]=3.0613 ; n[ 6, 4, 12, 4]=7 ; c[ 6, 4, 12, 4]=5.18 ;
    # e[ 4, 6, 4, 12]=-10.869 ; v[ 4, 6, 4, 12]=3.0613 ; n[ 4, 6, 4, 12]=7 ; c[ 4, 6, 4, 12]=5.18 ;
    # e[ 12, 4, 6, 4]=-10.869 ; v[ 12, 4, 6, 4]=3.0613 ; n[ 12, 4, 6, 4]=7 ; c[ 12, 4, 6, 4]=5.18 ;
    # e[ 4, 12, 4, 6]=-10.869 ; v[ 4, 12, 4, 6]=3.0613 ; n[ 4, 12, 4, 6]=7 ; c[ 4, 12, 4, 6]=5.18 ;

    # #2H Subsurface RuO2

    # #Two-Diagonal
    
    # #Blank
    # e[ 1, 11, 1, 8]=-1.428 ; v[ 1, 11, 1, 8]=6.9304 ; n[ 1, 11, 1, 8]=2 ; c[ 1, 11, 1, 8]=8.59 ;
    # e[ 11, 1, 8, 1]=-1.428 ; v[ 11, 1, 8, 1]=6.9304 ; n[ 11, 1, 8, 1]=2 ; c[ 11, 1, 8, 1]=8.59 ;
    # e[ 1, 8, 1, 11]=-1.428 ; v[ 1, 8, 1, 11]=6.9304 ; n[ 1, 8, 1, 11]=2 ; c[ 1, 8, 1, 11]=8.59 ;
    # e[ 8, 1, 11, 1]=-1.428 ; v[ 8, 1, 11, 1]=6.9304 ; n[ 8, 1, 11, 1]=2 ; c[ 8, 1, 11, 1]=8.59 ;
    
    # #0
    # e[ 1, 11, 1, 10]=-3.423 ; v[ 1, 11, 1, 10]=6.5263 ; n[ 1, 11, 1, 10]=3 ; c[ 1, 11, 1, 10]=7.54 ;
    # e[ 11, 1, 10, 1]=-3.423 ; v[ 11, 1, 10, 1]=6.5263 ; n[ 11, 1, 10, 1]=3 ; c[ 11, 1, 10, 1]=7.54 ;
    # e[ 1, 10, 1, 11]=-3.423 ; v[ 1, 10, 1, 11]=6.5263 ; n[ 1, 10, 1, 11]=3 ; c[ 1, 10, 1, 11]=7.54 ;
    # e[ 10, 1, 11, 1]=-3.423 ; v[ 10, 1, 11, 1]=6.5263 ; n[ 10, 1, 11, 1]=3 ; c[ 10, 1, 11, 1]=7.54 ;
    # e[ 9, 1, 11, 1]=-3.423 ; v[ 9, 1, 11, 1]=6.5263 ; n[ 9, 1, 11, 1]=3 ; c[ 9, 1, 11, 1]=7.54 ;
    # e[ 1, 9, 1, 11]=-3.423 ; v[ 1, 9, 1, 11]=6.5263 ; n[ 1, 9, 1, 11]=3 ; c[ 1, 9, 1, 11]=7.54 ;
    # e[ 11, 1, 9, 1]=-3.423 ; v[ 11, 1, 9, 1]=6.5263 ; n[ 11, 1, 9, 1]=3 ; c[ 11, 1, 9, 1]=7.54 ;
    # e[ 1, 11, 1, 9]=-3.423 ; v[ 1, 11, 1, 9]=6.5263 ; n[ 1, 11, 1, 9]=3 ; c[ 1, 11, 1, 9]=7.54 ;
    
    # #01
    # e[ 1, 13, 1, 10]=-5.431 ; v[ 1, 13, 1, 10]=6.1651 ; n[ 1, 13, 1, 10]=4 ; c[ 1, 13, 1, 10]=7.38 ;
    # e[ 13, 1, 10, 1]=-5.431 ; v[ 13, 1, 10, 1]=6.1651 ; n[ 13, 1, 10, 1]=4 ; c[ 13, 1, 10, 1]=7.38 ;
    # e[ 1, 10, 1, 13]=-5.431 ; v[ 1, 10, 1, 13]=6.1651 ; n[ 1, 10, 1, 13]=4 ; c[ 1, 10, 1, 13]=7.38 ;
    # e[ 10, 1, 13, 1]=-5.431 ; v[ 10, 1, 13, 1]=6.1651 ; n[ 10, 1, 13, 1]=4 ; c[ 10, 1, 13, 1]=7.38 ;
    # e[ 9, 1, 12, 1]=-5.431 ; v[ 9, 1, 12, 1]=6.1651 ; n[ 9, 1, 12, 1]=4 ; c[ 9, 1, 12, 1]=7.38 ;
    # e[ 1, 9, 1, 12]=-5.431 ; v[ 1, 9, 1, 12]=6.1651 ; n[ 1, 9, 1, 12]=4 ; c[ 1, 9, 1, 12]=7.38 ;
    # e[ 12, 1, 9, 1]=-5.431 ; v[ 12, 1, 9, 1]=6.1651 ; n[ 12, 1, 9, 1]=4 ; c[ 12, 1, 9, 1]=7.38 ;
    # e[ 1, 12, 1, 9]=-5.431 ; v[ 1, 12, 1, 9]=6.1651 ; n[ 1, 12, 1, 9]=4 ; c[ 1, 12, 1, 9]=7.38 ;
    
    # #012
    # e[ 1, 13, 2, 10]=-7.137 ; v[ 1, 13, 2, 10]=5.8457 ; n[ 1, 13, 2, 10]=5 ; c[ 1, 13, 2, 10]=6.28 ;
    # e[ 13, 1, 10, 2]=-7.137 ; v[ 13, 1, 10, 2]=5.8457 ; n[ 13, 1, 10, 2]=5 ; c[ 13, 1, 10, 2]=6.28 ;
    # e[ 2, 10, 1, 13]=-7.137 ; v[ 2, 10, 1, 13]=5.8457 ; n[ 2, 10, 1, 13]=5 ; c[ 2, 10, 1, 13]=6.28 ;
    # e[ 10, 2, 13, 1]=-7.137 ; v[ 10, 2, 13, 1]=5.8457 ; n[ 10, 2, 13, 1]=5 ; c[ 10, 2, 13, 1]=6.28 ;
    # e[ 9, 3, 12, 1]=-7.137 ; v[ 9, 3, 12, 1]=5.8457 ; n[ 9, 3, 12, 1]=5 ; c[ 9, 3, 12, 1]=6.28 ;
    # e[ 3, 9, 1, 12]=-7.137 ; v[ 3, 9, 1, 12]=5.8457 ; n[ 3, 9, 1, 12]=5 ; c[ 3, 9, 1, 12]=6.28 ;
    # e[ 12, 1, 9, 3]=-7.137 ; v[ 12, 1, 9, 3]=5.8457 ; n[ 12, 1, 9, 3]=5 ; c[ 12, 1, 9, 3]=6.28 ;
    # e[ 1, 12, 3, 9]=-7.137 ; v[ 1, 12, 3, 9]=5.8457 ; n[ 1, 12, 3, 9]=5 ; c[ 1, 12, 3, 9]=6.28 ;
    
    # #0123
    # e[ 2, 13, 2, 10]=-8.656 ; v[ 2, 13, 2, 10]=5.5799 ; n[ 2, 13, 2, 10]=6 ; c[ 2, 13, 2, 10]=6.32 ;
    # e[ 13, 2, 10, 2]=-8.656 ; v[ 13, 2, 10, 2]=5.5799 ; n[ 13, 2, 10, 2]=6 ; c[ 13, 2, 10, 2]=6.32 ;
    # e[ 2, 10, 2, 13]=-8.656 ; v[ 2, 10, 2, 13]=5.5799 ; n[ 2, 10, 2, 13]=6 ; c[ 2, 10, 2, 13]=6.32 ;
    # e[ 10, 2, 13, 2]=-8.656 ; v[ 10, 2, 13, 2]=5.5799 ; n[ 10, 2, 13, 2]=6 ; c[ 10, 2, 13, 2]=6.32 ;
    # e[ 9, 3, 12, 3]=-8.656 ; v[ 9, 3, 12, 3]=5.5799 ; n[ 9, 3, 12, 3]=6 ; c[ 9, 3, 12, 3]=6.32 ;
    # e[ 3, 9, 3, 12]=-8.656 ; v[ 3, 9, 3, 12]=5.5799 ; n[ 3, 9, 3, 12]=6 ; c[ 3, 9, 3, 12]=6.32 ;
    # e[ 12, 3, 9, 3]=-8.656 ; v[ 12, 3, 9, 3]=5.5799 ; n[ 12, 3, 9, 3]=6 ; c[ 12, 3, 9, 3]=6.32 ;
    # e[ 3, 12, 3, 9]=-8.656 ; v[ 3, 12, 3, 9]=5.5799 ; n[ 3, 12, 3, 9]=6 ; c[ 3, 12, 3, 9]=6.32 ;
    
    # #01234
    # e[ 2, 13, 4, 10]=-10.190 ; v[ 2, 13, 4, 10]=4.4611 ; n[ 2, 13, 4, 10]=7 ; c[ 2, 13, 4, 10]=5.57 ;
    # e[ 13, 2, 10, 4]=-10.190 ; v[ 13, 2, 10, 4]=4.4611 ; n[ 13, 2, 10, 4]=7 ; c[ 13, 2, 10, 4]=5.57 ;
    # e[ 4, 10, 2, 13]=-10.190 ; v[ 4, 10, 2, 13]=4.4611 ; n[ 4, 10, 2, 13]=7 ; c[ 4, 10, 2, 13]=5.57 ;
    # e[ 10, 4, 13, 2]=-10.190 ; v[ 10, 4, 13, 2]=4.4611 ; n[ 10, 4, 13, 2]=7 ; c[ 10, 4, 13, 2]=5.57 ;
    # e[ 9, 4, 12, 3]=-10.190 ; v[ 9, 4, 12, 3]=4.4611 ; n[ 9, 4, 12, 3]=7 ; c[ 9, 4, 12, 3]=5.57 ;
    # e[ 4, 9, 3, 12]=-10.190 ; v[ 4, 9, 3, 12]=4.4611 ; n[ 4, 9, 3, 12]=7 ; c[ 4, 9, 3, 12]=5.57 ;
    # e[ 12, 3, 9, 4]=-10.190 ; v[ 12, 3, 9, 4]=4.4611 ; n[ 12, 3, 9, 4]=7 ; c[ 12, 3, 9, 4]=5.57 ;
    # e[ 3, 12, 4, 9]=-10.190 ; v[ 3, 12, 4, 9]=4.4611 ; n[ 3, 12, 4, 9]=7 ; c[ 3, 12, 4, 9]=5.57 ;
    
    # #012345
    # e[ 4, 13, 4, 10]=-10.704 ; v[ 4, 13, 4, 10]=2.9391 ; n[ 4, 13, 4, 10]=8 ; c[ 4, 13, 4, 10]=5.30 ;
    # e[ 13, 4, 10, 4]=-10.704 ; v[ 13, 4, 10, 4]=2.9391 ; n[ 13, 4, 10, 4]=8 ; c[ 13, 4, 10, 4]=5.30 ;
    # e[ 4, 10, 4, 13]=-10.704 ; v[ 4, 10, 4, 13]=2.9391 ; n[ 4, 10, 4, 13]=8 ; c[ 4, 10, 4, 13]=5.30 ;
    # e[ 10, 4, 13, 4]=-10.704 ; v[ 10, 4, 13, 4]=2.9391 ; n[ 10, 4, 13, 4]=8 ; c[ 10, 4, 13, 4]=5.30 ;
    # e[ 9, 4, 12, 4]=-10.704 ; v[ 9, 4, 12, 4]=2.9391 ; n[ 9, 4, 12, 4]=8 ; c[ 9, 4, 12, 4]=5.30 ;
    # e[ 4, 9, 4, 12]=-10.704 ; v[ 4, 9, 4, 12]=2.9391 ; n[ 4, 9, 4, 12]=8 ; c[ 4, 9, 4, 12]=5.30 ;
    # e[ 12, 4, 9, 4]=-10.704 ; v[ 12, 4, 9, 4]=2.9391 ; n[ 12, 4, 9, 4]=8 ; c[ 12, 4, 9, 4]=5.30 ;
    # e[ 4, 12, 4, 9]=-10.704 ; v[ 4, 12, 4, 9]=2.9391 ; n[ 4, 12, 4, 9]=8 ; c[ 4, 12, 4, 9]=5.30 ;
    
    # #Two Horizontal 1
    
    # #Blank
    # e[ 1, 8, 1, 8]=-2.196 ; v[ 1, 8, 1, 8]=6.8394 ; n[ 1, 8, 1, 8]=2 ; c[ 1, 8, 1, 8]=5.44 ;
    # e[ 8, 1, 8, 1]=-2.196 ; v[ 8, 1, 8, 1]=6.8394 ; n[ 8, 1, 8, 1]=2 ; c[ 8, 1, 8, 1]=5.44 ;
    # e[ 11, 1, 11, 1]=-2.196 ; v[ 11, 1, 11, 1]=6.8394 ; n[ 11, 1, 11, 1]=2 ; c[ 11, 1, 11, 1]=5.44 ;
    # e[ 1, 11, 1, 11]=-2.196 ; v[ 1, 11, 1, 11]=6.8394 ; n[ 1, 11, 1, 11]=2 ; c[ 1, 11, 1, 11]=5.44 ;
    
    # #0
    # e[ 1, 8, 1, 10]=-4.184 ; v[ 1, 8, 1, 10]=6.2842 ; n[ 1, 8, 1, 10]=3 ; c[ 1, 8, 1, 10]=4.47 ;
    # e[ 8, 1, 10, 1]=-4.184 ; v[ 8, 1, 10, 1]=6.2842 ; n[ 8, 1, 10, 1]=3 ; c[ 8, 1, 10, 1]=4.47 ;
    # e[ 1, 10, 1, 8]=-4.184 ; v[ 1, 10, 1, 8]=6.2842 ; n[ 1, 10, 1, 8]=3 ; c[ 1, 10, 1, 8]=4.47 ;
    # e[ 10, 1, 8, 1]=-4.184 ; v[ 10, 1, 8, 1]=6.2842 ; n[ 10, 1, 8, 1]=3 ; c[ 10, 1, 8, 1]=4.47 ;
    # e[ 12, 1, 11, 1]=-4.184 ; v[ 12, 1, 11, 1]=6.2842 ; n[ 12, 1, 11, 1]=3 ; c[ 12, 1, 11, 1]=4.47 ;
    # e[ 1, 12, 1, 11]=-4.184 ; v[ 1, 12, 1, 11]=6.2842 ; n[ 1, 12, 1, 11]=3 ; c[ 1, 12, 1, 11]=4.47 ;
    # e[ 11, 1, 12, 1]=-4.184 ; v[ 11, 1, 12, 1]=6.2842 ; n[ 11, 1, 12, 1]=3 ; c[ 11, 1, 12, 1]=4.47 ;
    # e[ 1, 11, 1, 12]=-4.184 ; v[ 1, 11, 1, 12]=6.2842 ; n[ 1, 11, 1, 12]=3 ; c[ 1, 11, 1, 12]=4.47 ;
    
    # #01
    # e[ 1, 10, 1, 10]=-6.247 ; v[ 1, 10, 1, 10]=6.1172 ; n[ 1, 10, 1, 10]=4 ; c[ 1, 10, 1, 10]=3.93 ;
    # e[ 10, 1, 10, 1]=-6.247 ; v[ 10, 1, 10, 1]=6.1172 ; n[ 10, 1, 10, 1]=4 ; c[ 10, 1, 10, 1]=3.93 ;
    # e[ 12, 1, 12, 1]=-6.247 ; v[ 12, 1, 12, 1]=6.1172 ; n[ 12, 1, 12, 1]=4 ; c[ 12, 1, 12, 1]=3.93 ;
    # e[ 1, 12, 1, 12]=-6.247 ; v[ 1, 12, 1, 12]=6.1172 ; n[ 1, 12, 1, 12]=4 ; c[ 1, 12, 1, 12]=3.93 ;
    
    # #012
    # e[ 1, 10, 2, 10]=-8.163 ; v[ 1, 10, 2, 10]=5.7937 ; n[ 1, 10, 2, 10]=5 ; c[ 1, 10, 2, 10]=3.37 ;
    # e[ 10, 1, 10, 2]=-8.163 ; v[ 10, 1, 10, 2]=5.7937 ; n[ 10, 1, 10, 2]=5 ; c[ 10, 1, 10, 2]=3.37 ;
    # e[ 2, 10, 1, 10]=-8.163 ; v[ 2, 10, 1, 10]=5.7937 ; n[ 2, 10, 1, 10]=5 ; c[ 2, 10, 1, 10]=3.37 ;
    # e[ 10, 2, 10, 1]=-8.163 ; v[ 10, 2, 10, 1]=5.7937 ; n[ 10, 2, 10, 1]=5 ; c[ 10, 2, 10, 1]=3.37 ;
    # e[ 12, 3, 12, 1]=-8.163 ; v[ 12, 3, 12, 1]=5.7937 ; n[ 12, 3, 12, 1]=5 ; c[ 12, 3, 12, 1]=3.37 ;
    # e[ 3, 12, 1, 12]=-8.163 ; v[ 3, 12, 1, 12]=5.7937 ; n[ 3, 12, 1, 12]=5 ; c[ 3, 12, 1, 12]=3.37 ;
    # e[ 12, 1, 12, 3]=-8.163 ; v[ 12, 1, 12, 3]=5.7937 ; n[ 12, 1, 12, 3]=5 ; c[ 12, 1, 12, 3]=3.37 ;
    # e[ 1, 12, 3, 12]=-8.163 ; v[ 1, 12, 3, 12]=5.7937 ; n[ 1, 12, 3, 12]=5 ; c[ 1, 12, 3, 12]=3.37 ;
    
    # #0123
    # e[ 2, 10, 2, 10]=-9.946 ; v[ 2, 10, 2, 10]=5.5189 ; n[ 2, 10, 2, 10]=6 ; c[ 2, 10, 2, 10]=0.13 ;
    # e[ 10, 2, 10, 2]=-9.946 ; v[ 10, 2, 10, 2]=5.5189 ; n[ 10, 2, 10, 2]=6 ; c[ 10, 2, 10, 2]=0.13 ;
    # e[ 12, 3, 12, 3]=-9.946 ; v[ 12, 3, 12, 3]=5.5189 ; n[ 12, 3, 12, 3]=6 ; c[ 12, 3, 12, 3]=0.13 ;
    # e[ 3, 12, 3, 12]=-9.946 ; v[ 3, 12, 3, 12]=5.5189 ; n[ 3, 12, 3, 12]=6 ; c[ 3, 12, 3, 12]=0.13 ;
    
    # #01234
    # e[ 2, 10, 4, 10]=-11.507 ; v[ 2, 10, 4, 10]=4.4197 ; n[ 2, 10, 4, 10]=7 ; c[ 2, 10, 4, 10]=2.74 ;
    # e[ 10, 2, 10, 4]=-11.507 ; v[ 10, 2, 10, 4]=4.4197 ; n[ 10, 2, 10, 4]=7 ; c[ 10, 2, 10, 4]=2.74 ;
    # e[ 4, 10, 2, 10]=-11.507 ; v[ 4, 10, 2, 10]=4.4197 ; n[ 4, 10, 2, 10]=7 ; c[ 4, 10, 2, 10]=2.74 ;
    # e[ 10, 4, 10, 2]=-11.507 ; v[ 10, 4, 10, 2]=4.4197 ; n[ 10, 4, 10, 2]=7 ; c[ 10, 4, 10, 2]=2.74 ;
    # e[ 12, 4, 12, 3]=-11.507 ; v[ 12, 4, 12, 3]=4.4197 ; n[ 12, 4, 12, 3]=7 ; c[ 12, 4, 12, 3]=2.74 ;
    # e[ 4, 12, 3, 12]=-11.507 ; v[ 4, 12, 3, 12]=4.4197 ; n[ 4, 12, 3, 12]=7 ; c[ 4, 12, 3, 12]=2.74 ;
    # e[ 12, 3, 12, 4]=-11.507 ; v[ 12, 3, 12, 4]=4.4197 ; n[ 12, 3, 12, 4]=7 ; c[ 12, 3, 12, 4]=2.74 ;
    # e[ 3, 12, 4, 12]=-11.507 ; v[ 3, 12, 4, 12]=4.4197 ; n[ 3, 12, 4, 12]=7 ; c[ 3, 12, 4, 12]=2.74 ;
    
    # #012345
    # e[ 4, 10, 4, 10]=-12.258 ; v[ 4, 10, 4, 10]=2.9568 ; n[ 4, 10, 4, 10]=8 ; c[ 4, 10, 4, 10]=2.31 ;
    # e[ 10, 4, 10, 4]=-12.258 ; v[ 10, 4, 10, 4]=2.9568 ; n[ 10, 4, 10, 4]=8 ; c[ 10, 4, 10, 4]=2.31 ;
    # e[ 12, 4, 12, 4]=-12.258 ; v[ 12, 4, 12, 4]=2.9568 ; n[ 12, 4, 12, 4]=8 ; c[ 12, 4, 12, 4]=2.31 ;
    # e[ 4, 12, 4, 12]=-12.258 ; v[ 4, 12, 4, 12]=2.9568 ; n[ 4, 12, 4, 12]=8 ; c[ 4, 12, 4, 12]=2.31 ;
    
    # #Two Horizontal 2
    
    # #Blank
    # e[ 1, 5, 1, 14]=-1.760 ; v[ 1, 5, 1, 14]=6.8169 ; n[ 1, 5, 1, 14]=2 ; c[ 1, 5, 1, 14]=7.77 ;
    # e[ 5, 1, 14, 1]=-1.760 ; v[ 5, 1, 14, 1]=6.8169 ; n[ 5, 1, 14, 1]=2 ; c[ 5, 1, 14, 1]=7.77 ;
    # e[ 1, 14, 1, 5]=-1.760 ; v[ 1, 14, 1, 5]=6.8169 ; n[ 1, 14, 1, 5]=2 ; c[ 1, 14, 1, 5]=7.77 ;
    # e[ 14, 1, 5, 1]=-1.760 ; v[ 14, 1, 5, 1]=6.8169 ; n[ 14, 1, 5, 1]=2 ; c[ 14, 1, 5, 1]=7.77 ;
    
    # #0
    # e[ 1, 5, 1, 16]=-3.596 ; v[ 1, 5, 1, 16]=6.7166 ; n[ 1, 5, 1, 16]=3 ; c[ 1, 5, 1, 16]=7.58 ;
    # e[ 5, 1, 16, 1]=-3.596 ; v[ 5, 1, 16, 1]=6.7166 ; n[ 5, 1, 16, 1]=3 ; c[ 5, 1, 16, 1]=7.58 ;
    # e[ 1, 16, 1, 5]=-3.596 ; v[ 1, 16, 1, 5]=6.7166 ; n[ 1, 16, 1, 5]=3 ; c[ 1, 16, 1, 5]=7.58 ;
    # e[ 16, 1, 5, 1]=-3.596 ; v[ 16, 1, 5, 1]=6.7166 ; n[ 16, 1, 5, 1]=3 ; c[ 16, 1, 5, 1]=7.58 ;
    # e[ 6, 1, 14, 1]=-3.596 ; v[ 6, 1, 14, 1]=6.7166 ; n[ 6, 1, 14, 1]=3 ; c[ 6, 1, 14, 1]=7.58 ;
    # e[ 1, 6, 1, 14]=-3.596 ; v[ 1, 6, 1, 14]=6.7166 ; n[ 1, 6, 1, 14]=3 ; c[ 1, 6, 1, 14]=7.58 ;
    # e[ 14, 1, 6, 1]=-3.596 ; v[ 14, 1, 6, 1]=6.7166 ; n[ 14, 1, 6, 1]=3 ; c[ 14, 1, 6, 1]=7.58 ;
    # e[ 1, 14, 1, 6]=-3.596 ; v[ 1, 14, 1, 6]=6.7166 ; n[ 1, 14, 1, 6]=3 ; c[ 1, 14, 1, 6]=7.58 ;
    
    # #01
    # e[ 1, 7, 1, 16]=-5.530 ; v[ 1, 7, 1, 16]=6.0476 ; n[ 1, 7, 1, 16]=4 ; c[ 1, 7, 1, 16]=6.74 ;
    # e[ 7, 1, 16, 1]=-5.530 ; v[ 7, 1, 16, 1]=6.0476 ; n[ 7, 1, 16, 1]=4 ; c[ 7, 1, 16, 1]=6.74 ;
    # e[ 1, 16, 1, 7]=-5.530 ; v[ 1, 16, 1, 7]=6.0476 ; n[ 1, 16, 1, 7]=4 ; c[ 1, 16, 1, 7]=6.74 ;
    # e[ 16, 1, 7, 1]=-5.530 ; v[ 16, 1, 7, 1]=6.0476 ; n[ 16, 1, 7, 1]=4 ; c[ 16, 1, 7, 1]=6.74 ;
    # e[ 6, 1, 15, 1]=-5.530 ; v[ 6, 1, 15, 1]=6.0476 ; n[ 6, 1, 15, 1]=4 ; c[ 6, 1, 15, 1]=6.74 ;
    # e[ 1, 6, 1, 15]=-5.530 ; v[ 1, 6, 1, 15]=6.0476 ; n[ 1, 6, 1, 15]=4 ; c[ 1, 6, 1, 15]=6.74 ;
    # e[ 15, 1, 6, 1]=-5.530 ; v[ 15, 1, 6, 1]=6.0476 ; n[ 15, 1, 6, 1]=4 ; c[ 15, 1, 6, 1]=6.74 ;
    # e[ 1, 15, 1, 6]=-5.530 ; v[ 1, 15, 1, 6]=6.0476 ; n[ 1, 15, 1, 6]=4 ; c[ 1, 15, 1, 6]=6.74 ;
    
    # #012
    # e[ 1, 7, 2, 16]=-7.237 ; v[ 1, 7, 2, 16]=5.7661 ; n[ 1, 7, 2, 16]=5 ; c[ 1, 7, 2, 16]=6.11 ;
    # e[ 7, 1, 16, 2]=-7.237 ; v[ 7, 1, 16, 2]=5.7661 ; n[ 7, 1, 16, 2]=5 ; c[ 7, 1, 16, 2]=6.11 ;
    # e[ 2, 16, 1, 7]=-7.237 ; v[ 2, 16, 1, 7]=5.7661 ; n[ 2, 16, 1, 7]=5 ; c[ 2, 16, 1, 7]=6.11 ;
    # e[ 16, 2, 7, 1]=-7.237 ; v[ 16, 2, 7, 1]=5.7661 ; n[ 16, 2, 7, 1]=5 ; c[ 16, 2, 7, 1]=6.11 ;
    # e[ 6, 3, 15, 1]=-7.237 ; v[ 6, 3, 15, 1]=5.7661 ; n[ 6, 3, 15, 1]=5 ; c[ 6, 3, 15, 1]=6.11 ;
    # e[ 3, 6, 1, 15]=-7.237 ; v[ 3, 6, 1, 15]=5.7661 ; n[ 3, 6, 1, 15]=5 ; c[ 3, 6, 1, 15]=6.11 ;
    # e[ 15, 1, 6, 3]=-7.237 ; v[ 15, 1, 6, 3]=5.7661 ; n[ 15, 1, 6, 3]=5 ; c[ 15, 1, 6, 3]=6.11 ;
    # e[ 1, 15, 3, 6]=-7.237 ; v[ 1, 15, 3, 6]=5.7661 ; n[ 1, 15, 3, 6]=5 ; c[ 1, 15, 3, 6]=6.11 ;
    
    # #0123
    # e[ 2, 7, 2, 16]=-8.676 ; v[ 2, 7, 2, 16]=5.5103 ; n[ 2, 7, 2, 16]=6 ; c[ 2, 7, 2, 16]=5.84 ;
    # e[ 7, 2, 16, 2]=-8.676 ; v[ 7, 2, 16, 2]=5.5103 ; n[ 7, 2, 16, 2]=6 ; c[ 7, 2, 16, 2]=5.84 ;
    # e[ 2, 16, 2, 7]=-8.676 ; v[ 2, 16, 2, 7]=5.5103 ; n[ 2, 16, 2, 7]=6 ; c[ 2, 16, 2, 7]=5.84 ;
    # e[ 16, 2, 7, 2]=-8.676 ; v[ 16, 2, 7, 2]=5.5103 ; n[ 16, 2, 7, 2]=6 ; c[ 16, 2, 7, 2]=5.84 ;
    # e[ 6, 3, 15, 3]=-8.676 ; v[ 6, 3, 15, 3]=5.5103 ; n[ 6, 3, 15, 3]=6 ; c[ 6, 3, 15, 3]=5.84 ;
    # e[ 3, 6, 3, 15]=-8.676 ; v[ 3, 6, 3, 15]=5.5103 ; n[ 3, 6, 3, 15]=6 ; c[ 3, 6, 3, 15]=5.84 ;
    # e[ 15, 3, 6, 3]=-8.676 ; v[ 15, 3, 6, 3]=5.5103 ; n[ 15, 3, 6, 3]=6 ; c[ 15, 3, 6, 3]=5.84 ;
    # e[ 3, 15, 3, 6]=-8.676 ; v[ 3, 15, 3, 6]=5.5103 ; n[ 3, 15, 3, 6]=6 ; c[ 3, 15, 3, 6]=5.84 ;
    
    # #01234
    # e[ 2, 7, 4, 16]=-10.159 ; v[ 2, 7, 4, 16]=4.3982 ; n[ 2, 7, 4, 16]=7 ; c[ 2, 7, 4, 16]=5.19 ;
    # e[ 7, 2, 16, 4]=-10.159 ; v[ 7, 2, 16, 4]=4.3982 ; n[ 7, 2, 16, 4]=7 ; c[ 7, 2, 16, 4]=5.19 ;
    # e[ 4, 16, 2, 7]=-10.159 ; v[ 4, 16, 2, 7]=4.3982 ; n[ 4, 16, 2, 7]=7 ; c[ 4, 16, 2, 7]=5.19 ;
    # e[ 16, 4, 7, 2]=-10.159 ; v[ 16, 4, 7, 2]=4.3982 ; n[ 16, 4, 7, 2]=7 ; c[ 16, 4, 7, 2]=5.19 ;
    # e[ 6, 4, 15, 3]=-10.159 ; v[ 6, 4, 15, 3]=4.3982 ; n[ 6, 4, 15, 3]=7 ; c[ 6, 4, 15, 3]=5.19 ;
    # e[ 4, 6, 3, 15]=-10.159 ; v[ 4, 6, 3, 15]=4.3982 ; n[ 4, 6, 3, 15]=7 ; c[ 4, 6, 3, 15]=5.19 ;
    # e[ 15, 3, 6, 4]=-10.159 ; v[ 15, 3, 6, 4]=4.3982 ; n[ 15, 3, 6, 4]=7 ; c[ 15, 3, 6, 4]=5.19 ;
    # e[ 3, 15, 4, 6]=-10.159 ; v[ 3, 15, 4, 6]=4.3982 ; n[ 3, 15, 4, 6]=7 ; c[ 3, 15, 4, 6]=5.19 ;
    
    # #12345
    # e[ 4, 7, 4, 16]=-10.751 ; v[ 4, 7, 4, 16]=2.9235 ; n[ 4, 7, 4, 16]=8 ; c[ 4, 7, 4, 16]=4.80 ;
    # e[ 7, 4, 16, 4]=-10.751 ; v[ 7, 4, 16, 4]=2.9235 ; n[ 7, 4, 16, 4]=8 ; c[ 7, 4, 16, 4]=4.80 ;
    # e[ 4, 16, 4, 7]=-10.751 ; v[ 4, 16, 4, 7]=2.9235 ; n[ 4, 16, 4, 7]=8 ; c[ 4, 16, 4, 7]=4.80 ;
    # e[ 16, 4, 7, 4]=-10.751 ; v[ 16, 4, 7, 4]=2.9235 ; n[ 16, 4, 7, 4]=8 ; c[ 16, 4, 7, 4]=4.80 ;
    # e[ 6, 4, 15, 4]=-10.751 ; v[ 6, 4, 15, 4]=2.9235 ; n[ 6, 4, 15, 4]=8 ; c[ 6, 4, 15, 4]=4.80 ;
    # e[ 4, 6, 4, 15]=-10.751 ; v[ 4, 6, 4, 15]=2.9235 ; n[ 4, 6, 4, 15]=8 ; c[ 4, 6, 4, 15]=4.80 ;
    # e[ 15, 4, 6, 4]=-10.751 ; v[ 15, 4, 6, 4]=2.9235 ; n[ 15, 4, 6, 4]=8 ; c[ 15, 4, 6, 4]=4.80 ;
    # e[ 4, 15, 4, 6]=-10.751 ; v[ 4, 15, 4, 6]=2.9235 ; n[ 4, 15, 4, 6]=8 ; c[ 4, 15, 4, 6]=4.80 ;
    
    # #3H Subsurface RuO2
    
    # #Blank
    # e[ 1, 8, 1, 14]=-2.178 ; v[ 1, 8, 1, 14]=6.7100 ; n[ 1, 8, 1, 14]=3 ; c[ 1, 8, 1, 14]=7.84 ;
    # e[ 8, 1, 14, 1]=-2.178 ; v[ 8, 1, 14, 1]=6.7100 ; n[ 8, 1, 14, 1]=3 ; c[ 8, 1, 14, 1]=7.84 ;
    # e[ 1, 14, 1, 8]=-2.178 ; v[ 1, 14, 1, 8]=6.7100 ; n[ 1, 14, 1, 8]=3 ; c[ 1, 14, 1, 8]=7.84 ;
    # e[ 14, 1, 8, 1]=-2.178 ; v[ 14, 1, 8, 1]=6.7100 ; n[ 14, 1, 8, 1]=3 ; c[ 14, 1, 8, 1]=7.84 ;
    # e[ 11, 1, 14, 1]=-2.178 ; v[ 11, 1, 14, 1]=6.7100 ; n[ 11, 1, 14, 1]=3 ; c[ 11, 1, 14, 1]=7.84 ;
    # e[ 1, 11, 1, 14]=-2.178 ; v[ 1, 11, 1, 14]=6.7100 ; n[ 1, 11, 1, 14]=3 ; c[ 1, 11, 1, 14]=7.84 ;
    # e[ 14, 1, 11, 1]=-2.178 ; v[ 14, 1, 11, 1]=6.7100 ; n[ 14, 1, 11, 1]=3 ; c[ 14, 1, 11, 1]=7.84 ;
    # e[ 1, 14, 1, 11]=-2.178 ; v[ 1, 14, 1, 11]=6.7100 ; n[ 1, 14, 1, 11]=3 ; c[ 1, 14, 1, 11]=7.84 ;
    
    # #0
    # e[ 1, 8, 1, 16]=-4.062 ; v[ 1, 8, 1, 16]=6.5806 ; n[ 1, 8, 1, 16]=4 ; c[ 1, 8, 1, 16]=7.29 ;
    # e[ 8, 1, 16, 1]=-4.062 ; v[ 8, 1, 16, 1]=6.5806 ; n[ 8, 1, 16, 1]=4 ; c[ 8, 1, 16, 1]=7.29 ;
    # e[ 1, 16, 1, 8]=-4.062 ; v[ 1, 16, 1, 8]=6.5806 ; n[ 1, 16, 1, 8]=4 ; c[ 1, 16, 1, 8]=7.29 ;
    # e[ 16, 1, 8, 1]=-4.062 ; v[ 16, 1, 8, 1]=6.5806 ; n[ 16, 1, 8, 1]=4 ; c[ 16, 1, 8, 1]=7.29 ;
    # e[ 12, 1, 14, 1]=-4.062 ; v[ 12, 1, 14, 1]=6.5806 ; n[ 12, 1, 14, 1]=4 ; c[ 12, 1, 14, 1]=7.29 ;
    # e[ 1, 12, 1, 14]=-4.062 ; v[ 1, 12, 1, 14]=6.5806 ; n[ 1, 12, 1, 14]=4 ; c[ 1, 12, 1, 14]=7.29 ;
    # e[ 14, 1, 12, 1]=-4.062 ; v[ 14, 1, 12, 1]=6.5806 ; n[ 14, 1, 12, 1]=4 ; c[ 14, 1, 12, 1]=7.29 ;
    # e[ 1, 14, 1, 12]=-4.062 ; v[ 1, 14, 1, 12]=6.5806 ; n[ 1, 14, 1, 12]=4 ; c[ 1, 14, 1, 12]=7.29 ;
    
    # #01
    # e[ 1, 10, 1, 16]=-5.990 ; v[ 1, 10, 1, 16]=6.0201 ; n[ 1, 10, 1, 16]=5 ; c[ 1, 10, 1, 16]=6.81 ;
    # e[ 10, 1, 16, 1]=-5.990 ; v[ 10, 1, 16, 1]=6.0201 ; n[ 10, 1, 16, 1]=5 ; c[ 10, 1, 16, 1]=6.81 ;
    # e[ 1, 16, 1, 10]=-5.990 ; v[ 1, 16, 1, 10]=6.0201 ; n[ 1, 16, 1, 10]=5 ; c[ 1, 16, 1, 10]=6.81 ;
    # e[ 16, 1, 10, 1]=-5.990 ; v[ 16, 1, 10, 1]=6.0201 ; n[ 16, 1, 10, 1]=5 ; c[ 16, 1, 10, 1]=6.81 ;
    # e[ 12, 1, 15, 1]=-5.990 ; v[ 12, 1, 15, 1]=6.0201 ; n[ 12, 1, 15, 1]=5 ; c[ 12, 1, 15, 1]=6.81 ;
    # e[ 1, 12, 1, 15]=-5.990 ; v[ 1, 12, 1, 15]=6.0201 ; n[ 1, 12, 1, 15]=5 ; c[ 1, 12, 1, 15]=6.81 ;
    # e[ 15, 1, 12, 1]=-5.990 ; v[ 15, 1, 12, 1]=6.0201 ; n[ 15, 1, 12, 1]=5 ; c[ 15, 1, 12, 1]=6.81 ;
    # e[ 1, 15, 1, 12]=-5.990 ; v[ 1, 15, 1, 12]=6.0201 ; n[ 1, 15, 1, 12]=5 ; c[ 1, 15, 1, 12]=6.81 ;
    
    # #012
    # e[ 1, 10, 2, 16]=-7.719 ; v[ 1, 10, 2, 16]=5.7433 ; n[ 1, 10, 2, 16]=6 ; c[ 1, 10, 2, 16]=5.94 ;
    # e[ 10, 1, 16, 2]=-7.719 ; v[ 10, 1, 16, 2]=5.7433 ; n[ 10, 1, 16, 2]=6 ; c[ 10, 1, 16, 2]=5.94 ;
    # e[ 2, 16, 1, 10]=-7.719 ; v[ 2, 16, 1, 10]=5.7433 ; n[ 2, 16, 1, 10]=6 ; c[ 2, 16, 1, 10]=5.94 ;
    # e[ 16, 2, 10, 1]=-7.719 ; v[ 16, 2, 10, 1]=5.7433 ; n[ 16, 2, 10, 1]=6 ; c[ 16, 2, 10, 1]=5.94 ;
    # e[ 12, 3, 15, 1]=-7.719 ; v[ 12, 3, 15, 1]=5.7433 ; n[ 12, 3, 15, 1]=6 ; c[ 12, 3, 15, 1]=5.94 ;
    # e[ 3, 12, 1, 15]=-7.719 ; v[ 3, 12, 1, 15]=5.7433 ; n[ 3, 12, 1, 15]=6 ; c[ 3, 12, 1, 15]=5.94 ;
    # e[ 15, 1, 12, 3]=-7.719 ; v[ 15, 1, 12, 3]=5.7433 ; n[ 15, 1, 12, 3]=6 ; c[ 15, 1, 12, 3]=5.94 ;
    # e[ 1, 15, 3, 12]=-7.719 ; v[ 1, 15, 3, 12]=5.7433 ; n[ 1, 15, 3, 12]=6 ; c[ 1, 15, 3, 12]=5.94 ;
    
    # #0123
    # e[ 2, 10, 2, 16]=-9.187 ; v[ 2, 10, 2, 16]=5.4001 ; n[ 2, 10, 2, 16]=7 ; c[ 2, 10, 2, 16]=5.97 ;
    # e[ 10, 2, 16, 2]=-9.187 ; v[ 10, 2, 16, 2]=5.4001 ; n[ 10, 2, 16, 2]=7 ; c[ 10, 2, 16, 2]=5.97 ;
    # e[ 2, 16, 2, 10]=-9.187 ; v[ 2, 16, 2, 10]=5.4001 ; n[ 2, 16, 2, 10]=7 ; c[ 2, 16, 2, 10]=5.97 ;
    # e[ 16, 2, 10, 2]=-9.187 ; v[ 16, 2, 10, 2]=5.4001 ; n[ 16, 2, 10, 2]=7 ; c[ 16, 2, 10, 2]=5.97 ;
    # e[ 12, 3, 15, 3]=-9.187 ; v[ 12, 3, 15, 3]=5.4001 ; n[ 12, 3, 15, 3]=7 ; c[ 12, 3, 15, 3]=5.97 ;
    # e[ 3, 12, 3, 15]=-9.187 ; v[ 3, 12, 3, 15]=5.4001 ; n[ 3, 12, 3, 15]=7 ; c[ 3, 12, 3, 15]=5.97 ;
    # e[ 15, 3, 12, 3]=-9.187 ; v[ 15, 3, 12, 3]=5.4001 ; n[ 15, 3, 12, 3]=7 ; c[ 15, 3, 12, 3]=5.97 ;
    # e[ 3, 15, 3, 12]=-9.187 ; v[ 3, 15, 3, 12]=5.4001 ; n[ 3, 15, 3, 12]=7 ; c[ 3, 15, 3, 12]=5.97 ;
    
    # #01234
    # e[ 2, 10, 4, 16]=-10.560 ; v[ 2, 10, 4, 16]=4.2983 ; n[ 2, 10, 4, 16]=8 ; c[ 2, 10, 4, 16]=5.44 ;
    # e[ 10, 2, 16, 4]=-10.560 ; v[ 10, 2, 16, 4]=4.2983 ; n[ 10, 2, 16, 4]=8 ; c[ 10, 2, 16, 4]=5.44 ;
    # e[ 4, 16, 2, 10]=-10.560 ; v[ 4, 16, 2, 10]=4.2983 ; n[ 4, 16, 2, 10]=8 ; c[ 4, 16, 2, 10]=5.44 ;
    # e[ 16, 4, 10, 2]=-10.560 ; v[ 16, 4, 10, 2]=4.2983 ; n[ 16, 4, 10, 2]=8 ; c[ 16, 4, 10, 2]=5.44 ;
    # e[ 12, 4, 15, 3]=-10.560 ; v[ 12, 4, 15, 3]=4.2983 ; n[ 12, 4, 15, 3]=8 ; c[ 12, 4, 15, 3]=5.44 ;
    # e[ 4, 12, 3, 15]=-10.560 ; v[ 4, 12, 3, 15]=4.2983 ; n[ 4, 12, 3, 15]=8 ; c[ 4, 12, 3, 15]=5.44 ;
    # e[ 15, 3, 12, 4]=-10.560 ; v[ 15, 3, 12, 4]=4.2983 ; n[ 15, 3, 12, 4]=8 ; c[ 15, 3, 12, 4]=5.44 ;
    # e[ 3, 15, 4, 12]=-10.560 ; v[ 3, 15, 4, 12]=4.2983 ; n[ 3, 15, 4, 12]=8 ; c[ 3, 15, 4, 12]=5.44 ;
    
    # #012345
    
    # e[ 4, 10, 4, 16]=-11.113 ; v[ 4, 10, 4, 16]=2.7986 ; n[ 4, 10, 4, 16]=9 ; c[ 4, 10, 4, 16]=4.80 ;
    # e[ 10, 4, 16, 4]=-11.113 ; v[ 10, 4, 16, 4]=2.7986 ; n[ 10, 4, 16, 4]=9 ; c[ 10, 4, 16, 4]=4.80 ;
    # e[ 4, 16, 4, 10]=-11.113 ; v[ 4, 16, 4, 10]=2.7986 ; n[ 4, 16, 4, 10]=9 ; c[ 4, 16, 4, 10]=4.80 ;
    # e[ 16, 4, 10, 4]=-11.113 ; v[ 16, 4, 10, 4]=2.7986 ; n[ 16, 4, 10, 4]=9 ; c[ 16, 4, 10, 4]=4.80 ;
    # e[ 12, 4, 15, 4]=-11.113 ; v[ 12, 4, 15, 4]=2.7986 ; n[ 12, 4, 15, 4]=9 ; c[ 12, 4, 15, 4]=4.80 ;
    # e[ 4, 12, 4, 15]=-11.113 ; v[ 4, 12, 4, 15]=2.7986 ; n[ 4, 12, 4, 15]=9 ; c[ 4, 12, 4, 15]=4.80 ;
    # e[ 15, 4, 12, 4]=-11.113 ; v[ 15, 4, 12, 4]=2.7986 ; n[ 15, 4, 12, 4]=9 ; c[ 15, 4, 12, 4]=4.80 ;
    # e[ 4, 15, 4, 12]=-11.113 ; v[ 4, 15, 4, 12]=2.7986 ; n[ 4, 15, 4, 12]=9 ; c[ 4, 15, 4, 12]=4.80 ;
        
    # #4H Subsurface RuO2
    
    # #Blank
    # e[ 1, 14, 1, 14]=-2.070 ; v[ 1, 14, 1, 14]=6.8459 ; n[ 1, 14, 1, 14]=4 ; c[ 1, 14, 1, 14]=6.82 ;
    # e[ 14, 1, 14, 1]=-2.070 ; v[ 14, 1, 14, 1]=6.8459 ; n[ 14, 1, 14, 1]=4 ; c[ 14, 1, 14, 1]=6.82 ;
    
    # #0
    # e[ 1, 14, 1, 16]=-4.047 ; v[ 1, 14, 1, 16]=6.4674 ; n[ 1, 14, 1, 16]=5 ; c[ 1, 14, 1, 16]=6.31 ;
    # e[ 14, 1, 16, 1]=-4.047 ; v[ 14, 1, 16, 1]=6.4674 ; n[ 14, 1, 16, 1]=5 ; c[ 14, 1, 16, 1]=6.31 ;
    # e[ 1, 16, 1, 14]=-4.047 ; v[ 1, 16, 1, 14]=6.4674 ; n[ 1, 16, 1, 14]=5 ; c[ 1, 16, 1, 14]=6.31 ;
    # e[ 16, 1, 14, 1]=-4.047 ; v[ 16, 1, 14, 1]=6.4674 ; n[ 16, 1, 14, 1]=5 ; c[ 16, 1, 14, 1]=6.31 ;
    # e[ 15, 1, 14, 1]=-4.047 ; v[ 15, 1, 14, 1]=6.4674 ; n[ 15, 1, 14, 1]=5 ; c[ 15, 1, 14, 1]=6.31 ;
    # e[ 1, 15, 1, 14]=-4.047 ; v[ 1, 15, 1, 14]=6.4674 ; n[ 1, 15, 1, 14]=5 ; c[ 1, 15, 1, 14]=6.31 ;
    # e[ 14, 1, 15, 1]=-4.047 ; v[ 14, 1, 15, 1]=6.4674 ; n[ 14, 1, 15, 1]=5 ; c[ 14, 1, 15, 1]=6.31 ;
    # e[ 1, 14, 1, 15]=-4.047 ; v[ 1, 14, 1, 15]=6.4674 ; n[ 1, 14, 1, 15]=5 ; c[ 1, 14, 1, 15]=6.31 ;
    
    # #01
    # e[ 1, 16, 1, 16]=-5.970 ; v[ 1, 16, 1, 16]=5.9760 ; n[ 1, 16, 1, 16]=6 ; c[ 1, 16, 1, 16]=5.96 ;
    # e[ 16, 1, 16, 1]=-5.970 ; v[ 16, 1, 16, 1]=5.9760 ; n[ 16, 1, 16, 1]=6 ; c[ 16, 1, 16, 1]=5.96 ;
    # e[ 15, 1, 15, 1]=-5.970 ; v[ 15, 1, 15, 1]=5.9760 ; n[ 15, 1, 15, 1]=6 ; c[ 15, 1, 15, 1]=5.96 ;
    # e[ 1, 15, 1, 15]=-5.970 ; v[ 1, 15, 1, 15]=5.9760 ; n[ 1, 15, 1, 15]=6 ; c[ 1, 15, 1, 15]=5.96 ;
    
    # #012
    # e[ 1, 16, 2, 16]=-7.542 ; v[ 1, 16, 2, 16]=5.6340 ; n[ 1, 16, 2, 16]=7 ; c[ 1, 16, 2, 16]=5.24 ;
    # e[ 16, 1, 16, 2]=-7.542 ; v[ 16, 1, 16, 2]=5.6340 ; n[ 16, 1, 16, 2]=7 ; c[ 16, 1, 16, 2]=5.24 ;
    # e[ 2, 16, 1, 16]=-7.542 ; v[ 2, 16, 1, 16]=5.6340 ; n[ 2, 16, 1, 16]=7 ; c[ 2, 16, 1, 16]=5.24 ;
    # e[ 16, 2, 16, 1]=-7.542 ; v[ 16, 2, 16, 1]=5.6340 ; n[ 16, 2, 16, 1]=7 ; c[ 16, 2, 16, 1]=5.24 ;
    # e[ 15, 3, 15, 1]=-7.542 ; v[ 15, 3, 15, 1]=5.6340 ; n[ 15, 3, 15, 1]=7 ; c[ 15, 3, 15, 1]=5.24 ;
    # e[ 3, 15, 1 ,15]=-7.542 ; v[ 3, 15, 1 ,15]=5.6340 ; n[ 3, 15, 1 ,15]=7 ; c[ 3, 15, 1 ,15]=5.24 ;
    # e[ 15, 1, 15, 3]=-7.542 ; v[ 15, 1, 15, 3]=5.6340 ; n[ 15, 1, 15, 3]=7 ; c[ 15, 1, 15, 3]=5.24 ;
    # e[ 1, 15, 3, 15]=-7.542 ; v[ 1, 15, 3, 15]=5.6340 ; n[ 1, 15, 3, 15]=7 ; c[ 1, 15, 3, 15]=5.24 ;
    
    # #0123
    # e[ 2, 16, 2, 16]=-8.866 ; v[ 2, 16, 2, 16]=5.3242 ; n[ 2, 16, 2, 16]=8 ; c[ 2, 16, 2, 16]=5.16 ;
    # e[ 16, 2, 16, 2]=-8.866 ; v[ 16, 2, 16, 2]=5.3242 ; n[ 16, 2, 16, 2]=8 ; c[ 16, 2, 16, 2]=5.16 ;
    # e[ 15, 3, 15, 3]=-8.866 ; v[ 15, 3, 15, 3]=5.3242 ; n[ 15, 3, 15, 3]=8 ; c[ 15, 3, 15, 3]=5.16 ;
    # e[ 3, 15, 3, 15]=-8.866 ; v[ 3, 15, 3, 15]=5.3242 ; n[ 3, 15, 3, 15]=8 ; c[ 3, 15, 3, 15]=5.16 ;
    
    # #01234
    # e[ 2, 16, 4, 16]=-10.171 ; v[ 2, 16, 4, 16]=4.2169 ; n[ 2, 16, 4, 16]=9 ; c[ 2, 16, 4, 16]=4.85 ;
    # e[ 16, 2, 16, 4]=-10.171 ; v[ 16, 2, 16, 4]=4.2169 ; n[ 16, 2, 16, 4]=9 ; c[ 16, 2, 16, 4]=4.85 ;
    # e[ 4, 16, 2, 16]=-10.171 ; v[ 4, 16, 2, 16]=4.2169 ; n[ 4, 16, 2, 16]=9 ; c[ 4, 16, 2, 16]=4.85 ;
    # e[ 16, 4, 16, 2]=-10.171 ; v[ 16, 4, 16, 2]=4.2169 ; n[ 16, 4, 16, 2]=9 ; c[ 16, 4, 16, 2]=4.85 ;
    # e[ 15, 4, 15, 3]=-10.171 ; v[ 15, 4, 15, 3]=4.2169 ; n[ 15, 4, 15, 3]=9 ; c[ 15, 4, 15, 3]=4.85 ;
    # e[ 4, 15, 3, 15]=-10.171 ; v[ 4, 15, 3, 15]=4.2169 ; n[ 4, 15, 3, 15]=9 ; c[ 4, 15, 3, 15]=4.85 ;
    # e[ 15, 3, 15, 4]=-10.171 ; v[ 15, 3, 15, 4]=4.2169 ; n[ 15, 3, 15, 4]=9 ; c[ 15, 3, 15, 4]=4.85 ;
    # e[ 3, 15, 4, 15]=-10.171 ; v[ 3, 15, 4, 15]=4.2169 ; n[ 3, 15, 4, 15]=9 ; c[ 3, 15, 4, 15]=4.85 ;
    
    # #012345
    # e[ 4, 16, 4, 16]=-10.655 ; v[ 4, 16, 4, 16]=2.7138 ; n[ 4, 16, 4, 16]=10 ; c[ 4, 16, 4, 16]=4.44 ;
    # e[ 16, 4, 16, 4]=-10.655 ; v[ 16, 4, 16, 4]=2.7138 ; n[ 16, 4, 16, 4]=10 ; c[ 16, 4, 16, 4]=4.44 ;
    # e[ 15, 4, 15, 4]=-10.655 ; v[ 15, 4, 15, 4]=2.7138 ; n[ 15, 4, 15, 4]=10 ; c[ 15, 4, 15, 4]=4.44 ;
    # e[ 4, 15, 4, 15]=-10.655 ; v[ 4, 15, 4, 15]=2.7138 ; n[ 4, 15, 4, 15]=10 ; c[ 4, 15, 4, 15]=4.44 ;

    c = c/100.0/16.0*area
    v = v[ 1, 5, 1, 5] + gamma*(v - v[ 1, 5, 1, 5])
    
    return e,v,n, name, occurrence

def write_xyz(xyz_file, sigma_old, n_hydrogen_total):
    #Transpose the lattice
    sigma = np.transpose(sigma_old)
 
    #Get the size of the array
    n1, n2 = np.shape(sigma_old)
    num_atoms = (n1-1)*(n2-1)*6 + n_hydrogen_total

    #Define the lattice constants of the 'primitive' cell per site
    delta_bridge_atop = 3.18781 #Angstroms
    delta_bridge_bridge = 3.283255 #Angstroms

    #This defines the lattice positions for the structure
    bridge_sites = [["Ru", 1.272485, 0, 0],
                    ["O", 2.544976, 1.593907, 0],
                    ["O", 0, 1.593907, 0],
                    ["O", 1.272485, 0, 2.01076],
                    ["Ru", 1.272485, 1.593907, 3.283253],
                    ["O", 1.272485, 0, 4.555745]]

    atop_sites = [["Ru", 1.272491, 1.593907, 0],
                  ["O", 1.272491, 0, 1.272468],
                  ["O", 0, 1.593907, 3.283253],
                  ["Ru", 1.272491, 0, 3.283253],
                  ["O", 2.544983, 1.593907, 3.283253],
                  ["O", 1.272491, 0, 5.294014]]

    #These are the hydrogens adsorped to their sites
    #h[0] = Site 2, h[1] = Site 3, h[2] = Site 6, h[3] = Site 7,
    #h[4] = Subsurface 1, h[5] = Subsurface 2
    hydrogen_sites = [["H", 2.022486, 0, 5.305721],
                      ["H", 0.522484, 0, 5.305721],
                      ["H", 1.272491, -0.749997, 6.043991],
                      ["H", 1.272491, 0.749997, 6.043991],
                      ["H", 0, 1.593907, 2.283253],
                      ["H", 2.544983, 1.593907, 2.283253]]

    #Create a dictionary for each sigma state with atom coordinates
    sigma_dict = {1:bridge_sites, 
                  2:bridge_sites + [hydrogen_sites[0]], 
                  3:bridge_sites + [hydrogen_sites[1]], 
                  4:bridge_sites + hydrogen_sites[0:2],
                  5:atop_sites,
                  6:atop_sites + [hydrogen_sites[2]],
                  7:atop_sites + [hydrogen_sites[3]],
                  8:atop_sites + [hydrogen_sites[4]],
                  9:atop_sites + [hydrogen_sites[4]] + [hydrogen_sites[2]],
                 10:atop_sites + [hydrogen_sites[4]] + [hydrogen_sites[3]],
                 11:atop_sites + [hydrogen_sites[5]],
                 12:atop_sites + [hydrogen_sites[5]] + [hydrogen_sites[2]],
                 13:atop_sites + [hydrogen_sites[5]] + [hydrogen_sites[3]],
                 14:atop_sites + hydrogen_sites[4:6],
                 15:atop_sites + hydrogen_sites[4:6] + [hydrogen_sites[2]],
                 16:atop_sites + hydrogen_sites[4:6] + [hydrogen_sites[3]]}

    #Print the header of the file
    xyz_file.write("%d\n\n"%(num_atoms))

    y_delta = 0
    z = 0
    #Loop over each site on the surface and read in the sigma value
    #to print to the trajectory file
    for ix in range(1,n1):
        x_delta = 0
        for iy in range(1,n2):
            values = sigma_dict[sigma[ix][iy]]
            for item in values:
                xyz_file.write("{0:5} {1:.7f} {2:.7f} {3:.7f}\n"
                            .format(item[0], item[1]+x_delta, item[2]+y_delta, item[3]))
            #Each time you move to another site you need to increase
            #the change of positions in the dictionary
            x_delta = x_delta + delta_bridge_atop
 
        y_delta = y_delta + delta_bridge_bridge

if __name__ == "__main__":
    main()
