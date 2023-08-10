import glob
import numpy as np
import ase.io
import argparse
import matplotlib.pyplot as plt
import re


def grep_file_last_occurrence(pattern, file_path):
    last_match_line = None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            energies=[]
            for line_number, line in enumerate(file, start=1):
                if re.search(pattern, line):
                    match_line = line.strip()
                    energy_line = [i for i in match_line.split()]
                    if match_line:
                        energies.append(float(energy_line[4]))
                    else:
                        print("Pattern not found in file", file_path)
        return energies
        
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Error occurred: {e}")



parser = argparse.ArgumentParser()
parser.add_argument("--rootdir",type=str,required=True)
parser.add_argument("--config",nargs='+',required=False)
args = parser.parse_args()

count =0 
fig = plt.figure()
if 'Nathan' not in args.rootdir:
    if vars(args)["config"] != None:
        for config in vars(args)["config"]:
            filename = args.rootdir+'/RuO2-110-0-'+config+'/RuO2-110-0-' + config + '.out'
            energies = []
            atoms = ase.io.read(filename,index=":",format="espresso-out")
            for atom in atoms:
                energies.append(atom.get_total_energy())

            try:
                energies = np.array(energies) - energies[0]
            except:
                print(filename, "has no energies")
                
            plt.plot(np.arange(0,len(energies)), energies, 'o', label='%s, has %s points'%(args.rootdir[:args.rootdir.find("_")+1] + filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")], len(energies)))
            
            with open(filename, 'r') as file:
                if "bfgs converged" in file.read():    
                    print("bfgs converged for", filename)
                else:
                    count += 1
        plt.legend(fontsize="7", loc ="upper right")
                        
    else:
        for filename in glob.glob(args.rootdir+'/RuO2-*/*.out'):
            with open(filename, 'r') as file:
                if "bfgs converged" not in file.read():
                    count += 1
                    energies = []
                    atoms = ase.io.read(filename,index=":",format="espresso-out")
                    for atom in atoms:
                        energies.append(atom.get_total_energy())

                    try:
                        energies = np.array(energies) - energies[0]
                    except:
                        print(filename, "has no energies")
                        
                    plt.plot(np.arange(0,len(energies)), energies, 'o', label='%s, has %s points'%(args.rootdir[:args.rootdir.find("_")+1] + filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")], len(energies)))
        plt.legend(fontsize="7", loc ="lower right")
else: 
    count = 0
    for filename in glob.glob(args.rootdir+'/RuO2-*/*.out'):
        Nenergies = []
        if "old" in filename:
            continue
        count += 1
        with open(filename, 'r') as file:
            Nenergies.append(grep_file_last_occurrence("!", filename))
        try:
            Nenergies = np.array(Nenergies) - Nenergies[0]
        except:
            print(filename, "has no energies for Nathan")
                    
        plt.plot(np.arange(0,len(Nenergies)), Nenergies, 'o', label='%s, has %s points'%(args.rootdir[:args.rootdir.find("_")+1] + filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")], len(Nenergies)))
        plt.legend(fontsize="7", loc ="lower right")
plt.savefig('monitor_relaxation.png',dpi=600)
print(count, "configurations do not have bfgs convergence")



