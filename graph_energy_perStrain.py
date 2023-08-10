import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import re

TS = 0.4063209313
convert_Ry2eV = 13.6057039763

def grep_file_last_occurrence(pattern, file_path):
    last_match_line = None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if re.search(pattern, line):
                    last_match_line = line.strip()
                    last_line = [i for i in last_match_line.split()]

        if last_match_line:
            #print(file_path, ":" , last_line[4])
            return last_line[4]
        else:
            print("Pattern not found in file", file_path)
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print(f"Error occurred: {e}")

def SortDict(Dict):
  Keys = list(Dict.keys())
  Keys.sort()
  sorted_dict = {i: Dict[i] for i in Keys}
  return sorted_dict

parser = argparse.ArgumentParser()
parser.add_argument("--config",type=str,required=True)
# parser.add_argument("--dir2",type=str,required=True)
args = parser.parse_args()

Strain_List = ["Strain0","Strain2", "StrainJin", "Strain1"]
count=0
List_EDict = {}

for arg in ["Strain0_Files","Strain2_Files", "StrainJin_Files", "Strain1_Files"]:
    E_0 = grep_file_last_occurrence("!", arg+'/RuO2-110-0-/RuO2-110-0-.out')
    E_H2 =  grep_file_last_occurrence("!", arg+'/H2/relax.out')
    E_Dict={}

    for filename in glob.glob(arg+'/RuO2-*/*.out'):
        with open(filename, 'r') as file:
            E_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("!", filename))})
            count +=1

    E_Dict=SortDict(E_Dict)

    for key in E_Dict:
        try:
            E_Dict.update({key: (float(E_Dict[key]) - (float(E_0)+ float(E_H2)*len(key)))/2*convert_Ry2eV}) 
            #print(key, ":", E_Dict[key])
        except TypeError:
            print("for structure", key, "there is no energy")
            
    List_EDict.update({arg[:arg.find("_")]:E_Dict})

Graph_EList = []


for Strain in Strain_List:
    Graph_EList.append(List_EDict[Strain][args.config]) 

plt.plot(["0%", "50%", "Jin%", "100%"], Graph_EList, 'o', label='%s'%(args.config))

plt.xlabel('Strain Percent towards TiO2', fontsize = 15)
plt.ylabel('Energies', fontsize = 15)

plt.legend()

plt.savefig('graph_energy_perStrain.png',dpi=600)


