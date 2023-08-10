import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import re

TS = 0.4063209313
convert_Ry2eV = 13.6057039763

def grep_file_last_occurrence(pattern, file_path, index):
    last_match_line = None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                if re.search(pattern, line):
                    last_match_line = line.strip()
                    last_line = [i for i in last_match_line.split()]

        if last_match_line:
            #print(file_path, ":" , last_line[4])
            return last_line[index]
        else:
            return "DNE"
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
parser.add_argument("--fe",type=str,required=True)
parser.add_argument("--dir1",type=str,required=False)
parser.add_argument("--dir2",type=str,required=False)
parser.add_argument("--config",nargs='+',required=False)
args = parser.parse_args()


dir1 = args.dir1
dir2 = args.dir2

if dir1 == None and dir2 == None:
    dir1 = "Nathan_Surface_Files"
    dir2 = "Strain0_Files"

count=0
List_EDict = {}
List_FDict = {}

for arg in [dir1,dir2]:
    E_0 = grep_file_last_occurrence("!", arg+'/RuO2-110-0-/RuO2-110-0-.out' , 4)
    E_H2 =  grep_file_last_occurrence("!", arg+'/H2/relax.out' , 4)
    E_Dict = {}
    F_Dict = {}
    v_Dict = {}
    G_Dict = {}
    
    if vars(args)["config"] != None:
        for config in vars(args)["config"]:
            filename = arg +'/RuO2-110-0-'+config+'/RuO2-110-0-' + config + '.out'
            with open(filename, 'r') as file:
                E_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("!", filename, 4))})
                v_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("the Fermi energy", filename, 4))})
                
                if "Nathan" not in filename:
                    G_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("Gauss", filename, 9))})
                
                count +=1
    else:
        for filename in glob.glob(arg+'/RuO2-*/*.out'):
            if "old" in filename:
                continue
            with open(filename, 'r') as file:
                E_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("!", filename, 4))})
                v_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("the Fermi energy", filename, 4))})
                
                if "Nathan" not in filename:
                    G_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("Gauss", filename, 9))})
                
                count +=1

    
    
    v_Dict=SortDict(v_Dict)    
    
    if len(G_Dict) > 1:
        for key in v_Dict:
            try:
                F_Dict.update({key: -(float(v_Dict[key])+float(G_Dict[key]))}) 
                #print(key, ":", F_Dict[key])
            except TypeError:
                print("for structure", key, "there is no fermi energy")
    else:
        for key in v_Dict:
            try:
                F_Dict.update({key: -(float(v_Dict[key]))}) 
                #print(key, ":", F_Dict[key])
            except TypeError:
                print("for structure", key, "there is no fermi energy")
        
    
    E_Dict=SortDict(E_Dict)
    i = 0
    for key in E_Dict:
        try:
            i += 1
            if "Nathan" not in arg:
                # E_Dict.update({key: -(float(E_Dict[key]) - (float(E_0)+ len(key)*(float(E_H2)-TS)))/2*convert_Ry2eV})    
                E_Dict.update({key: (float(E_Dict[key]) - (float(E_0) + len(key)*(float(E_H2)+0.05)))/2*convert_Ry2eV})              
            else:
                E_Dict.update({key: (float(E_Dict[key]) - (float(E_0) + len(key)*(float(E_H2))))/2*convert_Ry2eV})                 
            # print(arg, key, len(key))
        except TypeError:
            print("for structure", key, "there is no energy")
    
    List_EDict.update({arg[:arg.find("_")]:E_Dict})
    List_FDict.update({arg[:arg.find("_")]:F_Dict})




Graph_Edir1 = []
Graph_Edir2 = []

Graph_Fdir1 = []
Graph_Fdir2 = []

for key in List_EDict[dir1[:dir1.find("_")]]:
    
    Graph_Edir1.append(List_EDict[dir1[:dir1.find("_")]][key])
    Graph_Edir2.append(List_EDict[dir2[:dir2.find("_")]][key])
    
    Graph_Fdir1.append(List_FDict[dir1[:dir1.find("_")]][key])
    Graph_Fdir2.append(List_FDict[dir2[:dir2.find("_")]][key])

if args.fe == 'energy':
    x, y = np.array(Graph_Edir1), np.array(Graph_Edir2)  #removed *1/0.79 for the corrective factor to try -TS: 0.4063209313
    legend = ["Energy"]
    #plt.plot(Graph_Edir1, Graph_Edir2, 'o', label = "Energy")
elif args.fe == 'fermi':
    x, y = np.array(Graph_Fdir1), np.array(Graph_Fdir2)
    legend = ["Fermi"]
    #plt.plot(Graph_Fdir1, Graph_Fdir2, 'o', label = "Fermi")
else:
    print('must choose: energy or fermi')

#find line of best fit
a, b = np.polyfit(x, y, 1)
#add points to plot
plt.scatter(x, y, color='purple')
#add line of best fit to plot
plt.plot(x, x, color='steelblue', linestyle='--', linewidth=2, label = "line of best fit")
#add fitted regression equation to plot
plt.text(-8,-2 , 'y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x', size=14)

print('y = ' + '{:.2f}'.format(b) + ' + {:.2f}'.format(a) + 'x' , float(E_H2))

#-8,-2 for energy
#3.5, 6 for fermi

plt.xlabel(dir1[:dir1.find("_")], fontsize = 15)
plt.ylabel(dir2[:dir2.find("_")], fontsize = 15)

plt.legend(legend)

plt.savefig('graph_compare_energies.png',dpi=600)

# print(List_EDict)









