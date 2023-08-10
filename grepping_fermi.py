import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
import re

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
parser.add_argument("--rootdir",type=str,required=True)
parser.add_argument("--name",type=str,required=False)
parser.add_argument("--gamma",nargs='+',required=False)
parser.add_argument("--pH",nargs='+',required=False)
args = parser.parse_args()

count_v=0
count_Gauss=0

v_Dict={}
G_Dict={}
F_Dict ={}

for filename in glob.glob(args.rootdir+'/RuO2-*/*.out'):
    if "old" in filename:
        continue
    with open(filename, 'r') as file:
        v_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("the Fermi energy", filename, 4))})
        count_v +=1
        if "Nathan" not in filename:
            G_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("Gauss", filename, 9))})
            count_Gauss += 1
        
v_Dict = SortDict(v_Dict)

if len(G_Dict) > 1:
        for key in v_Dict:
            try:
                F_Dict.update({key: -(float(v_Dict[key])+float(G_Dict[key]))}) 
                print(key, ":", F_Dict[key])
            except TypeError:
                print("for structure", key, "there is no fermi energy")
else:
    for key in v_Dict:
        try:
            F_Dict.update({key: -(float(v_Dict[key]))}) 
            print(key, ":", F_Dict[key])
        except TypeError:
            print("for structure", key, "there is no fermi energy")


print("There are",count_v, "Configurations with Fermi energy and", count_Gauss, "Configurations with Gaussian correction")
