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
parser.add_argument("--rootdir",type=str,required=True)
parser.add_argument("--name",type=str,required=False)
parser.add_argument("--gamma",nargs='+',required=False)
parser.add_argument("--pH",nargs='+',required=False)
args = parser.parse_args()

count=0

E_0 = grep_file_last_occurrence("!", args.rootdir+'/RuO2-110-0-/RuO2-110-0-.out')
E_H2 =  grep_file_last_occurrence("!", args.rootdir+'/H2/relax.out')
E_Dict={}

for filename in glob.glob(args.rootdir+'/RuO2-*/*.out'):
    if "old" in filename:
            continue
    with open(filename, 'r') as file:
        E_Dict.update({filename[len(filename) - filename[::-1].find("-0-") :filename.find(".")]:(grep_file_last_occurrence("!", filename))})
        count +=1

E_Dict=SortDict(E_Dict)

for key in E_Dict:
    try:
        E_Dict.update({key: (float(E_Dict[key]) - (float(E_0)+ (float(E_H2) + 0.05)*len(key)))/2*convert_Ry2eV}) #removed *1/0.79 for the Strain0 to try adding + 0.05 as correction to E_H2
        print(key, ":", E_Dict[key])
    except TypeError:
        print("for structure", key, "there is no energy")
    
print("There are", count, "configurations")
