#!/usr/bin/env python3
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir",type=str,required=True)
parser.add_argument("--n",type=str,required=True)
parser.add_argument("--name",type=str,required=False)
parser.add_argument("--gamma",nargs='+',required=False)
parser.add_argument("--pH",nargs='+',required=False)
args = parser.parse_args()

n = args.n
sh = f"RuO2_110_MC_{n}.sh"

rootdir= args.rootdir
name = args.name

#Writing gamma and pH arguments as a list args:
gamma = args.gamma
pH = args.pH
bash2 = ''

if vars(args)["gamma"] != None:
    gamma_str = ''
    for i in gamma:
        gamma_str = gamma_str + f'{i}'+" "
    bash2 = bash2 + f"""--gamma {gamma_str}"""
    
if vars(args)["pH"] != None:
    pH_str = ''
    pH_id_str = ''
    for i in pH:
        pH_str = pH_str + f'{i}' + " "
        pH_id_str = pH_id_str + f"|{i}"
    bash2 = bash2 + f"""--pH {pH_str}"""

#Writing the name as it will show in the queue
if pH != None:
    pH_id = pH_id_str
else:
    pH_id = "|0.3"

Queue_id = f"MC{n}_{rootdir[0:2]}{rootdir[6]}_pH{pH_id}"

if name != None:
    name = name
else:
    name = args.rootdir[:args.rootdir.find('_')]

#; Method 1

#subprocess.run(["qsub", "bash_script_method1.sh", "-v", "rootdir=$rootdir,$strain"])
#subprocess.run(["bash", "bash_script_method1.sh", rootdir])

#Bash file for method 1
    #!/bin/bash
    #rootdir=$1
    #echo "$rootdir"

#; Method 2

bash = f"""#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l pmem=4gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N {Queue_id}
#PBS -A ixd4_g_g_bc_default

cd $PBS_O_WORKDIR

python RuO2_110_MC_{n}.py --rootdir {rootdir} --name {name} """


with open(sh,"w") as bashfile:
    bashfile.write(bash)
    bashfile.write(bash2)

subprocess.run(["qsub", sh])
