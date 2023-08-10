#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l pmem=4gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N MC3_StJ_pH|13
#PBS -A ixd4_g_g_bc_default

cd $PBS_O_WORKDIR

python RuO2_110_MC_3.py --rootdir StrainJin_Files/ --name StrainJin --pH 13 