#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l pmem=4gb
#PBS -l walltime=48:00:00
#PBS -j oe
#PBS -N MC2_St2_pH|13
#PBS -A ixd4_g_g_bc_default

cd $PBS_O_WORKDIR

python RuO2_110_MC_2.py --rootdir Strain2_Files/ --name Strain2 --pH 13 