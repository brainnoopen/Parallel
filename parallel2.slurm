#! /bin/sh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=parallel2.txt
#SBATCH --time=00:03:00
#SBATCH --partition=cloud

module load OpenMPI/3.1.0-GCC-8.1.0
mpiexec ./parallel2 -2.0 1.0 -1.0 1.0 1000 10000
