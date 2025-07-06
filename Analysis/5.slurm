#!/bin/bash -l
#

# start 2 MPI processes
#SBATCH --ntasks=2
# allocate nodes for 6 hours
#SBATCH --time=24:00:00
# job name 
#SBATCH --job-name=logfiles/5<
# do not export environment variables
#SBATCH --export=NONE

# do not export environment variables
unset SLURM_EXPORT_ENV



module use -a  /apps/modules/data/tools
module load python/3.8-anaconda
# source activate /home/wecapstor1/caph/mppi045h/anaconda3/envs/gammapy-dev
source /home/wecapstor1/caph/mppi045h/anaconda3/etc/profile.d/conda.sh
conda activate  /home/wecapstor1/caph/mppi045h/anaconda3/envs/gammapy-dev

srun --mpi=pmi2  python 5_asimov_surface.py


# log into tiny
#sbatch.tinyfat elephant.slurm  
# scontrol.tinyfat show job <jpbID>
# salloc.tinyfat --time=24:00:00 



