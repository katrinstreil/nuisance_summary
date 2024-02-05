#!/bin/bash -l

for mass in {0..20}
do
    sbatch.tinyfat 7a.slurm
done


# bash analysis-loop.sh
