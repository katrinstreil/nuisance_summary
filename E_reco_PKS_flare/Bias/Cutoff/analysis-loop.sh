#!/bin/bash -l

for mass in {0..28}
do
    sbatch.tinyfat /home/wecapstor1/caph/mppi045h/DM/III-analysis/analysis.sh  $mass 
done


#bash analysis-loop.sh