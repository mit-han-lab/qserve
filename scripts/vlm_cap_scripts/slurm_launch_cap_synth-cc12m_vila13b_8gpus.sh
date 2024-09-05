#!/bin/bash

#SBATCH -N 1                                                                        #number of nodes
#SBATCH -J qserve-vlm_captioner                                                     #job name
#SBATCH --array=0-272%20                                                          
#SBATCH --dependency singleton                                                     
#SBATCH --gpus-per-node 8
#SBATCH --exclusive                                                                # important for efficiency


set -e
set -x

idx=$SLURM_ARRAY_TASK_ID
total=$SLURM_ARRAY_TASK_COUNT
job_id=$idx

jname=$idx-of-$total
logdir=/home/haotiant/workspace/projects/qserve-vila-captioner/logs
mkdir -p $logdir

cmd=/home/haotiant/workspace/projects/qserve-vila-captioner/scripts/vlm_cap_scripts/run_cap_synth-cc12m_vila13b_8gpus.sh
srun -e $logdir/$jname.log -o $logdir/$jname.log \
    bash -c "bash ${cmd} $job_id"
