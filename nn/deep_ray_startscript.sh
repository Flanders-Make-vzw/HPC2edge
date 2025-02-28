#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --job-name=num_samples_64
#SBATCH --output=num_samples_64.out
#SBATCH --error=num_samples_64.err
#SBATCH --partition=dp-esb
#SBATCH --nodes=64
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclusive

ml --force purge
ml Stages/2023 CUDA/11.7 Python PostgreSQL/14.4

. .env/bin/activate

COMMAND="ray_tune.py --scheduler NEVERGRAD --num-samples 64 --par-workers 4 --max-iterations 1 --data-dir '/p/project1/jureap57/AM_data/RAISE_LPBF_C027.hdf5' "

echo $COMMAND

export DB_NAME=""
export DB_USER=""
export DB_PASS=""
export DB_HOST=""
export DB_PORT=""

#sleep 1
# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export NCCL_SOCKET_IFNAME="ib0"

num_gpus=1

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1


####### this part is taken from the ray example slurm script #####
set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

port=27747

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" --redis-password='5241590000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    #sleep 5
done

echo "Ready"

python -u $COMMAND
