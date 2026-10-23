DELETE BRANCH ON UPSTREAM
git branch –D branch-name (delete from local)
git push origin :branch-name (delete from remote)


USUE FULL COMMANDS

sbatch <bash_script.sh>
Run a batch job (background job)

srun -p <partition> -q <qos> <command>
Run a command in  agiven partition / QoS (e.g., python3 train.py)

srun -p <partition> -q <qos> --pty bash
Request an interactive shell in a given parition / QoS (for debugging only)

scancel <job_id>
Cancel a job in the queue

scontrol requeue <job_id>
Requeue (i.e., cancel and rerun) a job

squeue -l
Show the Slurm queue, including all submitted and pending jobs

squeue --me
List the current jobs of the user (yourself)

sinfo
Show information about Slurm's nodes and partitions (resource configuration)

sacctmgr show association
Show user account properties

sacctmgr show reservations
Show resource reservations

sacctmgr show qos
Show available QoS levels