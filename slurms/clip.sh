#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=mars_clip_training
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_training.out
#SBATCH --error=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_training.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL

# Create a unique directory for this job in scratch space
SCRATCH_DIR="/scratch/general/vast/u1475870/clip_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Log directory: $LOG_DIR"

# Set up scratch directory for the job 
cd $SCRATCH_DIR

# Copy the script from home to scratch
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/clip.py .

# Load required modules first
module load cuda
module load cudnn

# Activate the virtual environment (assuming it's created with venv)
source /uufs/chpc.utah.edu/common/home/$USER/clip_project/venv/bin/activate

# Print Python and environment info for debugging
which python
python --version
pip list

# Check GPU availability
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
if [ $? -eq 0 ]; then
    echo "GPU is available" >> $LOG_DIR/gpu_info.txt
else
    echo "GPU is not available" >> $LOG_DIR/gpu_info.txt
fi

# Run the CLIP training script
python clip.py > $LOG_DIR/clip_training_output.txt 2>&1

# Deactivate the virtual environment
deactivate

# Copy results back to the home directory
mkdir -p /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
cp $LOG_DIR/clip_training_output.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
cp best_model.pt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/

echo "Job finished on $(date)"
