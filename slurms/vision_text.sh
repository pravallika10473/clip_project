#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=mars_vision_text_training
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_vision_text_training.out
#SBATCH --error=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_vision_text_training.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue  # Allow the job to be requeued
#SBATCH --open-mode=append  # Append to output files if requeued

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/clip_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR

echo "Job started/resumed on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Copy necessary files (only if they don't exist)
if [ ! -f vision_text.py ]; then
    cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/vision_text.py .
fi

# Load required modules
module purge  # Clear any existing modules
module load cuda/11.1.1
module load cudnn

# Activate virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/clip_project/venv/bin/activate

# Print environment info
which python
python --version
pip list
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1

# Run training
python vision_text.py > >(tee -a $LOG_DIR/vision_text_training_output.txt) 2>&1

# Check if the job completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Copy results back only if training was successful
    mkdir -p /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp best_model.pt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/vision_text_best_model.pt
    cp $LOG_DIR/vision_text_training_output.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
else
    echo "Training was interrupted, job will be requeued if possible"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"