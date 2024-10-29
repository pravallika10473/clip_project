#!/bin/bash
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --job-name=mars_clip_training
#SBATCH --time=12:00:00
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_training.out
#SBATCH --error=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_training.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/clip_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
mkdir -p $LOG_DIR
mkdir -p "$SCRATCH_DIR/similarity_matrices"

echo "Job started/resumed on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Copy necessary files
if [ ! -f clip.py ]; then
    cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/clip.py .
    cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/requirements.txt .
fi

# Load required modules
module purge
module load cuda/11.1.1
module load cudnn

# Activate virtual environment
source /uufs/chpc.utah.edu/common/home/$USER/clip_project/venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt

# Print environment info
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Pip packages:"
pip list

# Check GPU and save info
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run training with real-time output
echo "Starting training..."
python ~/clip_project/clip.py 2>&1 | tee $LOG_DIR/clip_training_output.txt

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully"
    
    # Copy results back
    mkdir -p /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp clip_best_model.pt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp -r similarity_matrices /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp $LOG_DIR/clip_training_output.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
else
    echo "Training was interrupted, job will be requeued if possible"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)"