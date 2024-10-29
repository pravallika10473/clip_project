#!/bin/bash
#SBATCH --account=yqu-gpu-np
#SBATCH --partition=yqu-gpu-np
#SBATCH --job-name=mars_clip_search
#SBATCH --time=4:00:00
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_search.out
#SBATCH --error=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_clip_search.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL

# Create a unique directory for this job in scratch space
SCRATCH_DIR="/scratch/general/vast/u1475870/clip_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
SEARCH_DIR="$SCRATCH_DIR/search_results"
mkdir -p $LOG_DIR
mkdir -p $SEARCH_DIR

echo "Job started on $(date)"
echo "Running on node: $SLURMD_NODENAME"
echo "Log directory: $LOG_DIR"

# Set up scratch directory for the job 
cd $SCRATCH_DIR

# Copy the necessary files from home to scratch
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/search_images.py .
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/clip.py .
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/best_model.pt outputs/

# Load required modules first
module load cuda
module load cudnn

# Activate the virtual environment
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

# Run the search script
python search_images.py > $LOG_DIR/clip_search_output.txt 2>&1

# Deactivate the virtual environment
deactivate

# Copy results back to the home directory
mkdir -p /uufs/chpc.utah.edu/common/home/$USER/clip_project/search_results/
cp -r $SEARCH_DIR/* /uufs/chpc.utah.edu/common/home/$USER/clip_project/search_results/
cp $LOG_DIR/clip_search_output.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/

echo "Job finished on $(date)"

