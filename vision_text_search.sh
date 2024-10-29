#!/bin/bash
#SBATCH --account=soc-gpu-np
#SBATCH --partition=soc-gpu-np
#SBATCH --job-name=mars_vision_text_search
#SBATCH --time=4:00:00
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_vision_text_search.out
#SBATCH --error=/scratch/general/vast/u1475870/clip_project/logs/%j/%j_vision_text_search.err
#SBATCH --mail-user=pravallikaslurm@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --requeue
#SBATCH --open-mode=append

# Create directories
SCRATCH_DIR="/scratch/general/vast/u1475870/clip_project/"
LOG_DIR="$SCRATCH_DIR/logs/$SLURM_JOB_ID"
SEARCH_DIR="$SCRATCH_DIR/vision_text_search_results"
mkdir -p $LOG_DIR
mkdir -p $SEARCH_DIR
mkdir -p "$SCRATCH_DIR/outputs"

echo "Job started/resumed on $(date)"
echo "Running on node: $SLURMD_NODENAME"

# Set up scratch directory
cd $SCRATCH_DIR

# Copy necessary files
echo "Copying necessary files..."
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/vision_text_search.py .
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/clip.py .
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/vision_text_best_model.pt outputs/
cp /uufs/chpc.utah.edu/common/home/$USER/clip_project/requirements.txt .

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

# Check GPU
nvidia-smi > $LOG_DIR/gpu_info.txt 2>&1
echo "GPU Info saved to: $LOG_DIR/gpu_info.txt"

# Run search
echo "Starting search..."
python vision_text_search.py 2>&1 | tee $LOG_DIR/search_output.txt

# Check if search completed successfully
if [ $? -eq 0 ]; then
    echo "Search completed successfully"
    
    # Copy results back
    mkdir -p /uufs/chpc.utah.edu/common/home/$USER/clip_project/vision_text_search_results/
    cp -r $SEARCH_DIR/* /uufs/chpc.utah.edu/common/home/$USER/clip_project/vision_text_search_results/
    cp $LOG_DIR/search_output.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
    cp $LOG_DIR/gpu_info.txt /uufs/chpc.utah.edu/common/home/$USER/clip_project/outputs/
else
    echo "Search was interrupted, job will be requeued if possible"
fi

# Deactivate virtual environment
deactivate

echo "Job ended on $(date)" 