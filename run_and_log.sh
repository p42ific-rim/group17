# chmod +x run_and_log.sh <- in case it says permission denied make sure it is known as a runable file

#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Generate a timestamp for the filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

echo "Starting Experiment... Output will be saved to $LOG_FILE"

# Run the experiment and pipe both stdout and stderr to the log file AND the screen
# We use PYTORCH_ENABLE_MPS_FALLBACK=1 just in case you are on a Mac
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 run_experiment.py 2>&1 | tee $LOG_FILE

echo "Experiment Complete. Log saved to $LOG_FILE"
