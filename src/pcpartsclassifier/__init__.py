import os
import sys
import logging

# logging string format
logging_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

# logging directory
log_dir = "logs"
log_filepath = os.path.join(log_dir, 'running_logs.log')
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logging.basicConfig(
  level=logging.INFO,
  format=logging_str,
  handlers=[
    logging.FileHandler(log_filepath),
    logging.StreamHandler(sys.stdout)
  ]
)

# logger function
logger = logging.getLogger('pcpartsclassifier')