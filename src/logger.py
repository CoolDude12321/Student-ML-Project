# =============================================================
# src/logger.py
# ------------------------------------------------------------  
# This module sets up logging for the application.
# It configures the logging format, log file location, and log level.
# =============================================================


# =============================================
# Importing the Necessary Libraries
# --------------------------------------------
# Logging Library
## Used to log messages for tracking events that happen during execution
# --------------------------------------------
# OS Library
## Used to interact with the operating system
# --------------------------------------------
# Datetime Library
## Used to handle date and time operations
# =============================================

import logging
import os
from datetime import datetime

# =============================================
# Logger Configuration
# --------------------------------------------
# This section sets up the logging configuration, including the log file path,
# log format, and log level.
# =============================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"log_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    force=True
)

logging.info("Logger initialized successfully")