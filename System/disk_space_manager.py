import os
import json
import shutil
import time
import logging

# Script Name: disk_space_manager.py
# Description: Starts diskspace manager
#
# Usage:
#   Called by systemd rule
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# ----------------------------------------------------

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
SAVE_DATA_FOLDER = os.path.join(BASE_DIR, 'save_data')
SAVE_JSON_PATH = os.path.join(BASE_DIR, "save.json")
MIN_FREE_SPACE_GB = 3.0
RETRY_INTERVAL = 60  # seconds to wait between retries

def check_save_folder_exists():
    """Check if save_data folder exists and wait until it does."""
    while not os.path.exists(SAVE_DATA_FOLDER):
        logging.warning(f"Save folder {SAVE_DATA_FOLDER} does not exist. Waiting...")
        time.sleep(RETRY_INTERVAL)
    return True

def check_disk_space(min_free_space_gb):
    """Check if disk space is below the threshold."""
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)
    logging.debug(f"Current free disk space: {free_gb:.2f} GB")
    return free_gb <= min_free_space_gb

def get_current_config():
    """Read the current config folder from save.json."""
    if not os.path.exists(SAVE_JSON_PATH):
        logging.warning(f"{SAVE_JSON_PATH} does not exist.")
        return ""
    try:
        with open(SAVE_JSON_PATH, "r") as f:
            data = json.load(f)
        return data.get("config_folder", "")
    except json.JSONDecodeError:
        logging.error(f"Error reading {SAVE_JSON_PATH}")
        return ""

def get_oldest_directory(folder_path, exclude=None):
    """Get the oldest directory in the folder, excluding a specific directory."""
    if not os.path.exists(folder_path):
        logging.warning(f"Directory {folder_path} does not exist.")
        return None

    try:
        dirs = [os.path.join(folder_path, d) for d in os.listdir(folder_path)]
        dirs = [d for d in dirs if os.path.isdir(d) and d != exclude]
        if not dirs:
            return None
        return min(dirs, key=os.path.getmtime)
    except (OSError, FileNotFoundError) as e:
        logging.error(f"Error accessing directory {folder_path}: {str(e)}")
        return None

def delete_oldest_runs(config_folder):
    """Delete the oldest 'run' directories in the config folder."""
    if not os.path.exists(config_folder):
        logging.warning(f"No 'runs' folder found in {config_folder}.")
        return

    while check_disk_space(MIN_FREE_SPACE_GB):
        oldest_run = get_oldest_directory(config_folder)
        if oldest_run:
            logging.info(f"Deleting oldest run: {oldest_run}")
            try:
                shutil.rmtree(oldest_run)
            except OSError as e:
                logging.error(f"Error deleting {oldest_run}: {str(e)}")
        else:
            logging.info("No run directories left to delete.")
            break

def manage_configs():
    """Manage the save_data folder and ensure disk space."""
    # First ensure the save folder exists
    if not check_save_folder_exists():
        return

    current_config = get_current_config()

    # Step 1: Delete the oldest config folder (if more than one exists)
    oldest_config = get_oldest_directory(SAVE_DATA_FOLDER, exclude=current_config)
    if oldest_config and check_disk_space(MIN_FREE_SPACE_GB):
        logging.info(f"Deleting oldest config folder: {oldest_config}")
        try:
            shutil.rmtree(oldest_config)
        except OSError as e:
            logging.error(f"Error deleting config folder {oldest_config}: {str(e)}")

    # Step 2: Manage the current config folder's runs
    if check_disk_space(MIN_FREE_SPACE_GB):
        delete_oldest_runs(current_config)

def main():
    logging.info(f"Starting disk space monitor for {SAVE_DATA_FOLDER}")
    logging.info("IT IS WORKING")

    while True:
        try:
            manage_configs()
        except Exception as e:
            logging.error(f"Unexpected error in manage_configs: {str(e)}")
        finally:
            time.sleep(RETRY_INTERVAL)

if __name__ == "__main__":
    logging.info("IT IS WORKING")
    main()