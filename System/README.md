# System Control Scripts

This directory contains the main control scripts for running and managing the system. These scripts perform various functions such as starting the captive portal, managing disk space, configuring WiFi and app login, and setting up the camera streaming. The only script intended to be manually called by the user is change_wifi_app_login, while the other scripts are automatically triggered by the system as needed.

## Available Scripts
1. **captive_start.sh**
    - Purpose: Starts the captive portal, which is responsible for managing network access and handling device connections.
    - Usage: This script is automatically called by the system during the startup process and does not require user intervention.
2. **change_wifi_app_login.py**
    - Purpose: Allows the user to change the WiFi network and app login credentials. This is the primary script that users will interact with.
    - Usage: Run this script whenever you need to change the WiFi network or update the app login credentials. The user will provide the new information when prompted.
```bash
./System/change_wifi_app_login.py
```
3. **disk_space_manager.py**
    - Purpose: Monitors the available disk space on the system and ensures that the disk does not fill up. If disk usage becomes too high, it will take necessary actions to free up space.
    - Usage: This script is automatically called by the system and runs in the background to ensure smooth operation.
4. **start_loopback_streams**
    - Purpose: Starts the camera streaming process, setting up the main camera and two virtual cameras. This enables the system to stream data to various endpoints.
    - Usage: This script is automatically executed by the system and does not require user interaction.

## System Overview
The system is designed to be hands-off for the user, with the exception of the `change_wifi_app_login` script. Once you run this script to configure WiFi and app login credentials, the other scripts will run automatically in the background:

1. `captive_start` initializes the captive portal during startup.
2. `disk_space_manager.py` monitors and manages disk usage to avoid overflow.
3. `start_loopback_streams` begins the camera streaming process.

The system is designed to operate smoothly with minimal user intervention, and all critical services will be handled automatically once configured.