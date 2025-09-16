#!/bin/bash
#
# Script Name: change_wifi_app_login.sh
# Description: Changes logins
#
# Usage:
#   Called by ./System/change_wifi_app_login.sh
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------

# Find base directory from save.json
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
SAVE_JSON="${BASE_DIR}/save.json"

# Check if save.json exists
if [[ ! -f "$SAVE_JSON" ]]; then
    echo "Error: save.json not found at $SAVE_JSON"
    exit 1
fi

# JSON file and hostapd config file paths
credentials_path="${BASE_DIR}/app/credentials.json"
hostapd_config="/etc/hostapd/hostapd.conf"

# Check if the script is run as root
if [[ $EUID -ne 0 ]]; then
    echo "This script must be run as root."
    exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "jq is not installed. Please install jq and try again."
    exit 1
fi

# Check if credentials.json exists
if [[ ! -f "$credentials_path" ]]; then
    echo "Error: credentials.json not found at $credentials_path"
    exit 1
fi

# Update JSON file
echo "Current credentials file content:"
cat "$credentials_path"

read -p "Enter new username: " new_username
read -p "Enter new password: " new_password

# Update the JSON file using jq
jq --arg USERNAME "$new_username" --arg PASSWORD "$new_password" \
   '.USERNAME = $USERNAME | .PASSWORD = $PASSWORD' \
   "$credentials_path" > temp.json && mv temp.json "$credentials_path"

echo "Updated credentials file content:"
cat "$credentials_path"

# Ask if the user wants to update WiFi settings
read -p "Do you want to change the WiFi name and password? (yes/no): " update_wifi

if [[ $update_wifi == "yes" ]]; then
    read -p "Enter new WiFi name (SSID): " wifi_name
    read -p "Enter new WiFi password: " wifi_password

    # Update the hostapd.conf file
    sed -i "s/^ssid=.*/ssid=${wifi_name}/" "$hostapd_config"
    sed -i "s/^wpa_passphrase=.*/wpa_passphrase=${wifi_password}/" "$hostapd_config"

    echo "Updated WiFi settings in $hostapd_config:"
    grep -E "^(ssid|wpa_passphrase)=" "$hostapd_config"

    # Reload system daemon and restart app
    echo "Reloading system daemon and restarting app service..."
    sudo systemctl daemon-reload
    sudo systemctl restart start_app
    sudo systemctl restart hostapd

    echo "WiFi and app service updates completed."
else
    echo "WiFi settings were not changed."
fi