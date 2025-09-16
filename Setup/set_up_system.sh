#!/bin/bash
# set -xe

source common_functions.sh

# Step 1: Install Required Packages
echo "Installing required packages..."
sudo apt install -y nginx
sudo apt install -y hostapd
sudo apt-get update
sudo apt install -y dnsmasq
sudo apt update
# Step 2: Install Python and GStreamer Related Packages
echo "Installing Python and GStreamer packages..."
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y python3-flask
sudo apt install -y python3-matplotlib
sudo apt install -y python3-numpy
sudo apt install -y python3-opencv
sudo apt install -y python3-pandas
sudo apt install -y python3-watchdog
sudo apt install -y python3-skimage
sudo apt-get -y install jq
sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav

# Step 3: Install wget, ffmpeg, v4l2loopback
sudo apt install -y wget
sudo apt install -y v4l2loopback-dkms
sudo apt install -y ffmpeg

sudo apt install ifupdown
# Step 4: Configure Network Interface
echo "Configuring network interface..."
INTERFACE_FILE="/etc/network/interfaces"
sudo bash -c "cat <<EOL > $INTERFACE_FILE
auto wlan0
iface wlan0 inet static
    address 192.168.0.1
    netmask 255.255.255.0
    network 192.168.0.0
    broadcast 192.168.0.255
EOL"

sudo sed -i '/^#net.ipv4.ip_forward=1/s/^#//' /etc/sysctl.conf
sudo rfkill unblock wifi

# Apply the network changes
echo "Applying network changes..."
sudo ip link set wlan0 down || true
sudo ip link set wlan0 up
sudo systemctl restart networking
# sudo ip addr add 192.168.0.1/24 dev wlan0

#Step 5 Set paths based on current system
# Get the current directory
CURRENT_DIR=$(pwd)

# Get the parent directory
# PARENT_DIR=$(dirname "$CURRENT_DIR")

# Define the path to your systemd service file
captive_portal_rule="Setup/rules/start_captive_portal.service"
gstreamer_service="Setup/rules/gstreamer.service"
start_piv_Service="Setup/rules/start_PIV_script.service"
start_app_service="Setup/rules/start_app.service"
disk_space_service="Setup/rules/disk_space_manager.service"

test_script="run_PIV.sh"
run_all_piv_script="test_PIV.sh"
# Update the ExecStart line in the systemd service file
# Use sed to replace the line that starts with ExecStart
sed -i "s|^ExecStart=.*|ExecStart=${CURRENT_DIR}/System/captive_start.sh|" "$captive_portal_rule"
sed -i "s|^ExecStart=.*|ExecStart=/bin/bash ${CURRENT_DIR}/System/start_loopback_streams.sh|" "$gstreamer_service"
sed -i "s|^ExecStart=.*|ExecStart=/bin/bash ${CURRENT_DIR}/run_PIV.sh|" "$start_piv_Service"
sed -i "s|^ExecStart=.*|ExecStart=/usr/bin/python3 ${CURRENT_DIR}/app/app.py|" "$start_app_service"
sed -i "s|^WorkingDirectory=.*|WorkingDirectory=${CURRENT_DIR}|" "$start_piv_Service"
sed -i "s|^WorkingDirectory=.*|WorkingDirectory=${CURRENT_DIR}|" "$start_app_service"

sed -i "s|^ExecStart=.*|ExecStart=/usr/bin/python3 ${CURRENT_DIR}/System/disk_space_manager.py|" "$disk_space_service"

sed -i "s|^PARENT_DIR=.*|PARENT_DIR='$CURRENT_DIR'|" "$test_script"
sed -i "s|^PARENT_DIR=.*|PARENT_DIR='$CURRENT_DIR'|" "$run_all_piv_script"

hostapd_config='Setup/configs/hostapd.conf'
# Prompt the user for the Wi-Fi name (SSID) and password
read -p "Enter the Wi-Fi name (SSID): " WIFI_NAME

# Validate SSID
while [[ -z "$WIFI_NAME" ]]; do
    echo "SSID cannot be empty."
    read -p "Enter the Wi-Fi name (SSID): " WIFI_NAME
done

# Keep prompting for password until valid
while true; do
    read -sp "Enter the Wi-Fi password (minimum 8 characters): " WIFI_PASSWORD
    echo  # Add a newline after password input

    if [[ ${#WIFI_PASSWORD} -ge 8 ]]; then
        break  # Exit loop if password is valid
    else
        echo "Password must be at least 8 characters long. Please try again."
    fi
done

# Update the hostapd.conf file
sed -i "s/^ssid=.*/ssid=${WIFI_NAME}/" "$hostapd_config"
sed -i "s/^wpa_passphrase=.*/wpa_passphrase=${WIFI_PASSWORD}/" "$hostapd_config"

# Confirm changes
echo "The hostapd.conf file has been updated successfully:"
grep -E "^(ssid|wpa_passphrase)=" "$hostapd_config"


# Step 5: Configure Services
echo "Copying service configuration files..."
sudo cp -R Setup/configs/nginx/sites-available/SPEC.com /etc/nginx/sites-available/SPEC.com
sudo ln -s /etc/nginx/sites-available/SPEC.com /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-available/default 
sudo rm /etc/nginx/sites-enabled/default

sudo cp Setup/configs/hostapd.conf /etc/hostapd/hostapd.conf
sudo cp Setup/configs/dnsmasq.conf /etc/dnsmasq.conf

# # Step 6: Restart and Debug Services
echo "Restarting services..."

# # # Restart dnsmasq
sudo systemctl restart dnsmasq # did not orginally install?????
if [[ $? -ne 0 ]]; then
    echo "Failed to restart dnsmasq, debugging..."
    sudo systemctl status dnsmasq
    sudo systemctl start dnsmasq
fi


sudo systemctl unmask hostapd.service
# Restart hostapd
sudo systemctl restart hostapd
if [[ $? -ne 0 ]]; then
    echo "Failed to restart hostapd, debugging..."
    sudo systemctl status hostapd
    sudo systemctl start hostapd
fi

# # Restart nginx
sudo systemctl restart nginx
if [[ $? -ne 0 ]]; then
    echo "Failed to restart nginx, debugging..."
    sudo systemctl status nginx
    sudo systemctl start nginx
fi

# # Step 7: Enable Services on Boot
echo "Enabling services on boot..."
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq
sudo systemctl stop nginx
sudo systemctl disable nginx


#Step 8: Start IMU
sudo raspi-config nonint do_i2c 0
sudo pip3 install --break-system-packages imusensor
sudo pip3 install --break-system-packages easydict
sudo pip3 install --break-system-packages flask_login
#Step 9: Make necessary scripts executable
sudo chmod +x run_PIV.sh
sudo chmod +x test_PIV.sh
sudo chmod +x System/start_loopback_streams.sh
sudo chmod +x System/captive_start.sh


JSON_FILE="app/credentials.json"
# Ask for the new username and password
echo "Enter the new USERNAME for web application :"
read USERNAME

echo "Enter the new PASSWORD for web application:"
read -s PASSWORD  # -s flag hides the input for password

# Update the JSON file using jq (make sure jq is installed)
jq --arg USERNAME "$USERNAME" --arg PASSWORD "$PASSWORD" \
   '.USERNAME = $USERNAME | .PASSWORD = $PASSWORD' \
   "$JSON_FILE" > temp.json && mv temp.json "$JSON_FILE"

echo "Username and password have been updated in $JSON_FILE"

# sudo cp $JSON_FILE /root/

# # Update the JSON file using jq (make sure jq is installed)
# jq --arg USERNAME "user" --arg PASSWORD "pw" \
#    '.USERNAME = $USERNAME | .PASSWORD = $PASSWORD' \
#    "$JSON_FILE" > temp.json && mv temp.json "$JSON_FILE"




#Copy defaults
cp Setup/defaults/default_config.json config.json
cp Setup/defaults/default_save.json save.json
cp Setup/defaults/default_monitor_file.txt monitor_file.txt
cp Setup/defaults/default_script.log script.log
cp Setup/defaults/default_saving_to_usb.log saving_to_usb.log

# Step 10: Set Up Startup Services
echo "Setting up startup services..."
sudo cp Setup/rules/start_app.service /etc/systemd/system/
sudo cp Setup/rules/start_PIV_script.service /etc/systemd/system/
sudo cp Setup/rules/start_captive_portal.service /etc/systemd/system/
sudo cp Setup/rules/loop_back.service /etc/systemd/system/
sudo cp Setup/rules/gstreamer.service /etc/systemd/system/
sudo cp Setup/rules/disk_space_manager.service /etc/systemd/system/

# Start the services
echo "Starting services..."
sudo systemctl start start_app.service
sudo systemctl start start_PIV_script.service
sudo systemctl start start_captive_portal.service
sudo systemctl start loop_back.service
sudo systemctl start gstreamer.service
sudo systemctl start disk_space_manager.service

# enable the services
echo "Starting services..."
sudo systemctl enable start_app.service
sudo systemctl enable start_PIV_script.service
sudo systemctl enable start_captive_portal.service
sudo systemctl enable loop_back.service
sudo systemctl enable gstreamer.service
sudo systemctl enable disk_space_manager.service


sudo systemctl daemon-reload
sudo cp Setup/configs/spec /etc/logrotate.d

enable_rtc_charging
result=$?

case $result in
    0)
        echo "RTC clock battery already setup, no changes needed"
        ;;
    1)
        echo "Error occurred during RTC clock battery setup"
        ;;
    2)
        echo "Changes made to RTC clock battery setup, system needs to reboot"
        ;;
esac



echo "System set up complete!"
echo "Captive portal setup completed!"
