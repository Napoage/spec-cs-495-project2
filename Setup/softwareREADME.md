# Software Explanation Setup Guide

## Running the Setup
To set up the system, run the following command from the main repository:

```bash
./Setup/set_up_system
```

## Overview
This script automates the installation and configuration of necessary components for the system. It installs required packages, configures network interfaces, updates systemd service files, and sets up a captive portal.

## Steps Explained

### 1. Install Required Packages
The script installs essential packages such as:
- `nginx`, `hostapd`, and `dnsmasq` for network and web services.
- Python and necessary libraries (`flask`, `matplotlib`, `numpy`, `opencv`, `pandas`, etc.).
- GStreamer for media streaming.
- `wget`, `ffmpeg`, and `v4l2loopback` for additional system utilities.

### 2. Configure Network Interface
The network interface is configured with a static IP for `wlan0` to establish a Wi-Fi hotspot. IP forwarding is also enabled.

### 3. Modify Systemd Service Files
The script updates systemd service files to ensure correct paths for execution:
- `start_captive_portal.service`
- `gstreamer.service`
- `start_PIV_script.service`
- `start_app.service`
- `disk_space_manager.service`

### 4. Set Up Wi-Fi Access Point
The script prompts the user to enter a Wi-Fi SSID and password, updating the `hostapd.conf` file accordingly.

### 5. Configure Services
- Nginx configurations are set up to serve the captive portal.
- `dnsmasq` and `hostapd` are configured and restarted.

### 6. Enable and Start Services
The script enables and starts required services to ensure persistence across reboots:
- `hostapd`
- `dnsmasq`
- `start_app.service`
- `start_PIV_script.service`
- `start_captive_portal.service`
- `loop_back.service`
- `gstreamer.service`
- `disk_space_manager.service`

### 7. Additional Configurations
- IMU-related packages are installed.
- Necessary scripts are made executable.
- Web application credentials are set via `credentials.json`.
- Default configuration files are copied.

### 8. Final Steps
- Systemd is reloaded to apply changes.
- Log rotation settings are updated.

## Completion
Once the script finishes running, the system is fully set up, and the captive portal is operational. If any issues arise, check the status of services using:

```bash
sudo systemctl status <service-name>
```

For debugging, restart failed services manually with:

```bash
sudo systemctl restart <service-name>
```

**System setup is now complete!**

