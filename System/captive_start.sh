#!/bin/bash
#
# Script Name: captive_start.sh
# Description: Sets up captive portal
#
# Usage:
#   Called by systemd rule
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------

sudo rfkill unblock wifi
sudo systemctl restart dnsmasq
sudo systemctl restart hostapd
sleep 10
sudo systemctl start nginx

