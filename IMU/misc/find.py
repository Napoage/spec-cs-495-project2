import smbus
"""
Script Description:
This Python script scans the I2C (Inter-Integrated Circuit) bus to detect connected devices. It utilizes the smbus library for I2C communication.

Components:
1. scan_i2c_bus() Function:
   - This function scans the I2C bus for connected devices.
   - It iterates through all possible device addresses (0 to 127) and attempts to read a byte from each address.
   - If reading from an address raises an IOError, it means no device is present at that address, and the address is skipped.
   - If reading is successful without raising an error, it indicates the presence of a device, and the device address (in hexadecimal format) is added to the list of detected devices.
   - Finally, the function returns the list of detected devices.

2. Main Block:
   - In the main block (if __name__ == "__main__":), the script first prints a message indicating that it is scanning the I2C bus.
   - It then calls the scan_i2c_bus() function to retrieve the list of detected devices.
   - If any devices are found, it prints each device's address in hexadecimal format.
   - If no devices are found, it prints a message indicating that no devices were detected.

Purpose:
This script is useful for identifying and listing all devices connected to the I2C bus of the system. It can be helpful for debugging, troubleshooting, or verifying the connectivity of I2C devices.
"""


def scan_i2c_bus():
    """
    Function to scan the I2C bus for connected devices.
    Returns:
        List: List of detected devices (addresses in hexadecimal format).
    """
    bus = smbus.SMBus(1)  # Change this to the appropriate bus number
    devices = []
    for address in range(0, 128):
        try:
            bus.read_byte(address)
            devices.append(hex(address))
        except IOError:
            pass
    return devices


if __name__ == "__main__":
    print("Scanning I2C bus...")
    devices = scan_i2c_bus()
    if devices:
        print("Devices found:")
        for device in devices:
            print(device)
    else:
        print("No devices found.")
