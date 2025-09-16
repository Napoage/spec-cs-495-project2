# Build Your Own SPEC

## Pre-Requisites
- Raspberry Pi 5 (8 GB ram tested)
- IMU and Camera as listed below
- Ability to solder IMU header pins if required
- Familiarity with Linux command line interface
- Basic understanding of running Python scripts
- Access to a local or online 3D printing service
- A passion for moving water


## Required/Suggested Hardware
Links are to places we have ordered hardware from previously, and are included simply as suggestions.
### [Please refer to the complete bill of materials (BOM) here for all parts.](/Setup/hardware/BOM_costed_01-22-2025.xlsx)
### Power for the System
- We have included a DC-DC power converter that accepts 12V/24V and outputs 5V to power the Pi. To power the system, you could use a car/marine battery, a USB battery pack, a solar power solution, or "plug in" with existing power solutions at the field site.
### A few of the parts
- If you want to [connect your Pi to a monitor during setup](https://www.tomshardware.com/how-to/set-up-raspberry-pi), you will need a monitor cable with microHDMI on one of the ends. You'll also need a keyboard and a mouse. It is also possible to perform a [headless install where you will remote into your Pi.](https://www.tomshardware.com/reviews/raspberry-pi-headless-setup-how-to,6028.html) 
- [Raspberry Pi 5 - 8 GB RAM](https://www.adafruit.com/product/5813)  (Pi 5 versions with less RAM and older versions have not been tested).
- [Official Raspberry Pi 27W USB C power supply](https://www.adafruit.com/product/5814). 
- [Raspberry Pi 5 fan](https://www.adafruit.com/product/5815).
- [microSD card](https://www.amazon.com/SanDisk-Extreme-microSDXC-Memory-Adapter/dp/B09X7C2GBC?ref_=ast_sto_dp&th=1) Choose the size that fits your field testing situation. We recommend 512GB for several months of storage, depending on your settings.
- Camera. [Here is the camera we tested](https://www.amazon.com/gp/product/B096VDL968/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&psc=1).  
- You will need female-female jumper wires and if you want to breadboard the project before installing in the 3D printed carrier, you'll need a few more parts. There are many sources for this. As an example of one that has far more than you need for this project, [check here](https://www.amazon.com/EL-CK-002-Electronic-Breadboard-Capacitor-Potentiometer/dp/B01ERP6WL4/ref=sxin_16_pa_sp_search_thematic_sspa?content-id=amzn1.sym.76d54fcc-2362-404d-ab9b-b0653e2b2239%3Aamzn1.sym.76d54fcc-2362-404d-ab9b-b0653e2b2239&crid=1KI6LO4IU0ISU&cv_ct_cx=breadboard%2Band%2Bjumper%2Bwire%2Bkit&dib=eyJ2IjoiMSJ9.PY5zqDWjxzh72O3tZAYULLkSNXMhfOqYD5HhSTMafx86uONQ4d7ZnqTSchC-slOWkKBFaJQONRApJu8di-BWBQ.RxnTzjKUuQMFK02MfCzxIKI6ZxDUkDKrSyotT8A1cfY&dib_tag=se&keywords=breadboard%2Band%2Bjumper%2Bwire%2Bkit&pd_rd_i=B01ERP6WL4&pd_rd_r=385481fa-dee1-4270-9b9e-1a6211d90fcd&pd_rd_w=Gt9FF&pd_rd_wg=1P0rx&pf_rd_p=76d54fcc-2362-404d-ab9b-b0653e2b2239&pf_rd_r=8R9RHQPT6CS3XV9X3F1M&qid=1731345326&sbo=RZvfv%2F%2FHxDF%2BO5021pAnSA%3D%3D&sprefix=breadboard%2Band%2Bjump%2Caps%2C175&sr=1-3-6024b2a3-78e4-4fed-8fed-e1613be3bcce-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9zZWFyY2hfdGhlbWF0aWM&th=1). 
- IMU (Inertial Measurement Unit) [MPU9250](https://www.amazon.com/HiLetgo-Gyroscope-Acceleration-Accelerator-Magnetometer/dp/B01I1J0Z7Y). Other IMUs may be used, but were not tested.
- Soldering iron and solder.
- [Camera Enclosure](https://www.amazon.com/OdiySurveil-Aluminum-Surveillance-Weatherproof-Adjustable/dp/B0811QX5YW/ref=asc_df_B0811QX5YW/) This is the enclosure that our 3D printed "sled" fits. Other enclosures not tested.
- 3D printed sled for mounting the Pi, camera and IMU. [Download STL files](/Setup/hardware/STREAM_CAMERA_SLED.zip). <br>
 <img src="/app/static/images/SPEC_sled_mount.jpg" alt="3D printed sled" width="400" /><br>



## Hardware Installation
In order to properly run the SPEC software, the IMU and camera need to be connected to the Pi.
##### 1. Wiring Diagram of Raspberry Pi and IMU

<img src="/app/static/images/spec_Pi_wiring.png" alt="Pi wiring" width="500" /><br>

Notes: The IMU in the wiring diagram does not have as many pins as the one we used for development.Also, the colors of the jumper used in the diagram do not have to be the ones you use. The diagram is for wire placement the colors chosen were used to help distinguish between the four 
wires.
[Here is the GPIO pinout for the Raspberry Pi 5.](https://www.hackatronic.com/raspberry-pi-5-pinout-specifications-pricing-a-complete-guide/)

##### 2. Raspberry Pi Fan
To keep the Pi at a safe operating temperature, we have added a fan and heatsink. To install this fan:
- Peel off the heatsink pad’s wrapper.
- Line up the heatsink-fan with the Pi – the cord of the heatsink-fan should be on the side of the Pi with the USB plug-ins
- Press down on the heatsink-fan corner connectors and push them through the predesigned holes
- Gently press down on the rest of the heat sink.
- Plug in heatsink/fan. Note: the plug on the Pi may have a cap on it; you simply need to take it off. 

<img src="/app/static/images/Pi_fan.png" alt="Pi fan install" width="500" /><br>
*Left: Pi before heatsink/fan, Middle: bottom of heatsink/fan, Right: Pi with heatsink/fan. The yellow squares represent the areas that need to be connected together. The red is where the cord is plugged in.* <br><br>

##### 3. Real Time Clock
The Raspberry Pi 5 includes a real time clock (RTC) module that will keep track of the time that is set in the operating system even if the unit is disconnected from power. The Pi will retrieve the correct time when it connects to the internet. You must plug a small coin battery into the Pi for the RTC to work. In the bill of materials, the official rechargeable battery with the appropriate 2-pin JST plug is listed. It is not recommended to use a non-rechargeable battery as the current consumption of the RTC module is high and will result in a short life for the battery. The figure below shows where to connect the battery, just to the right of the USB-C power connector.
<img src="/app/static/images/Pi_with_battery.png" alt="Pi battery install" width="500" /><br>

##### 4. 3D Printed Sled
In order to mount all of the pieces of the SPEC system together, we have a 3D-Printed sled that holds all the devices. Below are some images showing assembly. You might need to use a bit of force to screw into the 3D printed standoffs.

- Gather all the required parts as shown below.<br>
<img src="/app/static/images/3D_sled_assy_1.jpg" alt="3D sled assembly" width="500" /><br>
- Start by putting together the sled.<br>
<img src="/app/static/images/sled_parts.jpg" alt="Sled Parts" width="500" /><br>
  - Screw in camera holder.<br>
  <img src="/app/static/images/camera_holder.png" alt="Camera Holder" width="500" /><br>
  - Screw in Pi holder.<br>
  <img src="/app/static/images/battery_holder.png" alt="Pi Holder" width="500" /><br>
  - Place RTC battery in holder and add cover.<br>
<img src="/app/static/images/rtc_holder.png" alt="RTC Holder" width="500" /><br>
- Screw in the Pi and IMU (two screws each) and connect them using the jumper wires, also plug in RTC battery.<br>
<img src="/app/static/images/Pi_on_sled.jpg" alt="Pi screwed into sled" width="500" /><br>
- Screw in the camera, make sure it is right side up. For this camera white cord pluggin is on the bottom. <br>
<img src="/app/static/images/Pi_camera.png" alt="Pi with camera" width="500" /><br>
- The suggested camera enclosure comes with two predrilled holes for the ethernet and power passthrough connectors. To also include a USB pass through power connector, we need to drill another hole in the center. You will also need to connect the DC-DC converter with the power passthrough and USB-C plug-in for the Pi.<br>
<img src="/app/static/images/enclosure.png" alt="Pi with camera" width="500" /><br>
<img src="/app/static/images/DC_DC.jpg" alt="Pi with camera" width="500" /><br>
- Once all the devices are secured to the sled and the passthrough connectors are in place the two pieces are ready to be put together. To make this an easily adjustable install we designed 3D-printed clip-ins for the sled that allow you to adjust without any extra tools.<br>
<img src="/app/static/images/clip_ins.png" alt="Pi with camera" width="500" /><br>

<img src="/app/static/images/SPEC_finished.jpg" alt="Pi with camera" width="500" /><br>


## Operating System Installation
Note: In order to properly run the SPEC software, the IMU and camera need to be connected to the Pi.

##### 1. Download and install the [Raspberry Pi Imager](https://www.raspberrypi.com/software/) on your computer and insert the microSD card
Select the operating system with or without the desktop. The desktop version is not necessary and will use more power (though you can turn off booting to desktop later). 

<img src="/app/static/images/RPi_imager_install_screen_1.png" alt="raspberry pi imager, select operating system" width="500" /><br>
*Desktop version, Raspberry Pi OS* <br><br>
<img src="/app/static/images/RPi_imager_install_screen_2.png" alt="raspberry pi imager, select operating system" width="500" /><br> 
*Non-desktop version, Raspberry Pi OS Lite* <br>
<!--![Raspberry Pi installer 1](/app/static/images/RPi_imager_install_screen_1.png)-->

##### 2.  During the imaging setup, you will be asked to apply customization settings. Select Yes (or Edit settings if this is not the first time using the software).
<img src="app/static/images/RPi_imager_install_OS_customisation_0.png" alt="raspberry pi imager, install customisation general tab" width="400" /><br>
<!--![Raspberry Pi installer 2](/app/static/images/RPi_imager_install_screen_2.png)-->
##### 3. Fill in all the fields in the General tab,and select Enable SSH with Use password authentication on the Services tab.
<img src="app/static/images/RPi_imager_install_OS_customisation_1.png" alt="raspberry pi imager, install customisation general tab" width="400" /><br>
<img src="app/static/images/RPi_imager_install_OS_customisation_2.png" alt="raspberry pi imager, install customisation general tab" width="400" /><br>
##### 4. When settings are complete, select Save and you will be prompted to create the image.
<img src="app/static/images/RPi_imager_install_OS_customisation_3.png" alt="raspberry pi imager, install customisation general tab" width="400" /><br>
##### 5. Power Up: This step assumes that you either have a monitor/keyboard/mouse plugged into the Pi, OR you have connected "headlessly" to the Pi. 
Once the image has been created, plug everything into the Pi EXCEPT the power. Place the SD card into the Pi and insert the power cable from the official Raspberry Pi power supply. 
##### 6. Confirm Pi is connected to the internet via wifi, or plug in an ethernet cable.

## SPEC Software Installation
##### 1. Clone the SPEC repository. 
`git clone https://gitlab.com/deepanalyticsllc/spec.git`

##### 2. Run the setup script that will install the SPEC software.
`cd spec`<br>
`./Setup/set_up_system.sh`<br>


##### 3. To confirm the install is successful, connect a device to its captive portal. For example, on a mobile device, look for the Pi's wifi access point that you previously set. If you reach the SPEC login page, the system is working.

## SPEC IMU Calibration, Camera Calibration and Camera Parameters
It is <strong><mark>critical</mark></strong> to calibrate the camera and the IMU before going out to your field site to set up the system. The camera calibration will enable you to obtain the focal length of your camera experimentally, which is especially useful if the camera manufacturer does not provide this parameter directly.

##### 1. IMU Calibration
- To perform the IMU's accelerometer calibration, you will run the IMU/misc/calibrate_imu.py script and place the system in 6 different positions for 2 seconds each. If this is challenging (and for some of us it was!) you can modify the IMU code to change the number of seconds in between positions.
  - *Intermediate skill - navigate to /usr/local/lib/python3.11/dist-packages/imusensor/MPU9250/MPU9250.py. Scroll to line 292 and edit 
  `time.sleep(2)' by changing the numeric value to be the number of seconds you want in between data taking. Save the file.
- To make things easy, the 6 positions you can use are illustrated below:

<img src="app/static/images/IMU_calib_positions.jpg" alt="IMU calibration positions" width="400" /><br>
- Run the calibration script `sudo python calibrate_imu.py`
  - The script walks you through placing the IMU/system in 6 different positions. When this step is finished, it will display the message "Accelerate calib success". Wait until the script completes calibration of the magnetometer (not used in SPEC, but done for completeness). When finished, you will see that the calibration data has been saved and loaded properly.
- To test the results, run the IMU/misc/testimu.py script. This will output accelerations in x, y, z. If your system is upright and on a flat surface, you should expect outputs of something like this:
Accel x: -0.04689449376828755 ; Accel y : 0.2007838316150461 ; Accel z : 9.803405120620532
roll: 1.1735931389023007 ; pitch : 0.2740142540259086 ; yaw : -149.5553912913779

Acceleration in z (downward) should be about 9.8 m/s^2^ and the pitch and roll angles should be about 0&deg;. If this is not the case, try IMU calibration again.

##### 2. Gather your camera specs
- f = Focal Length <br>
    f may be provided by the camera/camera sensor manufacturer. If not, we will
    determine f via the camera calibration matrix results and parameters set in 
    Camera Parameters page of the web app.

- Sensor Height and Width <br>
 These paramters are either specified in the sensor documentation,
    or calculated from the sensor diagonal and the sensor's aspect ratio.

- Image Resolution <br>
    The default is 1920x1080, even though our system's camera is capable of 3840x2160. The reduced
    resolution will reduce image size and improve processing times. We performed our testing at 1920x1080 and 
    have set this as the maximum available.

- Sensor Pixel Size in mm <br>
  This parameter should be provided by the manufacturer.

##### 3.Perform camera calibration
[A brief description of how camera calibration works is found here.](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Download and print a chessboard for the process](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)
- The next step is to take about 10-15 photos of the chessboard while it is in various positions. 
  - We recommend pasting the checkerboard to a piece of cardboard. And we also recommend moving the chessboard around and keeping the camera stationary.
  - On the Pi, proceed to Image_Processing/camera_calibration in a VSCode terminal or a Linux terminal. Run the take_photos.py script that will take a photo each time you press enter.
  `sudo python take_photos.py`
  - You can see that these photos are saved in Image_Processing/camera_calibration/images_for_calibration.

- Once the photos are taken, open the calibration.py script. If you are using the calibration chessboard linked above, your settings should remain as<br> 
`chessboardSize = (9,6)`<br>
`frameSize = (1920,1080)`<br>
`size_of_chessboard_squares_mm = 25`<br>
- If you used a different chessboard:
  - Count the inner corners, the intersections where the black and white squares meet. Count along the rows and along the columns to determine the size.

- Run the calibration.py script
`sudo python calibration.py`
- The output of this script is your camera calibration matrix

[[1.02444441e+03 0.00000000e+00 9.91167483e+02]<br>
 [0.00000000e+00 1.02434832e+03 6.04988705e+02]<br>
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
- The number in the [first row, first column], or the [0,0] element, of your camera calibration is the focal length in the x-direction in pixels. The number in the [second row, second column], or the [1,1] element, is the focal length in the y-direction in pixels. These should be close, so choose one.  
- On the camera parameters page, we calculate your focal length in mm by multiplying your reduced resolution camera pixel size (also calculated for you on the camera parameters page). For example, element [0,0] is 1.02444e+03. Our camera reduced resolution pixel size is 0.0029 mm. focal length = 1024.44*0.0029 = 2.97 mm.

##### 4. Configure Camera Parameters
- Connect to the SPEC app via wifi.
- Log in with the user credentials you selected.
- Proceed to Main Menu, Setup and Calibrations, and Camera Parameters.

<img src="app/static/images/camera_parameters.png" alt="camera parameters" width="250" /><br>
- Fill in your sensor height, width and max resolution parameters. The reduced width and height should be set at 
1920x1080 for best system performance, and is the maximum setting available. 
- Fill in your full resolution pixel size in mm as provided in manufacturer specs.
- Enter the [0,0] element from the camera calibration matrix.
- The SPEC app has now calculated your reduced resolution pixel size as well as your focal length. 
- Hit UPDATE CONFIGURATION to save these settings to config.json, and you're done with setup.

##### Optional: Customizing Background Image
The SPEC web application has a default background image that you may wish to change. Rename your image to app_bg.jpg and place it in the /app/static/images folder. You will need to restart the app service manually via
`sudo systemctl restart start_app.service`
or simply reboot or power cycle the Pi.










