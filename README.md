<img src="/app/static/images/SPEC_logo.png" alt="SPEC logo" width="500"/><br>

# SPEC
Set up PIV (Particle Image Velocimetry) Edge Camera (SPEC)


## Description
The SPEC is a fully contained and continuous Particle Image Velocimetry (PIV) calculation system. Once a user sets it up on a riverbank, SPEC will aquire images at a user specified rate and use them to perform PIV calculations. SPEC is also set up to be customized by the user depending on the river. A web-application stored on SPEC and accessed by connecting to SPEC’s Wi-Fi access point allows easy setup and customization on site 

This repository includes the code for system configuration and calibration, image capture, image processing and undistorting, homography, and PIV analysis for a field site. The results of the pipeline are 2D velocity vector fields, represented in a plot overlaid on the measurement area.

### SPEC System

![fully labeled system](/app/static/images/full_labeled_system.png)     ![deploy SPEC](/app/static/images/deployed_system.jpg)

### Web-Application Login 

![Web-app](/app/static/images/web-app.png)

### Input and Outputs of a PIV calculation

![field site](/app/static/images/field_site.jpg)  ![velocity vectors](/app/static/images/field_site_vectors.jpg)

## Build/Installation
All materials and diagrams needed to build a SPEC camera system are located in the [Setup folder](Setup/) of this repository's directory, including all necessary documentation. 

## Usage
SPEC was developed to be a bank-mounted camera system to measure river surface velocities. The system is a completely self-contained data collection and processing unit. We designed this system with citizen science in mind, keeping the cost of materials low and all 3D-printed models and code are available open source.

Citizen science was also a priority when planning the deployment process in the field. A user just needs a SPEC system and a place to mount it. Then using the web-application a user can customize the parameters used during the PIV calculations for the current river conditions.

More information can be found in the [Docs folder](Docs/) of this repository.

## Authors and acknowledgement
SPEC was a research and development project engineered by Deep Analytics LLC. Funding for this project was provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.

This PIV algorithm of this project was ported from TRiVIA:
- Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): U.S. Geological software release, 
    https://doi.org/10.5066/P9AD3VT3.

## License
SPEC is licensed under the Creative Commons Zero v1.0 Universal [LICENSE.txt](LICENSE.txt)

## Project status
The SPEC project is ongoing and the team at Deep Analytics is committed to refining the software, adding new features, and responding to user needs.

## Disclaimer
No warranty, expressed or implied, is made by Deep Analytics LLC as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. Furthermore, the software is released on condition that Deep Analytics LLC shall not be held liable for any damage resulting from its authorized or unauthorized use.

Any use of trade, firm, or product names is for descriptive purposes only and does not imply endorsement by the U.S. Government

## Support
The [Docs folder](Docs/) in this repository includes a full user manual that includes everything from explanation of the background of the code, build/setup directions, and web-application use. There is also a separate web-application doc containing the same information as in the user-manual.

For technical questions or issues regarding SPEC, please fill out an issue form on this repository


