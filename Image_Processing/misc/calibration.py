import numpy as np
import cv2 as cv
import glob
import pickle
import shutil
'''
Camera calibration can be done prior to setting camera parameters in the web app, but in order to calculate the focal
length from the calibration matrix, you will need the reduced resolution camera pixel size from the camera parameters web app page.

Brief description of how calibration works found here: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
1. Download and print a chessboard for this calibration process https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
2. Collect at least 10 images of the chessboard from various perspectives using the script camera_calibration/take_photos.py
3. Set the chessboardSize (based on small text on printout) and your frameSize (image resolution) below.
4. Run this script (calibration.py) to generate and save the camera and distortion matrices. The camera calibration matrix will be printed out.
5. The [0,0] element of your camera calibration is the focal length in the x direction in pixels. The [1,1] element is the focal 
   length in the y direction in pixels. These should be close, so choose one and multiply by your reduced resolution camera pixel size (that 
   is shown as the last parameter on the camera parameters web page) to get your focal length in mm.
   For example, element [0,0] is 1.02017e+03. Our camera reduced resolution pixel size is 0.0029 mm. focal length = 1020.17*0.0029 = 2.96 mm
'''
################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

# User sets these parameters
chessboardSize = (9,6)
frameSize = (1920,1080)
size_of_chessboard_squares_mm = 25


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)


objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob('images_for_calibration/*.jpg')
print (images)

for image in images:
    print ("Processing ", image)
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imwrite('grayscale_image.jpg', gray)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize,flags=cv.CALIB_CB_ADAPTIVE_THRESH +
                                               cv.CALIB_CB_FAST_CHECK +
                                               cv.CALIB_CB_NORMALIZE_IMAGE +
                                               cv.CALIB_CB_EXHAUSTIVE)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)


cv.destroyAllWindows()

#################################### CALIBRATION ################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open( "../calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "../cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "../dist.pkl", "wb" ))

# Save calibrations to Offsite Processing folder for 
# later independent processing.
source = "../cameraMatrix.pkl"
destination = "../../Offsite_Processing/cameraMatrix.pkl"
source2 = "../dist.pkl"
destination2 = "../../Offsite_Processing/dist.pkl"
shutil.copy(source, destination)
shutil.copy(source2, destination2)

with open('../cameraMatrix.pkl', 'rb') as file:
    data = pickle.load(file)

# Print the contents of the file
print ("\n Camera calibration matrix \n")
print(data)

