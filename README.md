#Installation Instructions

I highly recommend creating a virtual environment and doing the installation on anaconda since that is what I did. Any other installation methods will have to be glued together by yourself.

1. Install Anaconda
2. conda creeate --name visor python=3.5
3. install pytorch via conda
4. pip install --user panda3d==1.10.3
5. pip install --user opencv-python
6. conda install matplotlib
7. pip install --user dlib
8. pip install imutils
9. pip install --ignore-installed --upgrade tensorflow-gpu
10. pip install scipy


				Face detection algorithms
###########################################################################################
###########################################################################################

http://dlib.net/face_detector.py.html
https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/

out_model_front+dlib:

	- dlib's frontal face detector
	- dlib's resnet 10 landmark detector
	- performance:

		avg 0.0659s fer frame

###########################################################################################

out_model_s3fd+bulat:

	- s3fd face detector. https://github.com/yxlijun/S3FD.pytorch
	- adrian bulat's landmark detector
	- https://github.com/1adrianb/face-alignment
	- performance:

		avg 1.9s per frame


###########################################################################################
out_model_front+dlib:
https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

	- opencv s3fd face detector
	- performance:

		avg 0.0659s+0.01s per frame

###########################################################################################

out_model_yolo+dlib

	- yolo face detector on cpu. https://github.com/iitzco/faced
	- dlib's resnet 10 landmark detector
	- performance:

		avg 0.0774s per frame

###########################################################################################

out_model_yolo+dlib

	- yolo face detector on cpu. https://github.com/iitzco/faced
	- dlib's resnet 10 5 point landmark detector
	- performance:

		avg 0.0738s per frame

###########################################################################################
###########################################################################################
###########################################################################################
SHADOW DETECTION

method1:
    http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
    opencv

method2:
    https://github.com/mykhailomostipan/shadow-removal/Shadow Detection and Removal Based on YCbCr Color Space.pdf
    https://github.com/mykhailomostipan/shadow-removal
