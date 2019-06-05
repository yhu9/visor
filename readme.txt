Attached are four folders with image results of the landmark detection. The folder names indicate the face detection algorithm and the landmark detection algorithm used.

out_model_front+dlib:

	- dlib's frontal face detector
	- dlib's resnet 10 landmark detector
	- performance:

		avg 0.0659s fer frame

out_model_s3fd+bulat:

	- s3fd face detector. https://github.com/yxlijun/S3FD.pytorch
	- adrian bulat's landmark detector
	- https://github.com/1adrianb/face-alignment
	- performance:

		avg 1.9s per frame


out_model_yolo+dlib

	- yolo face detector on cpu. https://github.com/iitzco/faced
	- dlib's resnet 10 landmark detector
	- performance:

		avg 0.0774s per frame

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
