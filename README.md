###########################################################################################
###########################################################################################
# Installation Instructions

I highly recommend creating a virtual environment and doing the installation on the anaconda virtual environment since that is what I did. That being said, I glued everything together using both pip and anaconda so make sure in intialize the conda virtual environment with the correct python version. Any other installation methods will have to be glued together by yourself.

1. Install Anaconda and pip
2. conda creeate --name visor python=3.5
3. install pytorch via conda version >= 1.1.0
4. pip install --user panda3d==1.10.3
5. install opencv for python >= 4.1.0. $pip install --user opencv-python
6. conda install matplotlib
7. pip install --user dlib
8. pip install imutils
9. pip install --ignore-installed --upgrade tensorflow-gpu
10. pip install scipy
11. conda install -c 1adrianb face_alignment

###########################################################################################
###########################################################################################
### Face detection algorithms

The best open source face detection algorithm I found as of 8-13-2019 was from opencv's deep learning library which came with a trained model. Besides that, dlib and some published papers also had useful implementations, but each one had its own drawbacks in terms of speed and performance. Some preliminary results are listed below using Intel® Xeon(R) CPU E5-1650 v3 @ 3.50GHz × 12. 
##### Face Detection Speed
| Title | Link | sec/frame | eyeballed performance | Description |
| ----- | ---- | --------- | --------------------- | ----------- |
| DLIB CNN | http://dlib.net/face_detector.py.html | 1.341s/frame | Med | CNN based face detection |
| YOLO CPU | https://github.com/iitzco/faced | 0.104s/frame | Low | YOLO object detection for faces only |
| OpenCV DNN | https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/ | 0.0345s/frame | High | OpenCV's DNN Library for face detection |

##### Landmark Detection
| Title | Link | sec/frame | eyeballed performance | Description |
| ----- | ---- | --------- | --------------------- | ----------- |
| DLIB resnet | https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/ | 0.00677s/frame | Med | Simple 2D landmark localization |
| DLIB resnet 5pts | https://www.pyimagesearch.com/2018/04/02/faster-facial-landmark-detector-with-dlib/ | 0.00313s/frame | Med | Simple 2D landmark for 5 pts |
| Haar Cascade Filter | https://pythonprogramming.net/haar-cascade-face-eye-detection-python-opencv-tutorial/ | 0.0332s/frame | Low | Haar Cascade filter for eye detection |
| Adrian Bulat | https://github.com/1adrianb/face-alignment | 0.550s/frame | High | How far are we from solving the 2D \& 3D Face Alignment problem? |

##### Shadow Detection
| Title | Link | sec/frame | eyeballed performance | Description |
| ----- | ---- | --------- | --------------------- | ----------- |
| YCbCr Color Removal | https://github.com/mykhailomostipan/shadow-removal/ | 0.0003s/frame | Low | Traditional method analyzing the luminescent color space |
