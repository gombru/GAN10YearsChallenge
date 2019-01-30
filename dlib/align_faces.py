# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

img_path = "/home/raulgomez/datasets/insta10YearsChallenge/faces_img_young_dlibFiltered/1960160211085587789.jpg"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor,  desiredLeftEye=(0.28, 0.28), desiredFaceWidth=286)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(img_path)
image = imutils.resize(image, width=286)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# show the original input image and detect faces in the grayscale
# image
cv2.imshow("Input", image)
rects = detector(gray, 2)
print("Num detections: " + str(len(rects)))

# loop over the face detections
for i, rect in enumerate(rects):
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    print((x, y, w, h))
    # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=286)
    faceAligned = fa.align(image, gray, rect)

    # display the output images
    # cv2.imshow("Original " + str(i), faceOrig)
    cv2.imshow("Aligned " + str(i), faceAligned)
cv2.waitKey(0)