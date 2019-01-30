import dlib
import cv2
import glob
from imutils import face_utils
import imutils
import numpy as np
import math

images_path = "../../../datasets/insta10YearsChallenge/faces_img_young/"
out_images_path = "../../../datasets/insta10YearsChallenge/faces_img_young_dlibFiltered_aligned/"
resize_size = 286

# Initialize dlib :
# Face detector (HOG-based) and then create the facial landmark predictor
print("Loading model")
shape_predictor = './shape_predictor_68_face_landmarks.dat'
print('Loading predictor from ' + shape_predictor)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# Initialize aligner
fa = face_utils.FaceAligner(predictor,  desiredLeftEye=(0.28, 0.28), desiredFaceWidth=286)

count = 0
total_accepted = 0
invalid_files = 0
eyes_discards = 0
nose_discards = 0
no_detection_discards = 0
alignment_errors = 0
size_discard = 0


nose_threshold = 0.2
max_rotation = 20

def get_rotation_by_eyes(landmarks):
    y_eye_left = landmarks[36:41][:,1].mean()
    y_eye_right = landmarks[42:47][:,1].mean()
    x_eye_left = landmarks[36:41][:,0].mean()
    x_eye_right = landmarks[42:47][:,0].mean()
    y_dist = y_eye_right - y_eye_left
    x_dist = x_eye_right - x_eye_left
    angle = np.arctan(y_dist/x_dist)
    return math.degrees(angle)

# def rotate_image(image, angle):
#     rotated = imutils.rotate(image, angle)
#     return rotated

print("Infering")
for filename in glob.glob(images_path + "/*.jpg"):
    # try:
    count+=1
    if count % 5 == 0:
        print("Accepted " + str(total_accepted) + " out of " + str(count) + " invalid files " + str(invalid_files) + " eyes " + str(eyes_discards) + " nose " + str(nose_discards) + " alignment errors " + str(alignment_errors) + " size discard " + str(size_discard) + " no detections " + str(no_detection_discards))

    accepted = True

    # Input image :
    image_young = cv2.imread(filename)
    image_old = cv2.imread(filename.replace('young','old'))

    if image_old.__sizeof__() < 20 or image_young.__sizeof__() < 20:
        invalid_files+=1
        continue # Check file exists

    # Discard by size
    if image_young.shape[0] < 100 or image_young.shape[1] < 100:
        size_discard += 1
        continue
    if image_old.shape[0] < 100 or image_old.shape[1] < 100:
        size_discard += 1
        continue

    # Resize images
    image_young = imutils.resize(image_young, width=resize_size)
    image_old = imutils.resize(image_old, width=resize_size)

    # Discard grayscale images
    if image_young.shape[2] != 3 or image_old.shape[2] != 3: continue
    gray_young = cv2.cvtColor(image_young, cv2.COLOR_BGR2GRAY)
    gray_old = cv2.cvtColor(image_old, cv2.COLOR_BGR2GRAY)

    # Face detection :
    # requires grayscale
    # rect contains as many boungding boxes as
    # faces detected in the image
    rect_young = detector(gray_young, 1)
    rect_old = detector(gray_old, 1)

    if len(rect_young) != 1 or len(rect_old) != 1:
        no_detection_discards+=1
        # print("Num detection: " + str(len(rect_young)) + ' , ' + str(len(rect_old)))
        continue # if more than one face is detected, discard

    rect_young = rect_young[0]
    rect_old = rect_old[0]

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy array
    shape_young = predictor(gray_young, rect_young)
    shape_young = face_utils.shape_to_np(shape_young)
    shape_old = predictor(gray_old, rect_old)
    shape_old = face_utils.shape_to_np(shape_old)

    # Convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    # (x_y, y_y, w_y, h_y) = face_utils.rect_to_bb(rect_young)
    # (x_o, y_o, w_o, h_o) = face_utils.rect_to_bb(rect_old)

    # for (x, y) in shape_old:
    #     cv2.circle(image_old, (x, y), 1, (0, 0, 255), -1)
    #     cv2.imwrite('_landmarks.jpg', image_old)
    #     # cv2.imshow("Facial landmarks", image_old)
    #     # cv2.waitKey(0)

    # Check the alignement between eyes landmarks in both young and old face.

    # If the difference in y position of the eyes in one of the images is too big, rotate it using that difference]
    rot_young = get_rotation_by_eyes(shape_young)
    rot_old = get_rotation_by_eyes(shape_old)
    if rot_young > max_rotation or rot_young < -max_rotation:
        # cv2.imwrite(out_images_path.replace('faces_img_young_dlibFiltered','rejected_eyes') + filename.split('/')[-1], image_young)
        accepted = False
        eyes_discards+=1
        continue
    if rot_old > max_rotation or rot_old < -max_rotation:
        # cv2.imwrite(out_images_path.replace('faces_img_young_dlibFiltered','rejected_eyes') + filename.split('/')[-1], image_old)
        accepted = False
        eyes_discards+=1
        continue

    # image_young = rotate_image(image_young,rot_young)
    # image_old = rotate_image(image_old,rot_old)


    # If the face is not centered (nose landmarks), remove pair
    # Get the mean x position of nose landmarks (27,29,30,31,34)
    img_y_var_allowed = image_young.shape[1] * nose_threshold
    img_o_var_allowed = image_old.shape[1] * nose_threshold
    mean_nose_x_young = shape_young[27:34][:,0].mean()
    mean_nose_x_old = shape_old[29:33][:,0].mean()
    if mean_nose_x_young > (image_young.shape[1]/2 + img_y_var_allowed) or mean_nose_x_young < (image_young.shape[1]/2 - img_y_var_allowed):
        # cv2.imwrite(out_images_path.replace('faces_img_young_dlibFiltered','rejected_nose') + filename.split('/')[-1], image_young)
        accepted = False
        nose_discards +=1
        continue
    if mean_nose_x_old > (image_old.shape[1]/2 + img_o_var_allowed) or mean_nose_x_old < (image_old.shape[1]/2 - img_o_var_allowed):
        # cv2.imwrite(out_images_path.replace('faces_img_young_dlibFiltered', 'rejected_nose') + filename.split('/')[-1],image_old)
        accepted = False
        nose_discards +=1
        continue

    # Faces are accepted, so align them
    try:
        young_aligned = fa.align(image_young, gray_young, rect_young)
        old_aligned = fa.align(image_old, gray_old, rect_old)
        cv2.imwrite(out_images_path + filename.split('/')[-1],young_aligned)
        cv2.imwrite(out_images_path.replace('young', 'old') + filename.split('/')[-1],old_aligned)
        total_accepted+=1
    except:
        print("Error aligning face")
        alignment_errors+=1
        continue

    # if accepted:
    #     cv2.imwrite(out_images_path + filename.split('/')[-1],image_young)
    #     cv2.imwrite(out_images_path.replace('young', 'old') + filename.split('/')[-1],image_old)
    #     total_accepted+=1



print("Accepted " + str(total_accepted) + " out of " + str(count) + " invalid files " + str(invalid_files) + " eyes " + str(eyes_discards) + " nose " + str(nose_discards) + " alignment errors " + str(alignment_errors) + " size discard " + str(size_discard) + " no detections " + str(no_detection_discards))
print("Accepted " + str(total_accepted) + " out of " + str(count))
print("DONE")