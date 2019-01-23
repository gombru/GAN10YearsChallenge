# Runs SFD for an image and plots detections
# https://github.com/clcarwin/SFD_pytorch
# Implementation of Single Shot Scale-invariant Face Detector, ICCV, 2017 (based on SSD)

import sys

sys.path.append('../')
from SFD import net_s3fd, detect
from SFD.bbox import *
import torch
import cv2
import glob

# indices_file = open("../../../ssd2/insta10YearsChallenge/anns/ids_filtered_by_metadata.csv","r")
images_path = "../../../hd/datasets/insta10YearsChallenge/data/"
detections_results_path = "../../../hd/datasets/insta10YearsChallenge/face_detections_rejected/"
cropped_faces_path = "../../../hd/datasets/insta10YearsChallenge/faces_img_young/"
accepted_images_file = open("../../../hd/datasets/insta10YearsChallenge/anns/accepted.csv", "w")

input_size = 512  # 640 # Multiscale approach, can try different sizes
net = 's3fd'
model = '../SFD/data/s3fd_convert.pth'
threshold = 0.5  # 0.5
iou_threshold = 0  # 0.3


def resize_image(img, size):
    h, w = img.shape[0], img.shape[1]
    ar = float(w) / h
    if h < w:
        new_h = size
        new_w = int(size * ar)
    else:
        new_w = size
        new_h = int(size / ar)
    img = cv2.resize(img, (new_w, new_h))
    return img


def save_result(img, bboxlist, accepted):
    img_draw = img.copy()
    for b in bboxlist:
        x1, y1, x2, y2, s = b
        cv2.rectangle(img_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
    folder = detections_results_path
    if accepted:
        folder = folder.replace('rejected', 'accepted')
        print(folder)
    cv2.imwrite(folder + filename.split('.')[-2].split('/')[-1] + '.jpg', img_draw)


print("Loading Net")
net = getattr(net_s3fd, net)()
net.load_state_dict(torch.load(model))
net.cuda()
net.eval()

i = 0
# for filename in indices_file:
print("Infering")
for filename in glob.glob(images_path + "/*.jpg"):

    i += 1
    if i % 1 == 0: print(i)
    img = cv2.imread(filename)
    img = resize_image(img, input_size)
    bboxlist = detect.detect(net, img)
    keep = nms(bboxlist, threshold)
    bboxlist = bboxlist[keep, :]

    # Get faces over threshold
    thresholded_bboxlist = []
    for b in bboxlist:
        if b[4] > threshold:
            thresholded_bboxlist.append(b)

    # Discard image if there are not 2 faces, or if there are more
    if len(thresholded_bboxlist) != 2:
        accepted = False
        save_result(img, thresholded_bboxlist, accepted)
        continue

    # Discard image if one of the bb is too small
    size_threshold = 512.0/5
    size_discarding = False
    for b in thresholded_bboxlist:
        if b[2] - b[0] < size_threshold or b[3] - b[1] < size_threshold:
            size_discarding = True
            break
    if size_discarding:
        accepted = False
        save_result(img, thresholded_bboxlist, accepted)
        continue

    # Identify left/right detections
    elif thresholded_bboxlist[0][0] < thresholded_bboxlist[1][0] and thresholded_bboxlist[0][2] < \
            thresholded_bboxlist[1][2]:
        left_face = thresholded_bboxlist[0]
        right_face = thresholded_bboxlist[1]
    elif thresholded_bboxlist[1][0] < thresholded_bboxlist[0][0] and thresholded_bboxlist[1][2] < \
            thresholded_bboxlist[0][2]:
        left_face = thresholded_bboxlist[1]
        right_face = thresholded_bboxlist[0]
    else:
        accepted = False
        save_result(img, thresholded_bboxlist, accepted)
        continue

    # Check that one detection is at left and the other at right and separation between detections
    left_face_center = left_face[0] + (left_face[2] - left_face[0]) / 2
    right_face_center = right_face[0] + (right_face[2] - right_face[0]) / 2
    min_separation = img.shape[1] * 0.15
    if left_face[2] -  right_face[0] < min_separation or left_face[2] > img.shape[1] / 2 or right_face[2] < img.shape[1] / 2:
        accepted = False
        save_result(img, thresholded_bboxlist, accepted)
        continue

    accepted = True
    # Write idx to file
    accepted_images_file.write(filename.split('/')[-1].strip('.jpg') + '\n')
    # Save image with detections
    save_result(img, thresholded_bboxlist, accepted)
    # Saved cropped faces
    padding_lf_h = (left_face[2] - left_face[0]) * 0.1
    padding_rf_h = (right_face[2] - right_face[0]) * 0.1
    padding_lf_v = (left_face[3] - left_face[1]) * 0.1
    padding_rf_v = (right_face[3] - right_face[1]) * 0.1
    try:
        cv2.imwrite(cropped_faces_path + filename.split('.')[-2].split('/')[-1] + '.jpg',
                    img[int(left_face[1])-padding_lf_v:int(left_face[3]+padding_lf_v), int(left_face[0])-padding_lf_h:int(left_face[2]+padding_lf_h), :])
        cv2.imwrite(cropped_faces_path.replace('young', 'old') + filename.split('.')[-2].split('/')[-1] + '.jpg',
                    img[int(right_face[1])-padding_rf_v:int(right_face[3])+padding_rf_v, int(right_face[0])-padding_rf_h:int(right_face[2])+padding_rf_h, :])
    except:
        print("Image ommited because of padding error")

# accepted_images_file.close()
print("DONE")

