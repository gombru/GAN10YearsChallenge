import glob
import random
from shutil import copyfile

img_dir = "/home/raulgomez/datasets/insta10YearsChallenge/faces_img_young_old/"
out_dir = "/home/raulgomez/datasets/insta10YearsChallenge/splits/"

num_test = 200
num_val = 200

imgs = []
for file in glob.glob(img_dir + "/*.jpg"):
    imgs.append(file.split('/')[-1])

random.shuffle(imgs)

for i,img in enumerate(imgs):
    if i < num_test: copyfile(img_dir + img, out_dir + 'test/' + img)
    elif i < num_test + num_val: copyfile(img_dir + img, out_dir + 'val/' + img)
    else: copyfile(img_dir + img, out_dir + 'train/' + img)

print("DONE")