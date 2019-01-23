
import glob
from PIL import Image
from joblib import Parallel, delayed
import os


faces_a = "/home/raulgomez/datasets/insta10YearsChallenge/faces_img_young/"
faces_b = "/home/raulgomez/datasets/insta10YearsChallenge/faces_img_old/"
out_dir = "/home/raulgomez/datasets/insta10YearsChallenge/faces_img_young_old/"

out_size = 286


def resize(path):
    try:
        img_name = path.split('/')[-1]
        face_a = Image.open(path)
        face_b = Image.open(faces_b + img_name)
        face_a = face_a.resize((out_size, out_size), Image.ANTIALIAS)
        face_b = face_b.resize((out_size, out_size), Image.ANTIALIAS)
        out_im = Image.new('RGB', (out_size*2, out_size))
        out_im.paste(face_a,(0,0))
        out_im.paste(face_b,(out_size,0))
        out_im.save(out_dir + img_name)

    except:
        print "Failed"
        return

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
Parallel(n_jobs=12)(delayed(resize)(file) for file in glob.glob(faces_a + "/*.jpg"))
print("DONE")