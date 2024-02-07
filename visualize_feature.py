import numpy as np
import cv2
import os
from extract_feature import get_feature_img
from tqdm import tqdm
import pdb
X = np.load('extracted_data/testX.npy', allow_pickle=True)
y = np.load('extracted_data/testY.npy')
FEATURE_DIR = 'feature_imgs'
BACT_PATH = os.path.join(FEATURE_DIR, 'bacteria')
SWAB_PATH = os.path.join(FEATURE_DIR, 'swab')
if not os.path.isdir(FEATURE_DIR):
    os.mkdir(FEATURE_DIR)

if not os.path.isdir(BACT_PATH):
    os.mkdir(BACT_PATH)

if not os.path.isdir(SWAB_PATH):
    os.mkdir(SWAB_PATH)

bact_count, swab_count = 0, 0
for i, x in tqdm(enumerate(X), total=len(X)):
    img = cv2.resize(get_feature_img([x[0]], [x[1]])[0] * 255, (128, 128))
    print(np.mean(img, axis=2))
    # pdb.set_trace()
    if y[i] == 0:
        cv2.imwrite(BACT_PATH + f'/{bact_count}.jpg', img)
        bact_count += 1
    else:
        cv2.imwrite(SWAB_PATH + f'/{swab_count}.jpg', img)
        swab_count += 1