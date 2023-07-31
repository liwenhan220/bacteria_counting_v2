from extract_feature import *
from generate_data import *
from cnn import *
import torch
import numpy as np
import cv2
import sys
import getopt
import pdb

argv = sys.argv[1:]

try:
    opts, args = getopt.getopt(argv, "m:i:o:n:", 
                                ["model_path=",
                                "image_path=",
                                "output_path=",
                                "model_number="])
except:
    print("Error")

MODEL_DIR = 'models'
acc1 = np.load('log/val_acc1.npy')
acc2 = np.load('log/val_acc2.npy')
acc = acc1 + acc2
MODEL_NUM = np.argmax(acc)
print('default model num: {}'.format(MODEL_NUM))
# MODEL_PATH = 'models/bact_model_49'
OUTPUT_PATH = "output_img.bmp"
INPUT_SHAPE_FILE = 'models/input_shape.npy'
WINDOW_SCALE = 1.7

# IMG_NAME = "raw_data/Ecoli-positive/E.coli + 1.bmp"
# IMG_NAME = "raw_data/Styphi-positive/S.typhi + 2.bmp"
# IMG_NAME = "raw_data/negative/swab-1.bmp"
# IMG_NAME = "raw_data/new negative/-1.bmp"
IMG_NAME = "raw_data/new negative/-small swab5-1.bmp"


for opt, arg in opts:
    if opt in ['-m', '--model_path']:
        MODEL_DIR = arg

    elif opt in ['-i', '--image_path']:
        IMG_NAME = arg

    elif opt in ['-o', '--output_path']:
        OUTPUT_PATH = arg
    
    elif opt in ['-n', '--model_number']:
        MODEL_NUM = arg

MODEL_PATH = MODEL_DIR + '/bact_model_{}'.format(MODEL_NUM)
INPUT_SHAPE = tuple(np.load(INPUT_SHAPE_FILE))
print('model name: {}'.format(MODEL_PATH))
print('img name: {}'.format(IMG_NAME))
print('output path: {}'.format(OUTPUT_PATH))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = LeNet5((3, *INPUT_SHAPE), 2)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device=device)
model.eval()
max_shape = INPUT_SHAPE

img = cv2.imread(IMG_NAME)
bacts, _ = generate_bacts(img, None, size = [10, 60], bias = BIAS, cover_corners= False, debug = False, debug_path = None, threshold=THRESHOLD, max_diameter=MAX_DIAMETER)


bact_count = 0
noise_count = 0
pbar = tqdm(bacts, total=len(bacts))
for bact in pbar:

    if bact.pad_img(max_shape) == -1:
        continue

    f_img = get_img(bact)
    tensor_img = torch.tensor(reshape_data(f_img.reshape(1, *f_img.shape))).to(device)
    output = model(tensor_img)[0]
    pbar.set_description('drawing bacteria on image')
    label = torch.argmax(output)
    if label == 0:
        color = [0, 255, 0]
        bact_count += 1
    else:
        color = [255, 0, 0]
        noise_count += 1
    img_shape = np.array(img.shape[:2])[::-1]
    cv2.imshow(IMG_NAME, cv2.resize(img, (img_shape * WINDOW_SCALE).astype(np.uint16)))
    cv2.waitKey(1)
    draw_bacteria(img, bact, color)

print('bacteria count: {}\nnoise count: {}\nbact rate: {}'.format(bact_count, noise_count, bact_count / (bact_count + noise_count)))
cv2.imwrite(OUTPUT_PATH, img)

input('PRESS ANY KEY TO END')

