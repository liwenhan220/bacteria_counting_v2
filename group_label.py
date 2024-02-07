from label_img import label_image
import numpy as np
import cv2
import os

INPUT_DIR = "extra_data/input_imgs_4"
OUTPUT_DIR = "extra_data/output_imgs_4"
MODEL_DIR = 'models'
LOG_DIR = 'logs'
COVER_CORNERS = False # for 2, 3, 4
MODEL_NUM = 100
PREDICTION_THRESHOLD = 0.5

# Define the font and scale
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_thickness = 10
font_color = (0, 0, 0)  # White color in BGR

acc1 = np.load(LOG_DIR + '/val_acc1.npy')
acc2 = np.load(LOG_DIR + '/val_acc2.npy')
acc = acc1 + acc2
print('default model num: {}'.format(MODEL_NUM))
INPUT_SHAPE_FILE = MODEL_DIR + '/input_shape.npy'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

PATH1 = OUTPUT_DIR + '/all'
PATH2 = OUTPUT_DIR + '/bact_only'

if not os.path.isdir(PATH1):
    os.mkdir(PATH1)

if not os.path.isdir(PATH2):
    os.mkdir(PATH2)  
print('output directory: {}'.format(OUTPUT_DIR))

images = os.listdir(INPUT_DIR)
bact_counts = []
noise_counts = []
image_names = []

for image in images:
    img = cv2.imread(INPUT_DIR + '/' + image)
    if img is None:
        continue
    bact_img, all_img, bact_count, noise_count = label_image(MODEL_DIR, MODEL_NUM, img, COVER_CORNERS, image_name=image, prediction_threshold=PREDICTION_THRESHOLD)
    
    position = (10, bact_img.shape[0] - 10)  # Bottom left corner

    # Put the text on the image
    cv2.putText(bact_img, image, position, font, font_scale, font_color, font_thickness)
    cv2.putText(all_img, image, position, font, font_scale, font_color, font_thickness)

    cv2.imwrite(PATH2 + '/' + image, bact_img)
    cv2.imwrite(PATH1 + '/' + image, all_img)
    bact_counts.append(bact_count)
    noise_counts.append(noise_count)
    image_names.append(image)

    np.save(OUTPUT_DIR + '/image_names', image_names)
    np.save(OUTPUT_DIR + '/bact_counts', bact_counts)
    np.save(OUTPUT_DIR + '/noise_counts', noise_counts)

