from extract_feature import *
import os
from hyperparameters import *

INPUT_FOLDER = "input_folder"
if not os.path.exists(INPUT_FOLDER):
    os.makedirs(INPUT_FOLDER)
    print("Prepared an input folder for you, please put images inside")
    quit()
DEBUG_PATH = "output_folder"

os.makedirs(DEBUG_PATH, exist_ok=True)

bacteria_generator = BacteriaGenerator(SIZE, MAX_DIAMETER, True, False)
bacteria_generator.debug = True
# input_img_path = "image2.jpg"
files = os.listdir(INPUT_FOLDER)
# img = cv2.imread(input_img_path)
for file in files:
    try:
        img = cv2.imread(os.path.join(INPUT_FOLDER, file))
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                            0, 
                                                            image_name = file, 
                                                            debug_path = DEBUG_PATH)
    except Exception as e:
        print(e)
