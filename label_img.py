from extract_feature import BacteriaGenerator, draw_bacteria, get_feature_img, get_symmetries
from hyperparameters import *
from generate_data import get_img
from tqdm import tqdm
from cnn import LeNet5, reshape_data
import torch
import numpy as np

def label_image(model_dir, model_num, orig_img, cover_corners = False, image_name = None):
    bacteria_generator = BacteriaGenerator(SIZE, MAX_DIAMETER, False, cover_corners)
    bact_img = orig_img.copy()
    all_img = orig_img.copy()

    INPUT_SHAPE_FILE = model_dir + '/input_shape.npy'

    MODEL_PATH = model_dir + '/bact_model_{}'.format(model_num)
    INPUT_SHAPE = tuple(np.load(INPUT_SHAPE_FILE))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = LeNet5((3, *INPUT_SHAPE), 2)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device=device)
    model.eval()
    max_shape = INPUT_SHAPE

    bacts, _ = bacteria_generator.generate_bacts(orig_img, 
                                                 None, image_name=image_name)
    bact_count = 0
    noise_count = 0
    pbar = tqdm(bacts, total=len(bacts))
    for bact in pbar:

        if bact.pad_img(max_shape) == -1:
            continue

        f_img = get_img(bact)
        f_img = get_feature_img([f_img[0]], [f_img[1]])[0]
        f_img = f_img.reshape(1, *f_img.shape)
        # f_img = np.array(get_symmetries(f_img))
        tensor_img = torch.tensor(reshape_data(f_img)).to(device)
        output = torch.mean(model(tensor_img), dim=0)
        pbar.set_description('drawing bacteria on image')
        if output[0] > output[1]:
            label = 0
        else:
            label = 1
        if label == 0:
            color = [0, 255, 0]
            bact_count += 1
            draw_bacteria(bact_img, bact, color)
        else:
            color = [255, 0, 0]
            noise_count += 1
        draw_bacteria(all_img, bact, color)
    return bact_img, all_img, bact_count, noise_count


