from extract_feature import *
import os
from my_threshold import *
from hyperparameters import *
from cnn import *
import torch
from train_model import metric_infos

# DEPRECATED METHODS FROM past generations

OUTPUT_PATH = 'results'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

bact_generator = BacteriaGenerator(FIRST_BIAS, SECOND_BIAS, SIZE, THRESHOLD, FIRST_BLOCK_SIZE, SECOND_BLOCK_SIZE, MAX_DIAMETER, True, False)

def draw_img(preprocess_key, path, name, o_img, crop_region = None):
    if os.path.exists(path + '/' + name):
        return
    img = o_img.copy()
    bacts, _ = bact_generator.f_generate_bacts(img, 0, preprocess_key)
    for bact in bacts:
        draw_bacteria(img, bact)
    if crop_region is None:
        cv2.imwrite(path + '/' + name, img)
    else:
        cv2.imwrite(path + '/' + name, img[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]])



# For Figure 3
def figure3():
    o_img = cv2.imread("extra_data/raw_inputs/E.coli + 10.bmp")
    # img = cv2.imread("raw_inputs/S.typhi + 14.bmp")
    # img = cv2.imread("raw_inputs/swab-4.bmp")
    PATH = OUTPUT_PATH + '/' + "Figure_3"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    img = roi(o_img)
    img = cv2.cvtColor(o_img, cv2.COLOR_BGR2GRAY)
    _, pimg = cv2.threshold(img, np.median(img.flatten()[img.flatten() != 0]) - 10, 255, cv2.THRESH_BINARY_INV)
    pimg2 = custom_adaptive_threshold(img, 51, 8)
    cv2.imwrite(PATH + '/raw_img.bmp', o_img)
    cv2.imwrite(PATH + '/custom_threshold_img.bmp', pimg2 * 255)
    cv2.imwrite(PATH + '/global_threshold.bmp', pimg)
    pimg3 = bact_generator.preprocess(o_img)
    cv2.imwrite(PATH +'/double_layer_treshold.bmp', pimg3 * 255)

def figure4():
    # o_img = cv2.imread("raw_inputs/E.coli + 9.bmp")
    o_img_ref = cv2.imread("extra_data/raw_inputs/S.typhi + 11.bmp")
    # o_img = cv2.imread("raw_inputs/swab-4.bmp")
    o_img = cv2.imread('extra_data/input_imgs_3/S LOG8-1.bmp')
    PATH = OUTPUT_PATH + '/' + "Figure_4"
    crop_region = ((230, 500), (400, 600))
    THRESH_FUNC = mean_point

    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    
    def preprocess_single(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = bg_normalize_img(img)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=max_point)
        return first_threshold

    def preprocess_double(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = bg_normalize_img(img)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=mean_point)
        first_filtered = cv2.bitwise_and(img, img, mask=first_threshold)
        img = custom_adaptive_threshold(first_filtered, SECOND_BLOCK_SIZE, const = SECOND_BIAS, mode = "foreground", threshold_point=mean_point)
        return img
    
    # bact_generator.max_diameter = 10
    draw_img(preprocess_single, PATH, 'single_reference.bmp', o_img_ref, crop_region)
    draw_img(preprocess_double, PATH, 'double_reference.bmp', o_img_ref, crop_region)
    draw_img(preprocess_single, PATH, 'single.bmp', o_img, crop_region)
    draw_img(preprocess_double, PATH, 'double.bmp', o_img, crop_region)
    # bact_generator.max_diameter = MAX_DIAMETER
    cv2.imwrite(PATH + '/raw.bmp', o_img)
    cv2.imwrite(PATH + '/raw_reference.bmp', o_img_ref)


def bg_norm_discussion():
    # o_img_ref = cv2.imread("extra_data/raw_inputs/E.coli + 5.bmp")
    o_img_ref = cv2.imread("extra_data/raw_inputs/S.typhi + 11.bmp")
    o_img = cv2.imread("extra_data/raw_inputs/swab-4.bmp")
    # crop_region = ((230, 500), (400, 600))
    PATH = OUTPUT_PATH + '/' + "Figure_5"
    THRESH_FUNC = mean_point
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    
    def preprocess_nobgnorm(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=mean_point)
        first_filtered = cv2.bitwise_and(img, img, mask=first_threshold)
        # img = first_threshold
        # img = custom_adaptive_threshold(first_filtered, 11, const=SECOND_BIAS, mode = "foreground", threshold_point=THRESH_FUNC)
        return first_threshold

    def preprocess_with_bgnorm(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = bg_normalize_img(img=img)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=mean_point)
        first_filtered = cv2.bitwise_and(img, img, mask=first_threshold)
        # img = first_threshold
        # img = custom_adaptive_threshold(first_filtered, 11, const=SECOND_BIAS, mode = "foreground", threshold_point=THRESH_FUNC)
        return first_threshold

    draw_img(preprocess_nobgnorm, PATH, 'no_bg_norm.bmp', o_img)
    draw_img(preprocess_with_bgnorm, PATH, 'bg_normed.bmp', o_img)
    draw_img(preprocess_nobgnorm, PATH, 'no_bg_ref.bmp', o_img_ref)
    draw_img(preprocess_with_bgnorm, PATH, 'bg_ref.bmp', o_img_ref)
    cv2.imwrite(PATH + '/raw.bmp', o_img)
    cv2.imwrite(PATH + '/raw_ref.bmp', o_img_ref)

def medianVSmean():
     # o_img = cv2.imread("extra_data/raw_inputs/E.coli + 9.bmp")
    o_img_ref = cv2.imread("extra_data/raw_inputs/S.typhi + 1.bmp")
    # o_img = cv2.imread("extra_data/raw_inputs/swab-4.bmp")
    o_img = cv2.imread('extra_data/raw_inputs/S.typhi + 11.bmp')
    o_img_3 = cv2.imread('extra_data/input_imgs_3/S LOG8-1.bmp')
    PATH = OUTPUT_PATH + '/' + "Median vs. Mean"
    crop_region = ((230, 500), (400, 600))
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    
    def mean_thresh(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=mean_point)
        return first_threshold

    def median_thresh(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=median_point)
        return first_threshold
    
    def max_thresh(orig_img):
        img = orig_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        first_threshold = custom_adaptive_threshold(img, FIRST_BLOCK_SIZE, const = FIRST_BIAS, mode="foreground", threshold_point=max_point)
        return first_threshold

    draw_img(median_thresh, PATH, 'median_thresh.bmp', o_img, crop_region)
    draw_img(mean_thresh, PATH, 'mean_thresh.bmp', o_img, crop_region)
    draw_img(max_thresh, PATH, 'max_thresh.bmp', o_img, crop_region)

    draw_img(median_thresh, PATH, 'median_thresh_reference.bmp', o_img_ref, crop_region)
    draw_img(mean_thresh, PATH, 'mean_thresh_reference.bmp', o_img_ref, crop_region)
    draw_img(max_thresh, PATH, 'max_thresh_reference.bmp', o_img_ref, crop_region)

    draw_img(median_thresh, PATH, 'median_thresh_3.bmp', o_img_3, crop_region)
    draw_img(mean_thresh, PATH, 'mean_thresh_3.bmp', o_img_3, crop_region)
    draw_img(max_thresh, PATH, 'max_thresh_3.bmp', o_img_3, crop_region)

    check_hist = cv2.cvtColor(o_img_3[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]], cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 3, 1)
    plt.hist(check_hist.flatten(), bins=50)
    plt.axvline(x = mean_point(check_hist, FIRST_BIAS))
    plt.title('mean')

    plt.subplot(1, 3, 2)
    plt.hist(check_hist.flatten(), bins=50)
    plt.axvline(x = median_point(check_hist, FIRST_BIAS))
    plt.title('median')

    plt.subplot(1, 3, 3)
    plt.hist(check_hist.flatten(), bins=50)
    plt.axvline(x = max_point(check_hist, FIRST_BIAS))
    plt.title('mode')
    plt.show()

    cv2.imwrite(PATH + '/raw.bmp', o_img)
    cv2.imwrite(PATH + '/raw_reference.bmp', o_img_ref)


def compare_bg_norm():
    PATH = OUTPUT_PATH + '/' + "BGNORM COMPARE"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    oimg1 = cv2.imread("extra_data/raw_inputs/E.coli + 1.bmp")
    # oimg1 = cv2.imread("extra_data/raw_inputs/S.typhi + 11.bmp")
    oimg2 = cv2.imread("extra_data/raw_inputs/swab-2.bmp")

    oimg1 = cv2.cvtColor(oimg1, cv2.COLOR_BGR2GRAY)
    oimg2 = cv2.cvtColor(oimg2, cv2.COLOR_BGR2GRAY)

    img1 = bg_normalize_img(oimg1)
    img2 = bg_normalize_img(oimg2)

    cv2.imwrite(PATH + '/light_raw.bmp', oimg1)
    cv2.imwrite(PATH + '/dark_raw.bmp', oimg2)
    cv2.imwrite(PATH + '/normed_light.bmp', img1)
    cv2.imwrite(PATH + '/normed_dark.bmp', img2)


def test_double_threshold():
    PATH = OUTPUT_PATH + '/' + "test_double_threshold"
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    o_img = cv2.imread('extra_data/input_imgs_3/S LOG8-1.bmp')

    bacts, _ = bact_generator.generate_bacts(o_img, 0)
    normed_img = o_img.copy()
    for bact in bacts:
        draw_bacteria(normed_img, bact)
    cv2.imwrite(PATH + "/double_threshold.bmp", normed_img)


if __name__ == "__main__":
    compare_bg_norm()
    figure3()
    figure4()
    test_double_threshold()
    medianVSmean()
    valX = np.load("extracted_data/testX.npy", allow_pickle=True)
    valX = get_feature_img(valX[:, 0], valX[:, 1]) # Normalize image with background mean
    valY = np.load('extracted_data/testY.npy', allow_pickle=True)
    valX = torch.Tensor(reshape_data(valX))
    valY = torch.Tensor(valY).long()
    model = LeNet5((3, 28, 28), 2)
    model.load_state_dict(torch.load('models/bact_model_50'))
    output = model(valX)
    infos = metric_infos(output, valY)
    print('TP: {}, FP: {}, TN: {}, FN: {}'.format(*infos))