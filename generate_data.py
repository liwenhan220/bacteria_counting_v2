from extract_feature import *
import os

BIAS = 5
MAX_DIAMETER = 10
THRESHOLD = 2.8
SWAB_THRESH = THRESHOLD
SIZE = [10, 100]

directories = ["extracted_data", "debug"]

def get_img(bact:Bacteria):
    # return (bact.img / 255.0).astype(np.float32)
    return bact.bg_normalized()

for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)

raw_folder = "raw_data/"

ecoli = "Ecoli-positive/"
styphi = "Styphi-positive/"
negative = "negative/"
new_neg = "new negative/"

test_bacts = []
train_bacts = []
max_shape = (28, 28) # images must be at least 28x28
test_split_index = 1

def draw_bact_hist(bacts, enabled = False):
    if enabled:
        plt.hist(np.mean(get_img(bacts[np.random.randint(0, len(bacts))]), axis=2).flatten())
        plt.show()

if __name__ == "__main__":
    # load ecoli data
    for i in range(1, 11):
        name = "E.coli + {}.bmp".format(i)
        input_img_path = raw_folder + ecoli + name
        img = cv2.imread(input_img_path)
        bacts, shape = generate_bacts(img, 0, bias = BIAS, size = SIZE, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = THRESHOLD, max_diameter=MAX_DIAMETER)
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)
        draw_bact_hist(bacts)

    # load styphi data
    for i in range(1, 17):
        name = "S.typhi + {}.bmp".format(i)
        input_img_path = raw_folder + styphi + name
        img = cv2.imread(input_img_path)
        bacts, shape = generate_bacts(img, 0, bias = BIAS, size = SIZE, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = THRESHOLD,  max_diameter=MAX_DIAMETER)
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)
        draw_bact_hist(bacts)

    # load negative data
    for i in range(1, 12):
        name = "swab-{}.bmp".format(i)
        input_img_path = raw_folder + negative + "swab-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        if i > 1:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)
        else:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)

        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)
        draw_bact_hist(bacts)

    # load new negative data part 1
    for i in range(1, 5):
        name = "-{}.bmp".format(i)
        input_img_path = raw_folder + new_neg + "-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        if i > 1:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)
        else:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)

        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)
        draw_bact_hist(bacts)

    # load new negative data part 2
    for i in range(1, 5):
        name = "-small swab5-{}.bmp".format(i)
        input_img_path = raw_folder + new_neg + "-small swab5-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        if i > 1:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)
        else:
            bacts, shape = generate_bacts(img, 1, bias = BIAS, size = SIZE, cover_corners= False, debug = True, debug_path = directories[1] + '/' + name, image_name=name, threshold = SWAB_THRESH, max_diameter=MAX_DIAMETER)

        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)
        draw_bact_hist(bacts)


    X = []
    y = []
    for bact in train_bacts:
        bact.pad_img(max_shape)
        X.append(get_img(bact))
        y.append(bact.label)

    np.save(directories[0] + "/trainX.npy", X)
    np.save(directories[0] + "/trainY.npy", y)

    testX = []
    testY = []

    for bact in test_bacts:
        bact.pad_img(max_shape)
        testX.append(get_img(bact))
        testY.append(bact.label)

    np.save(directories[0] + "/testX.npy", testX)
    np.save(directories[0] + "/testY.npy", testY)
