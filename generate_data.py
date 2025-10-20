from extract_feature import *
import os
from hyperparameters import *

directories = ["extracted_data", "debug"]

# returns a list of images
def get_img(bact:Bacteria):
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
bacteria_generator = BacteriaGenerator(SIZE, MAX_DIAMETER, True, True)

if __name__ == "__main__":
    # load ecoli data
    for i in range(1, 11):
        name = "E.coli + {}.bmp".format(i)
        input_img_path = raw_folder + ecoli + name
        img = cv2.imread(input_img_path)
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                         0, 
                                                         image_name = name, 
                                                         debug_path = directories[1])
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)

    # load styphi data
    for i in range(1, 17):
        name = "S.typhi + {}.bmp".format(i)
        input_img_path = raw_folder + styphi + name
        img = cv2.imread(input_img_path)
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                         0, 
                                                         debug_path = directories[1], 
                                                         image_name=name)
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)

    bacteria_generator.cover_corners = False
    # load negative data
    for i in range(1, 12):
        name = "swab-{}.bmp".format(i)
        input_img_path = raw_folder + negative + "swab-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                         1, 
                                                         image_name = name, 
                                                         debug_path = directories[1])

        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)

    # load new negative data part 1
    for i in range(1, 5):
        name = "-{}.bmp".format(i)
        input_img_path = raw_folder + new_neg + "-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                         1, 
                                                         image_name = name, 
                                                         debug_path = directories[1])
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)

    # load new negative data part 2
    for i in range(1, 5):
        name = "-small swab5-{}.bmp".format(i)
        input_img_path = raw_folder + new_neg + "-small swab5-{}.bmp".format(i)
        img = cv2.imread(input_img_path)
        bacts, shape = bacteria_generator.generate_bacts(img, 
                                                         1, 
                                                         image_name = name, 
                                                         debug_path = directories[1])
        if i <= test_split_index:
            test_bacts += bacts
        else:
            train_bacts += bacts
        max_shape = max_box(max_shape, shape)


    X = []
    y = []
    pbar = tqdm(train_bacts, total=len(train_bacts))
    for bact in pbar:
        pbar.set_description('saving training data')
        bact.pad_img(max_shape)
        bact_imgs = get_img(bact)
        X.append(bact_imgs)
        y.append(bact.label)

    np.save(directories[0] + "/trainX.npy", np.array(X, dtype=object))
    np.save(directories[0] + "/trainY.npy", y)

    testX = []
    testY = []
    
    pbar = tqdm(test_bacts, total = len(test_bacts))
    for bact in pbar:
        pbar.set_description('saving validation data')
        bact.pad_img(max_shape)
        img = get_img(bact)
        testX.append(img)
        testY.append(bact.label)

    np.save(directories[0] + "/testX.npy", np.array(testX, dtype=object))
    np.save(directories[0] + "/testY.npy", testY)
