import cv2
import matplotlib.pyplot as plt
import numpy as np

# deprecated

def set_to_freq(input_set):
    return np.bincount(input_set)

def smooth(freq_list):
    conv_result = np.convolve(freq_list, np.ones(10), mode='same')
    return conv_result

def derivative(freq_list, gap_length = 5):
    conv_result = np.convolve(freq_list, [1, *([0] * gap_length), -1], mode = "same")
    return conv_result

def local_maxes(derv):
    results = [0]
    for i in range(1, len(derv)-1):
        if derv[i-1] > 0 and derv[i+1] < 0:
            results.append(i)
    return results

def max_point(img, const):
    freqs = smooth(set_to_freq(img.flatten()))
    return np.argmax(freqs) - const
    # derv = smooth(derivative(freqs))
    # maxes = local_maxes(derv)
    # if len(maxes) == 0:
    #     return 0
    # return max(maxes) - const

def median_point(img, const):
    if (img != 0).sum(0).sum(0) == 0: 
        return 0

    # img_mean = img.sum(0).sum(0) / (img != 0).sum(0).sum(0)
    img_median = np.median(img.flatten()[img.flatten() != 0])
    thresh_value = img_median - const

    return thresh_value

def mean_point(img, const):
    if (img != 0).sum(0).sum(0) == 0: 
        return 0
    
    img_mean = img.sum(0).sum(0) / (img != 0).sum(0).sum(0)
    thresh_value = img_mean - const

    return thresh_value



def custom_adaptive_threshold(gray, block_size, const, mode="foreground", threshold_point = median_point):
    if mode != "foreground" and mode != "background":
        return None
    thresholded = np.zeros_like(gray)
    for x in range(0, gray.shape[0], block_size):
        for y in range(0, gray.shape[1], block_size):
            neighborhood = gray[x:x+block_size, y:y+block_size]
            threshold_value = threshold_point(neighborhood, const)
            if mode == "foreground":
                thresholded[x:x+block_size, y:y+block_size] = np.logical_and(neighborhood <= threshold_value, neighborhood != 0)
            elif mode == "background":
                thresholded[x:x+block_size, y:y+block_size] = np.logical_and(neighborhood > threshold_value, neighborhood != 0)

    return thresholded

# def test():
#     # img = cv2.imread("raw_inputs/E.coli + 4.bmp")
#     # img = cv2.imread("raw_inputs/S.typhi + 1.bmp")
#     img = cv2.imread("raw_inputs/swab-2.bmp")
#     plt.subplot(1, 2, 1)
#     freqs = smooth(set_to_freq(img.flatten()))
#     plt.plot(np.arange(0, len(freqs)), freqs)

#     derv = smooth(derivative(freqs))
#     plt.subplot(1, 2, 2)
#     plt.plot(np.arange(0, len(derv)), derv)

#     maxes = local_maxes(derv)
#     print(maxes)
#     plt.subplot(1, 2, 1)
#     plt.axvline(x = threshold_point(img, 25))
#     plt.show()

#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

#     thresh_img= custom_adaptive_threshold(img, 451, 25)
#     print(img.shape)
#     # _, thresh_img = cv2.threshold(img, threshold_point(img), 255, cv2.THRESH_BINARY_INV)
#     cv2.imshow('', thresh_img)
#     cv2.waitKey(1)
#     input()
#     # plt.imshow(thresh_img)
#     # plt.show()


