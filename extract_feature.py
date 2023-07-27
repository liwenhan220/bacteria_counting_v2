import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm
import math
import pdb

# Included to bound each bateria by a box
class Bounds:
    def __init__(self):
        self.init = False
        self.left = -1
        self.right = -1
        self.top = -1
        self.bottom = -1

    def add(self, coord):
        # coord in form (x, y)
        x, y = coord

        # Init part
        if not self.init:
            self.left = y
            self.right = y
            self.top = x
            self.bottom = x
            self.init = True
        
        # Update boundaries
        else:
            self.left = min(self.left, y)
            self.right = max(self.right, y)
            self.top = min(self.top, x)
            self.bottom = max(self.bottom, x)
    
    # shape of the bound
    def suggest_shape(self):
        return (self.bottom - self.top + 1, self.right - self.left + 1)

    # return informations needed for the bound
    def info(self):
        return (self.left, self.right, self.top, self.bottom)

    # Convert to string (easier to print)
    def __str__(self):
        return "left: {} right: {} top: {} bottom: {}".format(self.left, self.right, self.top, self.bottom)

# Bateria object
class Bacteria:
    def __init__(self):
        self.coords = [] # coordinates of the bacteria
        self.x_coords = [] # X values of the coordinate
        self.y_coords = [] # y values of the coordinate
        self.img = None # img to store bacteria
        self.bounds = Bounds() # box to bound bateria
        self.label = None 
        self.part_img = None
        self.bg_mean = None
        self.part_img_shape = (28, 28)

    def add_coord(self, x, y):
        # add to my lists
        self.coords.append([x, y])
        self.x_coords.append(x)
        self.y_coords.append(y)

        # update bounds
        self.bounds.add((x, y))

    def get_pixel_mean(self):
        return self.pixel_mean

    def retrieve_pixels(self, img):
        self.img = np.zeros((*(self.bounds.suggest_shape()), 3)).astype(np.uint8)
        self.pixels = []

        for x, y in self.coords:
            # Copy pixel
            self.img[x-self.bounds.top][y-self.bounds.left][0] = img[x][y][0]
            self.img[x-self.bounds.top][y-self.bounds.left][1] = img[x][y][1]
            self.img[x-self.bounds.top][y-self.bounds.left][2] = img[x][y][2]
    
    def pad_img(self, desired_shape):
        height, width = self.img_shape()
        desired_height, desired_width = desired_shape
        if desired_width < width or desired_height < height:
            print("Cannot perform padding with desired shape smaller or equal to {}".format(self.img_shape()))
            return -1
        hdiff, wdiff = abs(desired_height - height), abs(desired_width - width)
        self.img = cv2.copyMakeBorder(self.img, top = hdiff // 2, bottom = (hdiff + 1) // 2, left = wdiff // 2, right= (wdiff + 1)//2, borderType=cv2.BORDER_CONSTANT, value=0)
        return 1

    # def get_partial_img(self, img:np.ndarray):
    #     l, r, t, b = self.bounds.info()
    #     self.part_img = img[t:b+1, l:r+1]
    #     return self.part_img
    
    def get_partial_img(self, img:np.ndarray):
        xc, yc = self.get_center()
        x_size, y_size = self.part_img_shape
        t = max(xc - int(x_size // 2), 0)
        l = max(yc - int(y_size // 2), 0)
        b = min(xc + math.ceil(x_size / 2), len(img))
        r = min(yc + math.ceil(y_size / 2), len(img[0]))
        self.part_img = img[t:b+1, l:r+1]
        return self.part_img
    
    def bg_img(self, bias = 3):
        pimg = cv2.cvtColor(self.part_img, cv2.COLOR_BGR2GRAY)

        _, pimg = cv2.threshold(pimg, np.mean(pimg) - bias, 255, cv2.THRESH_BINARY)
        mask = (pimg/255).astype(np.uint8)
        bg_img = cv2.bitwise_and(self.part_img, self.part_img, mask=mask)
        return bg_img
    
    def get_bg_mean(self):
        if self.bg_mean is None:
            b_img = self.bg_img()

            pixel_sum = b_img.sum(0).sum(0)
            count = (b_img!=0).sum(0).sum(0)
            self.bg_mean = pixel_sum / count
        return self.bg_mean
    
    def bg_normalized(self):
        return ((self.img - self.get_bg_mean()) / 255.0).astype(np.float32)
    
    def img_mean(self, img):
        g_img = np.mean(img, axis = 2)
        f_img = g_img.flatten()
        return f_img.sum() / (f_img != 0).sum()
        
    def imshow(self):
        plt.imshow(self.img)
        plt.show()

    def size(self):
        return len(self.x_coords)
    
    def img_shape(self):
        if self.img is not None:
            return (self.img.shape[0], self.img.shape[1])
        else:
            return self.bounds.suggest_shape()
    
    def get_feature(self):
        return self.img

    def is_boundary(self, x, y):
        for i, j in [[1,0], [0,1], [1,1], [1,-1]]:
            for a in [1, -1]:
                xx = x + i * a
                yy = y + j * a
                if [xx, yy] not in self.coords:
                    return True
        return False
    
    def dist(self, pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    
    def get_end_pts(self):
        max_d = 0
        init_pt = self.coords[0]
        max_pt = init_pt
        for pt in self.coords:
            dist = self.dist(pt, init_pt)
            if dist > max_d:
                max_pt = pt
                max_d = dist
        
        max_d = 0
        init_pt = max_pt
        for pt in self.coords:
            dist = self.dist(pt, init_pt)
            if dist > max_d:
                max_pt = pt
                max_d = dist
        return init_pt, max_pt
    
    def get_center(self):
        pt1, pt2 = self.get_end_pts()
        return [int((pt1[0] + pt2[0])/2), int((pt1[1] + pt2[1])/2)]
    
    def est_diameter(self):
        return self.dist(*self.get_end_pts())



#####################
# Frontier object
class Frontier:
    def __init__(self):
        self.record = {}
        self.frontier = []

    def add(self, item):
        self.frontier.append(item)
        self.record[str(item)] = True

    def pop(self):
        item = self.frontier[0]
        self.record[str(item)] = False
        self.frontier = self.frontier[1:]
        return item
    
    def contains(self, item):
        key = str(item)
        return (key in self.record) and self.record[key]

    def isEmpty(self):
        return len(self.frontier) == 0
    

#####################
# Utilities 
    
# largest box to include both shapes
def max_box(shape1, shape2):
    return (max(shape1[0], shape2[0]), max(shape1[1], shape2[1]))

# Put elements in src to tar
def copy_pixel(src, tar):
    for i in range(len(src)):
        tar[i] = int(src[i])

def out_of_bound(img, x):
    return x < 0 or x >= len(img)
                                                               
# add a point to frontier and avoid duplicate points
def add_to_frontier(frontier, successors):
    for successor in successors:
        if not frontier.contains(successor):
            frontier.add(successor)

# To prepare a function that returns non-zero unvisited neighbors of a position on the image
def successor_init(img: np.ndarray):
    def getSuccessor(x:int, y:int, visited):
        successors = []
        for i, j in [[1,0], [0,1]]:
            for d in [1, -1]:
                xx = x + i * d
                yy = y + j * d
                if (xx < 0 or xx >= len(img) or yy < 0 or yy >= len(img[0]) or img[xx][yy] == 0 or visited[xx][yy]):
                    continue
                else:
                    successors.append([xx, yy])
        return successors
    return getSuccessor

def find_bacteria(x: int, y: int, visited: np.ndarray, img: np.ndarray) -> Bacteria:
    if img[x][y] == 0 or (visited[x][y]):
        visited[x][y] = True
        return None
    frontier = Frontier()
    frontier.add([x, y])
    bact = Bacteria()
    get_succ = successor_init(img)
    while not frontier.isEmpty():
        # pop a node
        node = frontier.pop()

        # use node to update rect
        xx, yy = node
        bact.add_coord(xx, yy)

        visited[xx][yy] = True       

        #get successors and push to frontier
        add_to_frontier(frontier, get_succ(xx, yy, visited)) 
    return bact
    
# pass thresholded img to make this work
def find_all_bact(img, size=[0, float('inf')], max_shape = None, image_name = "current image") -> List[Bacteria]:
    visited = np.full(img.shape, False)

    bacts = []

    pbar = tqdm(range(len(img)), total=len(img))
    for i in pbar:
        pbar.set_description('processing {}'.format(image_name))
        for j in range(len(img[i])):
            bact = find_bacteria(i, j, visited, img)
            visited[i][j] = True
            if bact is None or bact.size() < min(size) or bact.size() > max(size):
                continue

            bact_shape = bact.bounds.suggest_shape()
            if max_shape is not None and (bact_shape[0] > max_shape[0] or bact_shape[1] > max_shape[1]):
                continue
            bacts.append(bact)

    return bacts

def cover_upper_left(img, is_threshold, x_cover_range=600, y_cover_range=450):
    color = 0 if is_threshold else [0, 0, 0]
    vertices = np.array([[0,0], [x_cover_range, 0], [0, y_cover_range]])
    cv2.fillPoly(img, pts=[vertices], color=color)
    return img

def cover_upper_right(img, is_threshold, x_cover_range=120, y_cover_range=30):
    color = 0 if is_threshold else [0, 0, 0]
    vertices = np.array([[img.shape[1]-x_cover_range-1, 0], [img.shape[1]-1, y_cover_range], [img.shape[1]-1, 0]])
    cv2.fillPoly(img, pts=[vertices], color=color)
    return img

def roi(img, is_threshold=False):
    cover_upper_left(img, is_threshold)
    cover_upper_right(img, is_threshold)
    return img

def draw_bacteria(img, bact: Bacteria, color=[0, 255, 0]):
    for x, y in bact.coords:
        if bact.is_boundary(x, y):
            for i, j in [[1,0], [0,1], [1,1], [1,-1]]:
                for w in [1, -1]:
                    xx = x + w * i
                    yy = y + w * j
                    if xx < 0 or yy < 0 or xx >= len(img) or yy >= len(img[0]):
                        continue
                    if [xx, yy] not in bact.coords:
                        img[xx][yy] = color
    # img[tuple(bact.get_center())] = [255, 255, 255]
    cv2.putText(img, str(abs(int(bact.img_mean(bact.img) - bact.img_mean(bact.bg_img())))), bact.get_center()[::-1], cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0,0,0],1)
    return img

def invert_img(img):
    new_img = np.ones((img.shape))
    new_img -= img
    return new_img

# Input a grayscaled image only
def bg_normalize_img(img):
    filtered = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 15, 0)
    masked = cv2.bitwise_and(img, img, mask=filtered)
    filtered_mean = masked.sum() / filtered.sum()

    normalized = (img / filtered_mean).astype(np.float32)

    nmax = np.max(normalized)
    nmin = np.min(normalized)

    new_max = 255.0
    new_min = 0.0

    final = (normalized - nmin) * ((new_max - new_min) / (nmax - nmin)) + new_min
    return final.astype(np.uint8)

def preprocess(orig_img, cover_corners=True, bias = 3):
    img = orig_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = bg_normalize_img(img=img)
    # plt.hist(img.flatten(), bins=20)
    # plt.show()
    img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 15, bias)
    img = invert_img(img)
    if cover_corners:
        img = roi(img, is_threshold=True)
    return img
    
def generate_bacts(img, label, debug=False, bias = 3, size = [10, 200], debug_path = "for_debug.bmp", image_name = "current image", cover_corners=True, threshold = 2.0, max_diameter = 15):
    processed_img = preprocess(img, cover_corners=cover_corners, bias= bias)
    if debug:
        debug_img = img.copy()
    bacts = find_all_bact(processed_img, size=size, image_name=image_name)
    bact_count = 0
    max_shape = (1, 1)
    final_bacts = []
    for bact in bacts:
        bact.retrieve_pixels(img)
        bact.get_partial_img(img)
        bact.get_bg_mean()
        bact.label = label

        diff = abs(bact.img_mean(bact.img) - bact.img_mean(bact.bg_img()))
        if diff < threshold:
            continue

        if bact.est_diameter() >= max_diameter:
            continue

        if debug:
            draw_bacteria(debug_img, bact, [0, 255, 0])
            bact_count += 1

        final_bacts.append(bact)
        max_shape = max_box(max_shape, bact.img_shape())
    if debug:
        cv2.imwrite(debug_path, debug_img)
    return final_bacts, max_shape
        
# Test background

def test_background():
    img = cv2.imread("-3.bmp")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = bg_normalize_img(img)
    print(img)
    plt.imshow(img)
    plt.show()

# test_background()