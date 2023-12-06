import pygame
from label_img import label_image
import cv2

def flip_channel(image_arr):
    results = []
    for img in image_arr:
        results.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return results

def print_bact_info(bact_count, noise_count):
    print('bacteria count: {}, noise count: {}'.format(bact_count, noise_count))
    print('Usage: press \'a\' to check all dots, press \'b\' to check bacteria dots')

IM_SHAPE = 1000, 800
COVER_CORNERS = False
MODEL_NUM = 49

screen = pygame.display.set_mode(IM_SHAPE)
pygame.display.set_caption('whatev')

running = True
for_display = None
blit = True
next_o = False
prev_b = False
while running:
    for event in pygame.event.get(pygame.DROPFILE):
        image_path = event.file
        image = cv2.imread(image_path)
        bact_img, all_img, bact_count, noise_count = label_image('models', 9, image, image_name=image_path)
        bact_img, all_img, image = flip_channel([bact_img, all_img, image])
        print_bact_info(bact_count, noise_count)
        blit = True
        for_display = all_img

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_b:
                if not next_o or not prev_b:
                    for_display = bact_img
                    next_o = True
                    prev_b = True
                else:
                    for_display = image
                    next_o = False 
                    prev_b = True

                blit = True
            elif event.key == pygame.K_a:
                if not next_o or prev_b:
                    for_display = all_img
                    next_o = True
                    prev_b = False
                else:
                    for_display = image
                    next_o = False
                    prev_b = False

                blit = True
        if blit and for_display is None:
            screen.fill((0,0,0))
            blit = False
        elif blit:
            # display the image
            surf = pygame.surfarray.make_surface(for_display)
            surf = pygame.transform.rotate(surf, 90)
            surf = pygame.transform.flip(surf, False, True)
            surf = pygame.transform.scale(surf, IM_SHAPE)
            screen.blit(surf, (0, 0))
            blit = False

        pygame.display.update()






