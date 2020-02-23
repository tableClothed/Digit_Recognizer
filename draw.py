import pygame
import tensorflow as tf
import cv2
import numpy as np
from keras.preprocessing import image

pygame.init()

window = pygame.display.set_mode((300, 300))
pygame.display.set_caption("Digit")
window.fill((255, 255, 255))

draw_on = False
last_pos = (0, 0)
black = (0, 0, 0)
radius = 10

clock = pygame.time.Clock()
crashed = False

model = tf.keras.models.load_model("MNIST.model")

def drawing(window, curr_pos, last_pos):
    dx = last_pos[0] - curr_pos[0]
    dy = last_pos[1] - curr_pos[1]
    dis = max(abs(dx), abs(dy))
    for i in range(dis):
        x = int(curr_pos[0] + float(i)/dis * dx)
        y = int(curr_pos[1] + float(i)/dis * dy)
        pygame.draw.circle(window, black, (x, y), radius) 

def guess():
    try:
        # img = image.load_img('digit.jpg', target_size=(28, 28))
        # img = image.img_to_array(img)                    # (height, width, channels)
        # img = img.reshape((1,)+ img.shape)
        # img = img.reshape(-1, 784)
        pygame.image.save(window, "digit.jpg")
        img = cv2.imread('digit.jpg', 0)
        img = cv2.bitwise_not(img)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, [1, 28, 28])
        print(img)
        predictions = model.predict(img)
        print(predictions)
        i = np.argmax(predictions)
        print("This number is probably a ", i)
    except Exception as e:
        print(e)


while not crashed:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            crashed = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw_on = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw_on = False
            guess()
            window.fill((255, 255, 255))
        if event.type == pygame.MOUSEMOTION:
            if draw_on:
                pygame.draw.circle(window, black, event.pos, radius)
                drawing(window, event.pos, last_pos)
            last_pos = event.pos
    
    pygame.display.update()
    clock.tick(60)
