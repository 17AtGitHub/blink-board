import cv2
import numpy as np

keyboard = np.zeros((600, 1000, 3), np.uint8)

key_set_1 = {0: 'Q', 1: 'W', 2: 'E', 3: 'R', 4: 'T',
             5: 'A', 6: 'S', 7: 'D', 8: 'F', 9: 'G',
             10: 'Z', 11: 'X', 12: 'C', 13: 'V', 14: 'B'}
def letter(letter_index, text, light):
    # keys
    # determining x and y coordinates of each key's top-left corner
    x = int(letter_index % 5) * 200
    y = int(letter_index / 5) * 200
    
    width = 200
    height = 200
    th = 3  # thickness
    if light:
        cv2.rectangle(keyboard, (x+th, y+th), (x+width-th, y+height-th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keyboard, (x+th, y+th), (x+width-th, y+height-th), (255, 0, 0), 2)    

    # text settings:
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_sz, _ = cv2.getTextSize(text, font_letter, font_scale, font_th)
    text_width, text_height = text_sz[0], text_sz[1]
    text_x = int((width - text_width)/2) + x
    text_y = int((height + text_height)/2) + y
    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255,0,0), font_th)

for i in range(15):
    if i == 5:
        light = True
    else: 
        light = False
    letter(i, key_set_1[i], light)

cv2.imshow('KB', keyboard)
cv2.waitKey(0)
cv2.destroyAllWindows()
