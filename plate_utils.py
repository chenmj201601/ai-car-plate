import numpy as np
import cv2


def plate_split(plate_file):
    img_character_dict = {}
    img_plate = cv2.imread(plate_file)
    img_plate = cv2.cvtColor(img_plate, cv2.COLOR_RGB2GRAY)
    th, binary_plate = cv2.threshold(img_plate, 175, 255, cv2.THRESH_BINARY)
    result = []
    for col in range(binary_plate.shape[1]):
        result.append(0)
        for row in range(binary_plate.shape[0]):
            result[col] = result[col] + binary_plate[row][col] / 255
    character_dict = {}
    num = 0
    i = 0
    while i < len(result):
        if result[i] == 0:
            i += 1
        else:
            index = i + 1
            while result[index] != 0:
                index += 1
            character_dict[num] = [i, index - 1]
            num += 1
            i = index

    for i in range(8):
        if i == 2:
            continue
        padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
        img_character = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]],
                               ((0, 0), (int(padding), int(padding))), 'constant', constant_values=(0, 0))
        img_character = cv2.resize(img_character, (20, 20))
        # cv2.imwrite(str(i) + '.png', img_character)
        img_character_dict[i] = img_character

    return img_character_dict


if __name__ == '__main__':
    img_character_dict = plate_split('test/001.png')
    for key in img_character_dict:
        print("key:{}, img shape:{}".format(key, img_character_dict[key].shape))
