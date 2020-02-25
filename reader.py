import numpy as np
import cv2
from gen_utils import gen_dataset


def img_mapper(sample):
    img, label = sample
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 0)
    img = img.flatten().astype('float32') / 255.0
    img = img.reshape(1, 20, 20)
    return img, label


def data_loader(img_list, batch_size=100, mode='train'):
    def reader():
        if mode == 'train':
            np.random.shuffle(img_list)

        batch_imgs = []
        batch_labels = []
        for info in img_list:
            path, label = info
            img, label = img_mapper((path, label))
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                img_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
                yield img_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            img_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('int64').reshape(-1, 1)
            yield img_array, labels_array

    return reader


if __name__ == '__main__':
    train_dataset, _, _ = gen_dataset()
    train_loader = data_loader(train_dataset)
    train_img, train_label = next(train_loader())
    print(train_img.shape)
    print(train_label.shape)
