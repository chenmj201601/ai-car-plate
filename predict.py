import numpy as np
import paddle.fluid as fluid
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from plate_utils import plate_split
from lenet import LeNet5
from gen_utils import gen_dataset


def parse_args():
    parser = argparse.ArgumentParser("Prediction Parameters")
    parser.add_argument(
        '--weight_file',
        type=str,
        default='car_plate_10',
        help='the path of model parameters')
    parser.add_argument(
        '--test_plate',
        type=str,
        default='test/001.png',
        help='the test plate')
    args = parser.parse_args()
    return args


args = parse_args()
WEIGHT_FILE = args.weight_file
TEST_PLATE = args.test_plate


def predict(model, params_file_path, test_plate):
    with fluid.dygraph.guard():
        model_state_dict, _ = fluid.load_dygraph(params_file_path)
        model.load_dict(model_state_dict)
        model.eval()
        _, _, character_mapper = gen_dataset()
        img_character_dict = plate_split(test_plate)
        plate = []
        for key in img_character_dict:
            img = img_character_dict[key]
            img = img.flatten().astype('float32') / 255.0
            img = img.reshape(1, 1, 20, 20)
            img = fluid.dygraph.to_variable(img)
            logits = model(img)
            result = fluid.layers.softmax(logits).numpy()
            result = np.argmax(result[0])
            result = character_mapper.get(result, '')
            plate.append(result)
            print("key:{}, result:{}".format(key, result))
        plt.imshow(Image.open(test_plate))
        print("\n车牌识别结果为：", end='')
        for i in range(len(plate)):
            print(plate[i], end='')
        print("\n")
        plt.show()


if __name__ == '__main__':
    test_plate = TEST_PLATE
    params_file_path = WEIGHT_FILE
    with fluid.dygraph.guard():
        model = LeNet5('LeNet5', num_classes=65)
    predict(model, params_file_path, test_plate)
