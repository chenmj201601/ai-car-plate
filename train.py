import datetime
import paddle.fluid as fluid
import numpy as np
import argparse
from reader import data_loader
from lenet import LeNet5
from figure_utils import draw_figure
from gen_utils import gen_dataset


def parse_args():
    parser = argparse.ArgumentParser("Training Parameters")
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=20,
        help='the epoch num')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='the batch size')
    args = parser.parse_args()
    return args


args = parse_args()
NUM_EPOCH = args.num_epoch
BATCH_SIZE = args.batch_size


def train(model):
    with fluid.dygraph.guard():
        print('start training ... ')
        begin = datetime.datetime.now()
        model.train()
        epoch_num = NUM_EPOCH
        all_acc = []
        all_val_acc = []
        all_loss = []
        all_val_loss = []
        opt = fluid.optimizer.Momentum(learning_rate=0.001, momentum=0.9)
        train_dataset, valid_dataset, _ = gen_dataset()
        train_loader = data_loader(train_dataset, batch_size=BATCH_SIZE, mode='train')
        valid_loader = data_loader(valid_dataset, batch_size=BATCH_SIZE, mode='valid')
        for epoch in range(epoch_num):
            accuracies = []
            losses = []
            for batch_id, data in enumerate(train_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                logits = model(img)
                cost = fluid.layers.cross_entropy(input=logits, label=label)
                acc = fluid.layers.accuracy(input=logits, label=label)
                avg_cost = fluid.layers.mean(cost)
                if batch_id % 50 == 0:
                    print("epoch: {}, batch_id: {}, acc is {}, loss is: {}".format(epoch,
                                                                                   batch_id,
                                                                                   acc.numpy(),
                                                                                   avg_cost.numpy()))

                # 反向传播，更新权重，清除梯度
                avg_cost.backward()
                opt.minimize(avg_cost)
                model.clear_gradients()
                accuracies.append(acc.numpy())
                losses.append(avg_cost.numpy())

            print("[train] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            all_acc.append(np.mean(accuracies))
            all_loss.append(np.mean(losses))

            # 每5轮过后保存模型参数
            if (epoch % 5 == 0) or (epoch == epoch_num - 1):
                fluid.save_dygraph(model.state_dict(), 'car_plate_{}'.format(epoch))

            model.eval()
            accuracies = []
            losses = []
            for batch_id, data in enumerate(valid_loader()):
                x_data, y_data = data
                img = fluid.dygraph.to_variable(x_data)
                label = fluid.dygraph.to_variable(y_data)
                logits = model(img)
                cost = fluid.layers.cross_entropy(input=logits, label=label)
                acc = fluid.layers.accuracy(input=logits, label=label)
                avg_cost = fluid.layers.mean(cost)
                accuracies.append(acc.numpy())
                losses.append(avg_cost.numpy())

            print("[validation] accuracy/loss: {}/{}".format(np.mean(accuracies), np.mean(losses)))
            all_val_acc.append(np.mean(accuracies))
            all_val_loss.append(np.mean(losses))

            # 绘图，每5轮绘制一个趋势图
            if (epoch % 5 == 0) or (epoch == epoch_num - 1):
                count = len(all_acc)
                if count > 2:
                    sub_all_acc = all_acc[2:]
                    sub_all_val_acc = all_val_acc[2:]
                    sub_all_loss = all_loss[2:]
                    sub_all_val_loss = all_val_loss[2:]
                    draw_figure(sub_all_acc, sub_all_val_acc, sub_all_loss, sub_all_val_loss)

            model.train()

        end = datetime.datetime.now()
        seconds = (end - begin).seconds
        print("finished. total cost {}".format(datetime.timedelta(seconds=seconds)))


if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = LeNet5('LeNet5', num_classes=65)
    train(model)
