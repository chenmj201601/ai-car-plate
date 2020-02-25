import matplotlib.pyplot as plt


# 绘制训练趋势图
def draw_figure(acc, val_acc, loss, val_loss):
    count = len(acc)
    epochs = range(1, count + 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig('car_plate_acc_{}.png'.format(count))

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('car_plate_loss_{}.png'.format(count))
    # plt.show()
