import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC


class LeNet5(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_classes):
        super(LeNet5, self).__init__(name_scope)
        self.conv1 = Conv2D(self.full_name(), num_filters=50, filter_size=5, stride=1)
        self.pool1 = Pool2D(self.full_name(), pool_size=2, pool_stride=1, pool_type='max')
        self.conv2 = Conv2D(self.full_name(), num_filters=32, filter_size=3, stride=1)
        self.pool2 = Pool2D(self.full_name(), pool_size=2, pool_stride=1, pool_type='max')
        self.fc1 = FC(self.full_name(), size=num_classes, act='softmax')

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.fc1(x)
        return x
