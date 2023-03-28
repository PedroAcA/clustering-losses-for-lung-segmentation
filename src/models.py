from torch import nn
from collections import OrderedDict

# Pytorch implementation of recurrent CNNs as described in Recurrent Convolutional Neural Network for Object Recognition


class RCNN(nn.Module):
	def __init__(self, ch_in, ch_out, time_steps, forward_kernel_size, activation_fn, normalization_fn, forward_stride=1, recurrent_kernel_size=None, recurrent_stride=None):
		super(RCNN, self).__init__()
		self.time_steps = max(time_steps, 0)
		rec_kernel = forward_kernel_size if not recurrent_kernel_size else recurrent_kernel_size
		rec_stride = forward_stride if not recurrent_stride else recurrent_stride
		self.fwd_conv = nn.Conv2d(ch_in, ch_out, forward_kernel_size, forward_stride, padding='same')
		self.rec_conv = nn.Conv2d(ch_out, ch_out, rec_kernel, rec_stride, padding='same')
		self.activation = activation_fn
		self.normalization = normalization_fn

	def forward(self, x):
		y = self.activation(self.normalization(self.fwd_conv(x)))
		for i in range(self.time_steps):
			y = self.activation(self.normalization(self.fwd_conv(x) + self.rec_conv(y)))

		return y


class ConvNet(nn.Module):
	def __init__(self, config):
		super(ConvNet, self).__init__()
		self.layers = OrderedDict()
		self.device = config.SYSTEM.DEVICE
		self.layers["rcnn_0"] = RCNN(1, 64, config.MODEL.TIME_STEPS, (3, 3), nn.ReLU(), nn.BatchNorm2d(64))
		for i in range(4):
			self.layers["rcnn_" + str(i+1)] = RCNN(64, 64, config.MODEL.TIME_STEPS, (3, 3), nn.ReLU(), nn.BatchNorm2d(64))

		for i in range(2):
			self.layers["cnn_" + str(i)] = nn.Conv2d(64, 64, (3, 3), 1, padding='same')
			self.layers["bn_" + str(i)] = nn.BatchNorm2d(64)
			self.layers["activation_" + str(i)] = nn.ReLU()

		self.layers["final_cnn"] = nn.Conv2d(64, config.MODEL.CLASSES, (3, 3), 1, padding='same')
		self.layers["final_bn"] = nn.BatchNorm2d(config.MODEL.CLASSES)
		self.layers["final_activation"] = nn.ReLU()
		self.layers['out'] = nn.Softmax(dim=1)
		self.model = nn.Sequential(self.layers)

	def forward(self, x):
		return self.model(x)





