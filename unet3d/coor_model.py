import torch
import torch.nn as nn

FIRST_CONV_DIM = 32
SEC_CONV_DIM = 32

class CoorNet(nn.Module):
	"""
	A simple CNN used to learn the coordinate that corresponds to the centre
	of the PCS in the given scan.
	"""

	def __init__(self, in_channels):
		super(CoorNet, self).__init__()
		self.in_channels = in_channels

		self.conv_layer = nn.Sequential(
			nn.Conv3d(in_channels, FIRST_CONV_DIM, 3),
			nn.BatchNorm3d(FIRST_CONV_DIM),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(FIRST_CONV_DIM, SEC_CONV_DIM, 3),
			nn.BatchNorm3d(SEC_CONV_DIM),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=2, stride=2),

			nn.Conv3d(SEC_CONV_DIM, SEC_CONV_DIM, 3),
			nn.BatchNorm3d(SEC_CONV_DIM),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=2, stride=2)
		)

		self.fc_layer = nn.Sequential(
			nn.Dropout(p=0.1),
			nn.Linear(22*26*22*SEC_CONV_DIM, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.1),
			nn.Linear(128, 3)
		)

	def forward(self, x):
		x = self.conv_layer(x)
		x = x.view(x.size(0), -1)
		x = self.fc_layer(x)
		return x

