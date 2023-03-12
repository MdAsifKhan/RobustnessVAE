import torch
import torch.nn as nn



class LinearLayer(nn.Module):
	def __init__(self, in_dim, out_dim, norm=False, activation='lrelu'):
		super(LinearLayer, self).__init__()
	
		self.affine = nn.Linear(in_dim, out_dim)
		if norm:
			self.norm = nn.BatchNorm1d(out_dim)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.affine(x)
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out


class ConvLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride, norm=False, activation='lrelu', pad=0, bias=True):
		super(ConvLayer, self).__init__()	
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, bias=bias)
		if norm:
			self.norm = nn.BatchNorm2d(out_channels)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.conv(x)
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out


class ConvTransposeLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, stride, norm=False, activation='lrelu', pad=0, outpad=0, bias=True):
		super(ConvTransposeLayer, self).__init__()
		self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=pad, output_padding=outpad, bias=bias)
		if norm:
			self.norm = nn.BatchNorm2d(out_channels)
		else:
			self.norm = None

		if activation=='relu':
			self.activation = nn.ReLU(inplace=True)
		elif activation=='lrelu':
			self.activation = nn.LeakyReLU(0.2, inplace=True)
		elif activation=='tanh':
			self.activation = nn.Tanh()
		elif activation=='sigmoid':
			self.activation = nn.Sigmoid()
		elif activation=='selu':
			self.activation = nn.SELU(inplace=True)
		elif activation == 'none':
			self.activation = None
		else:
			assert 0,'Unsupported activation {}'.format(activation)

	def forward(self, x):
		out = self.convt(x)
		if self.norm:
			out = self.norm(out)
		if self.activation:
			out = self.activation(out)
		return out

class ResLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel, norm=False, activation='relu', pad_type='zero'):
		super(ResLayer, self).__init__()
		res = []
		res.append(ConvLayer(in_channels ,out_channels, kernel, 1, 1, norm=norm, activation=activation, pad_type=pad_type))
		res.append(ConvLayer(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type))
		self.res = nn.Sequential(*res)

	def forward(self, x):
		inp = x
		out = self.res(x)
		out += inp
		return out

class Reshape(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape

	def forward(self, x):
		return x.view(*self.shape)


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()

from torch.nn.utils import weight_norm

class ChannelNorm(nn.Module):
	def __init__(self):
		super(ChannelNorm, self).__init__()

	def forward(self, x):
		max_vals, _ = torch.max(torch.abs(x), 2, keepdim=True)
		max_vals  = max_vals + 1e-5
		x = x / max_vals
		return x

class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()
		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
										stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.LeakyReLU(0.2, inplace=True)
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
										stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.LeakyReLU(0.2, inplace=True)
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
									self.conv2, self.chomp2, self.relu2, self.dropout2)
		
		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
		
		self.relu = nn.LeakyReLU(0.2, inplace=True)
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)
		return self.relu(out + res)	


