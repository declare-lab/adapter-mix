import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

from transformers.adapters import AdapterConfig
from transformers.adapters.modeling import Adapter

class PrefixEncoder(nn.Module):  
	#code from P-tuning-v2
	#https://github.com/THUDM/P-tuning-v2/blob/main/model/prefix_encoder.py
	r'''
	The torch.nn model to encode the prefix

	Input shape: (batch-size, prefix-length)

	Output shape: (batch-size, prefix-length, 2*layers*hidden)
	'''
	def __init__(self, config, num_hidden_layers, hidden_size):
		super().__init__()
		self.prefix_projection = config["adapters"]['prefix_tuning']["prefix_projection"]
		if self.prefix_projection:
			# Use a two-layer MLP to encode the prefix
			self.embedding = torch.nn.Embedding(config.prefix_seq_len, config.hidden_size)
			self.trans = torch.nn.Sequential(
				torch.nn.Linear(config.hidden_size, config.prefix_hidden_size),
				torch.nn.Tanh(),
				torch.nn.Linear(config.prefix_hidden_size, config.num_hidden_layers * 2 * config.hidden_size)
			)
		else:
			# self.embedding = torch.nn.Embedding(config["adapter"]["prefix_seq_len"], config.num_hidden_layers * 2 * config.hidden_size)
			self.embedding = torch.nn.Embedding(config["adapters"]['prefix_tuning']["prefix_seq_len"], num_hidden_layers * 2 * hidden_size)

	def forward(self, prefix: torch.Tensor):
		if self.prefix_projection:
			prefix_tokens = self.embedding(prefix)
			past_key_values = self.trans(prefix_tokens)
		else:
			prefix = prefix.to(self.embedding.weight.device)
			past_key_values = self.embedding(prefix)
		return past_key_values

class SELayer(nn.Module):
	def __init__(self, channel, reduction=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool1d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid()
		)

	def forward(self, x):
		resdiual = x
		b, c, _= x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1)
		# return resdiual + x * y.expand_as(x)
		return x * y.expand_as(x)

class AdapterBlock(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, bias=False):
		super(AdapterBlock, self).__init__()
		self.layer_norm1 = nn.LayerNorm(in_dim)
		self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, bias=bias,groups=out_dim, padding='same')
		self.relu1 = nn.ReLU(inplace=True)
		# self.se1 = SELayer(out_dim)
		self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=stride, bias=False, groups=out_dim, padding='same')
		# self.se2 = SELayer(out_dim)
		self.conv3 = nn.Conv1d(out_dim, in_dim, kernel_size=3, stride=stride, bias=bias,groups=out_dim, padding='same')
		# self.relu2 = nn.ReLU(inplace=True)
		self.se3 = SELayer(in_dim)
		# self.layer_norm2 = nn.LayerNorm(out_dim)
		# self.dropout = nn.Dropout(p=0.1)
	def forward(self, x, residual_input):
		out = self.layer_norm1(x)
		out = torch.transpose(out,-1,-2)
		out = self.conv1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.se3(out)
		# out = self.dropout(out)
		out = torch.transpose(out,-1,-2)
		out = residual_input + out   #skip connection
		return out

class AdapterBlock_(nn.Module):
	def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, bias=False):
		super(AdapterBlock_, self).__init__()
		self.layer_norm1 = nn.LayerNorm(256)
		self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
		self.relu1 = nn.ReLU(inplace=True)
		# self.se1 = SELayer(out_dim)
		self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=5, stride=stride, bias=False, groups=out_dim, padding='same')
		# self.se2 = SELayer(out_dim)
		self.conv3 = nn.Conv1d(out_dim, in_dim, kernel_size=3, stride=stride, bias=bias, padding='same')
		# self.relu2 = nn.ReLU(inplace=True)
		self.se3 = SELayer(out_dim)
		self.layer_norm2 = nn.LayerNorm(out_dim)
	def forward(self, x, residual_input):
		out = self.layer_norm1(x)
		out = self.conv1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.conv3(out)
		out = self.se3(out)
		out = residual_input + out   #skip connection
		return out


class BottleneckAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(BottleneckAdapter, self).__init__()
		self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
		self.bottleneck_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		output, down, up = self.bottleneck_adapter(x, residual_input)
		return output