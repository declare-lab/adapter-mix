import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math

from utils.tools import make_positions
from transformers.adapters.modeling import Adapter
from transformers.adapters import AdapterConfig, CompacterConfig, PrefixTuningConfig


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table """

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array(
        [get_posi_angle_vec(pos_i) for pos_i in range(n_position)]
    )

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.0

    return torch.FloatTensor(sinusoid_table)


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None, positions=None, **kwargs):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.shape[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = make_positions(input, self.padding_idx) if positions is None else positions
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs):
        return inputs * inputs.sigmoid()


class GLU(nn.Module):
    """
    The gating mechanism is called Gated Linear Units (GLU), which was first introduced for natural language processing
    in the paper “Language Modeling with Gated Convolutional Networks”
    """
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        outputs, gate = inputs.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class LinearNorm(nn.Module):
    """ LinearNorm Projection """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x):
        x = self.linear(x)
        return x


class ConvBlock(nn.Module):
    """ 1D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=None, normalization=nn.BatchNorm1d, activation=nn.ReLU, transpose=False):
        super(ConvBlock, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=int((kernel_size - 1) / 2),
                dilation=1,
                w_init_gain="tanh",
                transpose=transpose
            ),
            normalization(out_channels),
            activation(),
        )
        self.dropout = dropout if dropout is not None else None
        self.transpose = transpose

    def forward(self, enc_input, mask=None):
        if not self.transpose:
            enc_input = enc_input.contiguous().transpose(1, 2)
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, training=True) # self.training)

        if not self.transpose:
            enc_output = enc_output.contiguous().transpose(1, 2)
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output


class ConvBlock2D(nn.Module):
    """ 2D Convolutional Block """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=None, normalization=nn.BatchNorm2d, activation=nn.ReLU, transpose=False):
        super(ConvBlock2D, self).__init__()

        self.conv_layer = nn.Sequential(
            ConvNorm2D(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, int((kernel_size - 1) / 2)),
                bias=False,
                w_init_gain="tanh",
                transpose=transpose,
            ),
            normalization(out_channels),
            activation(),
        )
        self.dropout = dropout if dropout is not None else None
        self.transpose = transpose

    def forward(self, enc_input, mask=None):
        """
        enc_input -- [B, H, W, C_in]
        mask -- [B, H]
        """
        if not self.transpose:
            enc_input = enc_input.contiguous().permute(0, 3, 1, 2) # [B, C_in, H, W]
        enc_output = self.conv_layer(enc_input)
        if self.dropout is not None:
            enc_output = F.dropout(enc_output, self.dropout, self.training)

        if not self.transpose:
            enc_output = enc_output.contiguous().permute(0, 2, 3, 1) # [B, H, W, C_out]
        if mask is not None:
            enc_output = enc_output.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)

        return enc_output


class ConvNorm(nn.Module):
    """ 1D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        if self.transpose:
            x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().transpose(1, 2)

        return x


class ConvNorm2D(nn.Module):
    """ 2D Convolution """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        transpose=False,
    ):
        super(ConvNorm2D, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )
        self.transpose = transpose

    def forward(self, x):
        """
        x -- [B, H, W, C] or [B, C, H, W]
        """
        if self.transpose:
            x = x.contiguous().permute(0, 3, 1, 2) # [B, C, H, W]
        x = self.conv(x)
        if self.transpose:
            x = x.contiguous().permute(0, 2, 3, 1) # [B, H, W, C]

        return x
    

class Condional_LayerNorm(nn.Module):

    def __init__(self,
                normal_shape,
                epsilon=1e-5
                ):
        super(Condional_LayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            self.normal_shape = normal_shape
        self.speaker_embedding_dim = 256
        self.epsilon = epsilon
        self.W_scale = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.W_bias = nn.Linear(self.speaker_embedding_dim, self.normal_shape)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)
    
    def forward(self, x, speaker_embedding):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= scale.unsqueeze(1)
        y += bias.unsqueeze(1)

        return y
    
class ResidualAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(ResidualAdapter, self).__init__()
		self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
		self.residual_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		output, down, up = self.residual_adapter(x, residual_input)
		return output

class MOA(nn.Module):
    def __init__(self,
                 number_of_adapters,
                 d_in,
                 c,
                 r):
        super(MOA,self).__init__()
        self.number_of_adapters = number_of_adapters
        self.d_in = d_in
        self.c = c
        self.r = r
        self.W_g = nn.Linear(self.d_in,self.number_of_adapters)
        self.adapters = nn.ModuleList([ResidualAdapter("residual_adapter",self.d_in,self.r) for i in range(number_of_adapters)])
        
    def forward(self,x,residual):
        _,n,_ = x.shape
        # x --> B,T,d
#        for i in range(2):
        X = self.W_g(x) # X----> B,T,e
        S = nn.Softmax(X) # -----> B,T,e
        # Computer Top k tokens
        k = int((n*self.c)/self.number_of_adapters)
        G,I = torch.topk(torch.transpose(S.dim,1,2),k)# G ---> B,e,k
        P = F.one_hot(I,num_classes=n) #  P ---> B,e,k,T
        X_in = torch.einsum('bijk,bkl->bijl',P.float(),x)
        X_in = torch.transpose(X_in,0,1)
        X_e = torch.zeros(X_in.shape,device=X_in.device)
        for i,layer in enumerate(self.adapters):
            X_e[i] =  layer(X_in[i],X_in[i])
        X_e = torch.transpose(X_e,0,1)
        X_out = torch.einsum('bijl,bij,bijd->bld',P.float(),G,X_e)
#            x = X_out
        X_out = X_out + residual   
        return X_out
        
        
        
        
        
