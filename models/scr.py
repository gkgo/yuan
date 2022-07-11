import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple
import numpy as np
from torch.nn import init
from torch import nn
import torch


class SCR(nn.Module):
    def __init__(self, planes=[640, 64, 64, 64, 640], stride=(1, 1, 1), ksize=3, do_padding=False, bias=False):
        super(SCR, self).__init__()
        self.ksize = _quadruple(ksize) if isinstance(ksize, int) else ksize
        padding1 = (0, self.ksize[2] // 2, self.ksize[3] // 2) if do_padding else (0, 0, 0)

        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv3d(planes[1], planes[2], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[2]),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv3d(planes[2], planes[3], (1, self.ksize[2], self.ksize[3]),
                                             stride=stride, bias=bias, padding=padding1),
                                   nn.BatchNorm3d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))

    def forward(self, x):
        b, c, h, w, u, v = x.shape
        x = x.view(b, c, h * w, u * v)

        x = self.conv1x1_in(x)   # [80, 640, hw, 25] -> [80, 64, HW, 25]

        c = x.shape[1]
        x = x.view(b, c, h * w, u, v)
        x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        c = x.shape[1]
        x = x.view(b, c, h, w)
        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        return x


class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size=(5, 5), padding=2):
        super(SelfCorrelationComputation, self).__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)
        identity = x

        x = self.unfold(x)  # b, cuv, h, w  # b,16000,25
        x = x.view(b, c, self.kernel_size[0], self.kernel_size[1], h, w)
        x = x * identity.unsqueeze(2).unsqueeze(2)  # b, c, u, v, h, w * b, c, 1, 1, h, w
        x = x.permute(0, 1, 4, 5, 2, 3).contiguous()  # b, c, h, w, u, v
        return x


class SelfCorrelationComputation1(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfCorrelationComputation1, self).__init__()

        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)



        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self,x, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        x = x.permute(0, 2, 3, 1).contiguous()  # （10，5，5，640）
        B, H, W, C = x.shape
        N = H * W
        x = x.view(B,N,C)
        queries = x
        keys = x
        values = x
        b_s, nq = queries.shape[:2]   # (50,49)
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k) (50,8,49,64)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)  (50,8,64,49)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v) (50,8,49,64)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)   #(50,8,49,49) # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v) (50,49,512)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        out = out.view(B,C,H,W)
        return out
