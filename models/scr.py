import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.functional import unfold, pad
import warnings
from timm.models.layers import trunc_normal_, DropPath



class SelfCorrelationComputation(nn.Module):
    def __init__(self, dim, kernel_size, num_heads,planes=[640, 64, 64, 64, 640],
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 mode=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = 2 * kernel_size - 1

        self.relu = nn.ReLU(inplace=True)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mode = mode
        self.rpb = nn.Parameter(torch.zeros(num_heads, self.rpb_size, self.rpb_size))
        trunc_normal_(self.rpb, std=.02)
        # RPB implementation by @qwopqwop200
        self.idx_h = torch.arange(0, kernel_size)
        self.idx_w = torch.arange(0, kernel_size)
        self.idx_k = ((self.idx_h.unsqueeze(-1) * self.rpb_size) + self.idx_w).view(-1)
        warnings.warn("This is the legacy version of NAT -- it uses unfold+pad to produce NAT, and is highly inefficient.")
        self.bn1 = nn.BatchNorm2d(640)
        self.conv1x1_in = nn.Sequential(nn.Conv2d(planes[0], planes[1], kernel_size=1, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[1]),
                                        nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(nn.Conv2d(planes[1], planes[2], kernel_size=3, bias=False, padding=0),
                                        nn.BatchNorm2d(planes[2]),
                                        nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(planes[2], planes[3],kernel_size=3, bias=False, padding=0),
                                   nn.BatchNorm2d(planes[3]),
                                   nn.ReLU(inplace=True))
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(planes[3], planes[4], kernel_size=1, bias=False, padding=0),
            nn.BatchNorm2d(planes[4]))


    def apply_pb(self, attn, height, width):

        num_repeat_h = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_w = torch.ones(self.kernel_size,dtype=torch.long)
        num_repeat_h[self.kernel_size//2] = height - (self.kernel_size-1)
        num_repeat_w[self.kernel_size//2] = width - (self.kernel_size-1)
        bias_hw = (self.idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*self.kernel_size-1)) + self.idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + self.idx_k
        # Index flip
        # Our RPB indexing in the kernel is in a different order, so we flip these indices to ensure weights match.
        bias_idx = torch.flip(bias_idx.reshape(-1, self.kernel_size**2), [0])
        return attn + self.rpb.flatten(1, 2)[:, bias_idx].reshape(self.num_heads, height * width, 1, self.kernel_size ** 2).transpose(0, 1)

    def forward(self, x):
        x = self.relu(x)
        x = F.normalize(x, dim=1, p=2)  # （10，640，5，5）
        x = x.permute(0, 2, 3, 1).contiguous()  # （10，5，5，640）
        # x = self.norm1(x)
        B, H, W, C = x.shape
        N = H * W
        num_tokens = int(self.kernel_size ** 2)  # 49
        pad_l = pad_t = pad_r = pad_b = 0
        Ho, Wo = H, W
        if N <= num_tokens:
            if self.kernel_size > W:
                pad_r = self.kernel_size - W
            if self.kernel_size > H:
                pad_b = self.kernel_size - H
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))  # （10，7，7，640）
            B, H, W, C = x.shape
            N = H * W
            assert N == num_tokens, f"Something went wrong. {N} should equal {H} x {W}!"
        x = self.qkv(x).reshape(B, H, W, 3 * C)  # (80,7,7,1920)
        q, x = x[:, :, :, :C], x[:, :, :, C:]
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # (80,25,1,1,640)
        pd = self.kernel_size - 1
        pdr = pd // 2

        if self.mode == 0:
            x = x.permute(0, 3, 1, 2).flatten(0, 1)
            x = x.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).permute(0, 3, 4, 1, 2)
            x = pad(x, (pdr, pdr, pdr, pdr, 0, 0), 'replicate')
            x = x.reshape(B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)
        elif self.mode == 1:
            Hr, Wr = H - pd, W - pd
            x = unfold(x.permute(0, 3, 1, 2),
                       kernel_size=(self.kernel_size, self.kernel_size),
                       stride=(1, 1),
                       padding=(0, 0)).reshape(B, 2 * C * num_tokens, Hr, Wr)
            x = pad(x, (pdr, pdr, pdr, pdr), 'replicate').reshape(
                B, 2, self.num_heads, C // self.num_heads, num_tokens, N)
            x = x.permute(1, 0, 5, 2, 4, 3)  # (2,80,25,1,9,640)
        else:
            raise NotImplementedError(f'Mode {self.mode} not implemented for NeighborhoodAttention2D.')
        k, v = x[0], x[1]

        attn = (q @ k.transpose(-2, -1))  # B x N x H x 1 x num_tokens
        attn = self.apply_pb(attn, H, W)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B x N x H x 1 x C # (80,25,1,1,640)
        x = x.reshape(B, H, W, C)  # (10，7，7，640)
        if pad_r or pad_b:
            x = x[:, :Ho, :Wo, :]
        x = self.proj_drop(self.proj(x))
        x = x.permute(0, 3, 1, 2).contiguous()  # （10，640，5，5）

        #
        x = self.conv1x1_in(x)  # [80, 640, hw, 25] -> [80, 64, hw, 25]

        # x = self.conv1(x)  # [80, 64, hw, 5, 5] --> [80, 64, hw, 3, 3]
        # x = self.conv2(x)  # [80, 64, hw, 3, 3] --> [80, 64, hw, 1, 1]

        x = self.conv1x1_out(x)  # [80, 64, h, w] --> [80, 640, h, w]
        #

        # x = self.bn1(x)



        return x



# class NonLocal(nn.Module):
#     def __init__(self, channel):
#         super(NonLocalBlock, self).__init__()
#         self.inter_channel = channel // 2
#         self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
#         self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1,0, False)
#         self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0, False)
#         self.softmax = nn.Softmax(dim=1)
#         self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, False)
#
#     def forward(self, x):
#         # [N, C, H , W]
#         b, c, h, w = x.size()
#         # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
#         x_phi = self.conv_phi(x).view(b, c, -1)
#         # 获取theta特征，维度为[N, H * W, C/2]
#         x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
#         # 获取g特征，维度为[N, H * W, C/2]
#         x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
#         # 对phi和theta进行矩阵乘，[N, H * W, H * W]
#         mul_theta_phi = torch.matmul(x_theta, x_phi)
#         # softmax拉到0~1之间
#         mul_theta_phi = self.softmax(mul_theta_phi)
#         # 与g特征进行矩阵乘运算，[N, H * W, C/2]
#         mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
#         # [N, C/2, H, W]
#         mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
#         # 1X1卷积扩充通道数
#         mask = self.conv_mask(mul_theta_phi_g)
#         out = mask + x # 残差连接
#         return out

