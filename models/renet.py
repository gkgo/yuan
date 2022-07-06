import torch
import torch.nn as nn
import torch.nn.functional as F


from models.resnet import ResNet
from models.cca import CCA
from models.scr import  SelfCorrelationComputation
# from models.others.se import SqueezeExcitation
# from models.others.lsa import LocalSelfAttention
# from models.others.nlsa import NonLocalSelfAttention
# from models.others.sce import SpatialContextEncoder


class RENet(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class)

        self.scr_module = self._make_scr_layer(planes=[640, 64, 64, 64, 640])
        self.cca_module = CCA(kernel_sizes=[3, 3], planes=[16, 1])
        self.lin = nn.Linear(640, 640)
        self.cca_1x1 = nn.Sequential(
            nn.Conv2d(self.encoder_dim, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def _make_scr_layer(self, planes):
        # stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()

        if self.args.self_method == 'scr':
            # corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
            # kernel_size=5,dim=640,num_heads=1
            # corr_block = SelfCorrelationComputation(kernel_size=3, dim=640, num_heads=1)
            corr_block = SelfCorrelationComputation(kernel_size=3,dim=640,num_heads=2)
            # self_block = SCR(planes=planes, stride=stride)
        # elif self.args.self_method == 'sce':
        #     planes = [640, 64, 64, 640]
        #     self_block = SpatialContextEncoder(planes=planes, kernel_size=kernel_size[0])
        # elif self.args.self_method == 'se':
        #     self_block = SqueezeExcitation(channel=planes[0])
        # elif self.args.self_method == 'lsa':
        #     self_block = LocalSelfAttention(in_channels=planes[0], out_channels=planes[0], kernel_size=kernel_size[0])
        # elif self.args.self_method == 'nlsa':
        #     self_block = NonLocalSelfAttention(planes[0], sub_sample=False)
        else:
            raise NotImplementedError

        if self.args.self_method == 'scr':
            layers.append(corr_block)
        # layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'cca':
            spt, qry = input
            return self.cca(spt, qry)
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def cca(self, spt, qry):
        spt = spt.squeeze(0)  # 移除数组中维度为1的维度

        # shifting channel activations by the channel mean
        spt = self.normalize_feature(spt)  # 原本的 # （5，640，5，5）
        qry = self.normalize_feature(qry)


        # way,c,h,w=spt.shape
        # spt1 = spt.view(way,c,h*w)  # vs
        # num_qry,C,H,W = qry.shape
        # qry1 = qry.view(num_qry,C,H*W)  # vq
        # spt2 = spt1.permute(0, 2, 1).contiguous()
        # spt2 = self.lin(spt2)  # 全连接层
        # spt2 = spt2.permute(0, 2, 1).contiguous()
        # Sh = torch.einsum('scj,qcm->sqjm', spt2, qry1)  # (5,5,25,25)
        # Sh = self.cca_module(Sh.reshape(-1, 1, h, w, H, W))
        # Sh1 =self.gaussian_normalize(Sh, dim=2)
        # Sh2 = self.gaussian_normalize(Sh, dim=3)
        # # corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)  # Eq.4（上面）（大小不变）
        # # corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        # # corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        # # corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)
        # Sh1 = F.softmax(Sh1, dim=2)  # (5,5,25,25)
        # Sh1 = Sh1.view(num_qry,way,h,w,H,W)
        # Sh2 = F.softmax(Sh2, dim=3)  # (5,5,5,5,5,5)
        # Sh2 = Sh2.view(num_qry, way, h, w, H, W)
        # Sh1 = Sh1.sum(dim=[4, 5])  # (5,5,5,5)
        # Sh2 = Sh2.sum(dim=[2, 3])
        # Sh1_att = Sh1.unsqueeze(2)*spt.unsqueeze(0)  # (5,5,640,5,5)
        # Sh2_att = Sh2.unsqueeze(2)*qry.unsqueeze(1)
        #
        # # spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)  # 公式5求最终的query embedding
        # # qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)
        # # Uq = spt1*Sh1
        # # Us = qry1*Sh2
        #
        # Sh1_pool = Sh1_att.mean(dim=[-1, -2])  # (5,5,640)
        # Sh2_pool = Sh2_att.mean(dim=[-1, -2])
        # qry_pooled = qry.mean(dim=[-1, -2])  # (5,640)
        # similarity_matrix = F.cosine_similarity(Sh1_pool, Sh2_pool, dim=-1)  # (5,5)
        # if self.training:
        #     return similarity_matrix / self.args.temperature, self.fc(qry_pooled)  # (5,5)相似度 （5，64）（batch，num—class）
        # else:
        #     return similarity_matrix / self.args.temperature


        # (S * C * Hs * Ws, Q * C * Hq * Wq) -> Q * S * Hs * Ws * Hq * Wq
        corr4d = self.get_4d_correlation_map(spt, qry)
        num_qry, way, H_s, W_s, H_q, W_q = corr4d.size()

        # corr4d refinement
        corr4d = self.cca_module(corr4d.view(-1, 1, H_s, W_s, H_q, W_q))  # （375，1，5，5，5，5）（用4维卷积细化）
        corr4d_s = corr4d.view(num_qry, way, H_s * W_s, H_q, W_q)  # （75，5，25，5，5）
        corr4d_q = corr4d.view(num_qry, way, H_s, W_s, H_q * W_q)  # （75，5，5，5，25）

        # normalizing the entities for each side to be zero-mean and unit-variance to stabilize training
        corr4d_s = self.gaussian_normalize(corr4d_s, dim=2)  # H_q * W_q可以看作一个特征向量
        corr4d_q = self.gaussian_normalize(corr4d_q, dim=4)  # 把H_q * W_q进行高斯归一化

        # applying softmax for each side
        corr4d_s = F.softmax(corr4d_s / self.args.temperature_attn, dim=2)  # Eq.4（上面）（大小不变）
        corr4d_s = corr4d_s.view(num_qry, way, H_s, W_s, H_q, W_q)
        corr4d_q = F.softmax(corr4d_q / self.args.temperature_attn, dim=4)
        corr4d_q = corr4d_q.view(num_qry, way, H_s, W_s, H_q, W_q)

        # suming up matching scores
        attn_s = corr4d_s.sum(dim=[4, 5])  # 最后2维 相当于最大池化
        attn_q = corr4d_q.sum(dim=[2, 3])  # 中间2维
        # 先求和（75，5，5，5）
        # applying attention
        spt_attended = attn_s.unsqueeze(2) * spt.unsqueeze(0)  # 公式5求最终的query embedding
        qry_attended = attn_q.unsqueeze(2) * qry.unsqueeze(1)
        # （75，5，640，5，5）
        # averaging embeddings for k > 1 shots
        if self.args.shot > 1:
            spt_attended = spt_attended.view(num_qry, self.args.shot, self.args.way, *spt_attended.shape[2:])
            qry_attended = qry_attended.view(num_qry, self.args.shot, self.args.way, *qry_attended.shape[2:])
            spt_attended = spt_attended.mean(dim=1)
            qry_attended = qry_attended.mean(dim=1)

        # In the main paper, we present averaging in Eq.(4) and summation in Eq.(5).
        # In the implementation, the order is reversed, however, those two ways become eventually the same anyway :)
        spt_attended_pooled = spt_attended.mean(dim=[-1, -2])  # (5,5,640)
        qry_attended_pooled = qry_attended.mean(dim=[-1, -2])
        # 再平均（对应公式4里的1/h*w）（75，5，640）
        qry_pooled = qry.mean(dim=[-1, -2])  # (5,640)

        similarity_matrix = F.cosine_similarity(spt_attended_pooled, qry_attended_pooled, dim=-1)  # 公式3（75，5）
        # 求余弦相似度，跟论文顺序不一样(5,5)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)  # (5,5)相似度 （5，64）（batch，num—class）
        else:
            return similarity_matrix / self.args.temperature

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)  # 求dim上的方差
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))  # （x原始-x平均）/根号下x_var
        return x

    def get_4d_correlation_map(self, spt, qry):
        '''
        The value H and W both for support and query is the same, but their subscripts are symbolic.
        :param spt: way * C * H_s * W_s
        :param qry: num_qry * C * H_q * W_q
        :return: 4d correlation tensor: num_qry * way * H_s * W_s * H_q * W_q
        :rtype:
        '''
        way = spt.shape[0]
        num_qry = qry.shape[0]

        # reduce channel size via 1x1 conv改变通道数量
        spt = self.cca_1x1(spt)
        qry = self.cca_1x1(qry)

        # normalize channels for later cosine similarity
        spt = F.normalize(spt, p=2, dim=1, eps=1e-8)
        qry = F.normalize(qry, p=2, dim=1, eps=1e-8)

        # num_way * C * H_p * W_p --> num_qry * way * H_p * W_p
        # num_qry * C * H_q * W_q --> num_qry * way * H_q * W_q
        spt = spt.unsqueeze(0).repeat(num_qry, 1, 1, 1, 1)  # 在0维度上复制num_qry
        qry = qry.unsqueeze(1).repeat(1, way, 1, 1, 1)  # 在第一维度上复制way
        # 使之大小都变为（75，5，64，5，5）
        similarity_map_einsum = torch.einsum('qncij,qnckl->qnijkl', spt, qry)  # （75，5，5，5，5，5）
        # 2 使用爱因斯坦求和
        return similarity_map_einsum

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)  # x-x.mean(1)行求平均值并在channal维上增加一个维度

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if self.args.self_method:
            identity = x  # (80,640,5,5)
            x = self.scr_module(x)

            if self.args.self_method == 'scr':
                x = x + identity   # 公式（2）
            x = F.relu(x, inplace=True)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x

# if __name__ == '__main__':
#     args = setup_run(arg_mode='train')  # 创建对象args
#     set_seed(args.seed)
#     model = RENet(args).cuda()  # 创建对象model并把数据传输到GPU里(调用renet)
#     model = nn.DataParallel(model, device_ids=args.device_ids)  # 如果有多GPU可以在多GPU上运行
#
#     if not args.no_wandb:
#         wandb.watch(model)
#     print(model)  # 使用wandb可视化工具来输出
