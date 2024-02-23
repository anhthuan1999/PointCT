
import torch
import torch.nn as nn
from lib.pointops.functions import pointops
import torch.nn.functional as F
from copy import copy, deepcopy
import numpy as np

class WeakLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes) 
        
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))
        self.linear_uni_spatial = nn.Sequential(nn.Linear(in_planes, 1),nn.BatchNorm1d(1),nn.ReLU(inplace=True))

        self.linear_add = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True), nn.Linear(3, out_planes))

        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                    nn.Linear(mid_planes, mid_planes),
                                    nn.BatchNorm1d(mid_planes),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(out_planes, out_planes))

        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=1)
        self.softmax3 = nn.Softmax(dim=1)
        self.bn1=nn.Sequential(nn.LayerNorm(out_planes),nn.ReLU(inplace=True))
        self.a = nn.Parameter(torch.zeros(size=(nsample, 1)),requires_grad=True)
        self.b = nn.Parameter(torch.zeros(size=(nsample, out_planes)),requires_grad=True)
        self.d = nn.Parameter(torch.zeros(size=(nsample,out_planes)),requires_grad=True)
        nn.init.xavier_uniform_(self.a.data)
        nn.init.xavier_uniform_(self.b.data)
        nn.init.xavier_uniform_(self.d.data)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo 

        x_r=x.clone()

        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_uni_spatial = self.linear_uni_spatial(x)
        device = torch.device("cuda:0")
        
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v_j = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)
        n, nsample, c = x_v_j.shape; s = self.share_planes

        e, x_k = x_k[:, :, 0:3], x_k[:, :, 3:] 
        
        # ================================================== Position Encoding ==================================================
        
        p_r=torch.clone(e)
        e_dis=torch.from_numpy(np.linalg.norm(e.detach().cpu().numpy(),axis=2)).unsqueeze(2).cuda()
        e_dis_xy=torch.from_numpy(np.linalg.norm(e.detach().cpu().numpy()[:,:,:2],axis=2)).unsqueeze(2).cuda()
        cos_theta_e=(e_dis_xy/e_dis).nan_to_num(0)
        
        cos_theta_a=(e[:,:,1].unsqueeze(2)/e_dis_xy).nan_to_num(0)
        all_feat=torch.cat([e_dis,cos_theta_a,cos_theta_e],dim=-1)\
        
        for i, layer in enumerate(self.linear_add): 
            all_feat = layer(all_feat.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(all_feat)

        for i, layer in enumerate(self.linear_p): 
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        
        p_r+=self.d*all_feat
        
        # ================================================== Weight ==================================================
        w_spatial_d = torch.matmul((x_k - x_q.unsqueeze(1)).transpose(0,2).contiguous(),x_uni_spatial)
        #w_spatial_d = torch.matmul(x_k.transpose(0,2).contiguous(),x_uni_spatial)
        w_spatial_d = self.softmax2(w_spatial_d)  # (n, nsample, c)
        w_spatial_e = torch.matmul(x_r,w_spatial_d.transpose(0,1).contiguous()).transpose(0,1).contiguous()
        
        w_spatial_f = w_spatial_d.transpose(0,2).contiguous()
        
        w_spatial_e = self.softmax3(w_spatial_e)
        
        w=torch.add(self.a*w_spatial_f,self.b*w_spatial_e)+ p_r

        for i, layer in enumerate(self.linear_w): 
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i in [0,3] else layer(w)
        
        w=self.softmax(w)

        # ================================================== Combine ==================================================
        
        n, nsample, c = x_v_j.shape; s = self.share_planes
        x = ((x_v_j+p_r) * w).sum(1).view(n, c)
        return x


class Downsampling(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3+in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, pxo):
        p, x, o = pxo 
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)
            n_p = p[idx.long(), :]
            x = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))
            x = self.pool(x).squeeze(-1)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))
        return [p, x, o]


class FeaturePropagation(nn.Module):
    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2*in_planes, in_planes), nn.BatchNorm1d(in_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes), nn.ReLU(inplace=True))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)
        return x


class WeakBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(WeakBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.weak = WeakLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.act = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout(0.5, inplace=True)

    def forward(self, pxo):
        p, x, o = pxo
        identity = x
        x = self.act((self.bn1(self.linear1(x))))
        x = self.act(self.bn2(self.weak([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = self.dp2(x)
        x += identity
        x = self.act(x)
        return [p, x, o]


class WeakSeg(nn.Module):
    def __init__(self, block, blocks, c=6, k=13,unitArch=[32, 64, 128, 256, 512]):
        super().__init__()
        self.c = c
        self.in_planes, planes = 6, unitArch

        self.num_heads=8

        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]

        self.enc1 = self._make_enc(block, planes[0], blocks[0], share_planes, stride=stride[0], nsample=nsample[0])
        self.enc2 = self._make_enc(block, planes[1], blocks[1], share_planes, stride=stride[1], nsample=nsample[1])
        self.enc3 = self._make_enc(block, planes[2], blocks[2], share_planes, stride=stride[2], nsample=nsample[2])
        self.enc4 = self._make_enc(block, planes[3], blocks[3], share_planes, stride=stride[3], nsample=nsample[3])
        self.enc5 = self._make_enc(block, planes[4], blocks[4], share_planes, stride=stride[4], nsample=nsample[4])
        self.dec5 = self._make_dec(block, planes[4], 1, share_planes, nsample=nsample[4], is_head=True)
        self.dec4 = self._make_dec(block, planes[3], 1, share_planes, nsample=nsample[3])
        self.dec3 = self._make_dec(block, planes[2], 1, share_planes, nsample=nsample[2])
        self.dec2 = self._make_dec(block, planes[1], 1, share_planes, nsample=nsample[1])
        self.dec1 = self._make_dec(block, planes[0], 1, share_planes, nsample=nsample[0])
        self.cls = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.BatchNorm1d(planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], k))
        self.in_planes=16*self.num_heads

    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(Downsampling(self.in_planes, planes * block.expansion, stride, nsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = []
        layers.append(FeaturePropagation(self.in_planes, None if is_head else planes * block.expansion))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, pxo,other=[None,None]):
        
        p0, x0, o0 = pxo
        x0 = torch.cat([x0,p0],1)
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


def weak_seg_repro(custom=None, **kwargs):
    
    model = WeakSeg(WeakBlock, [1, 3, 3, 6, 3], **kwargs)
    return model

    