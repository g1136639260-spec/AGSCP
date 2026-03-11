import torch
import torch.nn as nn
import torch.nn.functional as F

BN = torch.nn.BatchNorm2d
Conv2d = torch.nn.Conv2d

__all__ = [
    'scalablelayer', 'convlayer', 'selayer', 'gatinglayer', 'attentionlayer',
    'crossconvlayer', 'crossconvhrnetlayer', 'convlayersedsc', 'crossconvhrnetlayermoe' , 'crossconvhrnetlayermoeup', 'crossconvhrnetlayerrouter', 'crossconvhrnetlayerrouterv2', 'crossconvlayerrouterv1', 'crossconvhrnetlayerrouter', 'crossconvlayerdsc', 'crossconvlayerdtf', 'crossconvlayertvam'
]


class Scalable_Layer(nn.Module):
    def __init__(self,
                 channel=None,
                 channel_wise=False,
                 element_wise=False,
                 **kwargs):
        super(Scalable_Layer, self).__init__()
        self.channel = channel
        self.channel_wise = channel_wise
        self.element_wise = element_wise
        if channel_wise:
            self.w = torch.nn.Parameter(torch.FloatTensor((channel)),
                                        requires_grad=True)
        elif element_wise:
            pass
        else:
            self.w = torch.nn.Parameter(torch.FloatTensor(1),
                                        requires_grad=True)
        self.w.data.fill_(0.00)

    def forward(self, x, y):
        if self.channel_wise:
            return self.w.view(1, self.channel, 1, 1) * x
        return self.w * x


class Conv_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 **kwargs):
        super(Conv_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)
        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        if not self.use_pooling:
            return self.relu(self.bn1(self.conv1(x)))
        else:
            return self.avg_pool(self.relu(self.bn1(self.conv1(x))))


class Cross_Conv_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(channel),
                nn.ReLU(inplace=True))


        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: dict, y: torch.tensor, detach=False):

        self_spatial_size = y.size()[-2:]
        channel_y = y.size()[-1:-2]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))
        out = sum(transformed_aux_features)+y

        return out


class Cross_Conv_Layer_routerv1(nn.Module):

    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer_routerv1, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(channel),
                nn.ReLU(inplace=True))


        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 1外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer1 = nn.Conv2d(256 * 2, 2, kernel_size=1)

        # 2内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 2外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer2 = nn.Conv2d(512 * 2, 2, kernel_size=1)

        # 3内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])
        # 3外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer3 = nn.Conv2d(1024 * 2, 2, kernel_size=1)

        device = torch.device('cuda:0')
        self.tau = nn.Parameter(torch.tensor(1.0, device=device))
        self.relu = nn.ReLU()

    def forward(self, x: dict, y: torch.Tensor, layer:str, detach=False):
        self_spatial_size = y.size()[-2:]
        channel_y = y.size()[-1:-2]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        D = y
        seg_feats = transformed_aux_features

        ##For trans layer1
        if layer == 'layer1':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer1, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer1(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]
        elif layer == 'layer2':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer2, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer2(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]
        else:
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer3, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer3(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]

        # 融合输出
        F_out = alpha_det * D + alpha_seg * F_seg + D

        return F_out


class Cross_Conv_Layer_routerv1_only_Irouter(nn.Module):

    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer_routerv1_only_Irouter, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(channel),
                nn.ReLU(inplace=True))


        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 1内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])



        # 2内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])



        # 3内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])


        device = torch.device('cuda:0')
        self.tau = nn.Parameter(torch.tensor(1.0, device=device))
        self.relu = nn.ReLU()

    def forward(self, x: dict, y: torch.Tensor, layer:str, detach=False):
        self_spatial_size = y.size()[-2:]
        channel_y = y.size()[-1:-2]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        D = y
        seg_feats = transformed_aux_features

        ##For trans layer1
        if layer == 'layer1':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer1, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]


        elif layer == 'layer2':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer2, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]


        else:
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer3, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores / self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]


        # 融合输出
        F_out = F_seg + D

        return F_out

class Cross_Conv_Layer_routerv1_only_Orouter(nn.Module):

    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer_routerv1_only_Orouter, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(channel),
                nn.ReLU(inplace=True))


        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)



        # 1外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer1 = nn.Conv2d(256 * 2, 2, kernel_size=1)



        # 2外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer2 = nn.Conv2d(512 * 2, 2, kernel_size=1)

        # 3外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer3 = nn.Conv2d(1024 * 2, 2, kernel_size=1)

        device = torch.device('cuda:0')
        self.tau = nn.Parameter(torch.tensor(1.0, device=device))
        self.relu = nn.ReLU()

    def forward(self, x: dict, y: torch.Tensor, layer:str, detach=False):
        self_spatial_size = y.size()[-2:]
        channel_y = y.size()[-1:-2]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        D = y
        seg_feats = transformed_aux_features

        ##For trans layer1
        if layer == 'layer1':
            # 加权聚合分割特征
            F_seg = sum(seg_feats)  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer1(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]
        elif layer == 'layer2':

            # 加权聚合分割特征
            F_seg = sum(seg_feats)  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer2(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]
        else:
            # -------- 内层路由 --------
            # 加权聚合分割特征
            F_seg = sum(seg_feats)  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer3(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores / self.tau, dim=1)  # [B,2,H,W]

            alpha_det = self.relu(task_weights[:, 0:1, :, :])  # [B,1,H,W]
            alpha_seg = self.relu(task_weights[:, 1:2, :, :])  # [B,1,H,W]

        # 融合输出
        F_out = alpha_det * D + alpha_seg * F_seg + D

        return F_out




class Cross_Conv_HRNet_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_HRNet_Layer, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]
        self.channel_to_spatial_ratio = {
            # For ResNet
            256: 4,
            512: 2,
            1024: 1,
            # For FMTB
            768: 1,
            192: 1,
            # For MTB
            64: 4,
            128: 2,
            320: 1,
        }

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[
                self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    nn.Conv2d(src_channel,
                              self.out_channel,
                              kernel_size=(1, 1),
                              bias=False),
                    BN(self.out_channel),
                    nn.Upsample(scale_factor=target_spatial_ratio //
                                src_spatial_ratio,
                                mode='nearest',
                                align_corners=None),
                )

            else:
                conv_layers = []
                conv_args = {
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                    'padding': 1,
                    'bias': False
                }
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 -
                               1):
                    conv_layers.append(
                        nn.Conv2d(src_channel, src_channel, **conv_args))
                    conv_layers.append(BN(src_channel))
                    conv_layers.append(nn.ReLU(False))
                conv_layers.append(
                    nn.Conv2d(src_channel, self.out_channel, **conv_args))
                conv_layers.append(BN(self.out_channel))

                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1e-3)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: dict, y: torch.tensor, detach=False):
        #detach:True x:即inp_feature[gv_patch]:[input(2,3,608,608),prelayer(2,64,304,464),layer1(2,256,152,232),layer2(2,512,76,116),layer3(2,1024,38,58),layer3(2,2048,19,29),pre_fc(2,2048),]
        #y即inp_feature[gv_global][layer1]:[layer1(2,256,152,232),]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):#self.aux_layers:['layer1']
            transformed_aux_features.append(self.convs[aux_layer_name](#对
                (x[aux_layer_name].detach() if detach else x[aux_layer_name])))

        out = sum(transformed_aux_features) + y

        return out



class Cross_Conv_HRNet_Layer_moe(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_HRNet_Layer_moe, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]
        self.channel_to_spatial_ratio = {
            # For ResNet
            256: 4,
            512: 2,
            1024: 1,
            # For FMTB
            768: 1,
            192: 1,
            # For MTB
            64: 4,
            128: 2,
            320: 1,
        }

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[
                self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    nn.Conv2d(src_channel,
                              self.out_channel,
                              kernel_size=(1, 1),
                              bias=False),
                    BN(self.out_channel),
                    nn.Upsample(scale_factor=target_spatial_ratio //
                                src_spatial_ratio,
                                mode='nearest',
                                align_corners=None),
                )

            else:
                conv_layers = []
                conv_args = {
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                    'padding': 1,
                    'bias': False
                }
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 -
                               1):
                    conv_layers.append(
                        nn.Conv2d(src_channel, src_channel, **conv_args))
                    conv_layers.append(BN(src_channel))
                    conv_layers.append(nn.ReLU(False))
                conv_layers.append(
                    nn.Conv2d(src_channel, self.out_channel, **conv_args))
                conv_layers.append(BN(self.out_channel))

                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1e-3)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 1内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 1外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer1 = nn.Conv2d(256 * 2, 2, kernel_size=1)

        # 2内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 2外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer2 = nn.Conv2d(512 * 2, 2, kernel_size=1)

        # 3内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])
        # 3外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer3 = nn.Conv2d(1024 * 2, 2, kernel_size=1)

        device = torch.device('cuda:0')
        self.tau = nn.Parameter(torch.tensor(1.0, device=device))


    def forward(self, x: dict, y: torch.tensor, layer: str, detach=False):
        #detach:True x:即inp_feature[gv_patch]:[input(2,3,608,608),prelayer(2,64,304,464),layer1(2,256,152,232),layer2(2,512,76,116),layer3(2,1024,38,58),layer3(2,2048,19,29),pre_fc(2,2048),]
        #y即inp_feature[gv_global][layer1]:[layer1(2,256,152,232),]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):#self.aux_layers:['layer1']
            transformed_aux_features.append(self.convs[aux_layer_name](#对
                (x[aux_layer_name].detach() if detach else x[aux_layer_name])))


        ###双层路由
        D = y
        seg_feats = transformed_aux_features


        ##For trans layer1
        if layer == 'layer1':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer1, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer1(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]
        elif layer == 'layer2':
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer2, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer2(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]
        else :
            # -------- 内层路由 --------
            seg_scores = [router(s) for router, s in zip(self.seg_router_layer3, seg_feats)]  # [B,1,H,W] * 3
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer3(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]

        # 融合输出
        F_out = alpha_det * D + alpha_seg * F_seg + D

        return F_out


class Gating_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 **kwargs):
        super(Gating_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)
        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        if not self.use_pooling:
            return self.relu(self.bn1(self.conv1(x))) * y
        else:
            return self.avg_pool(self.relu(self.bn1(self.conv1(x)))) * y


class Attention_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 use_cross=False,
                 **kwargs):
        super(Attention_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn2 = BN(channel)

        self.tanh = nn.Tanh()

        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_cross = use_cross

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0.001, b=0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=0.001, b=0.01)

    def forward(self, x, y):
        if self.use_cross:
            return self.tanh(
                self.relu(self.bn1(self.conv1(x))) *
                self.relu(self.bn2(self.conv2(y)))) + self.tanh(
                    self.avg_pool(self.relu(self.bn1(self.conv1(x)))) *
                    self.avg_pool(self.relu(self.bn2(self.conv2(y)))))
        if not self.use_pooling:
            return self.tanh(
                self.relu(self.bn1(self.conv1(x))) *
                self.relu(self.bn2(self.conv2(y))))
        else:
            return self.tanh(
                self.avg_pool(self.relu(self.bn1(self.conv1(x)))) *
                self.avg_pool(self.relu(self.bn2(self.conv2(y)))))


class MATNLayer(nn.Module):
    def __init__(self, channel, kernel_size=1, stride=1, padding=0, **kwargs):
        super(MATNLayer, self).__init__()
        self.conv1 = Conv2d(2 * channel,
                            channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(channel,
                            channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.bn2 = BN(channel)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = Conv2d(channel,
                            channel,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn3 = BN(channel)

        init_weights = [0.9, 0.1]
        self.conv1.weight = nn.Parameter(
            torch.cat([
                torch.eye(channel) * init_weights[0],
                torch.eye(channel) * init_weights[1]
            ],
                      dim=1).view(channel, -1, 1, 1))
        self.conv1.bias.data.fill_(0)
        self.conv2.weight = nn.Parameter(
            torch.cat([torch.eye(channel) * init_weights[0]],
                      dim=1).view(channel, -1, 1, 1))
        self.conv2.bias.data.fill_(0)

    def forward(self, x, y):
        y = torch.cat([x, y], dim=1)
        y = self.sigmoid(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(y))))))
        x = x * y
        return self.relu(self.bn3(self.conv3(x)))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, a=0.0001, b=0.00001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.uniform(m.weight, a=0.0001, b=0.00001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, _):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayerPolicy(nn.Module):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SELayerPolicy, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x, _):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Conv_Layer_SE_DSC(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 **kwargs):
        super(Conv_Layer_SE_DSC, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)

        #SE-weight
        reduction = 16
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x, y):
        aux_map = x
        main_map = y
        B, C, _, _ = x.size()
        main_avg_channel_out = self.Sigmoid(self.fc(self.avg_pool(main_map)))
        aux_avg_channel_out = self.Sigmoid(self.fc(self.avg_pool(aux_map)))
        main_exp = main_avg_channel_out.flatten(2)  # (B,C,1)
        aux_exp = aux_avg_channel_out.flatten(2).permute(0, 2, 1)  # (B,1,C)

        main_holo = torch.log(aux_exp + 1e-8) / (torch.log(main_exp * aux_exp + 1e-8))  # (B,C,C)
        aux_holo = torch.log(main_exp + 1e-8) / (torch.log(main_exp * aux_exp + 1e-8))  # (B,C,C)

        main_holo = main_holo.mean(dim=2).view(B, C, 1, 1)  # input= (B,C,1),output=(B,C,1,1)
        aux_holo = aux_holo.mean(dim=1).view(B, C, 1, 1)  # input= (B,1,C),output=(B,C,1,1)

        # 将holo变为比例、权重。main_holo, aux_holo: (B, C, 1, 1)
        # 先拼接得到 (B, 2, C, 1, 1)
        weights = torch.stack([main_holo, aux_holo], dim=1)  # (B, 2, C, 1, 1)
        # 沿 dim=1 做归一化，使得 main+aux = 1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        # 拆分回来
        main_holo, aux_holo = weights[:, 0], weights[:, 1]  # (B, C, 1, 1)

        main_holo_weight = main_holo * main_avg_channel_out
        aux_holo_weight = aux_holo * aux_avg_channel_out

        main_attention_evoluation_weight = main_holo_weight * main_avg_channel_out
        aux_attention_evoluation_weight = aux_holo_weight * aux_avg_channel_out

        out=(self.relu(self.bn1(self.conv1(x)))*main_attention_evoluation_weight)+(self.relu(self.bn1(self.conv1(y)))*aux_attention_evoluation_weight)

        return out


class Cross_Conv_Layer_DSC(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer_DSC, self).__init__()

        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(channel),
                nn.ReLU(inplace=True))


        self.use_pooling = use_pooling

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.Sigmoid = nn.Sigmoid()


    def forward(self, x: dict, y: torch.Tensor,  detach=False):

        self_spatial_size = y.size()[-2:]
        channel_y = y.size()[-1:-2]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        aux_map = sum(transformed_aux_features)
        main_map = y
        B, C, _, _ = aux_map.size()
        main_avg_channel_out = self.avg_pool(main_map)
        aux_avg_channel_out = self.avg_pool(aux_map)
        main_exp = main_avg_channel_out.flatten(2)  # (B,C,1)
        aux_exp = aux_avg_channel_out.flatten(2).permute(0, 2, 1)  # (B,1,C)

        main_holo = torch.log(aux_exp + 1e-8) / (torch.log(main_exp * aux_exp + 1e-8))  # (B,C,C)
        aux_holo = torch.log(main_exp + 1e-8) / (torch.log(main_exp * aux_exp + 1e-8))  # (B,C,C)

        main_holo = main_holo.mean(dim=2).view(B, C, 1, 1)  # input= (B,C,1),output=(B,C,1,1)
        aux_holo = aux_holo.mean(dim=1).view(B, C, 1, 1)  # input= (B,1,C),output=(B,C,1,1)

        # 将holo变为比例、权重。main_holo, aux_holo: (B, C, 1, 1)
        # 先拼接得到 (B, 2, C, 1, 1)
        weights = torch.stack([main_holo, aux_holo], dim=1)  # (B, 2, C, 1, 1)
        # 沿 dim=1 做归一化，使得 main+aux = 1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        # 拆分回来
        main_holo, aux_holo = weights[:, 0], weights[:, 1]  # (B, C, 1, 1)

        out=(aux_map*main_holo)+(y*aux_holo)

        return out


class Cross_Conv_Layer_TVAM(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        # 注意这里调用父类初始化，假设父类是 nn.Module
        super(Cross_Conv_Layer_TVAM, self).__init__()

        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()

        self.out_channel = self.layer2channel[self.name]

        # 1. 辅助分支的特征变换层
        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                nn.Conv2d(src_channel,
                          channel,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True))

        self.use_pooling = use_pooling

        # ================== TVAM 迁移部分 ==================
        # 原代码中使用 Linear(768, 768)，这里用 1x1 卷积代替
        # 用于将特征投影到交互空间
        self.attn_main_proj = nn.Conv2d(channel, channel, kernel_size=1)
        self.attn_aux_proj = nn.Conv2d(channel, channel, kernel_size=1)

        # 最后的融合权重生成 (可选，根据需要调整)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: dict, y: torch.Tensor, detach=False):
        """
        x: 辅助层特征字典
        y: 主层特征 (Main / Visual)
        """
        self_spatial_size = y.size()[-2:]

        # 1. 准备辅助特征 (Aux / Touch)
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            # 对齐尺寸
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            # 变换通道
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        # 将所有辅助特征相加，作为一个整体 "Aux Map" (对应原代码的 Touch)
        aux_map = sum(transformed_aux_features)
        main_map = y  # (对应原代码的 Visual)

        # ================== TVAM 核心逻辑实现 ==================

        # --- Step 1: 特征投影 (Projection) ---
        # 对应: proj_features = torch.tanh(self.attn_proj(features))
        # 维度: (B, C, H, W)
        proj_main = torch.tanh(self.attn_main_proj(main_map))
        proj_aux = torch.tanh(self.attn_aux_proj(aux_map))

        # --- Step 2: 空间注意力交互 (Spatial Attention) ---
        # 原代码: spatial_score = einsum(..., ...) -> 逐元素相乘
        # 维度: (B, C, H, W)
        spatial_interaction = proj_main * proj_aux

        # 计算 Spatial Softmax
        # 我们需要将 (H, W) 拉平，在空间维度上做 Softmax
        B, C, H, W = spatial_interaction.size()
        # reshape 为 (B, C, H*W)
        spatial_score_flat = spatial_interaction.view(B, C, -1)
        # dim=2 表示在空间维度归一化
        spatial_attn_score = F.softmax(spatial_score_flat, dim=2)
        # 恢复为 (B, C, H, W)，这是空间注意力图
        spatial_attn_map = spatial_attn_score.view(B, C, H, W)

        # --- Step 3: 空间特征聚合 (Spatial Pooling) ---
        # 原代码: sum(spatial_attn_score * proj_features, dim=2)
        # 这里的 sum 相当于加权全局池化
        # 维度: (B, C, H, W) -> sum -> (B, C)
        global_main = torch.sum(spatial_attn_map * proj_main, dim=(2, 3))
        global_aux = torch.sum(spatial_attn_map * proj_aux, dim=(2, 3))

        # --- Step 4: 通道/全局交互 (原 Temporal Attention) ---
        # 原代码对应时间维度交互，这里因为没有时间轴，我们将其理解为“通道维度的全局交互”
        # 维度: (B, C) * (B, C) -> (B, C)
        channel_interaction = global_main * global_aux

        # 生成最终的融合权重
        # 原代码是对 Time 做 Softmax。这里只有 1 个时间步，Softmax 没意义。
        # 我们改用 Sigmoid 生成门控权重 (0~1)
        fusion_weight = self.sigmoid(channel_interaction)  # (B, C)

        # 将权重扩展回 (B, C, 1, 1) 以便广播相乘
        fusion_weight = fusion_weight.view(B, C, 1, 1)

        # ================== 最终输出 ==================
        # 这是一个基于注意力的加权融合
        # 策略：利用计算出的互注意力权重来增强主特征和辅助特征

        out = (main_map * fusion_weight) + (aux_map * fusion_weight)
        # 或者保留残差连接： out = main_map + (aux_map * fusion_weight)

        return out


from torch.nn.init import calculate_gain
class FilterNorm(nn.Module):
    def __init__(self, in_channels, kernel_size, filter_type='spatial',
                 nonlinearity='linear', running_std=False, running_mean=False):
        assert filter_type in ('spatial', 'channel', 'new')
        assert in_channels >= 1
        super(FilterNorm, self).__init__()
        self.in_channels = in_channels
        self.filter_type = filter_type
        self.runing_std = running_std
        self.runing_mean = running_mean
        std = calculate_gain(nonlinearity) / kernel_size
        if running_std:
            self.std = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2) * std, requires_grad=True)
        else:
            self.std = std
        if running_mean:
            self.mean = nn.Parameter(
                torch.randn(in_channels * kernel_size ** 2), requires_grad=True)

    def forward(self, x):
        if self.filter_type == 'spatial':
            b, _, h, w = x.size()
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]
        elif self.filter_type == 'channel':
            b = x.size(0)
            c = self.in_channels
            x = x.reshape(b, c, -1)
            x = x - x.mean(dim=2).reshape(b, c, 1)
            x = x / (x.std(dim=2).reshape(b, c, 1) + 1e-10)
            x = x.reshape(b, -1)
            if self.runing_std:
                x = x * self.std[None, :]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :]

        elif self.filter_type == "new":
            b, _, h, w = x.size()
            #print(f"x.shape : {x.shape}")
            x = x.reshape(b, self.in_channels, -1, h, w)
            x = x - x.mean(dim=2).reshape(b, self.in_channels, 1, h, w)
            x = x / (x.std(dim=2).reshape(b, self.in_channels, 1, h, w) + 1e-10)
            x = x.reshape(b, _, h, w)
            if self.runing_std:
                x = x * self.std[None, :, None, None]
            else:
                x = x * self.std
            if self.runing_mean:
                x = x + self.mean[None, :, None, None]

        else:
            raise RuntimeError('Unsupported filter type {}'.format(self.filter_type))
        return x





# 2. 你的 DTF 模块 (稍作修改以适配迁移)

class DTF(nn.Module):
    def __init__(self, dim, kernel_size, stride=1, padding=1, groups=1, prompt_cfg=None):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.prompt_cfg = prompt_cfg

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1)
        self.chan_FilterNorm = FilterNorm(
            in_channels=dim,
            kernel_size=kernel_size,
            filter_type='channel',
            nonlinearity='relu',
            running_std=True,
            running_mean=True
        )

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv(self.pool(x))
        weight = self.chan_FilterNorm(weight)
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)

        x_reshaped = x.reshape(1, b * c, h, w)
        # 正确：groups 应该等于总通道数 (BatchSize * Channels)
        out = F.conv2d(x_reshaped, weight, stride=self.stride, padding=self.padding, groups=b * c)
        out = out.view(b, c, out.shape[-2], out.shape[-1])
        return out


# ================= 3. 最后才是你的 Cross_Conv_Layer_DTF =================

class Cross_Conv_Layer_DTF(nn.Module):
    def __init__(self,
                 channel,  # 输出/主特征通道数 (例如 768)
                 dtf_rank=32,  # DTF 的瓶颈维度 (论文中 r=16, 32, 64) [cite: 287]
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_Layer_DTF, self).__init__()

        self.name = name
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]

        # 1. 基础特征对齐层 (Aux -> Main Channel)
        self.convs = torch.nn.ModuleDict()
        for aux_layer in self.aux_layers:
            src_channel = layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                nn.Conv2d(src_channel, channel, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            )

        # 2. TADFormer 核心组件: Bottleneck DTF [cite: 235]
        # 结构: Down-Project -> DTF -> Up-Project
        # 这就是论文图 4(b) 中的 "TA-Module"
        self.dtf_rank = dtf_rank

        # 降维 (Down-Projection)
        self.down_proj = nn.Sequential(
            nn.Conv2d(channel, dtf_rank, kernel_size=1, bias=False),
            # nn.ReLU() # 可选，论文公式 (1) 是线性投影，但通常加非线性更好
        )

        # 动态任务滤波器 (Dynamic Task Filter)
        # 注意: kernel_size=3 是 DTF 的典型设置，用于捕获空间上下文
        self.dtf = DTF(dim=dtf_rank, kernel_size=3, padding=1)

        # 升维 (Up-Projection)
        self.up_proj = nn.Sequential(
            nn.Conv2d(dtf_rank, channel, kernel_size=1, bias=False),
        )

        # 可选: 最后的融合权重或门控
        self.fusion_gate = nn.Sigmoid()

    def forward(self, x: dict, y: torch.Tensor, detach=False):
        """
        x: 辅助层特征字典 (Auxiliary Features)
        y: 主特征 (Main / Task-Agnostic Feature)
        """
        self_spatial_size = y.size()[-2:]  # (H, W)

        # --- Step 1: 聚合辅助特征 ---
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            # 插值对齐尺寸
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            # 卷积对齐通道
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))

        # 获得初始的 Task-Adapted Feature (Aux Map)
        # 对应论文中将 Task Prompts 或其他特征聚合 [cite: 228]
        aux_map = sum(transformed_aux_features)

        # --- Step 2: 应用 TADFormer 的 TA-Module (DTF) ---
        # 流程: Input -> Down -> DTF(Dynamic Conv) -> Up -> Output

        # 1. 降维
        f_down = self.down_proj(aux_map)  # (B, r, H, W)

        # 2. DTF 动态滤波
        # 这里 DTF 会根据 f_down 的内容动态生成卷积核并处理 f_down
        # "extracting input-context sensitive representations" [cite: 351]
        f_refined = self.dtf(f_down)  # (B, r, H, W)

        # 3. 升维
        f_up = self.up_proj(f_refined)  # (B, C, H, W)

        # --- Step 3: 融合 (Fusion) ---
        # 论文中通常是残差连接: Output = Main + Task_Specific
        # 这里我们将处理后的 Aux (Task-Specific) 加到 Main (y) 上

        # 方式 A: 直接相加 (类似 ResNet/LoRA)
        out = y + f_up

        # 方式 B: 门控相加 (更稳健，类似于你之前的代码)
        # out = y + f_up * self.fusion_gate(f_up)

        return out




class Cross_Conv_HRNet_Layer_moe_up(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_HRNet_Layer_moe_up, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]
        self.channel_to_spatial_ratio = {
            # For ResNet
            256: 4,
            512: 2,
            1024: 1,
            # For FMTB
            768: 1,
            192: 1,
            # For MTB
            64: 4,
            128: 2,
            320: 1,
        }

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[
                self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    nn.Conv2d(src_channel,
                              self.out_channel,
                              kernel_size=(1, 1),
                              bias=False),
                    BN(self.out_channel),
                    nn.Upsample(scale_factor=target_spatial_ratio //
                                src_spatial_ratio,
                                mode='nearest',
                                align_corners=None),
                )

            else:
                conv_layers = []
                conv_args = {
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                    'padding': 1,
                    'bias': False
                }
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 -
                               1):
                    conv_layers.append(
                        nn.Conv2d(src_channel, src_channel, **conv_args))
                    conv_layers.append(BN(src_channel))
                    conv_layers.append(nn.ReLU(False))
                conv_layers.append(
                    nn.Conv2d(src_channel, self.out_channel, **conv_args))
                conv_layers.append(BN(self.out_channel))

                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1e-3)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 1内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256*2, 64*2, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(64*2, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 1外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer1 = nn.Conv2d(256 * 2, 2, kernel_size=1)

        # 2内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512*2, 128*2, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(128*2, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])

        # 2外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer2 = nn.Conv2d(512 * 2, 2, kernel_size=1)

        # 3内层路由器: 给每个分割特征图生成分数图
        self.seg_router_layer3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1024*2, 256*2, kernel_size=1, bias=False),  # 降维
                nn.ReLU(inplace=True),
                nn.Conv2d(256*2, 1, kernel_size=1)  # 输出分数图
            )
            for _ in range(3)
        ])
        # 3外层路由器: 给检测和分割整体生成平衡分数图
        self.task_router_layer3 = nn.Conv2d(1024 * 2, 2, kernel_size=1)

        device = torch.device('cuda:0')
        self.tau = nn.Parameter(torch.tensor(1.0, device=device))


    def forward(self, x: dict, y: torch.tensor, layer: str, detach=False):
        #detach:True x:即inp_feature[gv_patch]:[input(2,3,608,608),prelayer(2,64,304,464),layer1(2,256,152,232),layer2(2,512,76,116),layer3(2,1024,38,58),layer3(2,2048,19,29),pre_fc(2,2048),]
        #y即inp_feature[gv_global][layer1]:[layer1(2,256,152,232),]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):#self.aux_layers:['layer1']
            transformed_aux_features.append(self.convs[aux_layer_name](#对
                (x[aux_layer_name].detach() if detach else x[aux_layer_name])))


        ###双层路由
        D = y
        seg_feats = transformed_aux_features


        ##For trans layer1
        if layer == 'layer1':
            # -------- 内层路由 --------
            seg_scores = [router(torch.cat([D, s], dim=1))  # 拼接 D 和 s
                          for router, s in zip(self.seg_router_layer1, seg_feats)]
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer1(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]
        elif layer == 'layer2':
            # -------- 内层路由 --------
            seg_scores = [router(torch.cat([D, s], dim=1))  # 拼接 D 和 s
                          for router, s in zip(self.seg_router_layer2, seg_feats)]
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer2(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]
        else :
            # -------- 内层路由 --------
            seg_scores = [router(torch.cat([D, s], dim=1))  # 拼接 D 和 s
                          for router, s in zip(self.seg_router_layer3, seg_feats)]
            seg_scores = torch.cat(seg_scores, dim=1)  # [B,3,H,W]
            seg_weights = F.softmax(seg_scores/self.tau, dim=1)  # 在3个特征图维度归一化

            # 加权聚合分割特征
            F_seg = sum(w * f for w, f in zip(seg_weights.split(1, dim=1), seg_feats))  # [B,C,H,W]

            # -------- 外层路由 --------
            fusion_input = torch.cat([D, F_seg], dim=1)  # [B,2C,H,W]
            task_scores = self.task_router_layer3(fusion_input)  # [B,2,H,W]
            task_weights = F.softmax(task_scores/self.tau, dim=1)  # [B,2,H,W]

            alpha_det = task_weights[:, 0:1, :, :]  # [B,1,H,W]
            alpha_seg = task_weights[:, 1:2, :, :]  # [B,1,H,W]

        # 融合输出
        F_out = alpha_det * D + alpha_seg * F_seg + D

        return F_out

class Cross_Conv_HRNet_Layer_router(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_HRNet_Layer_router, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]

        # ------------------------------
        # ① 设定空间分辨率比例表
        # ------------------------------
        self.channel_to_spatial_ratio = {
            256: 4, 512: 2, 1024: 1,
            768: 1, 192: 1,
            64: 4, 128: 2, 320: 1,
        }

        # ------------------------------
        # ② 各辅助层 -> 主层的映射变换模块
        # ------------------------------
        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    nn.Conv2d(src_channel, self.out_channel, kernel_size=1, bias=False),
                    BN(self.out_channel),
                    nn.Upsample(scale_factor=target_spatial_ratio // src_spatial_ratio, mode='nearest')
                )
            else:
                conv_layers = []
                conv_args = dict(kernel_size=3, stride=2, padding=1, bias=False)
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 - 1):
                    conv_layers += [nn.Conv2d(src_channel, src_channel, **conv_args), BN(src_channel), nn.ReLU(False)]
                conv_layers += [nn.Conv2d(src_channel, self.out_channel, **conv_args), BN(self.out_channel)]
                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        # ------------------------------
        # ③ Cross-Layer Attention Module (CLAM)
        # ------------------------------
        # 根据不同辅助层生成可学习的融合权重 α_ij
        self.cross_layer_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channel, len(self.aux_layers), kernel_size=1),
            nn.Softmax(dim=1)
        )

        # ------------------------------
        # ④ Dynamic Feature Gating 模块 (类似 SE-block)
        # ------------------------------
        reduction = max(4, self.out_channel // 16)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.out_channel * 2, reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, self.out_channel, 1, bias=True),
            nn.Sigmoid()
        )

        # ------------------------------
        # 参数初始化
        # ------------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


    def forward(self, x: dict, y: torch.Tensor, detach=False):
        """
        x: 包含多层辅助特征的dict，例如 {'layer1': tensor, 'layer2': tensor, ...}
        y: 当前主任务的特征图
        detach: 若为True，则辅助任务梯度不反传
        """
        # ------------------------------
        # Step 1. 对辅助任务特征进行变换
        # ------------------------------
        aux_feats = []
        for aux_layer_name in sorted(self.aux_layers):
            feat = x[aux_layer_name].detach() if detach else x[aux_layer_name]
            aux_feats.append(self.convs[aux_layer_name](feat))

        # ------------------------------
        # Step 2. Cross-Layer Attention (CLAM)
        # 计算各层辅助特征的权重 α_ij 并加权融合
        # ------------------------------
        if len(aux_feats) > 1:
            stack_feats = torch.stack(aux_feats, dim=1)  # [B, L, C, H, W]
            # 对每个辅助层生成通道注意力
            weights = self.cross_layer_attention(aux_feats[-1])  # [B, L, 1, 1]
            weights = weights.unsqueeze(2)  # -> [B, L, 1, 1, 1]
            print('weights:',weights)
            fused_aux = (stack_feats * weights).sum(dim=1)  # [B, C, H, W]
        else:
            fused_aux = aux_feats[0]

        # ------------------------------
        # Step 3. 动态通道门控 (Gating)
        # ------------------------------
        # 将主任务特征 y 与辅助特征 fused_aux 拼接，计算门控权重
        gate_input = torch.cat([y, fused_aux], dim=1)
        gate_weight = self.gate(gate_input)  # [B, C, 1, 1]
        # print('gate_weight:', gate_weight)
        # 计算主辅比例，使总和为 2
        alpha = 2 * gate_weight
        beta = 2 - alpha
        # ------------------------------
        # Step 4. 特征融合输出
        # ------------------------------
        out = alpha * y + beta * fused_aux
        print(alpha,beta)
        return out


import torch
import torch.nn as nn

BN = nn.BatchNorm2d


# ----------------------------------------------------------
# C3 模块定义（YOLO 系列中的跨阶段部分，用作映射单元）
# ----------------------------------------------------------
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True,k=1,s=1,p=0):
        super(C3, self).__init__()
        hidden_c = int(c2 * 0.5)
        self.cv1 = nn.Conv2d(c1, hidden_c, kernel_size=k,stride=s,padding=p, bias=False)
        self.cv2 = nn.Conv2d(c1, hidden_c,kernel_size=k,stride=s,padding=p, bias=False)
        self.cv3 = nn.Conv2d(2 * hidden_c, c2, 1, 1, bias=False)
        self.bn = BN(c2)
        self.act = nn.SiLU(inplace=True)
        self.m = nn.Sequential(*[
            Bottleneck(hidden_c, hidden_c, shortcut) for _ in range(n)
        ])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        out = torch.cat((y1, y2), dim=1)
        out = self.cv3(out)
        return self.act(self.bn(out))


# ----------------------------------------------------------
# Bottleneck 模块（C3 内部使用）
# ----------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super(Bottleneck, self).__init__()
        hidden_c = c2
        self.conv1 = nn.Conv2d(c1, hidden_c, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_c, c2, 3, 1, 1, bias=False)
        self.bn = BN(c2)
        self.act = nn.SiLU(inplace=True)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.act(self.bn(self.conv2(self.act(self.conv1(x)))))
        return x + y if self.add else y


# ----------------------------------------------------------
# 改进后的 Cross_Conv_HRNet_Layer_router_v2
# ----------------------------------------------------------
# class Cross_Conv_HRNet_Layer_router_v2(nn.Module):
#     def __init__(self, channel, name=None, layer2channel=None, layer2auxlayers=None, **kwargs):
#         super(Cross_Conv_HRNet_Layer_router_v2, self).__init__()
#         self.name = name
#         self.layer2channel = layer2channel
#         self.layer2auxlayers = layer2auxlayers
#         self.aux_layers = self.layer2auxlayers[name]
#         self.convs = torch.nn.ModuleDict()
#         self.out_channel = self.layer2channel[self.name]
#
#         self.channel_to_spatial_ratio = {
#             256: 4, 512: 2, 1024: 1,
#             768: 1, 192: 1,
#             64: 4, 128: 2, 320: 1,
#         }
#
#         for aux_layer in self.aux_layers:
#             src_channel = self.layer2channel[aux_layer]
#             src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
#             target_spatial_ratio = self.channel_to_spatial_ratio[self.out_channel]
#
#             # ✅ 确保 align_layer 输出 self.out_channel 通道
#             if target_spatial_ratio >= src_spatial_ratio:
#                 align_layer = nn.Sequential(
#                     nn.Conv2d(src_channel, self.out_channel, 1, bias=False),
#                     BN(self.out_channel),
#                     nn.Upsample(scale_factor=target_spatial_ratio // src_spatial_ratio, mode='nearest')
#                 )
#             else:
#                 conv_layers = []
#                 conv_args = dict(kernel_size=3, stride=2, padding=1, bias=False)
#                 for _ in range(src_spatial_ratio // target_spatial_ratio // 2 - 1):
#                     conv_layers += [nn.Conv2d(src_channel, src_channel, **conv_args), BN(src_channel), nn.ReLU(False)]
#                 conv_layers += [nn.Conv2d(src_channel, self.out_channel, **conv_args), BN(self.out_channel)]
#                 align_layer = nn.Sequential(*conv_layers)
#
#             # ✅ 现在输入输出通道一致
#             mapping = nn.Sequential(
#                 align_layer,
#                 C3(self.out_channel, self.out_channel)
#             )
#             self.convs[aux_layer] = mapping
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x: dict, y: torch.Tensor, detach=False):
#         aux_feats = []
#         for aux_layer_name in sorted(self.aux_layers):
#             feat = x[aux_layer_name].detach() if detach else x[aux_layer_name]
#             aux_feats.append(self.convs[aux_layer_name](feat))
#
#         fused_aux = sum(aux_feats)
#         out = y + fused_aux
#         return out



# class Cross_Conv_HRNet_Layer_router_v2(nn.Module):
#     def __init__(self,
#                  channel,
#                  kernel_size=1,
#                  stride=1,
#                  padding=0,
#                  use_pooling=False,
#                  name=None,
#                  layer2channel=None,
#                  layer2auxlayers=None,
#                  **kwargs):
#         super(Cross_Conv_HRNet_Layer_router_v2, self).__init__()
#         self.name = name
#         self.layer2channel = layer2channel
#         self.layer2auxlayers = layer2auxlayers
#         self.aux_layers = self.layer2auxlayers[name]
#         self.convs = torch.nn.ModuleDict()
#         self.out_channel = self.layer2channel[self.name]
#         self.channel_to_spatial_ratio = {
#             # For ResNet
#             256: 4,
#             512: 2,
#             1024: 1,
#             # For FMTB
#             768: 1,
#             192: 1,
#             # For MTB
#             64: 4,
#             128: 2,
#             320: 1,
#         }
#
#         for aux_layer in self.aux_layers:
#             src_channel = self.layer2channel[aux_layer]
#             src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
#             target_spatial_ratio = self.channel_to_spatial_ratio[self.out_channel]
#
#             # -------------------------- 修正1：重新计算调整倍数与场景判断 --------------------------
#             if target_spatial_ratio > src_spatial_ratio:
#                 # 目标特征更小 → 需要下采样（缩小辅助特征）
#                 # 步长 = 辅助特征下采样倍数 / 目标特征下采样倍数（确保步长为正整数）
#                 down_stride = src_spatial_ratio // target_spatial_ratio
#                 # 验证步长合法性（防止整数除法导致步长为0）
#                 assert down_stride > 0, f"下采样步长必须为正，当前计算为 {down_stride}（src={src_spatial_ratio}, target={target_spatial_ratio}）"
#                 # 定义下采样卷积（步长=down_stride，一次卷积缩小 down_stride 倍）
#                 downsample = nn.Conv2d(
#                     src_channel, src_channel, 3, stride=down_stride, padding=1, bias=False
#                 )
#             elif target_spatial_ratio < src_spatial_ratio:
#                 # 目标特征更大 → 需要上采样（放大辅助特征）
#                 # 上采样倍数 = 目标特征下采样倍数 / 辅助特征下采样倍数（确保倍数为正整数）
#                 up_scale = target_spatial_ratio // src_spatial_ratio
#                 assert up_scale > 0, f"上采样倍数必须为正，当前计算为 {up_scale}（src={src_spatial_ratio}, target={target_spatial_ratio}）"
#                 # 定义上采样层
#                 upsample = nn.Upsample(
#                     scale_factor=up_scale, mode='nearest', align_corners=None
#                 )
#             else:
#                 # 分辨率一致 → 无需上/下采样
#                 downsample = None
#                 upsample = None
#
#             # -------------------------- 修正2：构建特征转换网络（基于正确的场景） --------------------------
#             layers = []
#             # 1. 下采样（仅在目标特征更小时执行，放在C3前减少计算量）
#             if target_spatial_ratio > src_spatial_ratio:
#                 layers.append(downsample)
#                 layers.append(BN(src_channel))
#                 layers.append(nn.SiLU(inplace=True))
#
#             # 2. C3模块（通道转换+特征融合，输入=src_channel，输出=self.out_channel）
#             layers.append(C3(c1=src_channel, c2=self.out_channel, n=1, shortcut=True))
#
#             # 3. 上采样（仅在目标特征更大时执行，放在C3后避免引入冗余信息）
#             if target_spatial_ratio < src_spatial_ratio:
#                 layers.append(upsample)
#
#             # 整合网络并存储
#             t = nn.Sequential(*layers)
#             self.convs[aux_layer] = t
#
#         # 参数初始化部分保持不变（C3内部的卷积/BN层会被自动初始化）
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal(m.weight, mean=0.0, std=1e-3)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, mean=0.0, std=1e-3)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 0.0)  # 注意：通常BN权重初始化为1，根据需求调整
#                 nn.init.constant_(m.bias, 0.0)
#
#     def forward(self, x: dict, y: torch.tensor, detach=False):
#         #detach:True x:即inp_feature[gv_patch]:[input(2,3,608,608),prelayer(2,64,304,464),layer1(2,256,152,232),layer2(2,512,76,116),layer3(2,1024,38,58),layer3(2,2048,19,29),pre_fc(2,2048),]
#         #y即inp_feature[gv_global][layer1]:[layer1(2,256,152,232),]
#         transformed_aux_features = []
#         for aux_layer_name in sorted(self.aux_layers):#self.aux_layers:['layer1']
#             transformed_aux_features.append(self.convs[aux_layer_name](#对
#                 (x[aux_layer_name].detach() if detach else x[aux_layer_name])))
#
#         out = sum(transformed_aux_features) + y
#
#         return out


class Cross_Conv_HRNet_Layer_router_v2(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None,
                 **kwargs):
        super(Cross_Conv_HRNet_Layer_router_v2, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]
        self.channel_to_spatial_ratio = {
            # For ResNet
            256: 4,
            512: 2,
            1024: 1,
            # For FMTB
            768: 1,
            192: 1,
            # For MTB
            64: 4,
            128: 2,
            320: 1,
        }

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[
                self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    C3(c1=src_channel, c2=self.out_channel, n=1,k=1,s=1,p=0, shortcut=True),
                    nn.Upsample(scale_factor=target_spatial_ratio //
                                src_spatial_ratio,
                                mode='nearest',
                                align_corners=None),
                )

            else:
                conv_layers = []
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 -
                               1):
                    conv_layers.append(
                        C3(c1=src_channel, c2=src_channel, n=1,k=3,s=2,p=1, shortcut=True))
                conv_layers.append(
                    C3(c1=src_channel, c2=self.out_channel, n=1,k=3,s=2,p=1, shortcut=True))
                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1e-3)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: dict, y: torch.tensor, detach=False):
        #detach:True x:即inp_feature[gv_patch]:[input(2,3,608,608),prelayer(2,64,304,464),layer1(2,256,152,232),layer2(2,512,76,116),layer3(2,1024,38,58),layer3(2,2048,19,29),pre_fc(2,2048),]
        #y即inp_feature[gv_global][layer1]:[layer1(2,256,152,232),]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):#self.aux_layers:['layer1']
            transformed_aux_features.append(self.convs[aux_layer_name](#对
                (x[aux_layer_name].detach() if detach else x[aux_layer_name])))

        out = sum(transformed_aux_features) + y

        return out



def scalablelayer(channel, **kwargs):
    return Scalable_Layer(channel, **kwargs)


def convlayer(channel, **kwargs):
    return Conv_Layer(channel, **kwargs)


def selayer(channel, **kwargs):
    return SELayer(channel, **kwargs)


def gatinglayer(channel, **kwargs):
    return Gating_Layer(channel, **kwargs)


def attentionlayer(channel, **kwargs):
    return Attention_Layer(channel, **kwargs)


def crossconvlayer(channel, **kwargs):
    return Cross_Conv_Layer(channel, **kwargs)

def crossconvlayerdsc(channel, **kwargs):
    return Cross_Conv_Layer_DSC(channel, **kwargs)

def crossconvlayerdtf(channel, **kwargs):
    return Cross_Conv_Layer_DTF(channel, **kwargs)

def crossconvlayertvam(channel, **kwargs):
    return Cross_Conv_Layer_TVAM(channel, **kwargs)

def crossconvlayerrouterv1(channel, **kwargs):
    return Cross_Conv_Layer_routerv1(channel, **kwargs)

def crossconvlayerrouterv1onlyirouter(channel, **kwargs):
    return Cross_Conv_Layer_routerv1_only_Irouter(channel, **kwargs)

def crossconvlayerrouterv1onlyorouter(channel, **kwargs):
    return Cross_Conv_Layer_routerv1_only_Orouter(channel, **kwargs)

def crossconvhrnetlayer(channel, **kwargs):
    return Cross_Conv_HRNet_Layer(channel, **kwargs)

def convlayersedsc(channel, **kwargs):
    return Conv_Layer_SE_DSC(channel, **kwargs)

def crossconvhrnetlayermoe(channel, **kwargs):
    return Cross_Conv_HRNet_Layer_moe(channel, **kwargs)

def crossconvhrnetlayermoeup(channel, **kwargs):
    return Cross_Conv_HRNet_Layer_moe_up(channel, **kwargs)

def crossconvhrnetlayerrouter(channel, **kwargs):
    return Cross_Conv_HRNet_Layer_router(channel, **kwargs)

def crossconvhrnetlayerrouterv2(channel, **kwargs):
    return Cross_Conv_HRNet_Layer_router_v2(channel, **kwargs)

