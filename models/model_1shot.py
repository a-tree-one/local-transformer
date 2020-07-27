import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# from .resnet_dilated import resnet101, resnet50
device_ids = [1, 1]


class ASPP(nn.Module):
    # have bias and relu, no bn
    def __init__(self, in_channel=512, depth=256):
        super().__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),)

        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, depth, 1, 1),
                                           )
        self.atrous_block6 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2),
                                           )
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4),
                                        )
        self.atrous_block18 = nn.Sequential(nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6))

        self.conv_1x1_output = nn.Sequential(nn.Conv2d(depth * 5, depth, 1, 1))

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = nn.functional.interpolate(image_features, size=size, mode='bilinear', align_corners=True)

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

    def inverse_trans(self, x_s_mask, f_size):
        '''
        To get the generalized inverse
        :param x_s_mask:
        :param f_size:
        :return:
        '''
        x_s_mask = F.interpolate(x_s_mask, (f_size, f_size), mode='bilinear')
        x_s_mask = x_s_mask.view(x_s_mask.size()[0], 1, -1)
        x_s_mask_t = x_s_mask.permute(0, 2, 1)
        x_s_mask_mul = torch.matmul(x_s_mask, x_s_mask_t) + 1e-5
        # x_s_mask_mul = x_s_mask_mul.cpu().data.numpy()
        x_s_mask_mul_i = torch.inverse(x_s_mask_mul)
        x_s_mask_gene_i = torch.matmul(x_s_mask_t, x_s_mask_mul_i)

        return x_s_mask_gene_i

    def forward(self, x_sup, x_que, x_sup_gt):

        f_size = x_sup.size()[2]

        x_sup_gt_gene_inv = self.inverse_trans(x_sup_gt, f_size)

        x_sup = x_sup.view(x_sup.size()[0], x_sup.size()[1], -1)
        x_que = x_que.view(x_que.size()[0], x_que.size()[1], -1)

        x_que_norm = torch.norm(x_que, p=2, dim=1, keepdim=True)
        x_sup_norm = torch.norm(x_sup, p=2, dim=1, keepdim=True)

        x_que_norm = x_que_norm.permute(0, 2, 1)

        x_qs_norm = torch.matmul(x_que_norm, x_sup_norm)

        x_que = x_que.permute(0, 2, 1)

        x_qs = torch.matmul(x_que, x_sup)

        x_qs = x_qs / (x_qs_norm+1e-5)

        R_qs = x_qs

        x_att = torch.matmul(x_qs, x_sup_gt_gene_inv)

        x_att = x_att.view(x_att.size(0), 1, f_size, f_size)
        x_att = cam_normalize(x_att)

        return x_att, R_qs


class UpSample(nn.Module):
    def __init__(self, in_channels):
        super(UpSample, self).__init__()
        self.up_0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
        )

        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(False),
        )

        self.up_1_dealing = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
            # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
        )

        self.up_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.up_2_dealing = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(0.3),

            # nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
        )

        self.up_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.up_3_dealing = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(0.3),

            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout2d(0.3),
        )

        self.up_4 = nn.Sequential(
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            # nn.Sigmoid()
            nn.UpsamplingBilinear2d((320, 320)),
        )
        # self.aspp = ASPP(128, 128)

        self.aspp1 = ASPP(64, 64)

    def forward(self, x):
        x_0 = self.up_0(x)
        x_0_out = x_0 + x

        # x_0_out = self.aspp(x_0_out)

        x_1 = self.up_1(x_0_out)

        x_1_dealing = self.up_1_dealing(x_1)
        x_1_out = x_1_dealing + x_1

        # x_1_out = self.aspp(x_1_out)

        x_2 = self.up_2(x_1_out)
        x_2_dealing = self.up_2_dealing(x_2)
        x_2_out = x_2_dealing + x_2
        x_2_out = self.aspp1(x_2_out)

        x_3 = self.up_3(x_2_out)
        x_3_dealing = self.up_3_dealing(x_3)
        x_3_out = x_3_dealing + x_3

        x_out = self.up_4(x_3_out)
        return x_out


def cam_normalize(cam):

    # x_temp = nn.functional.relu(cam)
    x_temp = cam
    x_min = torch.min(x_temp.view(x_temp.size(0), x_temp.size(1), -1), 2)[0].unsqueeze(2).unsqueeze(2)
    x_min = x_min.expand(x_temp.size())

    x_max = torch.max(x_temp.view(x_temp.size(0), x_temp.size(1), -1), 2)[0].unsqueeze(2).unsqueeze(2)
    x_max = x_max.expand(x_temp.size())
    # x_temp = x_temp - x_min
    x_temp_que = (x_temp - x_min) / (x_max - x_min + 1e-5)

    return x_temp_que


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.feature = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])
        for p in self.feature.parameters():
            p.requires_grad = False

        self.feature_transformer = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1),
            nn.ReLU(),
        )

        self.embedding_transformer = nn.Sequential(
            nn.Conv2d(1024, 1000, kernel_size=1),
            # nn.Conv2d(1024, 1024, kernel_size=1),
            # nn.Conv2d(2048, 2048, kernel_size=1),
            # nn.Sigmoid()
        )

        self.upsample_que = UpSample(in_channels=2048)

        self.transformer = Transformer()

    def feature_trun(self, x_s_3, x_s_mask, f_size):
        x_s_mask_3 = F.interpolate(x_s_mask, (f_size, f_size), mode='bilinear', align_corners=True)
        x_s_mask_3 = x_s_mask_3.expand_as(x_s_3)
        x_s_3 = x_s_3 * x_s_mask_3
        return x_s_3

    def att2mask(self, x_q_3_att):
        x_q_att_tra_3 = torch.ones(x_q_3_att.size()).type(torch.FloatTensor).to(device=device_ids[0]) - x_q_3_att
        x_q_att_mask_3 = torch.cat((x_q_att_tra_3, x_q_3_att), 1)
        x_q_att_mask_3 = F.interpolate(x_q_att_mask_3, (320, 320), mode='bilinear', align_corners=True)
        return x_q_att_mask_3

    def reference_one(self, x_q, img_size):

        x_q = F.interpolate(x_q, (img_size, img_size), mode='bilinear', align_corners=True)
        # x_s = F.interpolate(x_s, (img_size, img_size), mode='bilinear', align_corners=True)
        # x_s_mask = F.interpolate(x_s_mask, (img_size, img_size), mode='bilinear', align_corners=True)

        # x_s_3 = self.feature(x_s)
        # x_s_3 = self.feature_trun(x_s_3, x_s_mask, f_size)
        x_q_3 = self.feature(x_q)

        x_q_3_f = self.feature_transformer(x_q_3)

        # att, R = self.transformer(self.embedding_transformer(x_s_3),
        #                           self.embedding_transformer(x_q_3),
        #                           x_s_mask)
        # att_mask = self.att2mask(att)

        # att = F.interpolate(att, (20, 20), mode='bilinear', align_corners=True)
        x_q_3 = F.interpolate(x_q_3, (20, 20), mode='bilinear', align_corners=True)

        x_q_3_f = F.interpolate(x_q_3_f, (20, 20), mode='bilinear', align_corners=True)

        return x_q_3, x_q_3_f

    def train_net(self, x_q, x_s, x_s_mask):
        q_f_1x0, q_f_t_1x0 = self.reference_one(x_q, 320)
        q_f_1x3, q_f_t_1x3 = self.reference_one(x_q, 416)
        q_f_0x7, q_f_t_0x7 = self.reference_one(x_q, 224)
        q_f = (q_f_0x7 + q_f_1x0 + q_f_1x3) / 3
        q_f_t = (q_f_t_1x0 + q_f_t_1x3 + q_f_t_0x7) / 3

        x_s_mask = x_s_mask.unsqueeze(1)

        x_s_3 = self.feature(x_s)
        x_s_3 = self.feature_trun(x_s_3, x_s_mask, 20)

        att, R = self.transformer(self.embedding_transformer(x_s_3),
                                  self.embedding_transformer(q_f),
                                  x_s_mask)
        att_mask = self.att2mask(att)

        q_f_att = q_f_t * att

        seg_q = self.upsample_que(q_f_att)

        return att_mask, seg_q, R

    def val_net(self, x_q, x_s, x_s_mask):

        q_f_1x0, q_f_t_1x0 = self.reference_one(x_q, 320)
        q_f_1x3, q_f_t_1x3 = self.reference_one(x_q, 416)
        q_f_0x7, q_f_t_0x7 = self.reference_one(x_q, 224)
        q_f = (q_f_0x7 + q_f_1x0 + q_f_1x3) / 3
        q_f_t = (q_f_t_1x0 + q_f_t_1x3 + q_f_t_0x7) / 3

        x_s_mask = x_s_mask.unsqueeze(1)

        x_s_3 = self.feature(x_s)
        x_s_3 = self.feature_trun(x_s_3, x_s_mask, 20)

        att, R = self.transformer(self.embedding_transformer(x_s_3),
                                  self.embedding_transformer(q_f),
                                  x_s_mask)
        # att_mask = self.att2mask(att)

        q_f_att = q_f_t * att

        seg_q = self.upsample_que(q_f_att)

        return seg_q, seg_q

    def forward(self, x_q, x_s, x_s_mask, is_train):
        if is_train:
            return self.train_net(x_q, x_s, x_s_mask)
        else:
            return self.val_net(x_q, x_s, x_s_mask)


if __name__ == '__main__':
    x = torch.Tensor(1, 3, 320, 320).cuda()
    m = torch.Tensor(1, 320, 320).cuda()
    model = Model().cuda()
    out = model(x, x, m)
    print(out)
