import torch
import torch.nn as nn
import torch.nn.functional as F

from fastmvsnet.nn.conv import *
import numpy as np


class ImageConv(nn.Module):
    def __init__(self, base_channels, in_channels=3):
        super(ImageConv, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 8 * base_channels
        self.conv0 = nn.Sequential(
            Conv2d(in_channels, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False)
        )


    def forward(self, imgs):
        out_dict = {}

        conv0 = self.conv0(imgs)
        out_dict["conv0"] = conv0
        conv1 = self.conv1(conv0)
        out_dict["conv1"] = conv1
        conv2 = self.conv2(conv1)
        out_dict["conv2"] = conv2

        return out_dict



class PropagationNet(nn.Module):
    def __init__(self, base_channels):
        super(PropagationNet, self).__init__()
        self.base_channels = base_channels

        self.img_conv = ImageConv(base_channels)
        
        self.conv1 = nn.Sequential(
            Conv2d(base_channels*4, base_channels * 4, 3, padding=1),
            Conv2d(base_channels * 4, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 2, 3, 1, padding=1),
            nn.Conv2d(base_channels*2, 9, 3, padding=1, bias=False)
        )

        self.unfold = nn.Unfold(kernel_size=(3,3), stride=1, padding=0)

    def forward(self, depth, img):
        img_featues = self.img_conv(img)
        img_conv2 = img_featues["conv2"]

        x = self.conv3(img_conv2)
        prob = F.softmax(x, dim=1)

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        depth_unfold = self.unfold(depth_pad)

        b, c, h, w = prob.size()
        prob = prob.view(b, 9, h*w)

        result_depth = torch.sum(depth_unfold * prob, dim=1)
        result_depth = result_depth.view(b, 1, h, w)
        return result_depth


class VolumeConv(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(VolumeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = base_channels * 8
        self.base_channels = base_channels
        self.conv1_0 = Conv3d(in_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_0 = Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv3_0 = Conv3d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.conv0_1 = Conv3d(in_channels, base_channels, 3, 1, padding=1)

        self.conv1_1 = Conv3d(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        self.conv2_1 = Conv3d(base_channels * 4, base_channels * 4, 3, 1, padding=1)

        self.conv3_1 = Conv3d(base_channels * 8, base_channels * 8, 3, 1, padding=1)
        self.conv4_0 = Deconv3d(base_channels * 8, base_channels * 4, 3, 2, padding=1, output_padding=1)
        self.conv5_0 = Deconv3d(base_channels * 4, base_channels * 2, 3, 2, padding=1, output_padding=1)
        self.conv6_0 = Deconv3d(base_channels * 2, base_channels, 3, 2, padding=1, output_padding=1)

        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, x):
        conv0_1 = self.conv0_1(x)

        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = self.conv4_0(conv3_1)

        conv5_0 = self.conv5_0(conv4_0 + conv2_1)
        conv6_0 = self.conv6_0(conv5_0 + conv1_1)

        conv6_2 = self.conv6_2(conv6_0 + conv0_1)

        return conv6_2


class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        depth_interval = depth_interval.view(-1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae


class Valid_MAELoss(nn.Module):
    def __init__(self, valid_threshold=2.0):
        super(Valid_MAELoss, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, pred_depth_image, gt_depth_image, depth_interval, before_depth_image):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        pred_height = pred_depth_image.size(2)
        pred_width = pred_depth_image.size(3)
        depth_interval = depth_interval.view(-1)
        mask_true = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        before_hight = before_depth_image.size(2)
        if before_hight != pred_height:
            before_depth_image = F.interpolate(before_depth_image, (pred_height, pred_width))
        diff = torch.abs(gt_depth_image - before_depth_image) / depth_interval.view(-1, 1, 1, 1)
        mask_valid = (diff < self.valid_threshold).type(torch.float)
        mask_valid = mask_true * mask_valid
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae
