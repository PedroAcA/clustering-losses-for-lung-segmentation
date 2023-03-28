from torch import nn
import torch
from functools import reduce
import operator
from torch.nn import functional as F

# class RFCM_loss from https://github.com/junyuchen245/Semi-supervised_FCM_Loss_for_Segmentation/blob/main/FCM_loss/torch/FCM_losses.py (om 14th, August 2022)


class RFCMLoss(nn.Module):
    def __init__(self, fuzzy_factor=2, regularizer_wt=0.0008, chkpt=None):
        '''
        Unsupervised Robust Fuzzy C-mean loss function for ConvNet based image segmentation
        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).
        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2
        :param regularizer_wt: weighting parameter for regularization, default value = 0
        Note that ground truth segmentation is NOT needed in this loss fuction, instead, the input_tensor image is required.
        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param image: input_tensor image to the ConvNet.
        '''
        super().__init__()
        self.fuzzy_factor = fuzzy_factor
        self.wt = regularizer_wt
        self.print_loss = False
        self.chkpt = chkpt

    def forward(self, y_pred, image):
        dim = len(list(y_pred.shape)[2:])
        assert dim == 3 or dim == 2, 'Supports only 3D or 2D!'
        num_clus = y_pred.shape[1]
        pred = torch.reshape(y_pred, (y_pred.shape[0], num_clus, reduce(operator.mul, list(y_pred.shape)[2:], 1))) #(bs, C, V)
        img = torch.reshape(image, (y_pred.shape[0], reduce(operator.mul, list(image.shape)[2:], 1))) #(bs, V)
        if dim == 3:
            kernel = torch.ones((1, 1, 3, 3, 3)).float().cuda()
            kernel[:, :, 1, 1, 1] = 0
        else:
            kernel = torch.ones((1, 1, 3, 3)).float().cuda()
            kernel[:, :, 1, 1] = 0

        J_1 = 0
        J_2 = 0
        for i in range(num_clus):
            mem = torch.pow(pred[:, i, ...], self.fuzzy_factor) #extracting membership function (bs, V)
            v_k = torch.sum(img * mem, dim=1, keepdim=True)/torch.sum(mem, dim=1, keepdim=True) #scalar
            J_1 += mem*torch.square(img - v_k) #(bs, V)
            J_in = 0
            for j in range(num_clus):
                if i==j:
                    continue
                mem_j = torch.pow(pred[:, j, ...], self.fuzzy_factor)
                mem_j = torch.reshape(mem_j, image.shape)
                if dim == 3:
                    res = F.conv3d(mem_j, kernel, padding=int(3 / 2))
                else:
                    res = F.conv2d(mem_j, kernel, padding=int(3 / 2))
                res = torch.reshape(res, (-1, reduce(operator.mul, list(image.shape)[2:], 1)))
                J_in += res #(bs, V)
            J_2 += mem * J_in #(bs, V)

        level_loss = torch.mean(J_1)
        spatial_loss = self.wt*torch.mean(J_2)
        if self.print_loss:
            detailed_losses = "Level loss: {}, reg*spatial_loss {}".format(level_loss, spatial_loss)
            if self.chkpt:
                self.chkpt.log_training(detailed_losses)

            print(detailed_losses)

        return level_loss + spatial_loss

    def set_display(self, display_loss):
        self.print_loss = display_loss


# levelsetLoss and gradientLoss2d for MS loss from https://github.com/jongcye/CNN_MumfordShah_Loss/blob/master/models/loss.py on 15th of AUg, 2022


class levelsetLoss(nn.Module):
    def __init__(self):
        super(levelsetLoss, self).__init__()

    def forward(self, output_tensor, target):
        # input size = batch x 1 (channel) x height x width
        outshape = output_tensor.shape
        tarshape = target.shape
        loss = 0.0
        for ich in range(tarshape[1]):
            target_ = torch.unsqueeze(target[:, ich], 1)
            target_ = target_.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pcentroid = torch.sum(target_ * output_tensor, (2, 3)) / torch.sum(output_tensor, (2, 3))
            pcentroid = pcentroid.view(tarshape[0], outshape[1], 1, 1)
            plevel = target_ - pcentroid.expand(tarshape[0], outshape[1], tarshape[2], tarshape[3])
            pLoss = plevel * plevel * output_tensor
            loss += pLoss

        loss = torch.sum(loss, dim=1) #sum across output softmax channels
        return torch.mean(loss)


class gradientLoss2d(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss2d, self).__init__()
        self.penalty = penalty

    def forward(self, input_tensor):
        dH = torch.abs(input_tensor[:, :, 1:, :] - input_tensor[:, :, :-1, :])
        dW = torch.abs(input_tensor[:, :, :, 1:] - input_tensor[:, :, :, :-1])
        if (self.penalty == "l2"):
            dH = dH * dH
            dW = dW * dW

        dH = torch.sum(dH, dim=1)#sum across output softmax channels
        dW = torch.sum(dW, dim=1)#sum across output softmax channels
        loss = torch.mean(dH) + torch.mean(dW)
        return loss


class MSLoss(nn.Module):
    def __init__(self, reg, chkpt=None):
        super(MSLoss, self).__init__()
        self.lambda_reg = reg
        self.level_set_loss = levelsetLoss()
        self.spatial_loss = gradientLoss2d()
        self.print_loss = False
        self.chkpt = chkpt

    def forward(self, prediction, target):
        level_loss = self.level_set_loss(prediction, target)
        spatial_loss = self.lambda_reg*self.spatial_loss(prediction)
        if self.print_loss:
            detailed_losses = "Level loss: {}, reg*spatial_loss {}".format(level_loss, spatial_loss)
            if self.chkpt:
                self.chkpt.log_training(detailed_losses)

            print(detailed_losses)

        return level_loss + spatial_loss

    def set_display(self, display_loss):
        self.print_loss = display_loss

