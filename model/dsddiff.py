import torch
import torch.nn as nn
from model import get_sdp
from model.reconstruction import CDFormer_SR
from model.encoder import DBCE_lr, DBCE_gt, denoise
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift


def make_model(args):
    return DSDDiff(args)


class DSDDiff(nn.Module):
    def __init__(self, args):
        super(DSDDiff, self).__init__()
        timesteps = 1000
        self.SR = CDFormer_SR(upscale=int(args.scale[0])).cuda()
        self.encoder = DBCE_gt(feats=8, scale=int(args.scale[0])).cuda()
        self.condition = DBCE_lr(feats=8, scale=int(args.scale[0])).cuda()
        self.denoise = denoise(feats=32, timesteps=timesteps).cuda()
        self.netG = get_sdp.SDCDM(denoise=self.denoise,
                                  condition=self.condition, feats=64, timesteps=timesteps, parameterization="x0"
                                  ).cuda()
        self.img_range = 1.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1).cuda()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _get_preprocessed(self, x):
        img = (x[0] / 255. - self.mean) * self.img_range
        gt = (x[1] / 255. - self.mean) * self.img_range
        return img, gt

    def _post_process(self, sr):
        return (sr / self.img_range + self.mean) * 255


    def _branch_train_off(self, img, gt):
        diff_loss = torch.tensor(0.0).cuda()
        deg_gt, spa_gt = self.encoder(img, gt)
      
        sr = self.SR(img, deg_gt, spa_gt)
        return diff_loss, deg_gt, spa_gt, deg_gt, spa_gt, self._post_process(sr)

    def _branch_train_on(self, img, gt):
        self.freeze_module(self.encoder)
        deg_gt, spa_gt = self.encoder(img, gt)
        diff_loss, deg_pred, spa_pred = self.netG(img, deg_gt=deg_gt, spa_gt=spa_gt)
        sr = self.SR(img, deg_pred, spa_pred)
        return diff_loss, deg_pred, spa_pred, deg_gt, spa_gt, self._post_process(sr)



    def _branch_eval_off(self, img, gt):
        #print("1111")
        deg, spa = self.encoder(img, gt)
        sr = self.SR(img, deg, spa)
        return self._post_process(sr)

    def _branch_eval_on(self, img, gt):

        _, _ = self.encoder(img, gt) 
        _, deg_diff, spa_diff = self.netG(img)
        sr = self.SR(img, deg_diff, spa_diff)
        return self._post_process(sr)
        
    def forward(self, x):
        img, gt = self._get_preprocessed(x)
        diff = x[2]

        phase = "train" if self.training else "eval"
        mode = diff
        func_name = f"_branch_{phase}_{mode}"
        
        handler = getattr(self, func_name)
        return handler(img, gt)
