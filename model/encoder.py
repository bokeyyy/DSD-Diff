from subprocess import check_output

import torch
import torch.nn as nn
import math
from sympy.strategies.branch import condition

import model.common as common
import torch.nn.functional as F
import numpy as np


class SpaDegInterBlock(nn.Module):
    def __init__(self, deg_dim=256, spa_dim=8, internal_dim=64, window_size=4):
        super().__init__()

        self.spatial_branch = SwinLikeSpatialBranch(
            deg_dim=deg_dim,
            spa_dim=spa_dim,
            head_dim=internal_dim,
            window_size=window_size
        )

        self.channel_branch = GlobalChannelBranch(
            deg_dim=deg_dim,
            spa_dim=spa_dim,
            internal_dim=internal_dim
        )

        self.spa_fusion = nn.Conv2d(spa_dim, spa_dim, 1)
        self.deg_fusion = nn.Linear(deg_dim, deg_dim)

    def forward(self, deg, spa):
        """
        deg: [B, 256]
        spa: [B, 8, 24, 24] -
        """
        deg_residual = deg
        spa_residual = spa

        spa_spatial_out = self.spatial_branch(deg, spa)

        deg_global_out, spa_global_vec = self.channel_branch(deg, spa)

        deg_out = self.deg_fusion(deg_global_out + deg_residual)

        spa_out = spa_spatial_out + spa_global_vec.unsqueeze(-1).unsqueeze(-1)
        spa_out = self.spa_fusion(spa_out + spa_residual)

        return deg_out, spa_out


class SwinLikeSpatialBranch(nn.Module):
    def __init__(self, deg_dim, spa_dim, head_dim, window_size=4):
        super().__init__()
        self.window_size = window_size
        self.head_dim = head_dim

        self.proj_in = nn.Conv2d(spa_dim, head_dim, 1)

        self.deg_mlp = nn.Linear(deg_dim, head_dim)

        self.cross_attn = WindowCrossAttention(dim=head_dim)
        self.norm1 = nn.LayerNorm(head_dim)


        self.ffn = nn.Sequential(
            nn.Linear(head_dim, head_dim * 4),
            nn.GELU(),
            nn.Linear(head_dim * 4, head_dim)
        )
        self.norm2 = nn.LayerNorm(head_dim)


        self.proj_out = nn.Conv2d(head_dim, spa_dim, 1)

    def forward(self, deg, spa):
        B, C, H, W = spa.shape


        x = self.proj_in(spa)  # [B, 64, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, HW, 64]


        x_windows = window_partition(x.view(B, H, W, self.head_dim), self.window_size)

        deg_emb = self.deg_mlp(deg)  # [B, 64]

        num_windows = x_windows.shape[0] // B
        deg_windows = deg_emb.unsqueeze(1).unsqueeze(0).repeat(1, num_windows, 1, 1)
        deg_windows = deg_windows.reshape(-1, 1, self.head_dim)  # [B*NumWins, 1, 64]


        attn_windows = self.cross_attn(x_windows, deg_windows)
        x_windows = self.norm1(x_windows + attn_windows)  # Residual

        x_windows = x_windows + self.ffn(x_windows)
        x_windows = self.norm2(x_windows)

        x = window_reverse(x_windows, self.window_size, H, W)  # [B, H, W, 64]

        x = x.permute(0, 3, 1, 2)  # [B, 64, H, W]
        return self.proj_out(x)


class WindowCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)  # K和V来自Deg
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        """
        x: [N, L, C] (Content Windows)
        context: [N, 1, C] (Degradation Vector)
        """
        B, L, C = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv_proj(context).reshape(B, 1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [B, heads, 1, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(x)



class GlobalChannelBranch(nn.Module):
    def __init__(self, deg_dim, spa_dim, internal_dim):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.spa_proj = nn.Linear(spa_dim, internal_dim)
        self.deg_proj = nn.Linear(deg_dim, internal_dim)


        self.fusion = nn.Sequential(
            nn.Linear(internal_dim * 2, internal_dim * 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(internal_dim * 2, internal_dim * 2),  # 输出融合特征
        )

        self.deg_mlp = nn.Sequential(
            nn.Linear(internal_dim, internal_dim * 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(internal_dim * 2, internal_dim)
        )
        self.spa_mlp = nn.Sequential(
            nn.Linear(internal_dim, internal_dim * 2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(internal_dim * 2, internal_dim)
        )

        self.deg_out = nn.Linear(internal_dim, deg_dim)
        self.spa_out = nn.Linear(internal_dim, spa_dim)

    def forward(self, deg, spa):
        b = deg.shape[0]
        # Pool & Project
        spa_vec = self.spa_proj(self.pool(spa).flatten(1))  # [B, 64]
        deg_vec = self.deg_proj(deg)  # [B, 64]

        # Interaction (Residual inside)
        combined = torch.cat([deg_vec, spa_vec], dim=1)
        fused = self.fusion(combined) + combined  # Residual

        deg_f, spa_f = torch.chunk(fused, 2, dim=1)

        # Feature Enhancement (FFN with Residual)
        deg_f = deg_f + self.deg_mlp(deg_f)
        spa_f = spa_f + self.spa_mlp(spa_f)

        # Output
        return self.deg_out(deg_f), self.spa_out(spa_f)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Encoder(nn.Module):
    def __init__(self, feats=32, scale=4):
        super(Encoder, self).__init__()
        self.scale = scale
        self.degradation_branch = nn.Sequential(
            nn.Conv2d(3 + 3 * scale * scale, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='d'),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='d'),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='d'),
            nn.Conv2d(feats, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
        )
        if self.scale == 4:
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(3, feats, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(feats, feats, kernel_size=3, stride=2, padding=1), 
                nn.LeakyReLU(0.1, True),
                common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
                common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
            )
        elif self.scale == 2:
            self.spatial_branch = nn.Sequential(
                nn.Conv2d(3, feats, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1, True),
                common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
                common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
            )
        self.pixel_unshuffle = nn.PixelUnshuffle(scale)
        self.down_conv = nn.Conv2d(3, feats * 2, kernel_size=4, stride=4, padding=0)

    def forward(self, lr, gt):
        # lr [B,3,48,48]
        # gt [B,3,192,192]

        gt0 = self.pixel_unshuffle(gt)
        # gt = self.down_conv(gt)  # [B,64,48,48]
        # print('gt', gt0.shape)
        # print(gt.shape)
        # x = torch.cat([lr, gt0], dim=1)
        # shared_feat = self.shared_encoder(x)  # [B,64,48,48]
        # print(lr.shape,gt0.shape)
        # print('111', lr.shape,gt0.shape)
        degradation = self.degradation_branch(torch.cat([lr, gt0], dim=1))  # [B,256]
        if self.scale == 4:
            spatial = self.spatial_branch(gt)
        elif self.scale == 2:
            spatial = self.spatial_branch(gt)  # [B,feat//2,gt.H//8,gt.W//8]   [2,32,24,24]
        # print('deg=',degradation.shape,'spa=',spatial.shape)
        return degradation, spatial


class EncoderOut(nn.Module):
    def __init__(self, feats=32, scale=4):
        super(EncoderOut, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 4, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 4, feats * 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.C = nn.Sequential(
            nn.Conv2d(feats, feats * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 4, feats * 8, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 8 * 2, feats * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 8, feats * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 8, feats * 8),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 8, feats * 8),
            nn.LeakyReLU(0.1, True)
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(scale)

    def forward(self, deg, spa):
        x1_ave = self.D(deg).squeeze(-1).squeeze(-1)
        x2_ave = self.C(spa).squeeze(-1).squeeze(-1)
        # print(x1_ave.shape,x2_ave.shape)
        fea = self.mlp(torch.cat([x1_ave, x2_ave], dim=1))
        return fea


class DBCE_gt(nn.Module):
    def __init__(self, spa_dim=8, feats=32, scale=4, deg_dim=256):
        super(DBCE_gt, self).__init__()
        self.encoder = Encoder(feats=feats, scale=scale)

        # Encoder[B, 8, H, W] -> Pool -> Flatten -> Linear -> [B, 256]
        self.deg_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feats, deg_dim),
            nn.LeakyReLU(0.1, True)
        )

        self.SDIB1 = SpaDegInterBlock(deg_dim=deg_dim, spa_dim=spa_dim)
        self.SDIB2 = SpaDegInterBlock(deg_dim=deg_dim, spa_dim=spa_dim)
        self.spa_adapter = nn.Conv2d(feats, spa_dim, kernel_size=1)

        self.deg_out_mlp = nn.Sequential(
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
        )


        self.c_out = nn.Sequential(
            nn.Conv2d(spa_dim, spa_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(spa_dim, spa_dim, kernel_size=3, padding=1),
            # nn.LeakyReLU(0.1, True),
        )

    def forward(self, lr, gt):
        deg_map, spa_map = self.encoder(lr, gt)
        # deg_map: [B, 8, 48, 48], spa_map: [B, 8, 24, 24]

        deg_vec = self.deg_adapter(deg_map)  # [B, 256]
        spa_map = self.spa_adapter(spa_map)

        deg_vec, spa_map = self.SDIB1(deg_vec, spa_map)
        deg_vec, spa_map = self.SDIB2(deg_vec, spa_map)

        deg_out = self.deg_out_mlp(deg_vec)  # [B, 256]
        spa_out = self.c_out(spa_map)  # [B, 3, 24, 24]

        return deg_out, spa_out


class DBCE_lr(nn.Module):
    def __init__(self, spa_dim=8, feats=32, scale=4, deg_dim=256):
        super(DBCE_lr, self).__init__()

        self.degradation_branch = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='d'),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='d'),
            nn.Conv2d(feats, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
        )

        self.spatial_branch = nn.Sequential(
            nn.Conv2d(3, feats, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
            common.ResBlock(common.default_conv, feats, kernel_size=3, mode='c'),
        )

        self.deg_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feats, deg_dim),
            nn.LeakyReLU(0.1, True)
        )
        self.spa_adapter = nn.Conv2d(feats, spa_dim, kernel_size=1)
        self.SDIB1 = SpaDegInterBlock(deg_dim=deg_dim, spa_dim=spa_dim)
        self.SDIB2 = SpaDegInterBlock(deg_dim=deg_dim, spa_dim=spa_dim)

        self.deg_out_mlp = nn.Sequential(
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
        )

        self.s_out = nn.Sequential(
            nn.Conv2d(spa_dim, spa_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=feats),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(spa_dim, spa_dim, kernel_size=3, padding=1),
        )

    def forward(self, lr):
        deg_map = self.degradation_branch(lr)  # [B, 8, 48, 48]
        spa_map = self.spatial_branch(lr)  # [B, 8, 48, 48]

        deg_vec = self.deg_adapter(deg_map)
        spa_map = self.spa_adapter(spa_map)

        deg_vec, spa_map = self.SDIB1(deg_vec, spa_map)
        deg_vec, spa_map = self.SDIB2(deg_vec, spa_map)

        deg_out = self.deg_out_mlp(deg_vec)
        spa_out = self.s_out(spa_map)

        return deg_out, spa_out


class TimeConditionAdjust(nn.Module):
    def __init__(self, channel=3, t_dim=256):
        super().__init__()
        self.channel = channel

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, channel)  # [B, feats] -> [B, channel]
        )

        self.modulator = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积
            nn.SiLU(),
            nn.Conv2d(channel, channel, kernel_size=1)
        )

    def forward(self, x_c, t_emb):

        t_proj = self.time_proj(t_emb)  # [B, c]
        t_proj = t_proj.view(-1, self.channel, 1, 1)  # [B, 32, 1, 1]
        # print('x_s=', x_s.shape)
        # print('x_s=',x_s.shape,'spa=', (self.modulator(x_c)).shape, 't=', t_embed.shape)
        modulated = x_c * (1 + self.modulator(x_c) * t_proj)

        return modulated


class TimeCondNoiseFuse(nn.Module):
    def __init__(self, channels=3, t_dim=256):
        super().__init__()
        self.channels = channels

        self.feature_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.cond_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.time_to_alpha = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, modulated_cond, t_emb):

        x_proj = self.feature_proj(x)  # [B, C, H, W]
        cond_proj = self.cond_proj(modulated_cond)  # [B, C, H, W]
        #print(x_proj.shape,cond_proj.shape)
        gate = self.gate(torch.cat([x_proj, cond_proj], dim=1))  # [B, C, H, W]

        alpha = self.time_to_alpha(t_emb)  # [B, 1]
        alpha = alpha.view(-1, 1, 1, 1)
        return x * (1 - alpha * gate) + modulated_cond * (alpha * gate)


class LightweightUNet(nn.Module):
    def __init__(self, in_c=8, feats=32, t_dim=256,):
        super().__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.feats = feats
        # self.time_embed = nn.Sequential(
        #     nn.Linear(1, feats),
        #     nn.SiLU(),
        #     nn.Linear(feats, feats)
        # )
        self.init_conv = nn.Conv2d(in_c, feats, kernel_size=3, padding=1)
        self.init_conv_c = nn.Conv2d(in_c, feats, kernel_size=3, padding=1)
        # down
        self.adjust_down = TimeConditionAdjust(feats, t_dim)
        self.fuse_down = TimeCondNoiseFuse(feats, t_dim)
        self.encoder1 = self._block(feats, feats * 2)  # 24x24 -> 24x24
        self.encoder2 = self._block(feats * 2, feats * 4)
        self.encoder3 = self._block(feats * 4, feats * 4)
        self.cond_encoder1 = self._block(feats, feats * 2)  # 24x24 -> 24x24
        self.cond_encoder2 = self._block(feats * 2, feats * 4)
        self.cond_encoder3 = self._block(feats * 4, feats * 4)

        self.downsample1 = nn.MaxPool2d(2)
        self.cond_downsample1 = nn.MaxPool2d(2)
        # mid
        self.adjust_mid = TimeConditionAdjust(feats * 4, t_dim)
        self.fuse_mid = TimeCondNoiseFuse(feats * 4, t_dim)
        self.encoder_mid1 = self._block(feats * 4, feats * 4)  # 24x24 -> 12x12
        self.encoder_mid2 = self._block(feats * 4, feats * 4)
        self.cond_encoder_mid1 = self._block(feats * 4, feats * 4)  # 24x24 -> 12x12
        self.cond_encoder_mid2 = self._block(feats * 4, feats * 4)

        # self.downsample2 = nn.MaxPool2d(2)

        # self.adjust_mid = TimeConditionAdjust(feats * 4)
        # self.fuse_mid = TimeCondNoiseFuse(feats * 4)
        # self.adjust_mid = TimeConditionAdjust(feats * 4)
        # self.fuse_mid = TimeCondNoiseFuse(feats * 4)
        # self.mid = self._block(feats * 4, feats * 4)  # 12x12 -> 12x12
        # up
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 12x12 -> 24x24
        self.cond_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 12x12 -> 24x24
        # skip
        self.skip_reducer = nn.Conv2d(feats * 8, feats * 4, kernel_size=1)

        self.adjust_up = TimeConditionAdjust(feats * 4)
        self.fuse_up = TimeCondNoiseFuse(feats * 4)
        self.decoder1 = self._block(feats * 4, feats * 2)
        self.decoder2 = self._block(feats * 2, feats)  # 24x24 -> 24x24
        self.decoder3 = self._block(feats, feats)
        self.final_conv = nn.Conv2d(feats, in_c, kernel_size=1)
        # self.final_conv_c = nn.Conv2d(feats, in_c, kernel_size=1)
        # self.final = nn.Conv2d(feats, feats, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.1, True),
            # nn.GroupNorm(4, out_channels),
            # nn.SiLU()
        )

    def forward(self, x, x_c, t):
        # print(t.shape)
        _, _, h, w = x.shape
        # t_embed = self.time_embed(t.unsqueeze(-1))  # (B, base_channels)
        # t_embed = t_embed.view(-1, self.in_channels, 1, 1).expand(-1, -1, h, w)
        # print(t_embed.shape)
        x = self.init_conv(x)
        x_c = self.init_conv_c(x_c)
        #print(x.shape,x_c.shape)
        x1 = self.fuse_down(x, self.adjust_down(x_c, t), t)  # (B, 32, 24, 24)
        # print(x1.shape)
        x1_skip = self.encoder3(self.encoder2(self.encoder1(x1)))
        x1 = self.downsample1(x1_skip)  # (B, 128, 12, 12)
        x_c = self.cond_downsample1(self.cond_encoder3(self.cond_encoder2(self.cond_encoder1(x_c))))
        # print(x1.shape)

        # x2 = self.fuse_down2(x1, self.adjust_down2(x_c, t), t)  # (B, 64, 12, 12)
        # x2 = self.downsample2(self.encoder4(self.encoder3(x2)))  # (B, 128, 6, 6)

        bottleneck = self.fuse_mid(x1, self.adjust_mid(x_c, t), t)  # (B, 128, 12, 12)
        # print(bottleneck.shape)
        bottleneck = self.encoder_mid2(self.encoder_mid1(bottleneck))  # (B, 128, 12, 12)
        x_c = self.cond_encoder_mid2(self.cond_encoder_mid1(x_c))
        # print(bottleneck.shape)

        x_up = self.upsample(bottleneck)  # (B, 128, 24, 24)
        x_c = self.cond_upsample(x_c)
        x_up = torch.cat([x_up, x1_skip], dim=1)
        x_up = self.skip_reducer(x_up)
        # print(x_up.shape)
        x_up = self.fuse_up(x_up, self.adjust_up(x_c, t), t)  # (B, 128, 24, 24)
        # print(x_up.shape)
        x_up = self.decoder3(self.decoder2(self.decoder1(x_up)))  # (B, 32, 24, 24)
        # print(x_up.shape)
        # x = self.decoder1(x)  # (B, 64, 24, 24)
        # x = self.decoder2(x)  # (B, 32, 24, 24)

        return self.final_conv(x_up)  # (B, 32, 24, 24)


class SpaToDegGate(nn.Module):
    def __init__(self, deg_dim=256, spa_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spa_dim, deg_dim * 2),
            nn.Tanh()
        )

    def forward(self, deg, spa):

        spa_global = torch.mean(spa, dim=(2, 3))  # [B, 32]
        scale, shift = self.mlp(spa_global).chunk(2, dim=1)  # [B, 256], [B, 256]
        return deg * (1 + scale) + shift



class DegToSpaCrossAttention(nn.Module):
    def __init__(self, deg_dim=256, spa_dim=3):
        super().__init__()
        self.deg_expand = nn.Sequential(
            nn.Linear(deg_dim, spa_dim * 4),  # [B, 128]
            nn.Unflatten(1, (spa_dim, 2, 2)),  # [B, 32, 2, 2]
            nn.Upsample(scale_factor=12, mode='bilinear')  # [B, 32, 24, 24]
        )
        self.attn = nn.MultiheadAttention(spa_dim, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm([spa_dim])

    def forward(self, spa, deg):
        deg_feat = self.deg_expand(deg)  # [B, 32, 24, 24]

        B, C, H, W = spa.shape
        spa_flat = spa.view(B, C, -1).permute(0, 2, 1)  # [B, HW, 32]
        deg_flat = deg_feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, 32]

        attn_out, _ = self.attn(
            query=spa_flat,
            key=deg_flat,
            value=deg_flat
        )  # [B, HW, 32]

        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        attn_out = self.norm(attn_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return spa + attn_out * 0.5


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class denoise(nn.Module):
    def __init__(self, feats=32, timesteps=5, deg_dim=256, spa_dim=8, t_dim=256, total_timesteps=1000):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim)
        )
        self.max_period = timesteps * 10
        self.total_timesteps = total_timesteps

        self.deg_net = nn.Sequential(
            nn.Linear(deg_dim * 2 + t_dim, deg_dim),
            nn.LayerNorm(deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LayerNorm(deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LayerNorm(deg_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(deg_dim, deg_dim),
            nn.LeakyReLU(0.1, True),
        )
        self.spa_net = LightweightUNet(in_c=spa_dim, feats=feats, t_dim=t_dim)

        self.deg2spa_attn = DegToSpaCrossAttention(deg_dim=deg_dim, spa_dim=spa_dim)

        self.deg_to_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(deg_dim, t_dim)
        )

    def forward(self, deg, spa, t, deg_c, spa_c):
        t_emb = self.time_mlp(t)

        current_t_val = t.float().mean()
        alpha = 1.0 - (current_t_val / self.total_timesteps)
        alpha = torch.clamp(alpha, 0.0, 1.0)


        deg_input = torch.cat([deg, t_emb, deg_c], dim=1)
        deg_pred = self.deg_net(deg_input)


        interaction_feat = self.deg2spa_attn(spa, deg_pred)
        spa_fused = spa + interaction_feat * alpha

        deg_emb = self.deg_to_time(deg_pred)
        global_cond = t_emb + deg_emb

        #print(spa_fused.shape,spa_c.shape)
        spa_pred = self.spa_net(spa_fused, spa_c, global_cond)

        return deg_pred, spa_pred


if __name__ == '__main__':
    # log_file = r'C:\Users\KEY\Desktop\实验模型存放\cdformer_x4_bicubic_iso\ori_encoder\output_gtencoder.log'
    # print_loss_plt(log_file)

    # lr = torch.randn(2, 3, 48, 48)
    # gt = torch.randn(2, 3, 192, 192)
    # el = DBCE_lr()
    # deg, spa = el(lr)
    # print(deg.shape, spa.shape)
    # eg = DBCE_gt()
    # deg, spa = eg(lr, gt)
    # print(deg.shape, spa.shape)
    # x = torch.randn(2, 8, 24, 24)
    # x_c = torch.randn(2, 8, 24, 24)
    # y = torch.randn(2, 256)
    # y_c = torch.randn(2, 256)
    # t=torch.randn(2)
    # unet = LightweightUNet()
    # new_x = unet(x, x_c, t)
    # de = denoise()
    # spa,deg = de(y, x, t, y_c, x_c)
    # print(spa.shape,deg.shape)
    # print(new_x.shape)
    # adjust = TimeConditionAdjust()
    # x_c = adjust(x_c, t)
    # print(x_c.shape)
    x = torch.randn(2, 3, 512, 512)
    y = torch.randn(2, 3, 128, 128)
    el = DBCE_lr()
    deg, spa = el(y)
    print(deg.shape, spa.shape)
    eg = DBCE_gt()
    deg, spa = eg(y, x)
    print(deg.shape, spa.shape)
