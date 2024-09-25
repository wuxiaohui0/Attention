import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 使用 nn.ModuleList 来存储子模块
        self.convs = nn.ModuleList(
            [nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1).to(self.device) for i in range(S)]
        )

        # 使用 nn.ModuleList 来存储子模块
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=False),  # 修改为 inplace=False
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ).to(self.device) for _ in range(S)
        ])

        self.softmax = nn.Softmax(dim=1).to(self.device)
        self.to(self.device)  # 将模型整体移动到指定设备

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
        self.to(self.device)  # 确保权重也被移动到正确的设备

    def forward(self, x):
        x = x.to(self.device)  # 将输入移到同一设备
        b, c, h, w = x.size()

        # Step1: SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w).clone()  # 使用 clone() 来避免 inplace 修改
        SPC_out_new = torch.zeros_like(SPC_out)  # 新的张量用于存储结果，避免 in-place 操作
        for idx, conv in enumerate(self.convs):
            output = SPC_out[:, idx, :, :, :].to(self.device)
            SPC_out_new[:, idx, :, :, :] = conv(output)

        # Step2: SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out_new[:, idx, :, :, :]))  # 使用新的 SPC_out_new
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out_new)

        # Step3: Softmax
        softmax_out = self.softmax(SE_out)

        # Step4: PSA
        PSA_out = SPC_out_new * softmax_out  # 使用新的 SPC_out_new
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out
