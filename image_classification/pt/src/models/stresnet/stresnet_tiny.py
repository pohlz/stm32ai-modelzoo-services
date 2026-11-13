# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2025 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import torch
import torch.nn as nn

activation_choice = "relu"

activation = {"relu" : nn.ReLU(inplace=True),
              "hswish" : nn.Hardswish(inplace=True) , 
              "silu" : nn.SiLU(inplace=True) ,
              }

# ---- BasicBlock definition ----
class BasicBlock(nn.Module):
    def __init__(self, conv1, bn1, conv2, bn2, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.drop_block = nn.Identity()
        
        # self.act1 = nn.ReLU(inplace=True)
        self.act1 = activation[activation_choice]
        self.aa = nn.Identity()
        self.conv2 = conv2
        self.bn2 = bn2
        # self.act2 = nn.ReLU(inplace=True)
        self.act2 = activation[activation_choice]
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop_block(out)
        out = self.act1(out)
        out = self.aa(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)
        return out



class STResNetTiny(nn.Module):
    
    def __init__(
        self,
        num_classes=1000,
    ):
        
        super(STResNetTiny, self).__init__()
        self.num_classes = num_classes
        
        # stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,3, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        # self.act1 = nn.ReLU(inplace=True)
        self.act1 = activation[activation_choice]
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # layer1
        self.layer1 = nn.Sequential(
            BasicBlock(
                conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                bn1=nn.BatchNorm2d(64),
                conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                bn2=nn.BatchNorm2d(64),
            ),
            BasicBlock(
                conv1=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                bn1=nn.BatchNorm2d(64),
                conv2=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                bn2=nn.BatchNorm2d(64),
            ),
        )

        # layer2
        self.layer2 = nn.Sequential(
            BasicBlock(
                conv1=nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                bn1=nn.BatchNorm2d(128),
                conv2=nn.Sequential(
                    nn.Conv2d(128, 96, kernel_size=1, bias=False),
                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(96, 128, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(128),
                downsample=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(128),
                ),
            ),
            BasicBlock(
                conv1=nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                bn1=nn.BatchNorm2d(128),
                conv2=nn.Sequential(
                    nn.Conv2d(128, 80, kernel_size=1, bias=False),
                    nn.Conv2d(80, 80, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(80, 128, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(128),
            ),
        )

        # layer3
        self.layer3 = nn.Sequential(
            BasicBlock(
                conv1=nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                bn1=nn.BatchNorm2d(256),
                conv2=nn.Sequential(
                    nn.Conv2d(256, 192, kernel_size=1, bias=False),
                    nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(192, 256, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(256),
                downsample=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(256),
                ),
            ),
            BasicBlock(
                conv1=nn.Conv2d(256, 256, kernel_size=3,padding=1, bias=False),
                bn1=nn.BatchNorm2d(256),
                conv2=nn.Sequential(
                    nn.Conv2d(256, 96, kernel_size=1, bias=False),
                    nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(96, 256, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(256),
            ),
        )

        # layer4
        self.layer4 = nn.Sequential(
            BasicBlock(
                conv1=nn.Sequential(
                    nn.Conv2d(256, 208, kernel_size=1, bias=False),
                    nn.Conv2d(208, 208, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.Conv2d(208, 512, kernel_size=1, bias=False),
                ),
                bn1=nn.BatchNorm2d(512),
                conv2=nn.Sequential(
                    nn.Conv2d(512, 88, kernel_size=1, bias=False),
                    nn.Conv2d(88, 88, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(88, 512, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(512),
                downsample=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(512),
                ),
            ),
            BasicBlock(
                conv1=nn.Sequential(
                    nn.Conv2d(512, 192, kernel_size=1, bias=False),
                    nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(192, 512, kernel_size=1, bias=False),
                ),
                bn1=nn.BatchNorm2d(512),
                conv2=nn.Sequential(
                    nn.Conv2d(512, 112, kernel_size=1, bias=False),
                    nn.Conv2d(112, 112, kernel_size=3, padding=1, bias=False),
                    nn.Conv2d(112, 512, kernel_size=1, bias=False),
                ),
                bn2=nn.BatchNorm2d(512),
            ),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, self.num_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
   

if __name__ == "__main__":
    model = STResNetTiny()