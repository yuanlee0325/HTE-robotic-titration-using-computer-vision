import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg16_bn, vgg11_bn
from torchvision.models import vgg 


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1', out_channels=2):
        super().__init__()

        self.encoder = vgg16_bn(weights=weights).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
    

def up_conv_ups(in_channels, out_channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear'),
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),padding=(1,1)),
        nn.ReLU(inplace=True)
    )
    
    
    
class UNet_ups(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1', out_channels=2):
        super().__init__()

        self.encoder = vgg16_bn(weights=weights).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv_ups(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv_ups(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv_ups(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv_ups(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv_ups(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
        
        
class UNet_ups_simple(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1', out_channels=2):
        super().__init__()

        self.encoder = vgg16_bn(weights=weights).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv_ups(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv_ups(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv_ups(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv_ups(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv_ups(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
        
        
        
class UNet_vgg11(nn.Module):
    def __init__(self,
                 weights=vgg.VGG11_BN_Weights.IMAGENET1K_V1,#'IMAGENET1K_V1',
                 num_classes=2,
                 up_conv_layer='transpose2d'):
        
        super().__init__()
        
        # encoder layers from vgg11_bn
        vggs=vgg11_bn(weights=weights)
        self.down_conv1 = nn.Sequential(*vggs.features[:3])
        self.down_conv2 = nn.Sequential(*vggs.features[4:8])
        self.down_conv3 = nn.Sequential(*vggs.features[8:15])
        self.down_conv4 = nn.Sequential(*vggs.features[15:22])
        self.down_conv5 = nn.Sequential(*vggs.features[22:])
        
        # bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                        nn.ReLU(inplace = True),
                                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.conv_bottleneck = self.__class__._conv(512,1024)
        
        # decoder layers
        if up_conv_layer == 'transpose2d' :
            self.up_conv6=self.__class__._up_conv_transpose2d(1024,512)
            self.conv6 = self.__class__._conv(1024,512)
            
            self.up_conv7=self.__class__._up_conv_transpose2d(512,256)
            self.conv7 = self._conv(768,256)
            
            self.up_conv8=self.__class__._up_conv_transpose2d(256,128)
            self.conv8 = self.__class__._conv(384,128)
            
            self.up_conv9=self.__class__._up_conv_transpose2d(128,64)
            self.conv9 = self.__class__._conv(192,64)
            
            self.up_conv10=self.__class__._up_conv_transpose2d(64,32)
            self.conv10 = self.__class__._conv(96,32)
            
        else : # else upsampling + conv2d
            self.up_conv6=self.__class__._up_conv_upsampling(1024,512)
            self.conv6 = self.__class__._conv(1024,512) # re-writing for clarity!
            
            self.up_conv7=self.__class__._up_conv_upsampling(512,256)
            self.conv7 = self.__class__._conv(768,256)
            
            self.up_conv8=self.__class__._up_conv_upsampling(256,128)
            self.conv8 = self.__class__._conv(384,128)
            
            self.up_conv9=self.__class__._up_conv_upsampling(128,64)
            self.conv9 = self._conv(192,64)
            
            self.up_conv10=self.__class__._up_conv_upsampling(64,32)
            self.conv10 = self.__class__._conv(96,32)
            
        self.out = nn.Conv2d(32, num_classes, kernel_size=(1,1))
        
        
    def forward(self,x):
        
        # encoders
        dblock1=self.down_conv1(x)
        dblock2=self.down_conv2(dblock1)
        dblock3=self.down_conv3(dblock2)
        dblock4=self.down_conv4(dblock3)
        dblock5=self.down_conv5(dblock4)
        
        # bottlenecks
        x = self.conv_bottleneck(self.bottleneck(dblock5))
        
        # decoders
        x = self.up_conv6(x)
        x = torch.cat([x, dblock5], dim=1)
        x = self.conv6(x)
        
        x = self.up_conv7(x)
        x = torch.cat([x, dblock4], dim=1)
        x = self.conv7(x)
        
        x = self.up_conv8(x)
        x = torch.cat([x, dblock3], dim=1)
        x = self.conv8(x)
        
        x = self.up_conv9(x)
        x = torch.cat([x, dblock2], dim=1)
        x = self.conv9(x)
        
        x = self.up_conv10(x)
        x = torch.cat([x, dblock1], dim=1)
        x = self.conv10(x)
    
        # readout
        x = self.out(x)
        
        return x
    
    @staticmethod
    def _up_conv_transpose2d(in_channels, out_channels):
        
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def _up_conv_upsampling(in_channels, out_channels):
        
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),padding=(1,1)),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    