import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(UNet, self).__init__()

    self.enc1 = DoubleConv(in_channels, 64)
    self.enc2 = DoubleConv(64, 128)
    self.enc3 = DoubleConv(128, 256)
    self.enc4 = DoubleConv(256, 512)
    self.bottleneck = DoubleConv(512, 1024)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
    self.dec4 = DoubleConv(1024, 512)
    self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.dec3 = DoubleConv(512, 256)
    self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.dec2 = DoubleConv(256, 128)
    self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.dec1 = DoubleConv(128, 64)

    self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
    enc1 = self.enc1(x)
    enc2 = self.enc2(self.pool(enc1))
    enc3 = self.enc3(self.pool(enc2))
    enc4 = self.enc4(self.pool(enc3))

    bottleneck = self.bottleneck(self.pool(enc4))

    dec4 = self.upconv4(bottleneck)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.dec4(dec4)

    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.dec3(dec3)

    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.dec2(dec2)

    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.dec1(dec1)

    return self.final_conv(dec1)


if __name__ == "__main__":
  model = UNet(in_channels=1, out_channels=1)  # Adjust channels as per your utils
  x = torch.randn(1, 1, 256, 256)  # Example input tensor
  y = model(x)
  print(y.shape)
