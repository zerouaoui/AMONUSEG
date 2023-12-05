import torch
import torch.nn as nn
from CONV import DoubleConv
from FFC import FFC_BN_ACT

class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)

class YNET(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_in, features=64):
        super(YNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio_in = ratio_in
        self.features = features

        # Spatial Encoder
        self.Conv1 = DoubleConv(self.in_channels, self.features)
        self.Conv2 = DoubleConv(self.features, self.features*2)
        self.Conv3 = DoubleConv(self.features*2, self.features*4)
        self.Conv4 = DoubleConv(self.features*4, self.features*4)


        self.MaxPool = nn.MaxPool2d(2, stride=2)

        # Spectral Decoder
        self.FFC_1 = FFC_BN_ACT(self.in_channels, self.features,
                         kernel_size=1,
                         ratio_gin=0,
                         ratio_gout=self.ratio_in
                         )

        self.FFC_2 = FFC_BN_ACT(self.features, self.features*2,
                         kernel_size=1,
                         ratio_gin=self.ratio_in,
                         ratio_gout=self.ratio_in
                         )

        self.FFC_3 = FFC_BN_ACT(self.features*2, self.features*4,
                         kernel_size=1,
                         ratio_gin=self.ratio_in,
                         ratio_gout=self.ratio_in
                         )

        self.FFC_4 = FFC_BN_ACT(self.features*4, self.features*4,
                         kernel_size=1,
                         ratio_gin=self.ratio_in,
                         ratio_gout=self.ratio_in
                         )

        # Bottle neck
        self.bottleneck = DoubleConv(self.features*8, self.features*16)

        # Decoder
        self.up_4 = nn.ConvTranspose2d(self.features*16, self.features*8,
                                      kernel_size=2, stride=2)

        self.decoder_4 = DoubleConv((self.features*8)*2, self.features*8)

        self.up_3 = nn.ConvTranspose2d(self.features*8, features * 4,
                                      kernel_size=2, stride=2)

        self.decoder_3 = DoubleConv((self.features * 6)*2, self.features*4)

        self.up_2 = nn.ConvTranspose2d(self.features * 4, self.features * 2,
                                       kernel_size=2, stride=2)

        self.decoder_2 = DoubleConv((self.features * 3) * 2, self.features * 2)

        self.up_1 = nn.ConvTranspose2d(self.features * 2, self.features,
                                      kernel_size=2, stride=2)

        self.decoder_1 = DoubleConv(self.features * 3, self.features)

        # Final Conv layer to generate the mask
        self.final_conv = nn.Conv2d(self.features, self.out_channels, kernel_size=1)

        # To concatenate Spectral features x_g and x_l
        self.catLayer = ConcatTupleLayer()



    def forward(self, x):

        # Spatial Encoder

        enc_1 = self.Conv1(x)
        enc_2 = self.Conv2(self.MaxPool(enc_1))
        enc_3 = self.Conv3(self.MaxPool(enc_2))
        enc_4 = self.Conv4(self.MaxPool(enc_3))

        # Spectral Encoder

        enc_f_1 = self.FFC_1(x)
        enc_l_1, enc_g_1 = enc_f_1

        enc_f_2 = self.FFC_2((self.MaxPool(enc_l_1), self.MaxPool(enc_g_1)))
        enc_l_2, enc_g_2 = enc_f_2

        enc_f_3 = self.FFC_3((self.MaxPool(enc_l_2), self.MaxPool(enc_g_2)))
        enc_l_3, enc_g_3 = enc_f_3

        enc_f_4 = self.FFC_4((self.MaxPool(enc_l_3), self.MaxPool(enc_g_3)))
        enc_l_4, enc_g_4 = enc_f_4
        enc_f_5 = self.catLayer((self.MaxPool(enc_l_4), self.MaxPool(enc_g_4)))

        enc = torch.cat((self.MaxPool(enc_4),enc_f_5), dim=1)

        # Bottleneck
        bottleneck = self.bottleneck(enc)

        # Decoder
        dec_4 = self.up_4(bottleneck)
        dec_4 = torch.cat((dec_4, enc_4, self.catLayer((enc_l_4, enc_g_4))), dim=1)
        dec_4 = self.decoder_4(dec_4)

        dec_3 = self.up_3(dec_4)
        dec_3 = torch.cat((dec_3, enc_3, self.catLayer((enc_l_3, enc_g_3))), dim=1)
        dec_3 = self.decoder_3(dec_3)

        dec_2 = self.up_2(dec_3)
        dec_2 = torch.cat((dec_2, enc_2, self.catLayer((enc_l_2, enc_g_2))), dim=1)
        dec_2 = self.decoder_2(dec_2)

        dec_1 = self.up_1(dec_2)

        dec_1 = torch.cat((dec_1, enc_1, self.catLayer((enc_l_1, enc_g_1))), dim=1)
        output = self.decoder_1(dec_1)

        # Generate the Mask
        output = self.final_conv(output)

        return output


if __name__ == "__main__":
    x = torch.randn((8, 3, 256, 256))
    model = YNET(in_channels=3, out_channels=1, ratio_in=0.5, features=64)

    preds = model(x)
    print(model)
    print(preds.shape)
    print(x.shape)










