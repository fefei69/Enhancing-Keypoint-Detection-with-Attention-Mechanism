# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tviz_models
import numpy as np 
from .spatial_softmax import SoftArgmaxPavlo

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.head_dim = in_channels // num_heads
        
        assert (
            self.head_dim * num_heads == in_channels
        ), "in_channels must be divisible by num_heads"
        
        # Define the linear transforms for query, key, and value as 1x1 convolutions
        self.conv_query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Linear layer to combine the output of multiple heads
        self.fc_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape
        
        # Pass the input through conv_query, conv_key, and conv_value and reshape
        q = self.conv_query(x).reshape(N, self.num_heads, self.head_dim, H * W)
        k = self.conv_key(x).reshape(N, self.num_heads, self.head_dim, H * W)
        v = self.conv_value(x).reshape(N, self.num_heads, self.head_dim, H * W)
        
        # Transpose q, k, v to get dimensions (N, num_heads, H*W, head_dim)
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        
        # Scaled dot-product attention
        query_key_dot = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention = self.softmax(query_key_dot)
        out = torch.matmul(attention, v)
        
        # Reshape and permute to get (N, C, H, W)
        out = out.permute(0, 1, 3, 2).reshape(N, C, H, W)
        
        # Apply the final linear transformation
        out = self.fc_out(out)
        return x + out


class Attention(nn.Module):
    def __init__(self, in_channels):
    
        super().__init__()   
        self.conv_query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels, kernel_size=1)  
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, C, H, W = x.shape
        
        q = self.conv_query(x).reshape(N, C, H*W)
        k = self.conv_key(x).reshape(N, C, H*W)   
        v = self.conv_value(x).reshape(N, C, H*W)
        
        # NOTE: The X in the formula is already added for you in the return line
        query_key_dot = torch.bmm(q, k.permute(0, 2, 1)) / (np.sqrt(C)) # without tuning temperature performs better
        attention = self.softmax(query_key_dot)
        attention = torch.bmm(attention, v)
        # Reshape the output to (N, C, H, W) before adding to the input volume
        attention = attention.reshape(N, C, H, W)
        #print("Attention: ",(x+attention).shape)
        return x + attention
  
class TransformerEncoderBlock(nn.Module):
    def __init__(self, in_channels, num_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(in_channels, num_heads)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        attn_output = self.mha(x)
        x = x + attn_output
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # print("Transformer:",x.shape)
        return x
  
  
class ResnetSimple(nn.Module):
    def __init__(
        self, n_keypoints=7, freeze=False, pretrained=True, full=False,
    ):
        super(ResnetSimple, self).__init__()
        net = tviz_models.resnet101(pretrained=pretrained)
        self.full = full
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        print("###############################")
        print('Model: ResnetSimple')

        # upconvolution and final layer
        BN_MOMENTUM = 0.1
        if not full:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )
        else:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
            )
            # This brings it up from 208x208 to 416x416
            self.upsample2 = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                ),
                nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, n_keypoints, kernel_size=1, stride=1),
            )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)

        if self.full:
            x = self.upsample2(x)

        return [x]


class DopeNetworkBelief(nn.Module):
    def __init__(
        self,
        n_keypoints=7,
        include_extractor=True,
        other=0,
        freeze=False,
        pretrained=True,
        feature_extractor="vgg",
        stage_out=6,
    ):
        super(DopeNetworkBelief, self).__init__()

        numBeliefMap = n_keypoints
        numAffinity = 0
        other = 0
        print("###############################")
        print('Model: DopeNetworkBelief')


        self.feature_extractor = feature_extractor

        # This is the where we decide to get things out
        self.stage_out = stage_out

        if include_extractor:
            if feature_extractor == "vgg":
                temp = tviz_models.vgg19(pretrained=pretrained).features
                self.vgg = nn.Sequential()
                j = 0
                r = 0
                for l in temp:
                    self.vgg.add_module(str(r), l)
                    r += 1
                    if r >= 23:
                        break
                if freeze:
                    for param in self.vgg.parameters():
                        param.requires_grad = False

                self.vgg.add_module(
                    str(r), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
                )
                self.vgg.add_module(str(r + 1), nn.ReLU(inplace=True))
                self.vgg.add_module(
                    str(r + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
                )
                self.vgg.add_module(str(r + 3), nn.ReLU(inplace=True))

            elif feature_extractor == "resnet":
                temp = models.resnet152(pretrained=pretrained)
                vgg = nn.Sequential()
                vgg.add_module("0", temp.conv1)
                vgg.add_module("1", temp.bn1)
                vgg.add_module("2", temp.relu)
                vgg.add_module("3", temp.maxpool)

                t = 4
                for module in temp.layer1:
                    vgg.add_module(str(t), module)
                    t += 1
                for module in temp.layer2:
                    vgg.add_module(str(t), module)
                    t += 1
                if freeze:
                    for param in vgg.parameters():
                        param.requires_grad = False

                # vgg1 = nn.Sequential()
                vgg.add_module(
                    str(t + 1), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
                )
                vgg.add_module(str(t + 2), nn.ReLU(inplace=True))
                vgg.add_module(
                    str(t + 3), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
                )
                vgg.add_module(str(t + 4), nn.ReLU(inplace=True))

                self.vgg = vgg

            # models

        # affinity bit of the model.
        other = 0
        numAffinity = 0

        # 2 is the belief map
        self.m1_2 = self.getModel(128 + other, numBeliefMap, True)
        self.m2_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m3_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m4_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m5_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )
        self.m6_2 = self.getModel(
            128 + numBeliefMap + numAffinity + other, numBeliefMap, False
        )

    def forward(self, x):

        out1 = self.vgg(x)
        out1_2 = self.m1_2(out1)

        if self.stage_out == 1:
            return [out1_2]

        out2 = torch.cat([out1_2, out1], 1)
        out2_2 = self.m2_2(out2)

        if self.stage_out == 2:
            return [out1_2, out2_2]

        out3 = torch.cat([out2_2, out1], 1)
        out3_2 = self.m3_2(out3)

        if self.stage_out == 3:
            return [out1_2, out2_2, out3_2]

        out4 = torch.cat([out3_2, out1], 1)
        out4_2 = self.m4_2(out4)

        if self.stage_out == 4:
            return [out1_2, out2_2, out3_2, out4_2]

        out5 = torch.cat([out4_2, out1], 1)
        out5_2 = self.m5_2(out5)

        if self.stage_out == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2]

        out6 = torch.cat([out5_2, out1], 1)
        out6_2 = self.m6_2(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2]

    def getModel(self, inChannels, outChannels, first=False):

        model = nn.Sequential()
        kernel = 7
        count = 10
        channels = 128
        padding = 3
        if first:
            padding = 1
            kernel = 3
            count = 6
        # kernel=conv
        model.add_module(
            "0",
            nn.Conv2d(
                inChannels, channels, kernel_size=kernel, stride=1, padding=padding
            ),
        )

        i = 0
        while i < count:
            i += 1
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            if i >= count - 1:
                toAdd = 0
                if first:
                    toAdd = 512 - channels
                model.add_module(
                    str(i),
                    nn.Conv2d(channels, channels + toAdd, kernel_size=1, stride=1),
                )
            else:
                model.add_module(
                    str(i),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel,
                        stride=1,
                        padding=padding,
                    ),
                )

        model.add_module(str(i + 1), nn.ReLU(inplace=True))
        model.add_module(
            str(i + 2),
            nn.Conv2d(channels + toAdd, outChannels, kernel_size=1, stride=1),
        )
        # print (model)
        # quit()
        return model


class DreamHourglassMultiStage(nn.Module):
    def __init__(
        self,
        n_keypoints,
        n_image_input_channels=3,
        internalize_spatial_softmax=True,
        learned_beta=True,
        initial_beta=1.0,
        n_stages=2,
        skip_connections=False,
        deconv_decoder=False,
        full_output=False,
    ):
        super(DreamHourglassMultiStage, self).__init__()

        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output
        
        print("###############################")
        print('Model: DreamHourglassMultiStage')


        if self.internalize_spatial_softmax:
            # This warning is because the forward code just ignores the second head (spatial softmax)
            # Revisit later if we need multistage networks where each stage has multiple output heads that are needed
            print(
                "WARNING: Keypoint softmax output head is currently unused. Prefer training new models of this type with internalize_spatial_softmax = False."
            )
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        assert isinstance(
            n_stages, int
        ), 'Expected "n_stages" to be an integer, but it is {}.'.format(type(n_stages))
        assert (
            0 < n_stages and n_stages <= 6
        ), "DreamHourglassMultiStage can only be constructed with 1 to 6 stages at this time."

        self.num_stages = n_stages

        # Stage 1
        self.stage1 = DreamHourglass(
            n_keypoints,
            n_image_input_channels,
            internalize_spatial_softmax,
            learned_beta,
            initial_beta,
            skip_connections=skip_connections,
            deconv_decoder=deconv_decoder,
            full_output=self.full_output,
        )

        # Stage 2
        if self.num_stages > 1:
            self.stage2 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 3
        if self.num_stages > 2:
            self.stage3 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 4
        if self.num_stages > 3:
            self.stage4 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 5
        if self.num_stages > 4:
            self.stage5 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

        # Stage 6
        if self.num_stages > 5:
            self.stage6 = DreamHourglass(
                n_keypoints,
                n_image_input_channels
                + n_keypoints,  # Includes the most previous stage
                internalize_spatial_softmax,
                learned_beta,
                initial_beta,
                skip_connections=skip_connections,
                deconv_decoder=deconv_decoder,
                full_output=self.full_output,
            )

    def forward(self, x, verbose=False):

        y_output_stage1 = self.stage1(x)
        y1 = y_output_stage1[0]  # Just keeping belief maps for now
        if self.num_stages == 1:
            return [y1]

        if self.num_stages > 1:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y1_upsampled = y1
            else:
                y1_upsampled = nn.functional.interpolate(
                    y1, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage2 = self.stage2(torch.cat([x, y1_upsampled], dim=1))
            y2 = y_output_stage2[0]  # Just keeping belief maps for now

            if self.num_stages == 2:
                return [y1, y2]

        if self.num_stages > 2:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y2_upsampled = y2
            else:
                y2_upsampled = nn.functional.interpolate(
                    y2, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage3 = self.stage3(torch.cat([x, y2_upsampled], dim=1))
            y3 = y_output_stage3[0]  # Just keeping belief maps for now

            if self.num_stages == 3:
                return [y1, y2, y3]

        if self.num_stages > 3:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y3_upsampled = y3
            else:
                y3_upsampled = nn.functional.interpolate(
                    y3, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage4 = self.stage4(torch.cat([x, y3_upsampled], dim=1))
            y4 = y_output_stage4[0]  # Just keeping belief maps for now

            if self.num_stages == 4:
                return [y1, y2, y3, y4]

        if self.num_stages > 4:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y4_upsampled = y4
            else:
                y4_upsampled = nn.functional.interpolate(
                    y4, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage5 = self.stage5(torch.cat([x, y4_upsampled], dim=1))
            y5 = y_output_stage5[0]  # Just keeping belief maps for now

            if self.num_stages == 5:
                return [y1, y2, y3, y4, y5]

        if self.num_stages > 5:
            # Upsample
            if self.deconv_decoder or self.full_output:
                y5_upsampled = y5
            else:
                y5_upsampled = nn.functional.interpolate(
                    y5, scale_factor=4
                )  # TBD: change scale factor depending on image resolution
            y_output_stage6 = self.stage6(torch.cat([x, y5_upsampled], dim=1))
            y6 = y_output_stage6[0]  # Just keeping belief maps for now

            if self.num_stages == 6:
                return [y1, y2, y3, y4, y5, y6]


# Based on DopeHourglassBlockSmall, not using skipped connections
class DreamHourglass(nn.Module):
    def __init__(
        self,
        n_keypoints,
        n_image_input_channels=3,
        internalize_spatial_softmax=True,
        learned_beta=True,
        initial_beta=1.0,
        skip_connections=False,
        deconv_decoder=False,
        full_output=False,
        model_type = 'vgg'
    ):
        super(DreamHourglass, self).__init__()
        self.n_keypoints = n_keypoints
        self.n_image_input_channels = n_image_input_channels
        self.internalize_spatial_softmax = internalize_spatial_softmax
        self.skip_connections = skip_connections
        self.deconv_decoder = deconv_decoder
        self.full_output = full_output
        self.model_type = model_type
        # TODO: add assertion error to invalid model name
        
        print("###############################")
        print(f'Model Type: {self.model_type}')
        # print("###############################")
        # print("DEBUG: Adding attention in init")
        if self.model_type == 'vgg_mha':
            #self.att1 = Attention(64)
            #self.att2 = Attention(128)
            #self.att3 = Attention(256)
            #self.att4 = Attention(512)
            #self.mha1 = MultiHeadAttention(64)
            #self.mha2 = MultiHeadAttention(128)
            #self.mha3 = MultiHeadAttention(256)
            #self.mha4 = MultiHeadAttention(512, num_heads=2)
            self.mha5 = MultiHeadAttention(512, num_heads=4)
            self.relu = nn.ReLU()
            
        elif self.model_type == 'vgg_att':
            #self.att1 = Attention(64)
            self.att2 = Attention(128)
            self.att3 = Attention(256)
            self.att4 = Attention(512)
            self.att5 = Attention(512)
            self.relu = nn.ReLU()
            
        elif self.model_type == 'vgg_attlast':
            self.attlast5 = Attention(512)
            self.relu = nn.ReLU()
            
        elif self.model_type == 'vgg_transformer':
            self.transformer_encoder = TransformerEncoderBlock(512, 4, 256)
            #self.transformer_att2 = Attention(128)
            self.relu = nn.ReLU()
            
        if self.internalize_spatial_softmax:
            self.n_output_heads = 2
            self.learned_beta = learned_beta
            self.initial_beta = initial_beta
        else:
            self.n_output_heads = 1
            self.learned_beta = False

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        vgg_t = tviz_models.vgg19(pretrained=True).features

        self.down_sample = nn.MaxPool2d(2)

        self.layer_0_1_down = nn.Sequential()
        self.layer_0_1_down.add_module(
            "0",
            nn.Conv2d(
                self.n_image_input_channels, 64, kernel_size=3, stride=1, padding=1
            ),
        )
        for layer in range(1, 4):
            self.layer_0_1_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_2_down = nn.Sequential()
        for layer in range(5, 9):
            self.layer_0_2_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_3_down = nn.Sequential()
        for layer in range(10, 18):
            self.layer_0_3_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_4_down = nn.Sequential()
        for layer in range(19, 27):
            self.layer_0_4_down.add_module(str(layer), vgg_t[layer])

        self.layer_0_5_down = nn.Sequential()
        for layer in range(28, 36):
            self.layer_0_5_down.add_module(str(layer), vgg_t[layer])

        # Head 1
        if self.deconv_decoder:
            # Decoder primarily uses ConvTranspose2d
            self.deconv_0_4 = nn.Sequential()
            self.deconv_0_4.add_module(
                "0",
                nn.ConvTranspose2d(
                    512,
                    256,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_4.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_4.add_module(
                "2", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_4.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_3 = nn.Sequential()
            self.deconv_0_3.add_module(
                "0",
                nn.ConvTranspose2d(
                    256,
                    128,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_3.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_3.add_module(
                "2", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_3.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_2 = nn.Sequential()
            self.deconv_0_2.add_module(
                "0",
                nn.ConvTranspose2d(
                    128,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_2.add_module("1", nn.ReLU(inplace=True))
            self.deconv_0_2.add_module(
                "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            self.deconv_0_2.add_module("3", nn.ReLU(inplace=True))

            self.deconv_0_1 = nn.Sequential()
            self.deconv_0_1.add_module(
                "0",
                nn.ConvTranspose2d(
                    64,
                    64,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1,
                ),
            )
            self.deconv_0_1.add_module("1", nn.ReLU(inplace=True))

        else:
            # Decoder primarily uses Upsampling
            self.upsample_0_4 = nn.Sequential()
            self.upsample_0_4.add_module("0", nn.Upsample(scale_factor=2))

            # should this go before the upsample?
            self.upsample_0_4.add_module(
                "4", nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_4.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_4.add_module(
                "6", nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )

            self.upsample_0_3 = nn.Sequential()
            self.upsample_0_3.add_module("0", nn.Upsample(scale_factor=2))
            self.upsample_0_3.add_module(
                "4", nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
            )
            self.upsample_0_3.add_module("5", nn.ReLU(inplace=True))
            self.upsample_0_3.add_module(
                "6", nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            )

            if self.full_output:
                self.upsample_0_2 = nn.Sequential()
                self.upsample_0_2.add_module("0", nn.Upsample(scale_factor=2))
                self.upsample_0_2.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_2.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_2.add_module("5", nn.ReLU(inplace=True))

                self.upsample_0_1 = nn.Sequential()
                self.upsample_0_1.add_module("00", nn.Upsample(scale_factor=2))
                self.upsample_0_1.add_module(
                    "2", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("3", nn.ReLU(inplace=True))
                self.upsample_0_1.add_module(
                    "4", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
                )
                self.upsample_0_1.add_module("5", nn.ReLU(inplace=True))

        # Output head - goes from [batch x 64 x height x width] -> [batch x n_keypoints x height x width]
        self.heads_0 = nn.Sequential()
        self.heads_0.add_module(
            "0", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("1", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "2", nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        )
        self.heads_0.add_module("3", nn.ReLU(inplace=True))
        self.heads_0.add_module(
            "4", nn.Conv2d(32, self.n_keypoints, kernel_size=3, stride=1, padding=1)
        )

        # Spatial softmax output of belief map
        if self.internalize_spatial_softmax:
            self.softmax = nn.Sequential()
            self.softmax.add_module(
                "0",
                SoftArgmaxPavlo(
                    n_keypoints=self.n_keypoints,
                    learned_beta=self.learned_beta,
                    initial_beta=self.initial_beta,
                ),
            )

    def forward(self, x):

        # Encoder
        # Conv1
        x_0_1 = self.layer_0_1_down(x)
        
        
        # Max pooling
        x_0_1_d = self.down_sample(x_0_1)
        
        # Conv2
        x_0_2 = self.layer_0_2_down(x_0_1_d)
        if self.model_type == "vgg_att":
            x_0_2 = self.relu(self.att2(x_0_2))
            
            
        # Max pooling
        x_0_2_d = self.down_sample(x_0_2)
        # Conv3
        x_0_3 = self.layer_0_3_down(x_0_2_d)
        
        if self.model_type == "vgg_att":
            x_0_3 = self.relu(self.att3(x_0_3))
        # Max pooling
        x_0_3_d = self.down_sample(x_0_3)
        # Conv4
        x_0_4 = self.layer_0_4_down(x_0_3_d)
        
        if self.model_type == "vgg_att":
            x_0_4 = self.relu(self.att4(x_0_4))
        
        # Max pooling
        x_0_4_d = self.down_sample(x_0_4)
        # Conv5
        x_0_5 = self.layer_0_5_down(x_0_4_d)
        
        if self.model_type == "vgg_att":
            x_0_5 = self.relu(self.att5(x_0_5))
        
        elif self.model_type == "vgg_attlast":
            x_0_5 = self.relu(self.attlast5(x_0_5))
        
        elif self.model_type == "vgg_mha":                  
            x_0_5 = self.relu(self.mha5(x_0_5))
            
        elif self.model_type == "vgg_transformer":
            x_0_5 = self.relu(self.transformer_encoder(x_0_5))

        if self.skip_connections:
            decoder_input = x_0_5 + x_0_4_d
        else:
            decoder_input = x_0_5

        # Decoder
        if self.deconv_decoder:
            y_0_5 = self.deconv_0_4(decoder_input)

            if self.skip_connections:
                y_0_4 = self.deconv_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_4 = self.deconv_0_3(y_0_5)

            if self.skip_connections:
                y_0_3 = self.deconv_0_2(y_0_4 + x_0_2_d)
            else:
                y_0_3 = self.deconv_0_2(y_0_4)

            if self.skip_connections:
                y_0_out = self.deconv_0_1(y_0_3 + x_0_1_d)
            else:
                y_0_out = self.deconv_0_1(y_0_3)

            if self.skip_connections:
                output_head_0 = self.heads_0(y_0_out + x_0_1)
            else:
                output_head_0 = self.heads_0(y_0_out)

        else:
            y_0_5 = self.upsample_0_4(decoder_input)

            if self.skip_connections:
                y_0_out = self.upsample_0_3(y_0_5 + x_0_3_d)
            else:
                y_0_out = self.upsample_0_3(y_0_5)

            if self.full_output:
                y_0_out = self.upsample_0_2(y_0_out)
                y_0_out = self.upsample_0_1(y_0_out)

            output_head_0 = self.heads_0(y_0_out)

        # Output heads
        outputs = []
        outputs.append(output_head_0)

        # Spatial softmax output of belief map
        if self.internalize_spatial_softmax:
            softmax_output = self.softmax(output_head_0)
            outputs.append(softmax_output)

        # Return outputs
        return outputs


if __name__ == "__main__":
    import torch

    # Unit test parameters
    n_keypoints = 7
    batch_size = 2

    # Unit testing network construction and forward
    print("DreamHourglassMultiStage: upsample decoder")
    for n in range(1, 7):
        print("{}-stage".format(n))
        net = DreamHourglassMultiStage(
            n_keypoints,
            n_stages=n,
            internalize_spatial_softmax=False,
            deconv_decoder=False,
            full_output=True,
        ).cuda()
        y = net(torch.zeros(batch_size, 3, 400, 400).cuda())
        print(y[-1].shape)
        print()
        del net, y
        torch.cuda.empty_cache()

    raise ()

    net_input_height = 400  # 480
    net_input_width = 400  # 640

    net_input = torch.zeros(batch_size, 3, net_input_height, net_input_width)
    print("net_input shape: {}".format(net_input.shape))

    print("ResnetSimple")
    net = ResnetSimple(n_keypoints).cuda()
    y = net(net_input.cuda())
    print(y[-1].shape)
    print()
    del net, y

    print("ResnetSimpleFull")
    net = ResnetSimple(n_keypoints, full=True).cuda()
    y = net(net_input.cuda())
    print(y[-1].shape)
    print()
    del net, y

    print("DOPE")
    net = DopeNetworkBelief().cuda()
    y = net(net_input.cuda())
    print(y[-1].shape)
    print()
    del net, y

    torch.cuda.empty_cache()

    # Testing DreamHourglass variations
    print("DreamHourglass")
    net1 = DreamHourglass(
        n_keypoints,
        internalize_spatial_softmax=False,
        skip_connections=False,
        deconv_decoder=False,
    ).cuda()
    y = net1(net_input.cuda())
    print(y[-1].shape)
    print()
    del net1, y

    net3 = DreamHourglass(
        n_keypoints,
        internalize_spatial_softmax=False,
        skip_connections=False,
        deconv_decoder=True,
    ).cuda()
    y = net3(net_input.cuda())
    print(y[-1].shape)
    print()
    del net3, y

    torch.cuda.empty_cache()

    net5 = DreamHourglass(
        n_keypoints,
        internalize_spatial_softmax=False,
        skip_connections=True,
        deconv_decoder=False,
    ).cuda()
    y = net5(net_input.cuda())
    print(y[-1].shape)
    print()
    del net5, y

    net7 = DreamHourglass(
        n_keypoints,
        internalize_spatial_softmax=False,
        skip_connections=True,
        deconv_decoder=True,
    ).cuda()
    y = net7(net_input.cuda())
    print(y[-1].shape)
    print()
    del net7, y

    torch.cuda.empty_cache()

    # Testing DreamHourglassMultiStage variations
    print("DreamHourglassMultiStage")
    for n in range(1, 7):
        print("{}-stage".format(n))
        net = DreamHourglassMultiStage(
            n_keypoints, n_stages=n, internalize_spatial_softmax=False
        ).cuda()
        y = net(net_input.cuda())
        print(y[-1].shape)
        print()
        del net, y
        torch.cuda.empty_cache()

    # Testing DreamHourglassMultiStage variations
    print("DreamHourglassMultiStage: deconvolutional decoder")
    for n in range(1, 7):
        print("{}-stage".format(n))
        net = DreamHourglassMultiStage(
            n_keypoints,
            n_stages=n,
            internalize_spatial_softmax=False,
            deconv_decoder=True,
        ).cuda()
        y = net(net_input.cuda())
        print(y[-1].shape)
        print()
        del net, y
        torch.cuda.empty_cache()
