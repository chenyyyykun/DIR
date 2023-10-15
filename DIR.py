import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from torch.autograd import Variable

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

from model import ResNet18

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)  

        patches = take_indexes(patches, forward_indexes)
        patches_1 = patches[:remain_T]
        patches_2 = patches[remain_T:]

        return patches_1,patches_2, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=8,
                 num_head=3,
                 mask_ratio=0.5,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches_1,patches_2, _ , backward_indexes = self.shuffle(patches)

        patches_1 = torch.cat([self.cls_token.expand(-1, patches_1.shape[1], -1), patches_1], dim=0)
        patches_1 = rearrange(patches_1, 't b c -> b t c')
        features_1 = self.layer_norm(self.transformer(patches_1))
        features_1 = rearrange(features_1, 'b t c -> t b c')

        patches_2 = torch.cat([self.cls_token.expand(-1, patches_2.shape[1], -1), patches_2], dim=0)
        patches_2 = rearrange(patches_2, 't b c -> b t c')
        features_2 = self.layer_norm(self.transformer(patches_2))
        features_2 = rearrange(features_2, 'b t c -> t b c')

        return features_1, features_2, backward_indexes

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=2,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        # patches = self.head(features)
        # mask = torch.zeros_like(patches)
        # mask[T:] = 1
        # mask = take_indexes(mask, backward_indexes[1:] - 1)
        # img = self.patch2img(patches)
        # mask = self.patch2img(mask)

        return features

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=8,
                 encoder_head=3,
                 decoder_layer=2,
                 decoder_head=3,
                 mask_ratio=0.5,
                 ) -> None:
        super().__init__()

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)
        # self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.img2patch = Rearrange('b c (h p1) (w p2) -> (h w) b (c p1 p2)',p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.b2c = Rearrange('c b f ->b c f')
        self.c2b = Rearrange('b c f ->c b f')
        self.conv1 = torch.nn.Conv1d(256,256,16,stride=16)

    def Loss(self,nat,pred):
        nat_patches = self.img2patch(nat)
        pred_patches = self.img2patch(pred)
        loss = (pred_patches - nat_patches)**2
        loss = loss.mean(dim=-1)
        loss = loss.sum()/(loss.size()[0]*loss.size()[1])  #200:batch_size; 256:patches_num
        return loss

    def forward(self, img_adv):
        features_1, features_2, backward_indexes = self.encoder(img_adv)
        pred_1 = self.decoder(features_1,  backward_indexes)
        pred_2 = self.decoder(features_2,  backward_indexes)
        pred_all = (pred_1 +pred_2)/2
        # pred_patches = self.head(pred_all)
        pred_b2c = self.b2c(pred_all)
        pred_conv = self.conv1(pred_b2c)
        pred_c2b = self.c2b(pred_conv)
        # pred = pred_c2b+pred_patches
        pred = pred_c2b
        # img_patches = self.img2patch(img_nat)
        # loss = self.Loss(img_patches,pred)
        img = self.patch2img(pred)
        return img




