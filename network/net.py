from statistics import mode
from unittest.mock import patch
import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from network.blocks.reassemble import Reassemble
from network.blocks.fusion import Fusion
from network.blocks.head import HeadDepth, HeadSeg

torch.manual_seed(0)

class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 num_heads          = 8,
                 read               = 'projection',
                 num_layers_encoder = 24,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 nclasses           = 3,
                 type               = "full",
                 model_timm         = "vit_large_patch16_384",
                 class_embedding_size = 256):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        # # Splitting img into patches
        # channels, image_height, image_width = image_size
        # assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_height // patch_size) * (image_width // patch_size)
        # patch_dim = channels * patch_size * patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, emb_dim),
        # )
        # # Embedding
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        # # Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        # self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        
        self.num_classes = nclasses
        self.patch_size = patch_size
        self.type_ = type

        self.class_embeddings = nn.Parameter(torch.randn(1, 1, nclasses, class_embedding_size))
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        print(emb_dim + class_embedding_size, emb_dim)
        self.emb_to_vit = nn.Linear(emb_dim + class_embedding_size, emb_dim)
        self.seg_patch_emb = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        # self.seg_patch_emb = timm.models.layers.PatchEmbed(img_size=image_size[-1],
        #                                                    patch_size=patch_size,
        #                                                    in_chans=1,
        #                                                    embed_dim=emb_dim)

        # Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        # Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def transformer_forward(self, model, x):
        x = model._pos_embed(x)
        
        if model.grad_checkpointing and not torch.jit.is_scripting():
            x = timm.models.helper.checkpoint_seq(model.blocks, x)
        else:
            x = model.blocks(x)
        x = model.norm(x)
        
        return model.forward_head(x)

    def segmentation_distill(self, segmentations):
        seg_patches = self.seg_patch_emb(segmentations.float()).to(torch.int64)
        print(seg_patches.min(), seg_patches.max(), seg_patches.unique())
        oh_patches = nn.functional.one_hot(seg_patches, num_classes=self.num_classes)
        class_distribution = oh_patches.sum(dim=-2, keepdim=True) / (self.patch_size**2)
        weighted_embeddings = class_distribution.transpose(-1, -2) * self.class_embeddings
        patch_embeddings = weighted_embeddings.sum(dim=-2)
        return patch_embeddings
    
    def forward(self, images, segmentations):
        # Pre-processing images
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # t = self.transformer_encoders(x)

        # Feed pre-processed images to transformer
        # t = self.transformer_encoders(img)
        
        model = self.transformer_encoders
        # flatten image into patch then patch into vector
        # img_patches size: b l=h*w/p^2 p^2*c
        img_patches = model.patch_embed(images)
        # TO DO: integrate segmentation result of the patch
        # l*768 -> l*1024 
        patch_embeddings = self.segmentation_distill(segmentations=segmentations)
        patches = torch.cat((img_patches, patch_embeddings), dim=-1)
        # l*1024 -> l*768
        print(patches.shape)
        vit_input = self.emb_to_vit(patches)
        t = self.transformer_forward(model, vit_input)
        
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result
        out_depth = None
        # out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        # return out_depth, out_segmentation
        return out_depth

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            # self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))
