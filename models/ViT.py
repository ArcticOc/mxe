import timm
import torch
from torch import nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=64,
        dim=768,
        depth=12,
        heads=6,
        mlp_ratio=4,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super(VisionTransformer, self).__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (img_size // patch_size) ** 2
        mlp_dim = mlp_ratio * dim

        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = num_patches

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()
        # self.layer_norm = nn.LayerNorm(dim)

        # self.mlp_head = nn.Linear(dim, dim)
        self.linear_proj = nn.Linear(dim, dim)

        self._init_weights()

    def _init_weights(self):
        # https://github.com/huggingface/pytorch-image-models/blob/4d4bdd64a996bf7b5919ec62f20af4a1c07d5848/timm/models/vision_transformer.py#L2107
        pretrained_model = timm.create_model('vit_small_patch16_224', pretrained=True)
        pretrained_state_dict = pretrained_model.state_dict()

        model_state_dict = self.state_dict()

        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if 'head' not in k}

        mapping = {
            'patch_embed.proj.weight': 'patch_embed.weight',
            'patch_embed.proj.bias': 'patch_embed.bias',
            'cls_token': 'cls_token',
            'pos_embed': 'pos_embedding',
        }

        for k in pretrained_state_dict:
            new_k = k
            if k in mapping:
                new_k = mapping[k]
            if new_k in model_state_dict:
                model_state_dict[new_k] = pretrained_state_dict[k]

        self.load_state_dict(model_state_dict)

    def forward(self, x):
        b, c, h, w = x.shape
        # Patch Embedding
        x = self.patch_embed(x)  # [B, dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        cls_tokens = self.cls_token.expand(b, -1, -1)  # [B, 1, dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, dim]
        x = x + self.pos_embedding[:, : (x.size(1))]
        x = self.dropout(x)

        # Transformer Encoder
        x = self.transformer(x)
        # x = self.layer_norm(x[:, 0])
        patch_tokens = x[:, 1:]  # [B, num_patches, dim]
        x = self.linear_proj(patch_tokens.mean(dim=1))
        # x = self.layer_norm(x)  # [B, dim]
        return x, w
