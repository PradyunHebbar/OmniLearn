import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_pytorch import StochasticDepth, RandomDrop, SimpleHeadAttention, TalkingHeadAttention, LayerScale

class PET(nn.Module):
    """Point-Edge Transformer"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_keep=7,
                 feature_drop=0.1,
                 projection_dim=128,
                 local=True, K=10,
                 num_local=2, 
                 num_layers=8, num_class_layers=2,
                 num_gen_layers=2,
                 num_heads=4, drop_probability=0.0,
                 simple=False, layer_scale=True,
                 layer_scale_init=1e-5,        
                 talking_head=False,
                 mode='classifier',
                 num_diffusion=3,
                 dropout=0.0,
                 class_activation=None):
        super(PET, self).__init__()
        self.num_feat = num_feat
        self.num_jet = num_jet
        self.num_classes = num_classes
        self.num_keep = num_keep
        self.feature_drop = feature_drop
        self.drop_probability = drop_probability
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.layer_scale_init = layer_scale_init
        self.mode = mode
        self.num_diffusion = num_diffusion
        self.ema = 0.999
        self.class_activation = class_activation

        self.input_features = nn.Linear(num_feat, projection_dim)
        self.input_points = nn.Linear(2, projection_dim)
        self.input_mask = nn.Linear(1, 1)
        self.input_jet = nn.Linear(num_jet, projection_dim)
        self.input_label = nn.Linear(num_classes, projection_dim)
        self.input_time = nn.Linear(1, projection_dim)

        self.body = self.PET_body(local, K, num_local, talking_head)
        self.classifier_head, self.classifier_regressor = self.PET_classifier(num_class_layers, num_jet, simple)
        self.generator_head = self.PET_generator(num_gen_layers, num_jet, simple)

        self.classifier = nn.ModuleList([self.body, self.classifier_head, self.classifier_regressor])
        self.generator = nn.ModuleList([self.body, self.generator_head])

        self.ema_body = self.PET_body(local, K, num_local, talking_head)
        self.ema_generator_head = self.PET_generator(num_gen_layers, num_jet, simple)

        self.pred_tracker = nn.Linear(num_classes, num_classes)
        self.loss_tracker = nn.Linear(1, 1)
        self.mse_tracker = nn.Linear(1, 1)
        self.gen_tracker = nn.Linear(1, 1)
        self.pred_smear_tracker = nn.Linear(num_classes, num_classes)
        self.mse_smear_tracker = nn.Linear(1, 1)

    def forward(self, x):
        if self.mode == 'generator':
            return self.generator(x)
        else:
            return self.classifier(x)

    def PET_body(self, local, K, num_local, talking_head):
        layers = []
        layers.append(RandomDrop(self.feature_drop if 'all' in self.mode else 0.0, num_skip=self.num_keep))
        layers.append(get_encoding(self.projection_dim))
        layers.append(FourierProjection(self.projection_dim))
        
        if local:
            for _ in range(num_local):
                layers.append(self.get_neighbors(K, self.projection_dim))
        
        for _ in range(self.num_layers):
            layers.append(nn.LayerNorm(self.projection_dim))
            if talking_head:
                layers.append(TalkingHeadAttention(self.projection_dim, self.num_heads, self.dropout))
            else:
                layers.append(SimpleHeadAttention(self.projection_dim, self.num_heads, self.dropout))
            if self.layer_scale:
                layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
            layers.append(StochasticDepth(self.drop_probability))
            layers.append(nn.LayerNorm(self.projection_dim))
            layers.append(nn.Linear(self.projection_dim, 2*self.projection_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
            if self.layer_scale:
                layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
            layers.append(StochasticDepth(self.drop_probability))
        
        return nn.Sequential(*layers)

    def PET_classifier(self, num_class_layers, num_jet, simple):
        layers = []
        regressor = []
        if simple:
            layers.append(nn.LayerNorm(self.projection_dim))
            layers.append(nn.AdaptiveAvgPool1d(1))
            layers.append(nn.Linear(self.projection_dim, self.projection_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(self.projection_dim, self.num_classes))
            regressor.append(nn.Linear(self.projection_dim, num_jet))
        else:
            layers.append(nn.Linear(num_jet, 2*self.projection_dim))
            layers.append(nn.GELU())
            layers.append(nn.Parameter(torch.zeros(1, self.projection_dim)))
            for _ in range(num_class_layers):
                layers.append(nn.LayerNorm(self.projection_dim))
                layers.append(SimpleHeadAttention(self.projection_dim, self.num_heads, self.dropout))
                if self.layer_scale:
                    layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
                layers.append(nn.LayerNorm(self.projection_dim))
                layers.append(nn.Linear(self.projection_dim, 2*self.projection_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(self.dropout))
                layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
                if self.layer_scale:
                    layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
            layers.append(nn.LayerNorm(self.projection_dim))
            layers.append(nn.Linear(self.projection_dim, self.num_classes))
            regressor.append(nn.Linear(self.projection_dim, num_jet))
        return nn.Sequential(*layers), nn.Sequential(*regressor)

    def PET_generator(self, num_layers, num_jet, simple):
        layers = []
        layers.append(nn.Linear(1, self.projection_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(num_jet, self.projection_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(self.num_classes, self.projection_dim))
        layers.append(nn.Dropout(self.feature_drop))
        
        if simple:
            layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.projection_dim, self.num_feat))
        else:
            for _ in range(num_layers):
                layers.append(nn.LayerNorm(self.projection_dim))
                layers.append(SimpleHeadAttention(self.projection_dim, self.num_heads, self.dropout))
                if self.layer_scale:
                    layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
                layers.append(nn.LayerNorm(self.projection_dim))
                layers.append(nn.Linear(self.projection_dim, 2*self.projection_dim))
                layers.append(nn.Linear(2*self.projection_dim, self.projection_dim))
                if self.layer_scale:
                    layers.append(LayerScale(self.layer_scale_init, self.projection_dim))
            layers.append(nn.LayerNorm(self.projection_dim))
            layers.append(nn.Linear(self.projection_dim, self.num_feat))
        
        return nn.Sequential(*layers)

    def get_neighbors(self, K, projection_dim):
        class Neighbors(nn.Module):
            def __init__(self, K, projection_dim):
                super(Neighbors, self).__init__()
                self.K = K
                self.mlp1 = nn.Sequential(
                    nn.Linear(2 * projection_dim, 2 * projection_dim),
                    nn.GELU()
                )
                self.mlp2 = nn.Sequential(
                    nn.Linear(2 * projection_dim, projection_dim),
                    nn.GELU()
                )

            def forward(self, points, features):
                B, N, _ = points.size()
                
                # Calculate pairwise distances
                diff = points.unsqueeze(2) - points.unsqueeze(1)
                distances = torch.sum(diff ** 2, dim=-1)
                
                # Find K nearest neighbors
                _, indices = torch.topk(-distances, k=self.K + 1, dim=-1)
                indices = indices[:, :, 1:]  # Exclude self
                
                # Gather neighbor features
                neighbor_features = torch.gather(features.unsqueeze(2).expand(-1, -1, self.K, -1),
                                                 dim=1,
                                                 index=indices.unsqueeze(-1).expand(-1, -1, -1, features.size(-1)))
                
                # Compute edge features
                edge_features = torch.cat([
                    features.unsqueeze(2).expand(-1, -1, self.K, -1),
                    neighbor_features - features.unsqueeze(2)
                ], dim=-1)
                
                # Apply MLPs
                edge_features = self.mlp1(edge_features)
                edge_features = torch.mean(edge_features, dim=2)  # Average pooling over neighbors
                local_features = self.mlp2(edge_features)
                
                return local_features

        return Neighbors(K, projection_dim)

def get_encoding(projection_dim):
    return nn.Sequential(
        nn.Linear(projection_dim, 2*projection_dim),
        nn.GELU(),
        nn.Linear(2*projection_dim, projection_dim),
        nn.GELU()
    )

def FourierProjection(projection_dim, num_embed=64):
    class FourierEmbedding(nn.Module):
        def __init__(self, num_embed, projection_dim):
            super(FourierEmbedding, self).__init__()
            self.num_embed = num_embed
            self.projection = nn.Sequential(
                nn.Linear(num_embed, 2*projection_dim, bias=False),
                nn.SiLU(),
                nn.Linear(2*projection_dim, projection_dim, bias=False),
                nn.SiLU()
            )

        def forward(self, x):
            half_dim = self.num_embed // 2
            emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
            emb = torch.exp(-emb * torch.arange(half_dim, dtype=torch.float32, device=x.device))
            emb = x * emb[None, :] * 1000.0
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
            return self.projection(emb)

    return FourierEmbedding(num_embed, projection_dim)

def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    b = torch.atan(torch.exp(-0.5 * logsnr_max))
    a = torch.atan(torch.exp(-0.5 * logsnr_min)) - b
    return -2. * torch.log(torch.tan(a * t + b))

def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma