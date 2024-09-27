import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_pytorch import StochasticDepth, TalkingHeadAttention, LayerScale, RandomDrop

class PET(nn.Module):
    """Point-Edge Transformer"""
    def __init__(self,
                 num_feat,
                 num_jet,      
                 num_classes=2,
                 num_keep=7, #Number of features that wont be dropped
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

        self.body = self.PET_body(local, K, num_local, talking_head)
        self.classifier_head = self.PET_classifier(num_class_layers, num_jet, simple)
        self.generator_head = self.PET_generator(num_gen_layers, simple)

        self.ema_body = None
        self.ema_generator_head = None

        self.pred_tracker = nn.CrossEntropyLoss()
        self.loss_tracker = nn.MSELoss()
        self.mse_tracker = nn.MSELoss()
        self.gen_tracker = nn.MSELoss()
        self.pred_smear_tracker = nn.CrossEntropyLoss()
        self.mse_smear_tracker = nn.MSELoss()

    def forward(self, x):
        if self.mode == 'generator':
            return self.generator(x)
        else:
            return self.classifier(x)

    def PET_body(self, local, K, num_local, talking_head):
        layers = []
        
        layers.append(RandomDrop(self.feature_drop if 'all' in self.mode else 0.0, num_skip=self.num_keep))
        layers.append(nn.Linear(self.num_feat, self.projection_dim))
        layers.append(nn.GELU())

        if local:
            for _ in range(num_local):
                layers.append(LocalFeatureExtractor(self.projection_dim, K))

        for _ in range(self.num_layers):
            layers.append(TransformerBlock(
                self.projection_dim,
                self.num_heads,
                self.drop_probability,
                self.layer_scale,
                self.layer_scale_init,
                talking_head
            ))

        return nn.Sequential(*layers)

    def PET_classifier(self, num_class_layers, num_jet, simple):
        if simple:
            return SimpleClassifier(self.projection_dim, self.num_classes, num_jet, self.class_activation)
        else:
            return ComplexClassifier(self.projection_dim, self.num_heads, num_class_layers, self.num_classes, num_jet, self.class_activation)

    def PET_generator(self, num_layers, simple):
        if simple:
            return SimpleGenerator(self.projection_dim, self.num_feat)
        else:
            return ComplexGenerator(self.projection_dim, self.num_heads, num_layers, self.num_feat)

    def train_step(self, inputs, optimizer):
        x, y = inputs
        batch_size = x['input_jet'].size(0)
        x['input_time'] = torch.zeros((batch_size, 1))

        loss = 0.0
        
        if self.mode == 'classifier' or 'all' in self.mode:
            body_output = self.body(x)
            y_pred, y_mse = self.classifier_head([body_output, x['input_jet']])
            loss_pred = F.cross_entropy(y_pred, y)
            loss += loss_pred
            if 'all' in self.mode:
                loss_mse = F.mse_loss(y_mse, x['input_jet'])
                loss += loss_mse

        if self.mode == 'generator' or 'all' in self.mode:
            t = torch.rand((batch_size, 1))
            _, alpha, sigma = get_logsnr_alpha_sigma(t)

            eps = torch.randn_like(x['input_features']) * x['input_mask'].unsqueeze(-1)
            mask_diffusion = torch.cat([
                torch.ones_like(eps[:, :, :self.num_diffusion], dtype=torch.bool),
                torch.zeros_like(eps[:, :, self.num_diffusion:], dtype=torch.bool)
            ], dim=-1)

            eps = torch.where(mask_diffusion, eps, torch.zeros_like(eps))
            perturbed_x = alpha.unsqueeze(1) * x['input_features'] + eps * sigma.unsqueeze(1)
            perturbed_x = torch.where(mask_diffusion, perturbed_x, torch.zeros_like(perturbed_x))

            perturbed_body = self.body([perturbed_x, perturbed_x[:, :, :2], x['input_mask'], t])
            v_pred = self.generator_head([perturbed_body, x['input_jet'], x['input_mask'], t, y])
            v_pred = v_pred[:, :, :self.num_diffusion].reshape(v_pred.size(0), -1)

            v = alpha.unsqueeze(1) * eps - sigma.unsqueeze(1) * x['input_features']
            v = v[:, :, :self.num_diffusion].reshape(v.size(0), -1)
            loss_part = torch.sum((v - v_pred)**2) / (self.num_diffusion * torch.sum(x['input_mask']))
            loss += loss_part

        if self.mode == 'all':
            y_pred_smear, y_mse_smear = self.classifier_head([perturbed_body, x['input_jet']])
            loss_pred_smear = F.cross_entropy(y_pred_smear, y)
            loss += alpha.pow(2) * loss_pred_smear

            loss_mse_smear = F.mse_loss(y_mse_smear, x['input_jet'])
            loss += alpha.pow(2) * loss_mse_smear

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_ema()

        return loss.item()

    def update_ema(self):
        if self.ema_body is None:
            self.ema_body = self.body
            self.ema_generator_head = self.generator_head
        else:
            for ema_param, param in zip(self.ema_body.parameters(), self.body.parameters()):
                ema_param.data = self.ema * ema_param.data + (1 - self.ema) * param.data
            for ema_param, param in zip(self.ema_generator_head.parameters(), self.generator_head.parameters()):
                ema_param.data = self.ema * ema_param.data + (1 - self.ema) * param.data

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, drop_probability, layer_scale, layer_scale_init, talking_head):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.GroupNorm(1, projection_dim)
        self.attention = TalkingHeadAttention(projection_dim, num_heads, 0.0) if talking_head else nn.MultiheadAttention(projection_dim, num_heads)
        self.drop1 = StochasticDepth(drop_probability)
        self.norm2 = nn.GroupNorm(1, projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(),
            nn.Dropout(drop_probability),
            nn.Linear(2 * projection_dim, projection_dim)
        )
        self.drop2 = StochasticDepth(drop_probability)
        self.layer_scale = layer_scale
        if layer_scale:
            self.layer_scale1 = LayerScale(layer_scale_init, projection_dim)
            self.layer_scale2 = LayerScale(layer_scale_init, projection_dim)

    def forward(self, x):
        x1 = self.norm1(x)
        attn_output = self.attention(x1, x1, x1)[0]
        if self.layer_scale:
            attn_output = self.layer_scale1(attn_output)
        x = x + self.drop1(attn_output)
        x2 = self.norm2(x)
        mlp_output = self.mlp(x2)
        if self.layer_scale:
            mlp_output = self.layer_scale2(mlp_output)
        x = x + self.drop2(mlp_output)
        return x

class LocalFeatureExtractor(nn.Module):
    def __init__(self, projection_dim, K):
        super(LocalFeatureExtractor, self).__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(2 * projection_dim, 2 * projection_dim),
            nn.GELU(),
            nn.Linear(2 * projection_dim, projection_dim),
            nn.GELU()
        )

    def forward(self, x, points):
        knn_fts = get_neighbors(points, x, self.K)
        knn_fts_center = x.unsqueeze(2).expand(-1, -1, self.K, -1)
        local = torch.cat([knn_fts - knn_fts_center, knn_fts_center], dim=-1)
        local = self.mlp(local)
        local = torch.mean(local, dim=2)
        return local

class SimpleClassifier(nn.Module):
    def __init__(self, projection_dim, num_classes, num_jet, class_activation):
        super(SimpleClassifier, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.jet_encoder = nn.Linear(num_jet, projection_dim)
        self.fc1 = nn.Linear(projection_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, num_classes)
        self.fc3 = nn.Linear(projection_dim, num_jet)
        self.activation = getattr(nn, class_activation)() if class_activation else nn.Identity()

    def forward(self, x, jet):
        x = self.global_pool(x.transpose(1, 2)).squeeze(-1)
        jet_encoded = self.jet_encoder(jet)
        x = self.fc1(x + jet_encoded)
        x = F.gelu(x)
        pred = self.activation(self.fc2(x))
        mse = self.fc3(x)
        return pred, mse

class ComplexClassifier(nn.Module):
    def __init__(self, projection_dim, num_heads, num_layers, num_classes, num_jet, class_activation):
        super(ComplexClassifier, self).__init__()
        self.conditional = nn.Linear(num_jet, 2 * projection_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
        self.layers = nn.ModuleList([TransformerBlock(projection_dim, num_heads, 0.0, True, 1e-5, False) for _ in range(num_layers)])
        self.norm = nn.GroupNorm(1, projection_dim)
        self.fc1 = nn.Linear(projection_dim, num_classes)
        self.fc2 = nn.Linear(projection_dim, num_jet)
        self.activation = getattr(nn, class_activation)() if class_activation else nn.Identity()

    def forward(self, x, jet):
        B = x.size(0)
        conditional = self.conditional(jet).unsqueeze(1)
        scale, shift = torch.chunk(conditional, 2, dim=-1)
        x = x * (1.0 + scale) + shift
        
        class_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        class_output = x[:, 0]
        pred = self.activation(self.fc1(class_output))
        mse = self.fc2(class_output)
        return pred, mse

class SimpleGenerator(nn.Module):
    def __init__(self, projection_dim, num_feat):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(projection_dim, 2 * projection_dim)
        self.fc2 = nn.Linear(2 * projection_dim, projection_dim)
        self.fc3 = nn.Linear(projection_dim, num_feat)

    def forward(self, x, jet, mask, time, label):
        cond = self.get_condition(jet, time, label)
        cond = cond.unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.fc1(x + cond)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x) * mask.unsqueeze(-1)
        return x

    def get_condition(self, jet, time, label):
        time_embedding = fourier_projection(time, self.projection_dim)
        jet_embedding = F.gelu(self.jet_encoder(jet))
        label_embedding = self.label_encoder(label)
        cond = F.gelu(self.cond_fc(torch.cat([time_embedding, jet_embedding, label_embedding], dim=-1)))
        return cond

class ComplexGenerator(nn.Module):
    def __init__(self, projection_dim, num_heads, num_layers, num_feat):
        super(ComplexGenerator, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_feat = num_feat

        self.time_proj = nn.Sequential(
            nn.Linear(projection_dim, 2 * projection_dim),
            nn.GELU(),
            nn.Linear(2 * projection_dim, projection_dim),
        )
        self.jet_proj = nn.Linear(4, projection_dim)  # Assuming jet has 4 features
        self.label_proj = nn.Linear(2, projection_dim)  # Assuming 2 classes

        self.cond_proj = nn.Sequential(
            nn.Linear(3 * projection_dim, 2 * projection_dim),
            nn.GELU(),
            nn.Linear(2 * projection_dim, projection_dim),
        )

        self.layers = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, 0.0, True, 1e-5, False)
            for _ in range(num_layers)
        ])

        self.final_proj = nn.Linear(projection_dim, num_feat)

    def forward(self, x, jet, mask, time, label):
        B, N, _ = x.shape
        
        time_emb = self.time_proj(fourier_projection(time, self.projection_dim))
        jet_emb = self.jet_proj(jet)
        label_emb = self.label_proj(label)
        
        cond = self.cond_proj(torch.cat([time_emb, jet_emb, label_emb], dim=-1))
        cond_token = cond.unsqueeze(1).expand(-1, N, -1)
        
        for layer in self.layers:
            x = layer(x + cond_token)
        
        x = self.final_proj(x) * mask.unsqueeze(-1)
        return x

def get_neighbors(points, features, K):
    B, N, _ = points.shape
    points_flat = points.view(B * N, -1)
    dist = torch.cdist(points_flat, points_flat)
    dist = dist.view(B, N, N)
    
    _, indices = torch.topk(-dist, k=K + 1, dim=-1)
    indices = indices[:, :, 1:]  # Exclude self
    
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).expand(-1, N, K)
    indices = torch.stack([batch_indices, indices], dim=-1)
    
    knn_fts = features[indices[..., 0], indices[..., 1]]
    return knn_fts

def fourier_projection(x, projection_dim, num_embed=64):
    half_dim = num_embed // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(-emb * torch.arange(half_dim, dtype=torch.float32))
    emb = emb.to(x.device)

    angle = x * emb * 1000.0
    embedding = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1) * x
    return embedding

def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b))

def get_logsnr_alpha_sigma(time):
    logsnr = logsnr_schedule_cosine(time)
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma

# Utility functions

def get_encoding(x, projection_dim, use_bias=True):
    x = nn.Sequential(
        nn.Linear(x.size(-1), 2 * projection_dim, bias=use_bias),
        nn.GELU(),
        nn.Linear(2 * projection_dim, projection_dim, bias=use_bias),
        nn.GELU()
    )(x)
    return x

# Main execution

if __name__ == "__main__":
    # Example usage
    num_feat = 13
    num_jet = 4
    batch_size = 32
    seq_len = 100

    model = PET(num_feat=num_feat, num_jet=num_jet)
    
    # Example input
    x = {
        'input_features': torch.randn(batch_size, seq_len, num_feat),
        'input_points': torch.randn(batch_size, seq_len, 2),
        'input_mask': torch.ones(batch_size, seq_len, 1),
        'input_jet': torch.randn(batch_size, num_jet),
        'input_time': torch.zeros(batch_size, 1),
    }
    y = torch.randint(0, 2, (batch_size,))

    # Forward pass
    output = model(x)
    print(f"Output shape: {output[0].shape}, {output[1].shape}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training step
    loss = model.train_step((x, y), optimizer)
    print(f"Training loss: {loss}")