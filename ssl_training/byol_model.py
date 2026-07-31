import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, projection_dim, hidden_dim=4096):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    """
    BYOL implementation including Online and Target networks.
    """
    def __init__(
        self, 
        backbone, 
        projection_dim=256, 
        projection_hidden_dim=4096, 
        moving_average_decay=0.99
    ):
        super(BYOL, self).__init__()
        self.online_encoder = backbone
        self.embedding_dim = backbone.embedding_dim

        # Online projection and prediction heads
        self.online_projector = MLP(self.embedding_dim, projection_dim, projection_hidden_dim)
        self.online_predictor = MLP(projection_dim, projection_dim, projection_hidden_dim)

        # Target network (starts as a copy of online network)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        self.tau = moving_average_decay

        # Freeze target network parameters
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self):
        """
        Momentum update of the target network.
        Target = Tau * Target + (1 - Tau) * Online
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.tau + param_q.data * (1. - self.tau)
        
        for param_q, param_k in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_k.data = param_k.data * self.tau + param_q.data * (1. - self.tau)

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def forward(self, image_one, image_two):
        # Online network projections and predictions
        _, online_feat_one = self.online_encoder(image_one)
        _, online_feat_two = self.online_encoder(image_two)
        
        online_proj_one = self.online_projector(online_feat_one)
        online_proj_two = self.online_projector(online_feat_two)
        
        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        # Target network projections
        with torch.no_grad():
            _, target_feat_one = self.target_encoder(image_one)
            _, target_feat_two = self.target_encoder(image_two)
            
            target_proj_one = self.target_projector(target_feat_one)
            target_proj_two = self.target_projector(target_feat_two)

        # Symmetric loss
        loss_one = self.regression_loss(online_pred_one, target_proj_two.detach())
        loss_two = self.regression_loss(online_pred_two, target_proj_one.detach())

        return (loss_one + loss_two).mean()
