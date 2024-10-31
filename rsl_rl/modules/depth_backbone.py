import torch
import torch.nn as nn
import sys
import torchvision

class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, env_cfg, depth_encoder_cfg) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        if env_cfg == None:
            self.combination_mlp = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    activation,
                                    nn.Linear(128, 32)
                                )
        else:
            self.combination_mlp = nn.Sequential(
                                        nn.Linear(32 + env_cfg.env.n_proprio, 128),
                                        activation,
                                        nn.Linear(128, 32)
                                    )
        self.rnn = nn.GRU(input_size=32, hidden_size=depth_encoder_cfg['hidden_dims'], batch_first=True)
        self.output_mlp = nn.Sequential(
                                nn.Linear(depth_encoder_cfg['hidden_dims'], 32),
                                last_activation
                            )
        # self.transformer = nn.Transformer(d_model=32, 
        #                                   nhead=8, 
        #                                   num_encoder_layers=6, 
        #                                   num_decoder_layers=6, 
        #                                   dim_feedforward=2048, 
        #                                   dropout=0.1, activation='relu')
        self.hidden_states = None
        self.dummy_param = nn.Parameter(torch.empty(0))
    def forward(self, depth_image, proprioception):
        depth_image = depth_image.to(self.dummy_param.device)
        depth_image = self.base_backbone(depth_image)
        proprioception = proprioception.to(depth_image.device)
        depth_latent = self.combination_mlp(torch.cat((depth_image, proprioception), dim=-1))
        depth_latent, self.hidden_states = self.rnn(depth_latent[:, None, :], self.hidden_states)
        depth_latent = self.output_mlp(depth_latent.squeeze(1))
        
        return depth_latent

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()

    
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, output_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent