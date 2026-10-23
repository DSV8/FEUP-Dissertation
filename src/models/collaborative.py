"""FM, MLP, and neural-CF scorers used by the pointwise CF objective."""

import torch
import torch.nn as nn
from .fusion import CROSS_ATTENTION, CrossAttentionFusion, Transformer, normalize_fusion_mode
from .tabular_encoders import TabularEncoder


class FM_model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.use_tabular = bool(config.get("use_tabular", True))
        self.fused_input = bool(config.get("fused_input", False)) or not self.use_tabular
        # Image embedding projections or fused embedding projections.
        self.user_image_projection = nn.Linear(
            config["image_embedding_dim"],
            config["projection_dim"],
        )
        self.item_image_projection = nn.Linear(
            config["image_embedding_dim"],
            config["projection_dim"],
        )
        # Tabular encoders are only used when inputs are raw image/tabular pairs.
        if self.fused_input:
            self.user_tab_encoder = nn.Identity()
            self.item_tab_encoder = nn.Identity()
        else:
            tab_in_features = config.get("tabular_in_features", 5)
            tab_dim = config.get("tabular_encoder_dim", config["projection_dim"])
            tab_nhead = config.get("tabular_nhead", 1)
            tab_layers = config.get("tabular_num_layers", 1)
            tab_encoder_type = config.get("tabular_encoder_type", "transformer")
            self.user_tab_encoder = TabularEncoder(tab_in_features, tab_dim, config["projection_dim"], nhead=tab_nhead, num_layers=tab_layers, encoder_type=tab_encoder_type)
            self.item_tab_encoder = TabularEncoder(tab_in_features, tab_dim, config["projection_dim"], nhead=tab_nhead, num_layers=tab_layers, encoder_type=tab_encoder_type)
        self.device = config["device"]

        # Total embedding dim: 2 * projection_dim when fused input is already merged or tabular is disabled; otherwise 4 * projection_dim.
        self.total_emb_dim = 2 * config["projection_dim"] if self.fused_input else 4 * config["projection_dim"]
        self.k = config["k"]
        self.interaction_dim = self.k + 1

        self.w0 = nn.Parameter(torch.zeros(1))
        self.w  = nn.Parameter(torch.randn(self.total_emb_dim))
        self.v = nn.Parameter(torch.randn(self.total_emb_dim, self.k))

    def init_weights(self):
        self.user_image_projection.weight.data.uniform_(0., 0.05)
        self.item_image_projection.weight.data.uniform_(0., 0.05)
        if not self.fused_input:
            self.user_tab_encoder.projection.weight.data.uniform_(0., 0.05)
            self.item_tab_encoder.projection.weight.data.uniform_(0., 0.05)

    def encode_features(self, user_img, item_img, user_tab, item_tab):
        user_image_embd = self.user_image_projection(user_img.to(self.device))
        item_image_embd = self.item_image_projection(item_img.to(self.device))
        if self.fused_input:
            return torch.cat([
                user_image_embd,
                item_image_embd,
            ], dim=1)
        user_tab_embd = self.user_tab_encoder(user_tab.to(self.device))
        item_tab_embd = self.item_tab_encoder(item_tab.to(self.device))

        return torch.cat([
            user_image_embd,
            user_tab_embd,
            item_image_embd,
            item_tab_embd,
        ], dim=1)

    def forward_components(self, x):
        user_img, item_img, user_tab, item_tab = x
        x_emb = self.encode_features(user_img, item_img, user_tab, item_tab)

        first_order_term = self.w0 + torch.sum(self.w * x_emb, dim=1, keepdim=True)
        xv = torch.matmul(x_emb, self.v)
        interaction_vector = 0.5 * (xv * xv - torch.matmul(x_emb * x_emb, self.v * self.v))
        interaction_term = torch.sum(interaction_vector, dim=1, keepdim=True)
        fm_branch_representation = torch.cat([first_order_term, interaction_vector], dim=1)
        score = (first_order_term + interaction_term).view(-1)

        return {
            "features": x_emb,
            "linear_term": first_order_term,
            "first_order_term": first_order_term,
            "interaction_vector": interaction_vector,
            "fm_branch_representation": fm_branch_representation,
            "score": score,
        }

    def forward(self, x):
        return self.forward_components(x)["score"]


class NN_model(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.use_tabular = bool(config.get("use_tabular", True))
        self.fused_input = bool(config.get("fused_input", False)) or not self.use_tabular
        # Image embedding projections or fused embedding projections.
        self.user_image_projection = nn.Linear(
            config["image_embedding_dim"],
            config["projection_dim"],
        )
        self.item_image_projection = nn.Linear(
            config["image_embedding_dim"],
            config["projection_dim"],
        )
        if self.fused_input:
            self.user_tab_encoder = nn.Identity()
            self.item_tab_encoder = nn.Identity()
        else:
            tab_in_features = config.get("tabular_in_features", 5)
            tab_dim = config.get("tabular_encoder_dim", config["projection_dim"])
            tab_nhead = config.get("tabular_nhead", 1)
            tab_layers = config.get("tabular_num_layers", 1)
            tab_encoder_type = config.get("tabular_encoder_type", "transformer")
            self.user_tab_encoder = TabularEncoder(tab_in_features, tab_dim, config["projection_dim"], nhead=tab_nhead, num_layers=tab_layers, encoder_type=tab_encoder_type)
            self.item_tab_encoder = TabularEncoder(tab_in_features, tab_dim, config["projection_dim"], nhead=tab_nhead, num_layers=tab_layers, encoder_type=tab_encoder_type)
        self.device = config["device"]

        # Total embedding dim: 2 * projection_dim when fused input is already merged or tabular is disabled; otherwise 4 * projection_dim.
        self.total_emb_dim = 2 * config["projection_dim"] if self.fused_input else 4 * config["projection_dim"]
        self.penultimate_dim = config["lay_4"]

        # Define a neural network with hidden layers
        self.fc1 = nn.Linear(self.total_emb_dim, config["lay_1"])
        self.fc2 = nn.Linear(config["lay_1"], config["lay_2"])
        self.fc3 = nn.Linear(config["lay_2"], config["lay_3"])
        self.fc4 = nn.Linear(config["lay_3"], config["lay_4"])
        self.fc5 = nn.Linear(config["lay_4"], 1)

        # Layer Normalization
        self.bn = nn.LayerNorm(self.total_emb_dim)

        # Activation functions
        self.dropout = nn.Dropout(config.get("dropout", 0.3))
        self.tanh = nn.Tanh()

    def encode_features(self, user_img, item_img, user_tab, item_tab):
        user_image_embd = self.user_image_projection(user_img.to(self.device))
        item_image_embd = self.item_image_projection(item_img.to(self.device))
        if self.fused_input:
            return torch.cat([
                user_image_embd,
                item_image_embd,
            ], dim=1)
        user_tab_embd = self.user_tab_encoder(user_tab.to(self.device))
        item_tab_embd = self.item_tab_encoder(item_tab.to(self.device))

        return torch.cat([
            user_image_embd,
            user_tab_embd,
            item_image_embd,
            item_tab_embd,
        ], dim=1)

    def forward_components(self, x):
        user_img, item_img, user_tab, item_tab = x
        x_emb = self.encode_features(user_img, item_img, user_tab, item_tab)

        # Apply Batch Normalization
        x_emb = self.bn(x_emb)

        hidden_1 = self.dropout(self.tanh(self.fc1(x_emb)))
        hidden_2 = self.dropout(self.tanh(self.fc2(hidden_1)))
        hidden_3 = self.dropout(self.tanh(self.fc3(hidden_2)))
        penultimate = self.dropout(self.tanh(self.fc4(hidden_3)))
        score = self.fc5(penultimate).view(-1)

        return {
            "features": x_emb,
            "penultimate": penultimate,
            "score": score,
        }

    def forward(self, x):
        return self.forward_components(x)["score"]


class NCF_model(nn.Module):
    def __init__(
        self,
        config: dict,
        nn_model: NN_model,
        fm_model: FM_model,
        freeze_nn: bool = False,
        freeze_fm: bool = False,
        fusion_mode: str = CROSS_ATTENTION,
    ):
        super().__init__()
        self.nn_model = nn_model
        self.fm_model = fm_model
        self.device = config['device']
        self.fusion_dim = config.get("ncf_fusion_dim", max(64, min(self.fm_model.total_emb_dim, self.nn_model.total_emb_dim)))
        self.fusion_dropout = config.get("ncf_dropout", config.get("dropout", 0.1))
        self.fusion_heads = config.get("ncf_fusion_heads", 2)
        self.fusion_layers = config.get("ncf_fusion_layers", 2)
        self.fusion_mode = normalize_fusion_mode(fusion_mode)
        self.fm_branch_dim = int(getattr(self.fm_model, "interaction_dim", getattr(self.fm_model, "k", 1) + 1))
        self.nn_branch_dim = int(getattr(self.nn_model, "penultimate_dim", getattr(self.nn_model, "lay_4", 1)))

        # Freeze nn_model parameters if freeze_nn is True
        if freeze_nn:
            for param in self.nn_model.parameters():
                param.requires_grad = False

        # Freeze fm_model parameters if freeze_fm is True
        if freeze_fm:
            for param in self.fm_model.parameters():
                param.requires_grad = False

        self.fm_projection = nn.Sequential(
            nn.Linear(self.fm_branch_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.fusion_dropout),
        )
        self.nn_projection = nn.Sequential(
            nn.Linear(self.nn_branch_dim, self.fusion_dim),
            nn.LayerNorm(self.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.fusion_dropout),
        )
        if self.fusion_mode == CROSS_ATTENTION:
            self.fusion_module = CrossAttentionFusion(
                embed_dim=self.fusion_dim,
                nhead=self.fusion_heads,
                dropout=self.fusion_dropout,
            )
        else:
            self.branch_token_embedding = nn.Parameter(torch.randn(1, 2, self.fusion_dim))
            self.fusion_transformer = Transformer(
                d_model=self.fusion_dim,
                nhead=self.fusion_heads,
                num_layers=self.fusion_layers,
                dim_feedforward=self.fusion_dim * 4,
                dropout=self.fusion_dropout,
                batch_first=True,
            )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, max(self.fusion_dim // 2, 1)),
            nn.GELU(),
            nn.Dropout(self.fusion_dropout),
            nn.Linear(max(self.fusion_dim // 2, 1), 1),
        )

    def forward(self, x):
        user_img, item_img, user_tab, item_tab = x
        # Fuse the FM interaction signal with the NN penultimate representation.
        fm_components = self.fm_model.forward_components((user_img, item_img, user_tab, item_tab))
        nn_components = self.nn_model.forward_components((user_img, item_img, user_tab, item_tab))

        fm_embedding = self.fm_projection(fm_components["fm_branch_representation"])
        nn_embedding = self.nn_projection(nn_components["penultimate"])

        if self.fusion_mode == CROSS_ATTENTION:
            fused_vector = self.fusion_module(fm_embedding, nn_embedding)
        else:
            # Stack the two branch embeddings to create a sequence of length 2 (two tokens), because the Transformer expects a sequence dimension
            # token 0 = FM branch representation, token 1 = NN branch representation.
            combined_output = torch.stack([fm_embedding, nn_embedding], dim=1)

            # Add learnable branch token embeddings to give the Transformer positional/identity information about which token represents FM and which represents NN.
            combined_output = combined_output + self.branch_token_embedding

            # Pass the two-token sequence through the Transformer encoder.
            # Each token can attend to the other, allowing the model to learn how to weight/combine them.
            combined_output = self.fusion_transformer(combined_output)

            # Pool the two tokens into a single fused representation by averaging them.
            # This fused vector now contains information from both branches, weighted by attention.
            fused_vector = combined_output.mean(dim=1)

        # Pass the fused representation through the final MLP head to produce a scalar ranking score.
        final_output = self.fusion_head(fused_vector)

        # Flatten to (batch_size) for compatibility with the loss function.
        return final_output.view(-1)
