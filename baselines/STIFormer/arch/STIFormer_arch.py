import math
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LocalSTEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, timestep_max, dropout, show_scores: bool = False):
        super().__init__()
        # Multiple GraphAttention heads
        self.ga_2 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max,
                                   time_filter=2, show_scores=show_scores, dropout=dropout)
        # self.ga_3 = GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max,
        #                            time_filter=3, show_scores=show_scores, dropout=dropout)
        self.ga_6= GraphAttention(in_channels=in_channels, out_channels=out_channels, timestep_max=timestep_max,
                                   time_filter=6, show_scores=show_scores, dropout=dropout)

        # Additional layers
        self.fc_res = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=1)
        self.fc_out = torch.nn.Conv2d(in_channels=out_channels * 2, out_channels=out_channels, kernel_size=(1, 1),
                                      stride=1, )
        # Normalization and dropout
        self.norm = torch.nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = x.transpose(1, 3)
        res = self.fc_res(x)  # B, C', N, T
        x = torch.cat([self.ga_2(x),self.ga_6(x)], dim=1 )
        x = self.fc_out(x)  # B, C', N, T'
        x = self.norm((x + res))  # B, C', N, T'
        x = self.dropout(x)  # B, C', N, T'
        x = x.transpose(1, 3)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads
        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
    def forward(self, query, key, value, ):
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]
        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)
        key = key.transpose( -1, -2)
        attn_score = ( query @ key) / self.head_dim ** 0.5  # (num_heads * batch_size, ..., tgt_length, src_length)
        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place
        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        out = self.out_proj(out)
        return out

class SelfAttentionLayer(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.argumented_linear = nn.Linear(model_dim, model_dim)
        self.act1 = nn.GELU()
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y=None, dim=-2, c=None, augment=False):
        x = x.transpose(dim, -2)
        augmented = None
        # x: (batch_size, ..., length, model_dim)
        if c is not None:
            residual = c
        else:
            residual = x
        if y is None:
            out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
            if augment is True:
                augmented = self.act1(self.argumented_linear(residual))
        else:
            y = y.transpose(dim, -2)
            out = self.attn(y, x, x)
        out = self.dropout1(out)
        if augmented is not None and augment is not False:
            out = self.ln1(residual + out + augmented)
        else:
            out = self.ln1(residual + out)
        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)
        out = out.transpose(dim, -2)
        return out


class STblock(nn.Module):
    def __init__(
            self, model_dim, feed_forward_dim, in_steps, num_heads, dropout, mask=False):
        super().__init__()
        self.te = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout, mask)
        self.st_lo =LocalSTEncoder(model_dim, model_dim, in_steps, dropout)
        self.st_gl = SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout, mask)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(model_dim)
    def forward(self, temporal_x, spatial_x):
        res = temporal_x
        x_te= self.te(temporal_x, dim=1)
        x_loc= self.st_lo(spatial_x)
        x_gl = self.st_gl(x_te,x_loc,dim=2)
        out = self.ln(x_gl + res )
        out = self.dropout(out)
        return out

class STIFormer(nn.Module):
    def __init__(
            self, num_nodes, in_steps, out_steps, steps_per_day, input_dim, output_dim, input_embedding_dim,tod_embedding_dim,
            ts_embedding_dim,dow_embedding_dim, time_embedding_dim, adaptive_embedding_dim, feed_forward_dim,num_heads,num_layers,
            dropout, use_mixed_proj,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.ts_embedding_dim = ts_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
                input_embedding_dim
                + tod_embedding_dim
                + dow_embedding_dim
                + ts_embedding_dim
                + time_embedding_dim
                + adaptive_embedding_dim
            # + node_dim
        )
        # self.mask_embedding = SpatioTemporalMaskEmbedding(in_steps, num_nodes, adaptive_embedding_dim, dropout)
        self.num_heads = num_heads
        self.use_mixed_proj = use_mixed_proj
        self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)
        self.STblock = nn.ModuleList(
            [
                STblock(self.model_dim, feed_forward_dim, in_steps, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        if self.input_embedding_dim > 0:
            self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if time_embedding_dim > 0:
            self.time_embedding = nn.Embedding(7 * steps_per_day, self.time_embedding_dim)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
        if self.ts_embedding_dim > 0:
            self.time_series_emb_layer = nn.Conv2d(
                in_channels=self.input_dim * self.in_steps,
                out_channels=self.ts_embedding_dim,
                kernel_size=(1, 1),
                bias=True
            )

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        x = history_data
        batch_size, _, num_nodes, _ = x.shape
        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        if self.time_embedding_dim > 0:
            tod = x[..., 1]
            dow = x[..., 2]
        x = x[..., : self.input_dim]
        if self.ts_embedding_dim > 0:
            input_data = x.transpose(1, 2).contiguous()
            input_data = input_data.view(
                batch_size, self.num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            # B L*3 N 1
            time_series_emb = self.time_series_emb_layer(input_data)
            time_series_emb = time_series_emb.transpose(1, -1).expand(batch_size, self.in_steps, self.num_nodes,
                                                                      self.ts_embedding_dim)
        x = self.input_proj(x)
        features = [x]
        if self.ts_embedding_dim > 0:
            features.append(time_series_emb)

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.time_embedding_dim > 0:
            time_emb = self.time_embedding(
                ((tod + dow * 7) * self.steps_per_day).long()
            )
            features.append(time_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
            # x_st,x_t = self.mask_embedding(adp_emb)
        x1 = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)
        x2 =x1
        # x1 = torch.cat([ x_st, x], dim=-1)
        # x2= torch.cat([x_t, x], dim=-1)
        for block in self.STblock:
            x= block(x1, x2)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)
        return out
