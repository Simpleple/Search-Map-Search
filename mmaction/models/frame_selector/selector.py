import torch
import pickle
import os.path as osp
import torch.nn as nn

from .pointernet import PointerNet


class MappingFunction(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, img_features):
        return self.network(img_features)


class MappingFunctionDropout(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, img_features):
        return self.network(img_features)


class MappingFunctionNorm(nn.Module):
    """Frame selector network that uses gumbel-softmax for sampling frames.

    Args:
        feature_dim (int): feature dimension
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.network = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.fc = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        x = self.norm(x + self.network(x))
        return torch.relu(self.fc(x))


class MappingFunctionTrans(nn.Module):

    def __init__(self, feature_dim, nhead=2):
        super().__init__()

        self.trans_layer = torch.nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead)

        self.network = nn.Sequential(
            # nn.Dropout(p=0.3),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.PReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, img_features, masks):
        img_features = img_features.transpose(0, 1)
        enc_out = self.trans_layer(img_features, src_key_padding_mask=masks).mean(dim=0)
        return self.network(enc_out)


class FrameSelector(nn.Module):
    """Frame selector network that uses gumbel-softmax for sampling frames.

    Args:
        feature_dim (int): feature dimension
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_features):
        return self.network(img_features)

class FrameSelectorSMART(nn.Module):
    """Frame selector network that uses gumbel-softmax for sampling frames.

    Args:
        feature_dim (int): feature dimension
    """

    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.single_sel = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

        self.fc1 = nn.Linear(feature_dim * 2, 1)
        self.fc2 = nn.Linear(feature_dim * 4, 1)
        self.lstm = nn.LSTM(feature_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.fc4 = nn.Linear(feature_dim * 4, 1)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward_single(self, input):
        return self.single_sel(input)

    def forward_global(self, input):
        # input shape -- (batch_size, # frames, feature dim * 2)
        num_frames = input.shape[1]

        alpha = torch.sigmoid(self.fc1(input))
        z_prime = (alpha * input).sum(dim=1, keepdim=True) / alpha.sum()

        beta = torch.sigmoid(self.fc2(torch.cat([input, z_prime.repeat(1, num_frames, 1)], dim=-1)))
        w = torch.zeros_like(input)
        for i in range(num_frames):
            w[:, i] = (beta[:, :(i+1), :] * input[:, :(i+1), :]).sum(dim=1)

        hidden = (torch.zeros(1, 1, self.hidden_dim).to(input.device), torch.zeros(1, 1, self.hidden_dim).to(input.device))
        h, _ = self.lstm(w.transpose(0, 1), hidden)
        h = h.transpose(0, 1)
        lam = torch.softmax(self.fc3(h), dim=-1)

        z_primeprime = (lam * w).sum(dim=1, keepdim=True) / lam.sum()

        gamma = torch.sigmoid(self.fc4(torch.cat([w, z_primeprime.repeat(1, num_frames, 1)], dim=-1)))

        c = torch.zeros_like(h)
        for i in range(num_frames):
            c[:, i] = (gamma[:, :(i+1), :] * h[:, :(i+1), :]).sum(dim=1)

        return self.cls(c)

    def forward_score(self, single_input, global_input):

        single_score = self.single_sel(single_input)

        num_frames = global_input.shape[1]

        alpha = torch.sigmoid(self.fc1(global_input))
        z_prime = (alpha * global_input).sum(dim=1, keepdim=True) / alpha.sum()

        beta = torch.sigmoid(self.fc2(torch.cat([global_input, z_prime.repeat(1, num_frames, 1)], dim=-1)))
        w = torch.zeros_like(global_input)
        for i in range(num_frames):
            w[:, i] = (beta[:, :(i+1), :] * global_input[:, :(i+1), :]).sum(dim=1)

        hidden = (torch.zeros(1, 1, self.hidden_dim).to(global_input.device), torch.zeros(1, 1, self.hidden_dim).to(global_input.device))
        h, _ = self.lstm(w.transpose(0, 1), hidden)
        h = h.transpose(0, 1)
        lam = torch.softmax(self.fc3(h), dim=-1)

        return single_score * lam


class FrameSelectorTrans(nn.Module):
    """Frame selector network that uses gumbel-softmax for sampling frames.

    Args:
        feature_dim (int): feature dimension
    """

    def __init__(self, input_dim, hidden_dim, nhead=8):
        super().__init__()

        self.trans_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, img_features, masks):
        img_features = img_features.transpose(0, 1)
        enc_out = self.trans_layer(img_features, src_key_padding_mask=masks)
        return self.network(enc_out).transpose(0, 1)

class TopKFrameSelector(nn.Module):

    def __init__(self, feature_dim, topk, feature_prefix, temperature=0.5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )
        self.topk = topk
        self.temperature = temperature
        self.feature_prefix = feature_prefix

    def forward(self, img_features):
        return self.network(img_features)

    def sample_frames(self, imgs, video_id):
        feature_file = open(osp.join(self.feature_prefix, video_id + '.pkl'), 'rb')
        frame_features = pickle.load(feature_file)
        num_frames = frame_features.shape[0]
        topk = min(num_frames, self.topk)
        probs = self.forward(torch.FloatTensor(frame_features).cuda()).squeeze()

        frame_dist = torch.distributions.RelaxedOneHotCategorical(
            self.temperature, probs=probs)
        frame_weights = frame_dist.rsample()
        frame_weights = frame_weights.clamp(0.0, 1.0)
        frame_indices = torch.topk(frame_weights.squeeze(), topk)[1]

        frames = []
        for i in range(topk):
            idx = frame_indices[i]

            one_hot_i = torch.zeros_like(frame_weights).scatter_(-1, idx, 1.0)
            one_hot_i = one_hot_i - frame_weights.detach() + frame_weights

            frame_i = torch.sum(imgs * one_hot_i.reshape(1, one_hot_i.shape[0], 1, 1, 1), dim=1)
            frames.append(frame_i)

        return torch.cat(frames, dim=0)


class FrameSelectorPointerNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, lstm_layers, dropout, decode_len, bidir=True):
        super().__init__()
        self.network = PointerNet(input_dim, hidden_dim, lstm_layers, dropout, decode_len, bidir=bidir)

    def forward(self, input, seq_len, target=None):
        return self.network(input, seq_len, target)