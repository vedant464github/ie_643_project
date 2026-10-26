import torch
import torch.nn as nn

from src.models.revin import RevIN

class GRUForecaster(nn.Module):
    """
    Lightweight GRU encoder -> linear head forecaster.
    Input: (batch, input_len) -> Output: (batch, horizon)
    """
    def __init__(self, input_len, horizon, hidden_size=32, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, input_len) -> (batch, input_len, 1)
        x = x.unsqueeze(-1)
        _, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]         # (batch, hidden)
        out = self.head(last_hidden)  # (batch, horizon)
        return out

    def get_features(self, x):
        """Exposes the encoder's final hidden state — needed later for CORAL/AdaRNN losses."""
        x = x.unsqueeze(-1)
        _, h_n = self.gru(x)
        return h_n[-1]

class DLinear(nn.Module):
    """
    DLinear (Zeng et al., 2023): moving-average trend/seasonal decomposition, one linear map each.
    Input: (batch, input_len) -> Output: (batch, horizon)
    """
    def __init__(self, input_len, horizon, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.seasonal = nn.Linear(input_len, horizon)
        self.trend = nn.Linear(input_len, horizon)

    def _components(self, x):
        # replicate-pad both ends so the moving average keeps length input_len
        pad = (self.kernel_size - 1) // 2
        x_pad = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, self.kernel_size - 1 - pad)], dim=1)
        trend = nn.functional.avg_pool1d(x_pad.unsqueeze(1), self.kernel_size, stride=1).squeeze(1)
        return self.seasonal(x - trend), self.trend(trend)

    def forward(self, x):
        seasonal_out, trend_out = self._components(x)
        return seasonal_out + trend_out

    def get_features(self, x):
        """No hidden encoder: features are the two component forecasts, concatenated (2 * horizon)."""
        return torch.cat(self._components(x), dim=-1)


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = torch.relu(self.conv1(nn.functional.pad(x, (self.causal_pad, 0))))
        y = torch.relu(self.conv2(nn.functional.pad(y, (self.causal_pad, 0))))
        return torch.relu(y + self.residual(x))


class TCNForecaster(nn.Module):
    """
    Dilated causal TCN encoder -> linear head on the last time step.
    Dilations 1..32 with kernel 3 give a receptive field of 253 steps (covers input_len=168).
    """
    def __init__(self, input_len, horizon, channels=32, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        blocks, in_ch = [], 1
        for d in dilations:
            blocks.append(TemporalBlock(in_ch, channels, kernel_size, d))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, horizon)

    def forward(self, x):
        return self.head(self.get_features(x))

    def get_features(self, x):
        return self.tcn(x.unsqueeze(1))[:, :, -1]  # (batch, channels) at the last time step


class PatchTSTForecaster(nn.Module):
    """
    Small PatchTST-style encoder (Nie et al., 2023): end-padded patches -> linear embedding + learned
    positions -> Transformer encoder -> mean-pool over patches -> linear head. Optional RevIN.
    """
    def __init__(self, input_len, horizon, patch_len=6, stride=3, d_model=32, n_layers=2,
                 n_heads=4, d_ff=64, dropout=0.0, revin=False):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        n_patches = (input_len + stride - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, horizon)
        self.revin = RevIN() if revin else None

    def _encode(self, x):
        x = torch.cat([x, x[:, -1:].repeat(1, self.stride)], dim=1)  # replicate-pad the end by one stride
        patches = x.unfold(1, self.patch_len, self.stride)            # (batch, n_patches, patch_len)
        return self.encoder(self.embed(patches) + self.pos).mean(dim=1)

    def forward(self, x):
        if self.revin is not None:
            x = self.revin.normalize(x)
        out = self.head(self._encode(x))
        if self.revin is not None:
            out = self.revin.denormalize(out)
        return out

    def get_features(self, x):
        """Mean-pooled encoder output (batch, d_model); RevIN-normalized input if enabled."""
        if self.revin is not None:
            x = self.revin.normalize(x)
        return self._encode(x)


BACKBONES = {
    "gru": GRUForecaster,
    "dlinear": DLinear,
    "tcn": TCNForecaster,
    "patchtst": PatchTSTForecaster,
}

def build_model(name, input_len, horizon, **kwargs):
    """Same constructor defaults for every domain; only input_len/horizon vary."""
    return BACKBONES[name](input_len, horizon, **kwargs)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
