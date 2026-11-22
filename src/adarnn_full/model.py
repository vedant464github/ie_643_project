"""
GRU backbone with the GRUForecaster dimensions (2 layers, hidden 32, linear head on the last-layer final hidden
state), additionally exposing the last-layer hidden state at every input time step for the per-step TDM.
Subclasses src.models.backbone.GRUForecaster without changing it: same parameters, same forward().
"""
import torch

from src.models.backbone import GRUForecaster


class GRUSeqForecaster(GRUForecaster):
    def hidden_states(self, x):
        """(batch, input_len) -> (batch, input_len, hidden): last GRU layer's hidden state h^t for t = 1..V."""
        out, _ = self.gru(x.unsqueeze(-1))
        return out

    def forward_with_states(self, x):
        """Prediction and all last-layer hidden states from one GRU pass. out[:, -1] == h_n[-1]."""
        hs = self.hidden_states(x)
        return self.head(hs[:, -1]), hs


def build_gru(input_len, horizon):
    return GRUSeqForecaster(input_len, horizon, hidden_size=32, num_layers=2)


def check_equivalence(model, x):
    """The prediction path must equal GRUForecaster.forward and head(get_features)."""
    with torch.no_grad():
        p1 = model(x)
        p2, hs = model.forward_with_states(x)
        p3 = model.head(model.get_features(x))
    assert torch.allclose(p1, p2, atol=1e-6) and torch.allclose(p1, p3, atol=1e-6)
    assert hs.shape == (x.shape[0], x.shape[1], model.hidden_size)
