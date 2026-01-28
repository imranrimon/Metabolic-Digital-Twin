import torch
import torch.nn as nn
import rtdl_revisiting_models as rtdl
import torchcde

class FTTransformerModel(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities):
        super().__init__()
        self.model = rtdl.FTTransformer(
            n_cont_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            d_block=192,
            d_out=192,
            n_blocks=3,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden_multiplier=4/3,
            ffn_dropout=0.1,
            residual_dropout=0.0
        )
        self.head = nn.Linear(192, 1)

    def forward(self, x_num, x_cat=None):
        # rtdl FTTransformer returns (n_samples, d_out) if used as a model
        # but the backbone version is what's usually called.
        x = self.model(x_num, x_cat)
        return self.head(x)

if __name__ == "__main__":
    # Quick test
    import rtdl_revisiting_models as rtdl
    import torch.nn as nn
    test_model = FTTransformerModel(4, [2, 2])
    x_n = torch.randn(2, 4)
    x_c = torch.zeros(2, 2).long()
    out = test_model(x_n, x_c)
    print(f"Success! Output shape: {out.shape}")

class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.pipeline = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_channels * input_channels)
        )

    def forward(self, t, z):
        batch = z.size(0)
        # Output must be (batch, hidden, input) for CDE
        return self.pipeline(z).view(batch, self.hidden_channels, self.input_channels)

class NeuralCDEModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        spline = torchcde.CubicSpline(coeffs)
        # Solve the CDE
        z0 = self.initial(spline.evaluate(spline.interval[0]))
        z_t = torchcde.cdeint(X=spline, z0=z0, func=self.func, t=spline.interval)
        return self.readout(z_t[:, -1])
