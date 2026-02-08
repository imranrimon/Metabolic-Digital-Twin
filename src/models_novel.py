import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer
    Replaces traditional linear+activation with learnable activation functions on edges.
    Based on: "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # B-spline coefficients for learnable activation functions
        # Shape: (out_features, in_features, grid_size)
        self.coefficients = nn.Parameter(torch.randn(out_features, in_features, grid_size) * 0.1)
        
        # Grid points for B-splines
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))
        
        # Base linear transformation (optional, for residual connection)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        
    def b_spline_basis(self, x: torch.Tensor, grid: torch.Tensor, k: int = 3) -> torch.Tensor:
        """
        Compute B-spline basis functions
        x: input tensor of shape (batch, in_features)
        Returns: basis of shape (batch, in_features, grid_size)
        """
        # Simplified cubic B-spline
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, grid_size)
        
        # Distance to grid points
        dist = torch.abs(x - grid)
        
        # Cubic B-spline kernel
        basis = torch.where(dist < 1,
                            (1 - 3 * dist**2 + 3 * dist**3),
                            torch.where(dist < 2,
                                        (2 - dist)**3,
                                        torch.zeros_like(dist)))
        return basis
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)
        Returns: (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # Compute B-spline basis
        basis = self.b_spline_basis(x, self.grid)  # (batch, in_features, grid_size)
        
        # Apply learnable coefficients
        # (batch, in_features, grid_size) @ (out_features, in_features, grid_size)
        # = (batch, out_features, in_features)
        basis = basis.unsqueeze(1)  # (batch, 1, in_features, grid_size)
        coeffs = self.coefficients.unsqueeze(0)  # (1, out_features, in_features, grid_size)
        
        # Element-wise multiplication and sum over grid
        activations = (basis * coeffs).sum(dim=-1)  # (batch, out_features, in_features)
        
        # Sum over input features
        output = activations.sum(dim=-1)  # (batch, out_features)
        
        # Add base linear transformation (residual)
        output = output + torch.matmul(x, self.base_weight.t())
        
        return output

class KANModel(nn.Module):
    """
    Kolmogorov-Arnold Network for Tabular Data
    Efficient alternative to FT-Transformer with learnable activations
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], output_dim: int = 1, 
                 grid_size: int = 5, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(KANLayer(prev_dim, hidden_dim, grid_size=grid_size))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.kan_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan_layers(x)
        return self.output_layer(x)


class TabNetEncoder(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning
    Based on Google's TabNet architecture for interpretable predictions
    """
    def __init__(self, input_dim: int, output_dim: int, n_d: int = 8, n_a: int = 8,
                 n_steps: int = 3, gamma: float = 1.3, n_independent: int = 2, n_shared: int = 2):
        super().__init__()
        # Will use pytorch-tabnet library for full implementation
        # This is a placeholder structure
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        
    def forward(self, x):
        # Placeholder - will be replaced with actual TabNet from library
        return x


class MambaBlock(nn.Module):
    """
    Mamba-like Block (Simplified Pure PyTorch Implementation)
    Implements the core Selective Scan mechanism using standard PyTorch operations.
    This avoids the complex CUDA build requirements of the official mamba-ssm package
    while retaining the architectural benefits for benchmarking.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = Variable_rank = (d_model + 15) // 16
        
        # Projects input to 2*d_inner (for x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # Activation
        self.activation = nn.SiLU()
        
        # State Space Model projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # S4D real initialization
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        
        # 1. Project to higher dim
        xz = self.in_proj(x) # (batch, seq_len, 2*d_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1) # (batch, seq_len, d_inner)
        
        # 2. Convolution
        x_branch = x_branch.transpose(1, 2) # (batch, d_inner, seq_len)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len] # Causal padding
        x_branch = x_branch.transpose(1, 2) # (batch, seq_len, d_inner)
        
        x_branch = self.activation(x_branch)
        z_branch = self.activation(z_branch)
        
        # 3. SSM (Simplified Selective Scan)
        # In a real efficient implementation, this would be a custom CUDA kernel
        # Here we use a sequential loop for compatibility
        
        # Project params
        x_dbl = self.x_proj(x_branch) # (batch, seq_len, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (batch, seq_len, d_inner)
        
        # Discretize A
        A = -torch.exp(self.A_log) # (d_inner, d_state)
        
        # Scan loop
        y = []
        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1) # (batch, d_inner, 1)
            dA = torch.exp(dt_t * A) # (batch, d_inner, d_state)
            dB = dt_t * B[:, t, :].unsqueeze(1) # (batch, d_inner, d_state)
            
            # h_t = A_bar * h_{t-1} + B_bar * x_t
            xt = x_branch[:, t, :].unsqueeze(-1) # (batch, d_inner, 1)
            h = dA * h + dB * xt
            
            # y_t = C_t * h_t
            Ct = C[:, t, :].unsqueeze(1) # (batch, 1, d_state)
            yt = (h * Ct).sum(dim=-1) # (batch, d_inner)
            y.append(yt)
            
        y = torch.stack(y, dim=1) # (batch, seq_len, d_inner)
        
        # Add residual connection D * x
        y = y + x_branch * self.D
        
        # 4. Multiply by gate and project out
        y = y * z_branch
        out = self.out_proj(y)
        
        return out


class MambaModel(nn.Module):
    """
    Full Mamba Architecture for Sequence Modeling
    """
    def __init__(self, d_input: int, d_output: int, d_model: int = 64, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Linear(d_input, d_model)
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, expand=2) # Using our custom pure-torch block
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(d_model, d_output)
        
    def forward(self, x):
        # x: (batch, seq_len, d_input)
        x = self.embedding(x)
        
        for layer in self.layers:
            # Residual connection handled inside MambaBlock in official repo, 
            # here we add it explicitly or assume block does it
            # Our block implementation does not have residual around it, so we add it here
            residual = x
            x = layer(x)
            x = x + residual
            
        x = self.norm(x)
        x = self.dropout(x)
        
        # Predict based on last time step
        out = self.output_head(x[:, -1, :])
        return out


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) - Simplified PyTorch Implementation
    For multi-horizon time series forecasting
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # Variable Selection Network (Simplified as Linear)
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        
        # LSTM Encoder-Decoder (Seq2Seq component)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        # Gate -> Add -> Norm (GLU-like)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GLU()
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Multi-Head Attention (Long-range dependencies)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Output layers
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # 1. Encoding
        encoded = self.input_encoder(x) # (batch, seq_len, hidden)
        
        # 2. LSTM (Local temporal processing)
        lstm_out, _ = self.lstm(encoded) # (batch, seq_len, hidden)
        
        # 3. Gating and Residual
        gated = self.gate(lstm_out)
        res1 = self.norm1(encoded + gated)
        
        # 4. Attention (Global temporal processing)
        attn_out, _ = self.attention(res1, res1, res1)
        res2 = self.norm2(res1 + self.dropout(attn_out))
        
        # 5. Output
        out = self.output_layer(res2[:, -1, :]) # Predict next step based on sequence
        return out


class MultimodalFusion(nn.Module):
    """
    End-to-end Multimodal Fusion Architecture
    Combines tabular and temporal encoders with cross-attention
    """
    def __init__(self, tabular_dim: int, temporal_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # Tabular encoder (KAN-based)
        self.tabular_encoder = KANModel(
            input_dim=tabular_dim,
            hidden_dims=[hidden_dim, hidden_dim // 2],
            output_dim=hidden_dim
        )
        
        # Temporal encoder (Mamba-based)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_dim, hidden_dim),
            MambaBlock(d_model=hidden_dim),
            MambaBlock(d_model=hidden_dim)
        )
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Multi-task heads
        self.risk_head = nn.Linear(hidden_dim, 1)  # Diabetes risk
        self.forecast_head = nn.Linear(hidden_dim, 1)  # Glucose prediction
        self.diet_head = nn.Linear(hidden_dim, 5)  # Diet recommendation (5 classes)
        
    def forward(self, tabular_input, temporal_input):
        """
        tabular_input: (batch, tabular_dim)
        temporal_input: (batch, seq_len, temporal_dim)
        """
        # Encode both modalities
        tab_emb = self.tabular_encoder(tabular_input)  # (batch, hidden_dim)
        temp_emb = self.temporal_encoder(temporal_input)  # (batch, seq_len, hidden_dim)
        
        # Cross-attention: use tabular as query, temporal as key/value
        tab_emb_expanded = tab_emb.unsqueeze(1)  # (batch, 1, hidden_dim)
        fused, _ = self.cross_attn(
            query=tab_emb_expanded,
            key=temp_emb,
            value=temp_emb
        )  # (batch, 1, hidden_dim)
        
        fused = fused.squeeze(1)  # (batch, hidden_dim)
        
        # Multi-task predictions
        risk = torch.sigmoid(self.risk_head(fused))
        forecast = self.forecast_head(fused)
        diet = self.diet_head(fused)
        
        return {
            'risk': risk,
            'forecast': forecast,
            'diet': diet,
            'embedding': fused
        }


# Test shapes
if __name__ == "__main__":
    # Test KAN
    kan = KANModel(input_dim=8, hidden_dims=[64, 32], output_dim=1)
    x = torch.randn(16, 8)
    out = kan(x)
    print(f"KAN output shape: {out.shape}")  # (16, 1)
    
    # Test Multimodal
    fusion = MultimodalFusion(tabular_dim=8, temporal_dim=2, hidden_dim=128)
    tab = torch.randn(16, 8)
    temp = torch.randn(16, 60, 2)
    outputs = fusion(tab, temp)
    print(f"Fusion risk: {outputs['risk'].shape}")  # (16, 1)
    print(f"Fusion forecast: {outputs['forecast'].shape}")  # (16, 1)
    print(f"Fusion diet: {outputs['diet'].shape}")  # (16, 5)
