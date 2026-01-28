import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Tabular Attention Blocks (1D Adaptations) ---

class ChannelAttention1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, 1) or (B, C)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(-1)
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1)).unsqueeze(-1)
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1D, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C, 1)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(concat))
        return x * out

class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM1D, self).__init__()
        self.ca = ChannelAttention1D(channels, reduction)
        self.sa = SpatialAttention1D(kernel_size)

    def forward(self, x):
        # Handle flat tabular data (B, C)
        flat = False
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            flat = True
        
        x = self.ca(x)
        x = self.sa(x)
        
        if flat:
            x = x.view(x.size(0), -1)
        return x

# --- Advanced Risk Prediction Model ---

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionResNetRisk(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3):
        super(AttentionResNetRisk, self).__init__()
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.initial_bn = nn.BatchNorm1d(hidden_dim)
        self.attention = CBAM1D(hidden_dim)
        
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.initial_bn(self.initial_fc(x)))
        x = self.attention(x)
        x = self.res_blocks(x)
        x = self.final_fc(x)
        return x # Return logits

class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1):
        super(EnhancedMLP, self).__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.pipeline = nn.Sequential(*layers)

    def forward(self, x):
        return self.pipeline(x)

# --- Advanced Glucose Forecasting Model ---

class STAttentionLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, output_dim=1):
        super(STAttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Temporal Attention
        self.attention_fc = nn.Linear(hidden_dim, 1)
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (B, SeqLen, InputDim)
        lstm_out, _ = self.lstm(x) # (B, SeqLen, HiddenDim)
        
        # Attention Weights
        attn_weights = F.softmax(self.attention_fc(lstm_out), dim=1) # (B, SeqLen, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1) # (B, HiddenDim)
        
        out = self.fc(context)
        return out

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, L, C)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

if __name__ == "__main__":
    # Test shapes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing modules on {device}")
    
    risk_model = AttentionResNetRisk(input_dim=8).to(device)
    dummy_tab = torch.randn(10, 8).to(device)
    print("Risk Prediction output shape:", risk_model(dummy_tab).shape)
    
    forecast_model = STAttentionLSTM().to(device)
    dummy_seq = torch.randn(10, 12, 1).to(device)
    print("Forecasting output shape:", forecast_model(dummy_seq).shape)
