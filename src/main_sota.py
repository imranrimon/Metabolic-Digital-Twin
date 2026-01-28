import torch
import json
import numpy as np
import pandas as pd
from models_sota import FTTransformerModel, NeuralCDEModel
from metabolic_rl import DQNAgent, MetabolicEnv

class MetabolicDigitalTwin:
    """
    The ultimate SOTA system:
    1. FT-Transformer for high-precision diabetes risk.
    2. Neural CDE for continuous biological process modeling.
    3. RL-Policy for proactive dietary optimization.
    """
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.risk_model = None
        self.trend_model = None
        self.policy_agent = None

    def load_sota(self, ft_path, cde_path, rl_path):
        # FT-Transformer: 4 num features (age, bmi, hba1c, glucose), 4 cats
        self.risk_model = FTTransformerModel(n_num_features=4, cat_cardinalities=[2, 8, 2, 2]).to(self.device)
        self.risk_model.load_state_dict(torch.load(ft_path))
        self.risk_model.eval()

        # Neural CDE: 2 channels (time, glucose)
        self.trend_model = NeuralCDEModel(input_channels=2, hidden_channels=32, output_channels=1).to(self.device)
        self.trend_model.load_state_dict(torch.load(cde_path))
        self.trend_model.eval()

        # RL Policy: 1 state dim (glucose), 5 actions (meal types)
        self.policy_agent = DQNAgent(state_dim=1, action_dim=5)
        self.policy_agent.model.load_state_dict(torch.load(rl_path))

    def get_digital_twin_profile(self, user_num, user_cat, glucose_coeffs):
        """
        user_num: [age, bmi, hba1c, current_glucose]
        user_cat: [gender, smoking, hypertension, heart_disease]
        glucose_coeffs: torchcde coefficients for past 1-hour glucose trend
        """
        # 1. Risk Assessment
        with torch.no_grad():
            x_n = torch.FloatTensor(user_num).unsqueeze(0).to(self.device)
            x_c = torch.LongTensor(user_cat).unsqueeze(0).to(self.device)
            risk_logit = self.risk_model(x_n, x_c)
            risk_prob = torch.sigmoid(risk_logit).item()

        # 2. Continuous Trend Forecasting
        with torch.no_grad():
            trend_val = self.trend_model(glucose_coeffs.to(self.device)).item()

        # 3. Proactive Policy Action
        current_g = user_num[3]
        action_idx = self.policy_agent.act([current_g])
        meal_types = ["Low Carb", "Balanced", "Moderate Carb", "High Fiber", "Custom Protocol"]
        recommended_action = meal_types[action_idx]

        return {
            "system": "Metabolic Digital Twin (v4-SOTA)",
            "precise_diabetes_risk": f"{risk_prob*100:.1f}%",
            "continuous_biological_trend": round(trend_val, 2),
            "proactive_recommendation": f"Adopt {recommended_action} meal strategy to maximize Time-in-Range.",
            "status": "Optimal Control Active" if 70 <= current_g <= 140 else "Corrective Action Required"
        }

if __name__ == "__main__":
    twin = MetabolicDigitalTwin()
    
    # Placeholder for test run (Paths assumed from training)
    ft_p = 'f:/Diabetics Project/ft_transformer_risk.pth'
    cde_p = 'f:/Diabetics Project/neural_cde_glucose.pth'
    rl_p = 'f:/Diabetics Project/metabolic_policy.pth'
    
    try:
        twin.load_sota(ft_p, cde_p, rl_p)
        
        # Test Input
        u_num = [45, 28.5, 6.2, 125.0]
        u_cat = [1, 3, 0, 0]
        # Dummy coeffs (would come from cde_preprocess in real app)
        dummy_coeffs = torch.randn(1, 60, 2) # Just for structure check
        
        # profile = twin.get_digital_twin_profile(u_num, u_cat, dummy_coeffs)
        # print(json.dumps(profile, indent=2))
        print("SOTA Digital Twin System Loaded.")
    except Exception as e:
        print(f"System not fully converged yet: {e}")
