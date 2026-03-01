import torch
import torch.nn as nn
import torch.nn.functional as F

# class TopkPredictor(nn.Module):
#     def __init__(self, input_dim=7168, vocab_size=256, hidden_dim=2048):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(hidden_dim, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, vocab_size)
#         )

#     def forward(self, x):
#         return self.net(x)  # shape: [batch_size, 256]

class TopkPredictor(nn.Module):
    def __init__(self, input_dim=7168, expert_num=256):
        super().__init__()
        self.expert_num = expert_num

        self.hidden_proj = nn.Linear(input_dim, 256)

        # inter_dim1 = 512
        # inter_dim2 = 128
        # self.hidden_proj = nn.Sequential(
        #     nn.Linear(input_dim, inter_dim1),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        # self.expert_proj = nn.Sequential(
        #     nn.Linear(expert_num, inter_dim2),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )
        # self.output_layer = nn.Sequential(
        #     nn.Linear(inter_dim1+inter_dim2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, expert_num)  # Predicting probability for 256 experts
        # )
    
    def forward(self, hidden_state, expert_mask = None):
        # hidden_state: [batch_size, hidden_dim]
        # expert_mask:  [batch_size, 256]
        
        # assert hidden_state.shape[1] == 7168, f"Expected hidden_state shape [batch_size, 7168], got {hidden_state.shape}"
        # assert expert_mask.shape[1] == 256, f"Expected expert_mask shape [batch_size, 256], got {expert_mask.shape}"
        # assert hidden_state.shape[0] == expert_mask.shape[0], f"Batch size mismatch: {hidden_state.shape[0]} vs {expert_mask.shape[0]}"

        # x1 = self.hidden_proj(hidden_state)
        # x2 = self.expert_proj(expert_mask)
        # x = torch.cat([x1, x2], dim=-1)
        # logits = self.output_layer(x)  # [batch_size, 256]
        logits = self.hidden_proj(hidden_state)  # [batch_size, 256]
        return logits  # Use BCEWithLogitsLoss

    def predict(self, hidden_state, expert_mask = None, top_k=8):
        logits = self.forward(hidden_state, expert_mask)
        probabilities = torch.sigmoid(logits)
        topk_preds = torch.topk(logits, k=top_k, dim=-1).indices
        topk_probs = torch.topk(probabilities, k=top_k, dim=-1).values
        return topk_preds, topk_probs # Return top-k predictions and their probabilities