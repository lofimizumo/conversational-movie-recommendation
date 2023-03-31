
import torch
import torch.nn as nn


class MyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12),
            num_layers=4
        )
        self.pooler = nn.Linear(512, 1)

    def forward(self, x):
        # Apply transformer
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        # Apply pooling to obtain output tensor of shape (batch_size, hidden_size)
        x = self.pooler(x).squeeze(2)
        return x


# Create an instance of MyTransformer
transformer = MyTransformer()

# Test the transformer with a sample input
x = torch.randn(2, 512, 768)
y = transformer(x)
print(y.shape)
