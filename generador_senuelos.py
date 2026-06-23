import torch
import torch.nn as nn
class DummySNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
model = DummySNN()
torch.save(model.state_dict(), 'data/modelo_ouroboros_real.pt')
spikes = {
    0: (torch.randn(1024, 64, 2) > 0.9).float(),
    1: (torch.randn(1024, 64, 2) > 0.9).float(),
    2: (torch.randn(1024, 64, 2) > 0.9).float(),
    3: (torch.randn(1024, 64, 2) > 0.9).float()
}
torch.save(spikes, 'data/spikes_inferencia.pt')
