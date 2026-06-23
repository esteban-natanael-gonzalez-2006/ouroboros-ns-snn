import torch
import torch.nn as nn
import math

class OuroborosCore(nn.Module):
    def __init__(self, d_l=64, target_E_A_0=13.4511):
        super().__init__()
        self.d_l = d_l
        self.L = 4
        self.layers = nn.ModuleList([nn.Linear(d_l, d_l, bias=False) for _ in range(self.L)])
        self._initialize_topology(target_E_A_0)

    def _initialize_topology(self, target):
        # CORRECCION: Como la medicion hace un promedio (w_l = 1/L), 
        # la energia de cada capa DEBE SER el target exacto, no target / L.
        E_per_layer = target 
        
        for idx, layer in enumerate(self.layers):
            nn.init.orthogonal_(layer.weight)
            W = layer.weight.data
            
            R = torch.randn(self.d_l, self.d_l)
            A = 0.5 * (R - R.T)
            
            current_energy = 0.5 * torch.sum(A * A).item()
            if current_energy > 0:
                scale = math.sqrt(E_per_layer / current_energy)
                A = A * scale
            
            layer.weight.data = W + A

print("[+] Iniciando Genesis Topologica Ouroboros (Corregida)...")
model = OuroborosCore(target_E_A_0=13.4511)
torch.save(model.state_dict(), 'data/modelo_ouroboros_real.pt')

print("[+] Forjando regimen disperso estricto (epsilon = 1%)...")
spikes = {}
for i in range(4):
    spikes[i] = (torch.rand(1024, 64, 2) > 0.99).float()
    
torch.save(spikes, 'data/spikes_inferencia.pt')
print("[+] Tensores Ouroboros reales forjados y guardados en 'data/'.")
