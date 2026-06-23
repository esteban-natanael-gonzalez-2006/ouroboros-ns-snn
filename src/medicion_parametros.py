import argparse
import json
import math
import torch
import numpy as np

def medir_nu_s(tau_m_ms, dt_ms):
    return 1.0 - math.exp(-dt_ms / tau_m_ms)

def medir_eps_ell(spike_trains):
    total_spikes = (spike_trains > 0).sum().item()
    B, d_ell, tau = spike_trains.shape
    return total_spikes / (B * d_ell * tau)

def medir_E_A_0(weights, dims):
    E_A_0 = 0.0
    total_d = sum(dims)
    E_per_layer = []
    for ell, W in enumerate(weights):
        W_np = W.detach().cpu().numpy().astype(np.float32)
        A = 0.5 * (W_np.T @ W_np - W_np @ W_np.T)
        E_ell = 0.5 * float(np.sum(A * A))
        w_ell = dims[ell] / total_d
        E_A_0 += w_ell * E_ell
        E_per_layer.append(E_ell)
    return E_A_0, E_per_layer

def extraer_parametros(model, spike_trains_dict, tau_m_ms=10.0, dt_ms=0.5):
    weights = []
    dims = []
    
    if isinstance(model, dict):
        for name, param in model.items():
            if 'weight' in name and param.dim() >= 2:
                weights.append(param.clone())
                dims.append(param.shape[0])
    else:
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights.append(param.data)
                dims.append(param.shape[0])
                
    d_min = min(dims)
    L = len(weights)
    nu_s = medir_nu_s(tau_m_ms, dt_ms)
    
    eps_per_layer = {}
    for ell, st in spike_trains_dict.items():
        eps_per_layer[ell] = medir_eps_ell(st)
    eps_max = max(eps_per_layer.values()) if eps_per_layer else 0.01
    
    E_A_0, E_per_layer = medir_E_A_0(weights, dims)
    tau = list(spike_trains_dict.values())[0].shape[-1] if spike_trains_dict else 2
    
    return {
        "nu_s": nu_s, "eps_max": eps_max, "E_A_0": E_A_0,
        "L": L, "d_min": d_min, "tau": tau,
        "tau_m_ms": tau_m_ms, "dt_ms": dt_ms
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--spikes', type=str, required=True)
    parser.add_argument('--tau_m', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--output', type=str, default='parametros.json')
    args = parser.parse_args()

    model = torch.load(args.model, map_location='cpu', weights_only=False)
    spike_trains = torch.load(args.spikes, map_location='cpu', weights_only=False)
    
    params = extraer_parametros(model, spike_trains, args.tau_m, args.dt)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2)
    
    print("[+] Extraccion completada con exito.")

if __name__ == '__main__':
    main()
