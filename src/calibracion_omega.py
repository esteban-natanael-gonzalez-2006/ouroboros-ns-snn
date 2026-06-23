import argparse
import json
import math

def constante_sobolev_discreta(d_min, C_P_continuo=0.7854):
    return C_P_continuo * math.sqrt(1.0 - 1.0 / d_min)

def constante_estiramiento(nu_s, eps_max):
    return 1.0 + eps_max / nu_s

def calibrar_omega(nu_s, eps_max, d_min, E_A_0):
    C_P_disc = constante_sobolev_discreta(d_min)
    C = constante_estiramiento(nu_s, eps_max)
    factor = (C * C_P_disc / nu_s) ** (2.0 / 3.0)
    return factor * math.sqrt(E_A_0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--omega_emp', type=float, default=6.0028)
    args = parser.parse_args()

    with open(args.params, 'r') as f:
        p = json.load(f)

    omega_cal = calibrar_omega(p['nu_s'], p['eps_max'], p['d_min'], p['E_A_0'])
    delta = abs(omega_cal - args.omega_emp) / args.omega_emp

    print("-" * 60)
    print("TABLA DE VALORES EMPIRICOS (Daemon-Core / Ouroboros)")
    print("-" * 60)
    print(f"{'tau_m_ms':<20} = {p['tau_m_ms']:.4f}")
    print(f"{'dt_ms':<20} = {p['dt_ms']:.4f}")
    print(f"{'nu_s':<20} = {p['nu_s']:.4f}")
    print(f"{'eps_max':<20} = {p['eps_max']:.4f}")
    print(f"{'d_min':<20} = {p['d_min']}")
    print(f"{'tau':<20} = {p['tau']}")
    print(f"{'L':<20} = {p['L']}")
    print(f"{'E_A_0':<20} = {p['E_A_0']:.4f}")
    print(f"{'Omega_cal':<20} = {omega_cal:.4f}")
    print(f"{'Omega_emp':<20} = {args.omega_emp:.4f}")
    print(f"{'delta_Omega':<20} = {delta:.4f} ({(delta*100):.2f}%)")
    print("-" * 60)
    
    if delta < 0.01:
        print("Veredicto: VALIDADO (Criterio A)")
    elif delta < 0.05:
        print("Veredicto: VALIDADO CON ADVERTENCIA (Criterio B)")
    else:
        print("Veredicto: FALSO (Criterio C)")

if __name__ == '__main__':
    main()
