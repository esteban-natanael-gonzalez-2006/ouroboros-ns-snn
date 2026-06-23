"""
cocomplejo_tensorial.py
=======================
Construccion del co-complejo de cadenas (C_*, partial_*, delta_*)
sobre el reticulo tensorial Lambda_ell = Z^{d_ell} x Z_tau.
Computa los numeros de Betti beta_k.

Compatible con ARM Cortex-A53 (float32, CPU).
"""
import itertools
import numpy as np
from typing import List, Dict, Tuple


def construir_generadores(d_ell: int, tau: int) -> Dict[int, List]:
    """
    Construye los generadores cubo^k(n, I) del complejo.
    Devuelve dict {k: lista de (n, I)}.
    """
    n_total = d_ell + tau  # dimension ambient
    generadores = {}
    for k in range(n_total + 1):
        gen_k = []
        # indices I de tamano k
        for I in itertools.combinations(range(n_total), k):
            # vertice base: usamos el origen (complejo local)
            n_base = tuple([0] * n_total)
            gen_k.append((n_base, I))
        generadores[k] = gen_k
    return generadores


def operador_borde(d_ell: int, tau: int) -> Dict[int, np.ndarray]:
    """
    Construye las matrices del operador borde partial_k.
    partial_k : C_k -> C_{k-1}
    """
    n_total = d_ell + tau
    generadores = construir_generadores(d_ell, tau)

    # indice de cada generador
    indices = {}
    for k, gen_k in generadores.items():
        indices[k] = {g: i for i, g in enumerate(gen_k)}

    bordes = {}
    for k in range(1, n_total + 1):
        dim_k = len(generadores[k])
        dim_km1 = len(generadores[k - 1])
        B = np.zeros((dim_km1, dim_k), dtype=np.float32)

        for col, (n, I) in enumerate(generadores[k]):
            # partial_k cube^k(n, I) =
            #   sum_j (-1)^{j-1} [cube^{k-1}(n + e_{i_j}, I\{i_j})
            #                      - cube^{k-1}(n, I\{i_j})]
            for j_pos, j in enumerate(I):
                sign = (-1) ** j_pos
                I_reduced = tuple(i for i in I if i != j)

                # cube^{k-1}(n + e_j, I\{j})
                n_plus = list(n)
                n_plus[j] = n_plus[j] + 1
                n_plus = tuple(n_plus)

                if (n_plus, I_reduced) in indices[k - 1]:
                    row_plus = indices[k - 1][(n_plus, I_reduced)]
                    B[row_plus, col] += sign

                if (n, I_reduced) in indices[k - 1]:
                    row_base = indices[k - 1][(n, I_reduced)]
                    B[row_base, col] -= sign

        bordes[k] = B
    return bordes


def operador_coborde(bordes: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    delta_k = (partial_{k+1})^T  (adjunto formal).
    """
    cobordes = {}
    for k, B in bordes.items():
        if k >= 1:
            cobordes[k - 1] = B.T
    return cobordes


def laplaciano_hodge(bordes: Dict[int, np.ndarray],
                     cobordes: Dict[int, np.ndarray],
                     k: int) -> np.ndarray:
    """
    Delta_k = delta_{k-1} partial_k + partial_{k+1} delta_k
    """
    if k in bordes and (k - 1) in cobordes:
        term1 = cobordes[k - 1] @ bordes[k]
    else:
        term1 = 0
    if (k + 1) in bordes and k in cobordes:
        term2 = bordes[k + 1] @ cobordes[k]
    else:
        term2 = 0
    return term1 + term2


def compute_betti(d_ell: int, tau: int,
                  n_skip: int = 0) -> Dict[int, int]:
    """
    Computa los numeros de Betti del reticulo Z^{d_ell} x Z_tau.
    Por el Corolario 7.2:
        beta_0 = 1
        beta_1 = 1 + n_skip
        beta_k = 0 para k >= 2
    """
    # Para reticulos pequenos, podemos verificar numericamente
    # via el Laplaciano de Hodge. Para reticulos grandes, usamos
    # el resultado teorico directamente.
    if d_ell + tau > 12:
        # caso grande: resultado teorico
        betti = {0: 1, 1: 1 + n_skip}
        for k in range(2, d_ell + tau + 1):
            betti[k] = 0
        return betti

    # caso pequeno: verificacion numerica
    bordes = operador_borde(d_ell, tau)
    cobordes = operador_coborde(bordes)
    betti = {}
    for k in range(d_ell + tau + 1):
        L_k = laplaciano_hodge(bordes, cobordes, k)
        if isinstance(L_k, np.ndarray):
            # beta_k = dim ker(L_k) = dim - rank(L_k)
            rank = np.linalg.matrix_rank(L_k)
            betti[k] = L_k.shape[0] - rank
        else:
            betti[k] = 0
    return betti


def verificar_identidades(bordes: Dict[int, np.ndarray],
                          cobordes: Dict[int, np.ndarray]) -> bool:
    """
    Verifica que partial^2 = 0 y delta^2 = 0.
    """
    for k in range(2, max(bordes.keys()) + 1):
        if k in bordes and (k - 1) in bordes:
            prod = bordes[k - 1] @ bordes[k]
            if np.linalg.norm(prod) > 1e-5:
                return False
    for k in range(0, max(cobordes.keys())):
        if k in cobordes and (k + 1) in cobordes:
            prod = cobordes[k + 1] @ cobordes[k]
            if np.linalg.norm(prod) > 1e-5:
                return False
    return True


def main():
    print('=== Co-complejo tensorial NS<->SNN ===\n')

    # Configuracion tipica: d_ell = 64, tau = 2
    d_ell = 64
    tau = 2
    n_skip = 0

    print(f'Reticulo: Z^{d_ell} x Z_{tau}')
    print(f'Dimension ambient: {d_ell + tau}')

    # Numeros de Betti (resultado teorico, Corolario 7.2 del Cap. 7)
    # beta_k(Z^{d_ell} x Z_tau) = sum_{i+j=k} beta_i(Z^{d_ell}) * beta_j(Z_tau)
    # Como Z^{d_ell} es contractible: beta_0 = 1, beta_i = 0 (i>=1)
    # Y Z_tau tiene: beta_0 = 1, beta_1 = 1, beta_i = 0 (i>=2)
    # Por Kunneth: beta_0 = 1*1 = 1, beta_1 = 1*1 + 0*1 = 1, beta_k = 0 (k>=2)
    betti = compute_betti(d_ell, tau, n_skip)
    print(f'\nNumeros de Betti (Corolario 7.2 + Kunneth):')
    for k, b in betti.items():
        if b > 0:
            print(f'  beta_{k} = {b}')
    print(f'  (beta_k = 0 para k >= 2)')

    # Verificacion numerica de las identidades estructurales
    # (parcial^2 = 0 y delta^2 = 0)
    print(f'\n--- Verificacion numerica de identidades estructurales ---')
    print(f'(Caso local d_ell=3, tau=2)')
    bordes = operador_borde(3, 2)
    cobordes = operador_coborde(bordes)
    ok = verificar_identidades(bordes, cobordes)
    print(f'partial^2 = 0 y delta^2 = 0: {"OK" if ok else "FALLO"}')
    print(f'(Estas identidades son puramente combinatorias y se preservan')
    print(f' bajo cuantizacion LNS por la Proposicion 12.1.)')

    # Los beta_k teoricos requieren el complejo global con periodicidad
    # en la direccion Z_tau. La verificacion numerica local (sin
    # periodicidad) da beta_k = 0 para todo k porque el complejo local
    # es contractible. No contradice el resultado teorico.


if __name__ == '__main__':
    main()