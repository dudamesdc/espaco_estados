import numpy as np
import sympy as sp
from scipy.linalg import expm
import control as ctrl


def parse_matrix(text: str):
    rows = [list(map(float, r.split())) for r in text.strip().split(";")]
    arr = np.array(rows)
    mat = sp.Matrix(rows).applyfunc(lambda x: sp.nsimplify(x, rational=False))
    return arr, mat

def parse_poles(text: str):
    poles = []
    for p in text.split(","):
        p = p.strip().replace("i", "j")
        poles.append(complex(p))
    result = []
    for p in poles:
        if p.imag == 0:
            result.append(p.real)
        else:
            result.append(p)
    return result

def _nsimplify_mat(M):
    return sp.Matrix(M).applyfunc(lambda x: sp.nsimplify(x, rational=False))

def sym_var(continuous=True):
    return sp.Symbol('s') if continuous else sp.Symbol('z')

def characteristic_poly(A_sym, continuous=True):
    n = A_sym.shape[0]
    var = sym_var(continuous)
    sI = var * sp.eye(n)
    sI_A = sI - A_sym
    poly = sI_A.det().expand()
    return {"var": var, "sI": sI, "sI_A": sI_A, "poly": poly}

def eigenvalues_np(A_np):
    return np.linalg.eigvals(A_np)

def controllability_steps(A_np, B_np, A_sym, B_sym):
    n = A_np.shape[0]
    steps = []
    cols_sym = []
    for i in range(n):
        col_sym = A_sym**i * B_sym
        steps.append((i, col_sym))
        cols_sym.append(col_sym)
    U_sym = cols_sym[0]
    for c in cols_sym[1:]:
        U_sym = U_sym.row_join(c)
    cols_np = [np.linalg.matrix_power(A_np, i) @ B_np for i in range(n)]
    U_np = np.hstack(cols_np)
    rank = np.linalg.matrix_rank(U_np)
    return {"steps": steps, "U_sym": U_sym, "U_np": U_np, "rank": rank, "n": n}

def observability_steps(A_np, C_np, A_sym, C_sym):
    n = A_np.shape[0]
    steps = []
    rows_sym = []
    for i in range(n):
        row_sym = C_sym * A_sym**i
        steps.append((i, row_sym))
        rows_sym.append(row_sym)
    V_sym = rows_sym[0]
    for r in rows_sym[1:]:
        V_sym = V_sym.col_join(r)
    rows_np = [C_np @ np.linalg.matrix_power(A_np, i) for i in range(n)]
    V_np = np.vstack(rows_np)
    rank = np.linalg.matrix_rank(V_np)
    return {"steps": steps, "V_sym": V_sym, "V_np": V_np, "rank": rank, "n": n}

def ackermann_gain_steps(A_np, B_np, A_sym, B_sym, poles, is_observer=False):
    """Calcula Ackermann aplicando o sinal negativo de controle se is_observer=False"""
    n = A_np.shape[0]
    ctrl_data = controllability_steps(A_np, B_np, A_sym, B_sym)
    U_np = ctrl_data["U_np"]
    U_sym = ctrl_data["U_sym"]
    rank = ctrl_data["rank"]
    if rank < n:
        raise ValueError("Sistema não é controlável – Ackermann não aplicável.")
    
    var = sp.Symbol('s')
    poles_sym = [sp.nsimplify(p, rational=False) for p in poles]
    factors_expr = sp.Mul(*[(var - p) for p in poles_sym], evaluate=False)
    expanded_expr = sp.expand(factors_expr)
    phi_poly = sp.Poly(expanded_expr, var)
    coeffs_sym = phi_poly.all_coeffs()
    degree = len(coeffs_sym) - 1
    
    A_powers = {}
    for exp in range(degree + 1):
        if exp == 0:
            A_powers[0] = sp.eye(n)
        else:
            A_powers[exp] = (A_sym ** sp.Integer(exp))
            
    phi_terms = []
    phi_A_sym = sp.zeros(n)
    for i, c in enumerate(coeffs_sym):
        exp = degree - i
        term = c * A_powers[exp]
        phi_terms.append((c, exp, term))
        phi_A_sym += term
        
    coeffs_float = [complex(c) for c in coeffs_sym]
    phi_A_np = np.zeros_like(A_np, dtype=complex)
    for i, c in enumerate(coeffs_float):
        phi_A_np += c * np.linalg.matrix_power(A_np, degree - i)
    phi_A_np = np.real_if_close(phi_A_np)
    
    U_inv_np = np.linalg.inv(U_np)
    U_inv_sym = _nsimplify_mat(U_inv_np)
    e_n = np.zeros((1, n))
    e_n[0, -1] = 1.0
    e_n_sym = sp.Matrix(e_n).applyfunc(lambda x: sp.nsimplify(x))
    
    # CÁLCULO FINAL DE K: APLICAÇÃO DO SINAL DA FOLHA DE FÓRMULAS
    K_np = np.real_if_close(e_n @ U_inv_np @ phi_A_np)
    if not is_observer:
        K_np = -K_np  # u = Kx exige o K invertido
        
    K_sym = _nsimplify_mat(K_np)
    
    return {
        "ctrl": ctrl_data,
        "poles_sym": poles_sym,
        "factors_expr": factors_expr,
        "expanded_expr": expanded_expr,
        "phi_poly": phi_poly,
        "coeffs_sym": coeffs_sym,
        "phi_terms": phi_terms,
        "phi_A_sym": phi_A_sym,
        "U_inv_sym": U_inv_sym,
        "e_n_sym": e_n_sym,
        "K_np": K_np,
        "K_sym": K_sym,
        "A_powers": A_powers,
        "n": n,
    }

def observer_gain_steps(A_np, C_np, A_sym, C_sym, poles):
    """Calcula L via Ackermann dual (transposta). Passa is_observer=True."""
    data = ackermann_gain_steps(A_np.T, C_np.T, A_sym.T, C_sym.T, poles, is_observer=True)
    L_np = data["K_np"].T
    L_sym = _nsimplify_mat(L_np)
    data["L_np"] = L_np
    data["L_sym"] = L_sym
    return data

def reference_tracker(A_np, B_np, C_np, A_sym, B_sym, C_sym, poles_aug):
    n = A_np.shape[0]
    p = C_np.shape[0]
    A_a_np = np.block([[A_np, np.zeros((n, p))],
                       [-C_np, np.zeros((p, p))]])
    B_a_np = np.vstack([B_np, np.zeros((p, B_np.shape[1]))])
    A_a_sym = _nsimplify_mat(A_a_np)
    B_a_sym = _nsimplify_mat(B_a_np)
    ack_data = ackermann_gain_steps(A_a_np, B_a_np, A_a_sym, B_a_sym, poles_aug, is_observer=False)
    K_a = ack_data["K_np"]
    K_x = K_a[:, :n]
    K_i = K_a[:, n:]
    return {
        "A_a_sym": A_a_sym, "B_a_sym": B_a_sym,
        "K_a_sym": ack_data["K_sym"],
        "K_x_sym": _nsimplify_mat(K_x),
        "K_i_sym": _nsimplify_mat(K_i),
        "ack": ack_data,
        "n": n, "p": p,
    }

def tf_to_ccf(num_coeffs, den_coeffs):
    n = len(den_coeffs) - 1
    a0 = den_coeffs[0]
    den_n = [c / a0 for c in den_coeffs]
    num_n = [c / a0 for c in num_coeffs]
    while len(num_n) < n + 1:
        num_n.insert(0, 0.0)
    a = den_n[1:]
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    for i in range(n):
        A[n - 1, i] = -a[n - 1 - i]
    B = np.zeros((n, 1))
    B[n - 1, 0] = 1.0
    C = np.zeros((1, n))
    for i in range(n):
        C[0, i] = num_n[n - i] - num_n[0] * a[n - 1 - i] if num_n[0] != 0 else num_n[n - i]
    D = np.array([[num_n[0]]])
    return A, B, C, D

def tf_to_ocf(num_coeffs, den_coeffs):
    A_c, B_c, C_c, D_c = tf_to_ccf(num_coeffs, den_coeffs)
    return A_c.T, C_c.T, B_c.T, D_c

def discretize(A_np, B_np, T, A_sym, B_sym):
    G_np = expm(A_np * T)
    n = A_np.shape[0]
    if np.linalg.matrix_rank(A_np) == n:
        H_np = np.linalg.solve(A_np, (G_np - np.eye(n))) @ B_np
    else:
        from scipy.integrate import quad_vec
        def integrand(t):
            return (expm(A_np * t) @ B_np).flatten()
        result, _ = quad_vec(integrand, 0, T)
        H_np = result.reshape(B_np.shape)
    G_sym = sp.Matrix(G_np).applyfunc(lambda x: sp.nsimplify(x, tolerance=1e-8))
    H_sym = sp.Matrix(H_np).applyfunc(lambda x: sp.nsimplify(x, tolerance=1e-8))
    return G_np, H_np, G_sym, H_sym
