import streamlit as st
import numpy as np
import sympy as sp
import control as ctrl
import streamlit as st
import numpy as np
import sympy as sp
import control as ctrl
from engine import (
    parse_matrix, parse_poles, characteristic_poly, eigenvalues_np,
    controllability_steps, observability_steps,
    ackermann_gain_steps, observer_gain_steps,
    reference_tracker, tf_to_ccf, tf_to_ocf, discretize, _nsimplify_mat
)
st.set_page_config(
    page_title="Controle – Espaço de Estados",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {background: linear-gradient(180deg,#0f0c29,#302b63,#24243e);}
    [data-testid="stSidebar"] * {color:#e0e0e0 !important;}
    .block-container {padding-top:1.5rem;}
    h1,h2,h3 {color:#7c4dff;}
    .stAlert {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📐 Controle")
page = st.sidebar.radio("Selecione o módulo:", [
    "1 · Análise (Espaço de Estados)",
    "2 · Alocação de Pólos / Observador",
    "3 · Seguidor de Referência",
    "4 · Formas Canônicas",
    "5 · Discretização",
])

# ==========================================
# FUNÇÕES DE RENDERIZAÇÃO (UI)
# ==========================================
def show_matrix(M_sym, label):
    """Exibe uma matriz nomeada em LaTeX."""
    st.latex(f"{label} = " + sp.latex(M_sym))

def render_controllability(A_sym, B_sym, ctrl_data, label_U=r"\mathcal{U}",
                           col_label="B", row_prefix="A"):
    """Renderiza passo a passo a montagem da Matriz de Controlabilidade."""
    n = ctrl_data["n"]
    st.markdown(f"**Definição:** ${label_U} = [{col_label} \\;\\; {row_prefix}{col_label}"
                f" \\;\\; {row_prefix}^2{col_label} \\;\\; \\cdots \\;\\;"
                f" {row_prefix}^{{n-1}}{col_label}]$")
    st.markdown("---")
    for i, col_sym in ctrl_data["steps"]:
        if i == 0:
            st.latex(f"{row_prefix}^{{{i}}}{col_label} = I \\cdot {col_label} = {col_label} = "
                     + sp.latex(col_sym))
        else:
            st.latex(f"{row_prefix}^{{{i}}}{col_label} = " + sp.latex(A_sym**i)
                     + r" \cdot " + sp.latex(B_sym) + " = " + sp.latex(col_sym))
    st.markdown("---")
    st.markdown(f"**Concatenando as colunas:**")
    show_matrix(ctrl_data["U_sym" if "U_sym" in ctrl_data else "V_sym"], label_U)
    st.latex(rf"\text{{posto}}({label_U}) = {ctrl_data['rank']}, \quad n = {n}")
    return ctrl_data["rank"] == n

def render_observability(A_sym, C_sym, obs_data, label_V=r"\mathcal{V}"):
    """Renderiza passo a passo a montagem da Matriz de Observabilidade."""
    n = obs_data["n"]
    st.markdown(r"**Definição:** $\mathcal{V} = \begin{bmatrix} C \\ CA \\ CA^2 \\"
                r" \vdots \\ CA^{n-1} \end{bmatrix}$")
    st.markdown("---")
    for i, row_sym in obs_data["steps"]:
        if i == 0:
            st.latex(r"CA^{0} = C \cdot I = C = " + sp.latex(row_sym))
        else:
            st.latex(f"CA^{{{i}}} = " + sp.latex(C_sym) + r" \cdot "
                     + sp.latex(A_sym**i) + " = " + sp.latex(row_sym))
    st.markdown("---")
    st.markdown("**Empilhando as linhas:**")
    show_matrix(obs_data["V_sym"], label_V)
    st.latex(rf"\text{{posto}}({label_V}) = {obs_data['rank']}, \quad n = {n}")
    return obs_data["rank"] == n

def render_ackermann(data, var_name="s", mat_name="A", col_name="B",
                     u_label=r"\mathcal{U}", gain_label="K", is_observer=False):
    """Renderiza TODOS os passos de Ackermann com o ajuste de sinal para K."""
    n = data["n"]
    st.markdown(f"**Passo 1 – Verificar controlabilidade do par $({mat_name}, {col_name})$**")
    show_matrix(data["ctrl"]["U_sym" if "U_sym" in data["ctrl"] else "V_sym"], u_label)
    st.latex(rf"\text{{posto}}({u_label}) = {data['ctrl']['rank']} = n = {n} \;\checkmark")
    st.markdown("---")
    
    st.markdown(f"**Passo 2 – Polinômio desejado $\\varphi({var_name})$**")
    poles = data["poles_sym"]
    root_parts = []
    for p in poles:
        root_parts.append(f"({var_name} - ({sp.latex(p)}))")
    roots_str = " \\cdot ".join(root_parts)
    st.latex(rf"\varphi({var_name}) = {roots_str}")
    st.latex(rf"\varphi({var_name}) = {sp.latex(data['expanded_expr'])}")
    st.markdown("---")
    
    st.markdown(f"**Passo 3 – Substituição de ${mat_name}$ no polinômio (Cayley-Hamilton)**")
    terms_str_parts = []
    for coef, exp, _ in data["phi_terms"]:
        coef_latex = sp.latex(coef)
        if exp == 0:
            terms_str_parts.append(f"({coef_latex}) \\cdot I")
        elif exp == 1:
            terms_str_parts.append(f"({coef_latex}) \\cdot {mat_name}")
        else:
            terms_str_parts.append(f"({coef_latex}) \\cdot {{{mat_name}}}^{{{exp}}}")
    formula_str = " + ".join(terms_str_parts)
    st.latex(rf"\varphi({mat_name}) = {formula_str}")
    
    A_powers = data["A_powers"]
    for coef, exp, term_mat in data["phi_terms"]:
        coef_latex = sp.latex(coef)
        if exp == 0:
            st.latex(f"({coef_latex}) \\cdot I = " + sp.latex(term_mat))
        else:
            power_label = f"{mat_name}" if exp == 1 else f"{{{mat_name}}}^{{{exp}}}"
            st.latex(f"{power_label} = " + sp.latex(A_powers[exp]))
            st.latex(f"({coef_latex}) \\cdot {power_label} = " + sp.latex(term_mat))
            
    st.markdown(f"**Resultado de $\\varphi({mat_name})$:**")
    show_matrix(data["phi_A_sym"], rf"\varphi({mat_name})")
    st.markdown("---")
    
    st.markdown(rf"**Passo 4 – Inversa da matriz de controlabilidade ${{{u_label}}}^{{-1}}$**")
    show_matrix(data["U_inv_sym"], rf"{{{u_label}}}^{{-1}}")
    st.markdown("---")
    
    st.markdown(rf"**Passo 5 – Cálculo de ${gain_label}$**")
    if is_observer:
        st.latex(rf"{gain_label} = e_n^T \cdot {{{u_label}}}^{{-1}} \cdot \varphi({mat_name})")
    else:
        st.latex(rf"{gain_label} = - e_n^T \cdot {{{u_label}}}^{{-1}} \cdot \varphi({mat_name})")
        
    st.markdown("Onde:")
    show_matrix(data["e_n_sym"], "e_n^T")
    st.markdown("**Resultado final:**")
    show_matrix(data["K_sym"], gain_label)


# ==========================================
# TELAS / INTERFACE (UI LOGIC)
# ==========================================
if page.startswith("1"):
    st.header("Análise de Sistemas no Espaço de Estados")
    st.info("Insira as matrizes separando colunas por espaço e linhas por `;`  \n"
            "Exemplo: `0 1 0; 0 0 1; -6 -11 -6`")
    col1, col2 = st.columns(2)
    with col1:
        A_txt = st.text_input("Matriz A", "0 1 0; 0 0 1; -6 -11 -6", key="a1")
        C_txt = st.text_input("Matriz C", "1 0 0", key="c1")
    with col2:
        B_txt = st.text_input("Matriz B", "0; 0; 1", key="b1")
        D_txt = st.text_input("Matriz D", "0", key="d1")
    cont_disc = st.radio("Tipo de sistema",
                         ["Contínuo (s)", "Discreto (z)"], horizontal=True, key="cd1")
    continuous = cont_disc.startswith("Cont")
    if st.button("▶  Calcular", key="btn1"):
        try:
            A_np, A_sym = parse_matrix(A_txt)
            B_np, B_sym = parse_matrix(B_txt)
            C_np, C_sym = parse_matrix(C_txt)
            D_np, D_sym = parse_matrix(D_txt)
            n = A_np.shape[0]
            v = 's' if continuous else 'z'
            st.subheader("Matrizes do Sistema")
            c1, c2, c3, c4 = st.columns(4)
            with c1: show_matrix(A_sym, "A")
            with c2: show_matrix(B_sym, "B")
            with c3: show_matrix(C_sym, "C")
            with c4: show_matrix(D_sym, "D")
            st.subheader("Polinômio Característico")
            cp = characteristic_poly(A_sym, continuous)
            st.markdown(f"**Passo 1:** Construir a matriz ${v}I$")
            show_matrix(cp["sI"], f"{v}I")
            st.markdown(f"**Passo 2:** Calcular ${v}I - A$")
            show_matrix(cp["sI_A"], f"{v}I - A")
            st.markdown(f"**Passo 3:** Calcular o determinante $\\det({v}I - A)$")
            st.latex(rf"\det({v}I - A) = \det\left(" + sp.latex(cp["sI_A"])
                     + r"\right)")
            st.latex(rf"\boxed{{\det({v}I - A) = {sp.latex(cp['poly'])}}}")
            st.subheader("Autovalores (Pólos)")
            eigs = eigenvalues_np(A_np)
            st.markdown("Raízes do polinômio característico:")
            for i, e in enumerate(eigs):
                e_sym = sp.nsimplify(complex(round(e.real, 6), round(e.imag, 6)),
                                     rational=False)
                st.latex(rf"\lambda_{{{i+1}}} = {sp.latex(e_sym)}")
            if continuous:
                stable = all(e.real < 0 for e in eigs)
                crit = r"\text{Re}(\lambda_i) < 0"
            else:
                stable = all(abs(e) < 1 for e in eigs)
                crit = r"|\lambda_i| < 1"
            if stable:
                st.success(f" Todos os pólos satisfazem ${crit}$ → sistema **estável**.")
            else:
                st.error(f" Nem todos os pólos satisfazem ${crit}$ → sistema **instável**.")
            st.subheader("Controlabilidade")
            ctrl_data = controllability_steps(A_np, B_np, A_sym, B_sym)
            is_ctrl = render_controllability(A_sym, B_sym, ctrl_data)
            if is_ctrl:
                st.success(" Sistema **controlável** (posto = n).")
            else:
                st.warning(" Sistema **não controlável** (posto < n).")
            st.subheader("Observabilidade")
            obs_data = observability_steps(A_np, C_np, A_sym, C_sym)
            is_obs = render_observability(A_sym, C_sym, obs_data)
            if is_obs:
                st.success(" Sistema **observável** (posto = n).")
            else:
                st.warning(" Sistema **não observável** (posto < n).")
        except Exception as ex:
            st.error(f"Erro: {ex}")

elif page.startswith("2"):
    st.header("Alocação de Pólos (Ackermann) e Observador")
    st.info("Insira matrizes e os pólos desejados separados por vírgula.\n"
            "Pólos complexos aceitos: `-0.5+0.5j` ou `-0.5+0.5i`")
    col1, col2 = st.columns(2)
    with col1:
        A_txt = st.text_input("Matriz A", "0 1 0; 0 0 1; -6 -11 -6", key="a2")
        B_txt = st.text_input("Matriz B", "0; 0; 1", key="b2")
    with col2:
        C_txt = st.text_input("Matriz C", "1 0 0", key="c2")
        poles_txt = st.text_input("Pólos desejados (K)", "-5, -6, -7", key="pk")
    
    tab_k, tab_l = st.tabs(["Ganho K (realimentação)", "Ganho L (observador)"])
    
    with tab_k:
        if st.button("▶  Calcular K", key="btn2k"):
            try:
                A_np, A_sym = parse_matrix(A_txt)
                B_np, B_sym = parse_matrix(B_txt)
                poles = parse_poles(poles_txt)
                data = ackermann_gain_steps(A_np, B_np, A_sym, B_sym, poles, is_observer=False)
                st.subheader("Resolução Passo a Passo – Fórmula de Ackermann")
                render_ackermann(data, var_name="s", mat_name="A",
                                 col_name="B", u_label=r"\mathcal{U}",
                                 gain_label="K", is_observer=False)
            except Exception as ex:
                st.error(f"Erro: {ex}")
                
    with tab_l:
        poles_obs_txt = st.text_input("Pólos do observador", "-10, -11, -12", key="pl")
        if st.button("▶  Calcular L", key="btn2l"):
            try:
                A_np, A_sym = parse_matrix(A_txt)
                C_np, C_sym = parse_matrix(C_txt)
                poles_o = parse_poles(poles_obs_txt)
                data = observer_gain_steps(A_np, C_np, A_sym, C_sym, poles_o)
                st.subheader("Resolução Passo a Passo – Observador (Ackermann Dual)")
                st.markdown(r"Aplica-se a fórmula de Ackermann ao par dual $(A^T, C^T)$ "
                            r"e obtém-se $L = K^T$.")
                st.markdown("---")
                render_ackermann(data, var_name="s", mat_name="A^T",
                                 col_name="C^T", u_label=r"\mathcal{V}^T",
                                 gain_label="K_{dual}", is_observer=True)
                st.markdown("---")
                st.markdown("**Ganho do Observador:**")
                st.latex(r"L = K_{dual}^T")
                show_matrix(data["L_sym"], "L")
            except Exception as ex:
                st.error(f"Erro: {ex}")

elif page.startswith("3"):
    st.header("Projeto de Seguidor de Referência (Entrada Degrau)")
    st.info("Adiciona um integrador ao sistema para rastreamento de referência "
            "com erro nulo em regime permanente.")
    col1, col2 = st.columns(2)
    with col1:
        A_txt = st.text_input("Matriz A", "0 1; -2 -3", key="a3")
        B_txt = st.text_input("Matriz B", "0; 1", key="b3")
    with col2:
        C_txt = st.text_input("Matriz C", "1 0", key="c3")
        poles_txt = st.text_input("Pólos desejados (sistema aumentado)", "-5, -6, -7", key="pa")
    
    if st.button("▶  Calcular Seguidor", key="btn3"):
        try:
            A_np, A_sym = parse_matrix(A_txt)
            B_np, B_sym = parse_matrix(B_txt)
            C_np, C_sym = parse_matrix(C_txt)
            poles = parse_poles(poles_txt)
            result = reference_tracker(A_np, B_np, C_np, A_sym, B_sym, C_sym, poles)
            st.subheader("1) Sistema Aumentado")
            st.markdown("Adicionamos a dinâmica do erro no topo do vetor de estados:")
            st.latex(r"A_a = \begin{bmatrix} 0 & C \\ 0 & A \end{bmatrix}, \quad B_a = \begin{bmatrix} 0 \\ B \end{bmatrix}")
            c1, c2 = st.columns(2)
            with c1: show_matrix(result["A_a_sym"], "A_a")
            with c2: show_matrix(result["B_a_sym"], "B_a")
            
            st.subheader("2) Ackermann no Sistema Aumentado")
            render_ackermann(result["ack"], var_name="s", mat_name="A_a",
                             col_name="B_a", u_label=r"\mathcal{U}_a",
                             gain_label="K_a", is_observer=False)
                             
            st.subheader("3) Separação dos Ganhos")
            st.latex(r"K_a = \begin{bmatrix} K_i & K_x \end{bmatrix}") # Ordem invertida
            c1, c2 = st.columns(2)
            with c1: show_matrix(result["K_i_sym"], "K_i")
            with c2: show_matrix(result["K_x_sym"], "K_x")
            st.markdown("**Lei de controle:**")
            st.latex(r"u = K_x\,x + K_i\,\xi, \quad \dot{\xi} = r - Cx")
        except Exception as ex:
            st.error(f"Erro: {ex}")

elif page.startswith("4"):
    st.header("Formas Canônicas (a partir da Função de Transferência)")
    st.info("Insira os coeficientes em ordem decrescente de potência, separados "
            "por espaço.\nEx: `1 5 6` → $s^2 + 5s + 6$")
    num_txt = st.text_input("Coeficientes do Numerador", "1 3", key="num")
    den_txt = st.text_input("Coeficientes do Denominador", "1 5 6", key="den")
    
    if st.button("▶  Converter", key="btn4"):
        try:
            num = list(map(float, num_txt.split()))
            den = list(map(float, den_txt.split()))
            s = sp.Symbol('s')
            num_poly = sum(c * s**(len(num)-1-i) for i, c in enumerate(num))
            den_poly = sum(c * s**(len(den)-1-i) for i, c in enumerate(den))
            st.subheader("Função de Transferência")
            st.latex(rf"G(s) = \frac{{{sp.latex(num_poly)}}}{{{sp.latex(den_poly)}}}")
            
            st.subheader("Forma Canônica Controlável (CCF)")
            A_c, B_c, C_c, D_c = tf_to_ccf(num, den)
            A_c_s = _nsimplify_mat(A_c)
            B_c_s = _nsimplify_mat(B_c)
            C_c_s = _nsimplify_mat(C_c)
            D_c_s = _nsimplify_mat(D_c)
            st.markdown("Na CCF, a última linha de $A$ contém $-a_n, \\ldots, -a_1$ "
                        "e $B = [0\\;\\cdots\\;0\\;1]^T$.")
            cc1, cc2 = st.columns(2)
            with cc1:
                show_matrix(A_c_s, "A_{ccf}")
                show_matrix(B_c_s, "B_{ccf}")
            with cc2:
                show_matrix(C_c_s, "C_{ccf}")
                show_matrix(D_c_s, "D_{ccf}")
                
            st.subheader("Forma Canônica Observável (OCF)")
            A_o, B_o, C_o, D_o = tf_to_ocf(num, den)
            A_o_s = _nsimplify_mat(A_o)
            B_o_s = _nsimplify_mat(B_o)
            C_o_s = _nsimplify_mat(C_o)
            D_o_s = _nsimplify_mat(D_o)
            st.markdown("$A_{ocf} = A_{ccf}^T$, $B_{ocf} = C_{ccf}^T$, $C_{ocf} = B_{ccf}^T$.")
            co1, co2 = st.columns(2)
            with co1:
                show_matrix(A_o_s, "A_{ocf}")
                show_matrix(B_o_s, "B_{ocf}")
            with co2:
                show_matrix(C_o_s, "C_{ocf}")
                show_matrix(D_o_s, "D_{ocf}")
        except Exception as ex:
            st.error(f"Erro: {ex}")

elif page.startswith("5"):
    st.header("Discretização de Sistemas Contínuos")
    st.info("Converte $(A, B)$ contínuo em $(G, H)$ discreto para um dado período de amostragem $T$.")
    col1, col2 = st.columns(2)
    with col1:
        A_txt = st.text_input("Matriz A", "0 1; -2 -3", key="a5")
        B_txt = st.text_input("Matriz B", "0; 1", key="b5")
    with col2:
        T = st.number_input("Período de amostragem T (s)", value=0.1,
                            min_value=0.001, format="%.4f", key="T5")
                            
    if st.button("▶  Discretizar", key="btn5"):
        try:
            A_np, A_sym = parse_matrix(A_txt)
            B_np, B_sym = parse_matrix(B_txt)
            st.subheader("Fórmulas")
            st.latex(r"G = e^{A T}")
            st.latex(r"H = \left(\int_{0}^{T} e^{A\tau}\,d\tau\right) B")
            st.markdown("Quando $A$ é invertível:")
            st.latex(r"H = A^{-1}(G - I)\,B")
            
            G_np, H_np, G_sym, H_sym = discretize(A_np, B_np, T, A_sym, B_sym)
            st.subheader("Resultados")
            st.latex(rf"T = {T}")
            c1, c2 = st.columns(2)
            with c1:
                show_matrix(A_sym, "A")
                show_matrix(G_sym, "G = e^{AT}")
            with c2:
                show_matrix(B_sym, "B")
                show_matrix(H_sym, "H")
                
            sys_c = ctrl.ss(A_np, B_np, np.eye(A_np.shape[0]), np.zeros_like(B_np))
            sys_d = ctrl.c2d(sys_c, T)
            G_ctrl = np.array(sys_d.A)
            H_ctrl = np.array(sys_d.B)
            st.subheader("Validação (python-control `c2d`)")
            st.latex(r"G_{ctrl} = " + sp.latex(_nsimplify_mat(G_ctrl)))
            st.latex(r"H_{ctrl} = " + sp.latex(_nsimplify_mat(H_ctrl)))
        except Exception as ex:
            st.error(f"Erro: {ex}")

st.sidebar.markdown("---")
st.sidebar.caption("UFRN – Controle · 2026")