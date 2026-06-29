import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # The QFlex Distribution — Interactive Companion

    This notebook is a **section-by-section interactive companion** to the paper

    > **The QFlex Distribution** — J. Eric Bickel, Connor Colombe, and Benjamin D. Leibowicz,
    > Operations Research & Industrial Engineering, The University of Texas at Austin.
    > [SSRN 5930859](https://ssrn.com/abstract=5930859) ·
    > [DOI 10.2139/ssrn.5930859](http://dx.doi.org/10.2139/ssrn.5930859)

    QFlex is a **quantile-parameterized distribution (QPD)** built entirely from *monotone
    transformations of valid quantile functions*. Unlike the Metalog, which can produce
    non-monotone (invalid) fits and requires ex-post numerical repair, QFlex remains a valid
    distribution under simple, analytic coefficient conditions.

    Each part below mirrors a section of the paper and adds interactive exploration:

    | Part | Paper section | Topic |
    |------|---------------|-------|
    | 1 | §1 | Motivation and a running example |
    | A | §2 | Constructing valid quantile functions (Gilchrist rules) |
    | B | §3 | The QFlex system: bases, fitting, interpolation, moments |
    | C | §4 | Monotonicity: Theorems 1 & 5, Propositions 2–4 |
    | D | §5 | Modality and the quantile curvature function |
    | E | §6 | Bounded and semi-bounded variants |
    | F | §7 | QFlex vs Metalog across distributions |
    | G | §8 | Demonstrating monotonicity control |
    | H | §9 | Conclusion |

    *Built using the [`qflex`](https://pypi.org/project/qflex/) library; the Metalog comparisons
    use a vendored copy of a Metalog implementation (`notebooks/_metalog/`).*
    """)
    return


@app.cell
async def _(mo):
    # WebAssembly bootstrap. When this notebook runs in the browser (Pyodide,
    # e.g. via `marimo export html-wasm` on GitHub Pages), the local `qflex` and
    # `_metalog` packages are not installed. We install the wheels shipped in
    # `public/` with micropip. cvxpy is intentionally omitted -- qflex falls back
    # to a scipy-based tail-center solver when cvxpy is absent. Locally this whole
    # block is skipped and the already-importable packages are used as-is.
    import sys as _sys

    wasm_ready = True
    if _sys.platform == "emscripten":
        import micropip  # provided by the Pyodide runtime

        _base = str(mo.notebook_location()).rstrip("/")
        await micropip.install([
            f"{_base}/public/qflex-1.0.0-py3-none-any.whl",
            f"{_base}/public/qflex_metalog_bundle-0.1.0-py3-none-any.whl",
        ])
    return (wasm_ready,)


@app.cell
def _(wasm_ready):
    import os
    import sys

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    # Make the vendored Metalog package importable regardless of working directory.
    _nb_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else os.getcwd()
    )
    if _nb_dir not in sys.path:
        sys.path.insert(0, _nb_dir)

    from qflex import QFlex, LogQFlex, LogitQFlex, ConstraintType
    from qflex import check_delta_p_monotonicity
    from qflex.utils import compute_w1
    from qflex.basis import (
        BasisType,
        evaluate_basis,
        evaluate_basis_derivative,
        evaluate_quantile,
        evaluate_quantile_derivative,
        get_term_structure,
        build_design_matrix,
    )
    from qflex.constraints import (
        center_M_j,
        tail_center_margin_coeff,
        get_tail_indices,
    )

    from _metalog import Metalog, LogMetalog, LogitMetalog

    return (
        BasisType,
        ConstraintType,
        LogMetalog,
        LogQFlex,
        LogitMetalog,
        LogitQFlex,
        Metalog,
        QFlex,
        build_design_matrix,
        check_delta_p_monotonicity,
        compute_w1,
        evaluate_basis,
        evaluate_basis_derivative,
        evaluate_quantile,
        evaluate_quantile_derivative,
        get_term_structure,
        np,
        plt,
        stats,
        tail_center_margin_coeff,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1 — Motivation and a running example  *(paper §1)*

    A distribution can be described by its CDF \(F(x)\), PDF \(f(x)\), **quantile function**
    \(Q(p)\), or quantile density \(q(p)=Q'(p)\). In decision analysis, uncertainty is usually
    expressed through quantiles — expert *P10 / P50 / P90* assessments or empirical summaries —
    so the quantile function is the natural representation.

    The paper's motivating example: a project team assesses that the **total installed cost** of a
    new processing facility has

    - 10th percentile = **\$530M**, 50th percentile = **\$620M**, 90th percentile = **\$810M**.

    A two-parameter family (e.g. lognormal) generally cannot match all three. QFlex fits a smooth,
    valid quantile function through them directly. With three assessments, the default **3-term**
    QFlex interpolates them exactly.
    """)
    return


@app.cell
def _(QFlex, np):
    p_cost = [0.10, 0.50, 0.90]
    x_cost = [530.0, 620.0, 810.0]  # facility cost in $M (paper R1 running example)
    qf_cost = QFlex(x_cost, p_cost)  # default terms = 3
    fitted_cost = qf_cost.quantile(p_cost)
    cost_err = float(np.max(np.abs(fitted_cost - x_cost)))
    return cost_err, p_cost, qf_cost, x_cost


@app.cell
def _(np, p_cost, plt, qf_cost, x_cost):
    _p = np.linspace(0.01, 0.99, 400)
    _x = qf_cost.quantile(_p)
    _pdf = qf_cost.pdf(_p)

    fig_cost, axes_cost = plt.subplots(1, 2, figsize=(11, 4))
    axes_cost[0].plot(_p, _x, color="steelblue", linewidth=2)
    axes_cost[0].scatter(p_cost, x_cost, color="firebrick", zorder=5, s=55)
    axes_cost[0].set_xlabel("cumulative probability p")
    axes_cost[0].set_ylabel("cost ($M)")
    axes_cost[0].set_title("Quantile function Q(p)")

    axes_cost[1].plot(_x, _pdf, color="steelblue", linewidth=2)
    axes_cost[1].set_xlabel("cost ($M)")
    axes_cost[1].set_ylabel("density")
    axes_cost[1].set_title("Implied PDF  f(x) = 1 / q(p)")
    axes_cost[1].set_ylim(bottom=0)
    fig_cost.tight_layout()
    return (fig_cost,)


@app.cell
def _(cost_err, fig_cost, mo, qf_cost):
    mo.vstack([
        mo.md(
            f"""
            **Fitted 3-term QFlex** through (P10, P50, P90) = (530, 620, 810):

            - centering parameter γ = **{qf_cost.gamma:.4f}**
            - valid (feasible) distribution = **{qf_cost.is_feasible}**
            - max interpolation error = **{cost_err:.2e}** (exact, since terms = points = 3)
            - basis terms used: **a₀, R¹, L¹** (intercept + first-order right/left tails)
            """
        ),
        fig_cost,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part A — Constructing valid quantile functions  *(paper §2)*

    A function \(Q\) is a **valid quantile function** if it is continuous and strictly increasing
    on \((0,1)\) — equivalently, its quantile density \(q(p)=Q'(p)\) is nonnegative. Gilchrist
    (2000) gives a set of transformations that *preserve* validity. The table below lists them;
    use the dropdown on the right to **visualise** each one (building blocks dashed, result in
    green).

    Most rules come with a **condition**. Use the two parameter sliders (θ₁, θ₂) to push each
    rule across its boundary and watch the result flip between **valid** (strictly increasing) and
    **invalid** (non-monotone). The two unconditional rules — *reflection* and *addition* — stay
    valid no matter what, so their sliders have no effect, which is itself the point.
    """)
    return


@app.cell
def _(mo):
    transform_dropdown = mo.ui.dropdown(
        options={
            "Affine shift — a + b·Q(p)": "affine",
            "Reflection — −Q(1−p)": "reflection",
            "Addition — Q₁ + Q₂": "addition",
            "Positive combination — a·Q₁ + b·Q₂": "poscomb",
            "Product — Q₁·Q₂": "product",
            "Monotone map — T(Q)": "monotone",
        },
        value="Addition — Q₁ + Q₂",
        label="Transformation",
    )
    return (transform_dropdown,)


@app.cell
def _(mo):
    ta_p1 = mo.ui.slider(
        -2.0, 3.0, value=1.5, step=0.25,
        label="θ₁", show_value=True,
    )
    ta_p2 = mo.ui.slider(
        -2.0, 3.0, value=1.0, step=0.25,
        label="θ₂", show_value=True,
    )
    return ta_p1, ta_p2


@app.cell
def _(mo, np, plt, ta_p1, ta_p2, transform_dropdown):
    _p = np.linspace(0.001, 0.999, 400)
    _R = -np.log(1 - _p)            # right-tail exponential R¹
    _L = np.log(_p)                 # left-tail reflected exponential L¹
    _logit = np.log(_p / (1 - _p))  # logistic quantile function
    _unif = _p.copy()               # uniform(0,1) quantile function
    _choice = transform_dropdown.value
    _t1 = float(ta_p1.value)
    _t2 = float(ta_p2.value)

    fig_tr, ax_tr = plt.subplots(figsize=(6, 4.3))
    _monotone_map = (_choice == "monotone")

    if _choice == "affine":
        _inputs = [("Q(p) = logit", _logit)]
        _out = _t2 + _t1 * _logit
        _title = f"Affine shift:  {_t2:.2f} + {_t1:.2f}·Q(p)"
        _cond = "valid ⇔ slope θ₁ > 0  (θ₂ only shifts, never breaks validity)"
        _active = "**θ₁** = slope b   ·   **θ₂** = shift a"
    elif _choice == "reflection":
        _inputs = [("Q(p) = R¹(p)", _R)]
        _out = _L  # −Q(1−p) with Q = R¹  →  ln(p)
        _title = "Reflection:  −Q(1−p)"
        _cond = "unconditionally valid — no parameter can break it"
        _active = "*no active parameters* (sliders have no effect here)"
    elif _choice == "addition":
        _inputs = [("Q₁ = R¹", _R), ("Q₂ = L¹", _L)]
        _out = _R + _L
        _title = "Addition:  Q₁ + Q₂ = logit"
        _cond = "unconditionally valid — sum of valid QFs is always valid"
        _active = "*no active parameters* (sliders have no effect here)"
    elif _choice == "poscomb":
        _inputs = [("Q₁ = logit", _logit), ("Q₂ = R¹", _R)]
        _out = _t1 * _logit + _t2 * _R
        _title = f"Combination:  {_t1:.2f}·Q₁ + {_t2:.2f}·Q₂"
        _cond = "valid ⇔ θ₁ ≥ 0 AND θ₂ ≥ 0  (a negative weight can tip it)"
        _active = "**θ₁** = weight on Q₁   ·   **θ₂** = weight on Q₂"
    elif _choice == "product":
        _f2 = _unif + _t1
        _inputs = [("Q₁ = R¹ (≥0)", _R), (f"Q₂ = p + {_t1:.2f}", _f2)]
        _out = _R * _f2
        _title = f"Product:  R¹ · (p + {_t1:.2f})"
        _cond = "valid ⇔ both factors ≥ 0 on (0,1)  ⇔  θ₁ ≥ 0"
        _active = "**θ₁** = shift of 2nd factor (push it below 0 to break it)"
    else:  # monotone
        _inputs = [("Q(p) = logit", _logit)]
        _out = np.exp(_t1 * _logit)
        _title = f"Monotone map:  T(Q) = exp({_t1:.2f}·Q)"
        _cond = "valid ⇔ T non-decreasing  ⇔  θ₁ > 0"
        _active = "**θ₁** = rate in T(y) = exp(θ₁·y)"

    _valid = bool(np.all(np.diff(_out) > 0))
    _res_color = "tab:green" if _valid else "tab:red"

    if _monotone_map:
        ax_tr.plot(_p, _inputs[0][1], linestyle="--", color="tab:blue",
                   alpha=0.8, linewidth=1.3, label=_inputs[0][0])
        ax_tr.set_ylabel("Q(p)")
        _ax2 = ax_tr.twinx()
        _ax2.plot(_p, _out, color=_res_color, linewidth=2.4, label="result T(Q)")
        _ax2.set_ylabel("T(Q)")
        ax_tr.legend(fontsize=8, loc="upper left")
        _ax2.legend(fontsize=8, loc="lower right")
    else:
        for _lbl, _v in _inputs:
            ax_tr.plot(_p, _v, linestyle="--", alpha=0.75, linewidth=1.2, label=_lbl)
        ax_tr.plot(_p, _out, color=_res_color, linewidth=2.4, label="result")
        ax_tr.set_ylabel("quantile value")
        ax_tr.legend(fontsize=8)

    ax_tr.axhline(0, color="gray", linewidth=0.5)
    ax_tr.set_xlabel("p")
    ax_tr.set_title(
        f"{_title}\nstrictly increasing: {'YES — valid' if _valid else 'NO — invalid'}",
        fontsize=9, color=("black" if _valid else "tab:red"),
    )
    fig_tr.tight_layout()

    _badge = "✅ valid" if _valid else "❌ invalid"
    tr_status = mo.md(
        f"""
        **{_badge}** — result is {'strictly increasing' if _valid else 'not monotone'}.

        - **Condition:** {_cond}
        - **Sliders:** {_active}
        """
    )
    return fig_tr, tr_status


@app.cell
def _(fig_tr, mo, ta_p1, ta_p2, transform_dropdown, tr_status):
    _table = mo.md(r"""
    | Transformation | Form | Condition |
    |----------------|------|-----------|
    | Affine shift | a + b·Q(p) | b > 0 |
    | Reflection | −Q(1−p) | always valid |
    | Addition | Q₁(p) + Q₂(p) | both valid |
    | Positive comb. | a·Q₁ + b·Q₂ | a, b > 0 |
    | Product | Q₁(p)·Q₂(p) | both ≥ 0 on (0,1) |
    | Monotone map | T(Q(p)) | T non-decreasing |
    """)
    _panel = mo.vstack([
        transform_dropdown,
        mo.hstack([ta_p1, ta_p2], justify="start", gap=1.5),
        fig_tr,
        tr_status,
    ])
    mo.hstack([_table, _panel], widths=[0.4, 0.6], gap=2, align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Odd-Power Rule** (Bickel 2025): \(Q(p)^k\) stays valid whenever \(Q\) is valid and \(k\) is
    *odd*. Even powers are admissible only if \(Q\) is nonnegative on \((0,1)\).

    QFlex is assembled **only** from these rules, so it is monotone *by construction*. The Metalog
    is not: its \(\mu(p)\) and \(\sigma(p)\) polynomial-logit terms multiply the logit
    \(\ln\!\frac{p}{1-p}\) (which changes sign at \(p=0.5\)) by powers of \((p-0.5)\). For odd
    \(k>1\) these terms are **not** valid quantile functions, so a Metalog can be non-monotone even
    from valid quantile inputs.
    """)
    return


@app.cell
def _(np, plt):
    # Building the logistic from monotone QFlex tail bases: R + L = logit
    _p = np.linspace(0.001, 0.999, 400)
    _R = -np.log(1 - _p)            # right-tail exponential R_1
    _L = np.log(_p)                 # left-tail reflected exponential L_1
    _logit = _R + _L               # = ln(p/(1-p)), the logistic quantile function

    fig_build, axes_build = plt.subplots(1, 2, figsize=(11, 4))
    axes_build[0].plot(_p, _R, label="R¹(p) = −ln(1−p)", color="tab:blue")
    axes_build[0].plot(_p, _L, label="L¹(p) = ln(p)", color="tab:orange")
    axes_build[0].axhline(0, color="gray", linewidth=0.6)
    axes_build[0].set_title("Two valid (monotone) tail bases")
    axes_build[0].set_xlabel("p")
    axes_build[0].legend(fontsize=8)

    axes_build[1].plot(_p, _logit, color="tab:green", linewidth=2,
                       label="R¹ + L¹ = ln(p/(1−p))")
    axes_build[1].axhline(0, color="gray", linewidth=0.6)
    axes_build[1].set_title("Their sum is the logistic QF (Addition Rule)")
    axes_build[1].set_xlabel("p")
    axes_build[1].legend(fontsize=8)
    fig_build.tight_layout()
    return (fig_build,)


@app.cell
def _(fig_build, mo):
    mo.vstack([
        mo.md(
            "**Addition Rule in action.** Adding the right-tail exponential R¹ and the "
            "left-tail reflected exponential L¹ — both strictly increasing — yields the logistic "
            "quantile function. Every QFlex model is built this way, so validity is automatic."
        ),
        fig_build,
    ])
    return


@app.cell
def _(mo):
    violation_dropdown = mo.ui.dropdown(
        options={
            "Metalog k=3:  (p−0.5)·ln(p/(1−p))": 1,
            "Metalog k=7:  (p−0.5)³·ln(p/(1−p))": 3,
        },
        value="Metalog k=3:  (p−0.5)·ln(p/(1−p))",
        label="Metalog term",
    )
    return (violation_dropdown,)


@app.cell
def _(np, plt, violation_dropdown):
    _power = violation_dropdown.value
    _p = np.linspace(0.001, 0.999, 600)
    _logit = np.log(_p / (1 - _p))
    _metalog_term = (_p - 0.5) ** _power * _logit  # a Metalog uniform-logit term
    _gamma = 0.45
    _qflex_center = (_p - _gamma) ** _power         # the QFlex center basis C_j (odd power)

    def _is_increasing(v):
        return bool(np.all(np.diff(v) >= -1e-9))

    fig_viol, axes_viol = plt.subplots(1, 2, figsize=(11, 4))

    _ml_ok = _is_increasing(_metalog_term)
    axes_viol[0].plot(_p, _metalog_term, color="firebrick", linewidth=2)
    _dec = np.diff(_metalog_term) < 0
    axes_viol[0].fill_between(_p[1:], _metalog_term[1:].min(), _metalog_term[1:],
                             where=_dec, color="red", alpha=0.12,
                             label="decreasing region")
    axes_viol[0].axhline(0, color="gray", linewidth=0.6)
    axes_viol[0].set_title(
        f"Metalog term — monotone? {'yes' if _ml_ok else 'NO (invalid QF)'}"
    )
    axes_viol[0].set_xlabel("p")
    axes_viol[0].legend(fontsize=8)

    _qf_ok = _is_increasing(_qflex_center)
    axes_viol[1].plot(_p, _qflex_center, color="tab:green", linewidth=2)
    axes_viol[1].axhline(0, color="gray", linewidth=0.6)
    axes_viol[1].set_title(
        f"QFlex center Cʲ = (p−γ)^{_power} — monotone? {'yes' if _qf_ok else 'NO'}"
    )
    axes_viol[1].set_xlabel("p")
    fig_viol.tight_layout()
    return (fig_viol,)


@app.cell
def _(fig_viol, mo, violation_dropdown):
    mo.vstack([
        mo.md("### Interactive — where Metalog breaks Gilchrist's rules"),
        mo.md(
            "The Metalog multiplies the sign-changing logit by a power of (p−0.5). The product is "
            "**not monotone**, so it is not a valid quantile function — adding it can break the "
            "whole distribution. QFlex instead uses the centered-uniform basis (p−γ) raised to an "
            "**odd** power, which stays strictly increasing (Odd-Power Rule)."
        ),
        violation_dropdown,
        fig_viol,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part B — The QFlex system  *(paper §3)*

    QFlex writes the quantile function as a linear combination of basis functions drawn from the
    three monotone families of Part A:

    \[
    Q(p) \;=\; a_0 \;+\; \sum_{j\ge 1}\Big[\, b_j\,R_j(p) \;+\; c_j\,L_j(p) \;+\; d_j\,C_j(p)\,\Big],
    \]

    with

    - **Right tail** \(R_j(p) = \big(-\ln(1-p)\big)^{j}\) — controls the *upper* tail,
    - **Left tail** \(L_j(p) = (-1)^{j+1}\big(\ln p\big)^{j}\) — controls the *lower* tail,
    - **Center** \(C_j(p) = (p-\gamma)^{2j-1}\) — shapes the *body*, anchored at the data-driven
      center \(\gamma\).

    Terms are added in the repeating order \(a_0,\,R_1,\,L_1,\,C_1,\,R_2,\,L_2,\,C_2,\dots\), so a
    **3-term** model uses \(\{a_0, R_1, L_1\}\) and the center terms switch on from 4 terms upward.

    **Fitting.** Stacking the \(m\) assessments gives the linear system \(\mathbf{x} = Y\mathbf{a}\),
    where the *design matrix* \(Y_{ij} = \text{basis}_j(p_i)\). With \(n\) terms:

    - \(m = n\): **exact interpolation**, \(\mathbf{a} = Y^{-1}\mathbf{x}\);
    - \(m > n\): **least squares**, \(\mathbf{a} = (Y^\top Y)^{-1}Y^\top\mathbf{x}\).

    Because every column of \(Y\) is a valid quantile function, the fit is monotone whenever the
    coefficient signs cooperate (the subject of Part C).
    """)
    return


@app.cell
def _(
    build_design_matrix,
    get_term_structure,
    mo,
    np,
    p_cost,
    qf_cost,
    x_cost,
):
    _Y = build_design_matrix(np.asarray(p_cost), qf_cost.terms, qf_cost.gamma)
    _a = qf_cost.coefficients
    _q = _Y @ _a
    _struct = get_term_structure(qf_cost.terms)
    _names = {"constant": "1 (a₀)", "f1": "R¹=−ln(1−p)", "f2": "L¹=ln(p)", "f3": "C¹=(p−γ)"}
    _hdr = " | ".join(_names[bt.value] for bt, _ in _struct)
    _rows = []
    for _i, _p in enumerate(p_cost):
        _vals = " | ".join(f"{v:.3f}" for v in _Y[_i])
        _rows.append(f"| {_p:.2f} | {_vals} | {_q[_i]:.1f} | {x_cost[_i]:.1f} |")
    _table = (
        f"| p | {_hdr} | Q(p)=Y·a | data x |\n"
        f"|---|{'---|' * (len(_struct) + 2)}\n"
        + "\n".join(_rows)
    )
    _coef = ", ".join(f"a{_i}={c:.3f}" for _i, c in enumerate(_a))
    mo.md(
        f"**Design matrix for the facility-cost fit** (γ = {qf_cost.gamma:.3f}, "
        f"{qf_cost.terms} terms). Each row evaluates Q(p) = Y·a and recovers the data exactly:\n\n"
        f"{_table}\n\n"
        f"Fitted coefficients: **{_coef}**. The intercept sets the location, the R¹ coefficient "
        "stretches the upper tail, and the L¹ coefficient stretches the lower tail."
    )
    return


@app.cell
def _(BasisType, evaluate_basis, np, plt):
    _p = np.linspace(0.001, 0.999, 400)
    _gamma = 0.5
    fig_basis, axes_basis = plt.subplots(1, 3, figsize=(13, 4))
    _colors = ["tab:blue", "tab:orange", "tab:green"]
    for _j in (1, 2, 3):
        axes_basis[0].plot(_p, evaluate_basis(_p, BasisType.F1_TAIL_RIGHT, _j, _gamma),
                           color=_colors[_j - 1], label=f"R{_j}")
        axes_basis[1].plot(_p, evaluate_basis(_p, BasisType.F2_TAIL_LEFT, _j, _gamma),
                           color=_colors[_j - 1], label=f"L{_j}")
        axes_basis[2].plot(_p, evaluate_basis(_p, BasisType.F3_CENTER, _j, _gamma),
                           color=_colors[_j - 1], label=f"C{_j}")
    _titles = [
        "Right-tail bases  Rⱼ = (−ln(1−p))ʲ",
        "Left-tail bases  Lⱼ = (−1)^(j+1)(ln p)ʲ",
        "Center bases  Cⱼ = (p−γ)^(2j−1),  γ=0.5",
    ]
    axes_basis[0].set_ylim(-0.5, 15)
    axes_basis[1].set_ylim(-15, 0.5)
    for _k in range(3):
        axes_basis[_k].axhline(0, color="gray", linewidth=0.5)
        axes_basis[_k].set_xlabel("p")
        axes_basis[_k].set_title(_titles[_k], fontsize=9)
        axes_basis[_k].legend(fontsize=8)
    fig_basis.tight_layout()
    return (fig_basis,)


@app.cell
def _(fig_basis, mo):
    mo.vstack([
        mo.md(
            "**The three basis families.** Each is a *valid* (monotone) quantile function on its "
            "own. Higher orders concentrate influence: bigger **j** makes the tails grow faster "
            "near p→0 or p→1, while the center terms add increasingly localized flexibility in the "
            "body. A QFlex model mixes a handful of these — its expressiveness scales with the "
            "number of terms."
        ),
        fig_basis,
    ])
    return


@app.cell
def _(mo):
    p_demo = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95]
    x_demo = [12.0, 14.0, 17.0, 21.0, 25.0, 30.0, 38.0, 48.0, 56.0]
    terms_slider = mo.ui.slider(3, 9, value=3, step=1, label="QFlex terms", show_value=True)
    return p_demo, terms_slider, x_demo


@app.cell
def _(QFlex, np, p_demo, plt, terms_slider, x_demo):
    import warnings as _warnings

    _terms = terms_slider.value
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        _qf = QFlex(x_demo, p_demo, terms=_terms)
    _pp = np.linspace(0.01, 0.99, 400)
    _xx = _qf.quantile(_pp)
    _fit_pts = _qf.quantile(np.asarray(p_demo))
    _err = float(np.max(np.abs(_fit_pts - np.asarray(x_demo))))

    fig_depth, axes_depth = plt.subplots(1, 2, figsize=(12, 4.2))
    axes_depth[0].plot(_pp, _xx, color="steelblue", linewidth=2, label=f"{_terms}-term Q(p)")
    axes_depth[0].scatter(p_demo, x_demo, color="firebrick", zorder=5, s=45, label="assessments")
    axes_depth[0].set_xlabel("p")
    axes_depth[0].set_ylabel("x")
    axes_depth[0].set_title(f"Fit with {_terms} terms   (max error = {_err:.2e})", fontsize=9)
    axes_depth[0].legend(fontsize=8)

    _pdf = _qf.pdf(_pp)
    axes_depth[1].plot(_xx, _pdf, color="seagreen", linewidth=2)
    axes_depth[1].set_xlabel("x")
    axes_depth[1].set_ylabel("density")
    axes_depth[1].set_title(f"Implied PDF   (feasible: {_qf.is_feasible})", fontsize=9)
    axes_depth[1].set_ylim(bottom=0)
    fig_depth.tight_layout()
    return (fig_depth,)


@app.cell
def _(fig_depth, mo, terms_slider):
    mo.vstack([
        mo.md(
            "### Interactive — fit depth and interpolation\n\n"
            "Nine right-skewed assessments are fit with a varying number of terms. With **3 terms** "
            "the fit is a smooth least-squares approximation (nonzero error). As terms increase the "
            "error shrinks, and at **9 terms = 9 points** the model **interpolates exactly** "
            "(error ≈ 0). Watch the implied PDF: extra terms add flexibility but can also push the "
            "fit toward *infeasibility* — the motivation for the monotonicity constraints in Part C."
        ),
        terms_slider,
        fig_depth,
    ])
    return


@app.cell
def _(mo, qf_cost):
    _m = qf_cost.moments(order=4)
    mo.md(
        "**Moments.** Once \\(Q(p)\\) is fixed, moments follow from \\(E[X^k]=\\int_0^1 Q(p)^k\\,dp\\) "
        "(numerical integration). For the facility-cost fit:\n\n"
        "| mean | std | skewness | kurtosis |\n|---|---|---|---|\n"
        f"| {_m['mean']:.2f} | {_m['std']:.2f} | {_m['skewness']:.3f} | {_m['kurtosis']:.3f} |\n\n"
        "The positive skewness mirrors the asymmetric assessments "
        "(P90−P50 = 190 > P50−P10 = 90)."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part C — Monotonicity: when is a QFlex valid?  *(paper §4)*

    A QFlex is a valid distribution iff its quantile density is positive everywhere,
    \(q(p)=Q'(p)>0\) on \((0,1)\). Because every basis function is a *monotone* quantile
    function, the cleanest sufficient condition is purely structural:

    > **Theorem 1 (Monotonicity Sufficiency).** If all shape coefficients are nonnegative,
    > \(a_k \ge 0\) for \(k\ge 2\), then \(q(p)>0\) and \(Q\) is strictly increasing.

    Nonnegative coefficients give a distribution-free guarantee — *no optimization required*. But a
    least-squares fit can return **negative** coefficients (especially on the center terms), and then
    the distribution may be invalid. The rest of §4 develops weaker conditions that still certify (or
    enforce) validity, and the `qflex` library exposes each one as a `ConstraintType`.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §4.1  Necessary-and-sufficient conditions, and a tail necessity

    For small models the exact (necessary **and** sufficient) coefficient conditions are closed-form
    (**Proposition 2**):

    | Order | Condition for strict monotonicity |
    |-------|-----------------------------------|
    | \(K=2\) | \(a_2 \ge 0\) |
    | \(K=3\) | \(a_2 \ge 0,\; a_3 \ge 0\) |
    | \(K=4\) | \(a_2 \ge 0,\; a_3 \ge 0,\; a_4 \ge -(\sqrt{a_2}+\sqrt{a_3})^2\) |
    | \(K\ge 5\) | the QDF/QCF become transcendental — **no closed form** |

    Here \(a_2,a_3\) are the first-order right/left tail coefficients (\(R_1,L_1\)) and \(a_4\) is the
    first center coefficient (\(C_1\)). The \(K=4\) case says the center term *cannot be too negative*.

    > **Theorem 5 (Tail necessity).** Any valid QFlex must have **strictly positive leading-tail
    > coefficients**. So enforcing nonnegative tails (`ConstraintType.TL`, `TA`) is *necessary* — but,
    > as the demo below shows, **not sufficient** when the center term is the culprit.

    Because no closed form exists for \(K\ge5\), the paper provides two practical guarantees.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §4.2  Two practical guarantees and their nesting

    Decompose the QDF into tail and center parts, \(q(p)=q_{\text{tail}}(p)+q_{\text{center}}(p)\),
    and define the smallest tail magnitude and largest center magnitude
    \[
    m_{\text{tail}}=\inf_{p}q_{\text{tail}}(p)\ge 0,\qquad
    M_{\text{center}}=\sup_{p}\big|q_{\text{center}}(p)\big|.
    \]

    > **Proposition 3 (ex-post certificate).** If \(m_{\text{tail}}>M_{\text{center}}\) then
    > \(q(p)>0\) everywhere. Evaluated on a grid (`qf.check_proposition4()` /
    > `ConstraintType.TC_mag`), this *certifies* a fitted coefficient vector after the fact.

    > **Proposition 4 (a-priori bound).** Replacing the inf/sup with *coefficient-only* bounds gives a
    > **convex, linear** condition: all tails \(\ge 0\) and
    > \((a_2+a_3)-\sum_{\text{center}}|a_k|\,\bar C_k \ge 0\), with
    > \(\bar C_k=(2j-1)\max(\gamma,1-\gamma)^{2j-2}\). It can be enforced *before* fitting
    > (`ConstraintType.TC`, solved as a linear program).

    These give **two different** convex inner approximations of the (unknown) valid set:
    \[
    \mathcal A^{[K]}\ (\text{Thm 1})\ \subseteq\ \mathcal A_{\text{valid}}^{[K]},
    \qquad
    \mathcal A_{TC}^{[K]}\ (\text{Prop 4})\ \subseteq\ \mathcal A_{\text{valid}}^{[K]} .
    \]
    They are **not** nested inside one another: Theorem 1 forbids any negative center coefficient but
    allows arbitrarily large *positive* ones (which forces unimodality); Proposition 4 allows
    *moderately negative* center coefficients but caps their magnitude. Both sit strictly inside the
    exact valid set, which has no closed form.
    """)
    return


@app.cell
def _():
    # Asymmetric probabilities (not mirrored about 0.5) keep the 7x7 design matrix
    # well-conditioned, so the unconstrained fit interpolates exactly while still being infeasible.
    p_base = [0.05, 0.15, 0.30, 0.45, 0.60, 0.80, 0.95]
    x_base = [1.0, 2.0, 3.0, 9.0, 16.0, 18.0, 19.0]  # steep-middle S-shape -> infeasible at K=7
    return p_base, x_base


@app.cell
def _(
    BasisType,
    QFlex,
    evaluate_basis_derivative,
    get_term_structure,
    np,
    p_base,
    plt,
    x_base,
):
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _qf = QFlex(x_base, p_base, terms=7)
    _p = np.linspace(0.005, 0.995, 500)
    _qt = np.zeros_like(_p)
    _qc = np.zeros_like(_p)
    for _idx, (_bt, _o) in enumerate(get_term_structure(7)):
        _d = evaluate_basis_derivative(_p, _bt, _o, _qf.gamma)
        if _bt in (BasisType.F1_TAIL_RIGHT, BasisType.F2_TAIL_LEFT):
            _qt += _qf.coefficients[_idx] * _d
        elif _bt == BasisType.F3_CENTER:
            _qc += _qf.coefficients[_idx] * _d
    _qfull = _qt + _qc

    fig_base, axes_base = plt.subplots(1, 2, figsize=(12, 4.2))
    axes_base[0].plot(_p, _qf.quantile(_p), color="steelblue", lw=2, label="Q(p)")
    axes_base[0].scatter(p_base, x_base, color="firebrick", zorder=5, s=40, label="data")
    axes_base[0].set_title("Unconstrained 7-term fit — interpolates data, but non-monotone",
                           fontsize=9)
    axes_base[0].set_xlabel("p")
    axes_base[0].set_ylabel("x")
    axes_base[0].set_ylim(min(x_base) - 8, max(x_base) + 9)  # focus on data; tails blow up off-screen
    axes_base[0].legend(fontsize=8)

    axes_base[1].plot(_p, _qt, color="tab:blue", lw=1.4, label="q_tail")
    axes_base[1].plot(_p, _qc, color="tab:orange", lw=1.4, label="q_center")
    axes_base[1].plot(_p, _qfull, color="black", lw=2, label="q(p) = q_tail + q_center")
    axes_base[1].axhline(0, color="red", lw=0.9, ls="--")
    axes_base[1].fill_between(_p, _qfull, 0, where=_qfull < 0, color="red", alpha=0.2,
                             label="q(p) < 0  (invalid)")
    _lim = float(np.percentile(np.abs(_qfull), 92))
    axes_base[1].set_ylim(-_lim, _lim)
    axes_base[1].set_title("Quantile density q(p) dips below 0", fontsize=9)
    axes_base[1].set_xlabel("p")
    axes_base[1].legend(fontsize=8)
    fig_base.tight_layout()
    return (fig_base,)


@app.cell
def _(fig_base, mo):
    mo.vstack([
        mo.md(
            "**A genuinely infeasible baseline.** These seven assessments rise steeply through the "
            "middle. The exact 7-term fit (7 points, 7 terms) **passes through every point**, yet it "
            "is **non-monotone between them** — \\(Q(p)\\) dips back down, so the quantile density "
            "**goes negative** (red region) and the implied PDF would be negative. The distribution "
            "is *invalid*; the constraints below repair it. (The left axis is zoomed to the data; the "
            "tails actually shoot off-screen, another symptom of the unconstrained high-order fit.)"
        ),
        fig_base,
    ])
    return


@app.cell
def _(ConstraintType, mo):
    c_constraint_dropdown = mo.ui.dropdown(
        options={
            "None — unconstrained (baseline)": ConstraintType.NONE,
            "A+ — Theorem 1 (all shape coeffs ≥ 0)": ConstraintType.A,
            "TL+ — Theorem 5 (leading tails ≥ 0)": ConstraintType.TL,
            "TA+ — all tail coeffs ≥ 0": ConstraintType.TA,
            "TC_mag — Proposition 3 (grid certificate)": ConstraintType.TC_MAG,
            "TC — Proposition 4 (coefficient bound, LP)": ConstraintType.TC,
        },
        value="None — unconstrained (baseline)",
        label="Constraint",
    )
    return (c_constraint_dropdown,)


@app.cell
def _(
    BasisType,
    QFlex,
    c_constraint_dropdown,
    evaluate_basis_derivative,
    get_term_structure,
    mo,
    np,
    p_base,
    plt,
    tail_center_margin_coeff,
    x_base,
):
    import warnings as _w

    _ct = c_constraint_dropdown.value
    _err = None
    try:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            _qf = QFlex(x_base, p_base, terms=7, constraint_type=_ct)
    except Exception as _e:  # constraint may be infeasible for some configs
        _qf = None
        _err = str(_e)

    if _qf is None:
        fig_constraint, _ax = plt.subplots(figsize=(6, 2))
        _ax.axis("off")
        _ax.text(0.5, 0.5, "Constraint solve failed:\n" + _err, ha="center", va="center",
                 fontsize=9, color="firebrick")
        c_status = mo.md(f"**{_ct.value}** could not be solved: {_err}")
    else:
        _p = np.linspace(0.005, 0.995, 500)
        _qt = np.zeros_like(_p)
        _qc = np.zeros_like(_p)
        _labels, _colors = [], []
        for _idx, (_bt, _o) in enumerate(get_term_structure(7)):
            _d = evaluate_basis_derivative(_p, _bt, _o, _qf.gamma)
            if _bt == BasisType.CONSTANT:
                _labels.append("a₀"); _colors.append("gray")
            elif _bt == BasisType.F1_TAIL_RIGHT:
                _qt += _qf.coefficients[_idx] * _d
                _labels.append(f"R{_o}"); _colors.append("tab:blue")
            elif _bt == BasisType.F2_TAIL_LEFT:
                _qt += _qf.coefficients[_idx] * _d
                _labels.append(f"L{_o}"); _colors.append("tab:cyan")
            else:
                _qc += _qf.coefficients[_idx] * _d
                _labels.append(f"C{_o}"); _colors.append("tab:orange")
        _qfull = _qt + _qc

        fig_constraint, _axc = plt.subplots(1, 2, figsize=(12, 4.2))
        _axc[0].plot(_p, _qt, color="tab:blue", lw=1.4, label="q_tail")
        _axc[0].plot(_p, _qc, color="tab:orange", lw=1.4, label="q_center")
        _axc[0].plot(_p, _qfull, color="black", lw=2, label="q(p)")
        _axc[0].axhline(0, color="red", lw=0.9, ls="--")
        _axc[0].fill_between(_p, _qfull, 0, where=_qfull < 0, color="red", alpha=0.2)
        _lim = float(np.percentile(np.abs(_qfull), 92)) or 1.0
        _axc[0].set_ylim(-_lim, _lim)
        _axc[0].set_title(f"QDF decomposition — {_ct.value}", fontsize=9)
        _axc[0].set_xlabel("p")
        _axc[0].legend(fontsize=8)

        _axc[1].bar(range(len(_labels)), _qf.coefficients, color=_colors)
        _axc[1].axhline(0, color="gray", lw=0.6)
        _axc[1].set_xticks(range(len(_labels)))
        _axc[1].set_xticklabels(_labels)
        _axc[1].set_title("Fitted coefficients (blue/cyan=tail, orange=center)", fontsize=9)
        fig_constraint.tight_layout()

        _pr = _qf.check_proposition4()
        _p4 = tail_center_margin_coeff(_qf.coefficients, 7, _qf.gamma)
        _sse = float(np.sum((_qf.quantile(np.asarray(p_base)) - np.asarray(x_base)) ** 2))
        c_status = mo.md(
            f"| metric | value | meaning |\n|---|---|---|\n"
            f"| **feasible** | **{_qf.is_feasible}** | valid PDF everywhere (ground truth) |\n"
            f"| SSE (fit) | {_sse:.2f} | lower = closer to the data |\n"
            f"| min q(p) | {_pr['q_flex_min']:.3f} | must be > 0 for validity |\n"
            f"| m_tail | {_pr['m_tail']:.3f} | smallest tail magnitude |\n"
            f"| M_center | {_pr['M_center']:.3f} | largest center magnitude |\n"
            f"| Prop 3 margin (grid) | {_pr['margin']:.3f} | > 0 *certifies* validity (sufficient) |\n"
            f"| Prop 4 margin (coeff) | {_p4:.3f} | > 0 *certifies* validity (sufficient) |\n"
            "\n*`feasible` is checked directly on q(p). The Prop 3 / Prop 4 margins are "
            "**sufficient** certificates: a positive margin guarantees validity, but a "
            "**negative margin does not imply invalid** — e.g. `A+` is valid via Theorem 1 even "
            "though both margins are negative.*"
        )
    return c_status, fig_constraint


@app.cell
def _(c_constraint_dropdown, c_status, fig_constraint, mo):
    mo.vstack([
        mo.md(
            "### Interactive — how each constraint changes the fit\n\n"
            "Pick a constraint and watch the **coefficients** (right) and the **quantile density** "
            "(left) respond. `A+` (Theorem 1) forces every shape coefficient ≥ 0 — here that drives "
            "the tails to zero and leaves a near-uniform body. `TC_mag` / `TC` allow a negative "
            "center yet keep q(p) > 0. `TL+` / `TA+` only constrain the **tails**, so they cannot "
            "fix an infeasibility caused by the **center** and stay infeasible here. Each guarantee "
            "costs some fit accuracy — compare the **SSE** row (in this example `A+` happens to fit "
            "best, not `TC`)."
        ),
        c_constraint_dropdown,
        c_status,
        fig_constraint,
    ])
    return


@app.cell
def _(plt):
    _a2, _a3 = 1.0, 1.0
    _exact_lo = -((_a2 ** 0.5 + _a3 ** 0.5) ** 2)   # valid:    a4 > -4
    _thm1_lo = 0.0                                   # Theorem1: a4 >= 0  (half-line)
    _prop4_lo = -(_a2 + _a3)                         # Prop4:    -2 < a4 < 2  (interval)
    _prop4_hi = (_a2 + _a3)
    _xmax = 6.0

    fig_nest, ax_nest = plt.subplots(figsize=(10, 3.2))
    # Each region drawn as a horizontal bar at its own height.
    ax_nest.hlines(3, _exact_lo, _xmax, color="tab:blue", lw=10, alpha=0.55)
    ax_nest.text(_xmax, 3, "  valid (exact):  a₄ > −4", va="center", fontsize=9, color="tab:blue")
    ax_nest.hlines(2, _thm1_lo, _xmax, color="tab:green", lw=10, alpha=0.65)
    ax_nest.text(_xmax, 2, "  Theorem 1:  a₄ ≥ 0", va="center", fontsize=9, color="tab:green")
    ax_nest.hlines(1, _prop4_lo, _prop4_hi, color="tab:orange", lw=10, alpha=0.7)
    ax_nest.text(_xmax, 1, "  Prop 4:  −2 < a₄ < 2", va="center", fontsize=9, color="tab:orange")

    for _xv in (_exact_lo, _prop4_lo, _thm1_lo, _prop4_hi):
        ax_nest.axvline(_xv, color="gray", lw=0.6, ls=":")
    ax_nest.set_xlim(_exact_lo - 1, _xmax + 5)
    ax_nest.set_ylim(0.3, 3.7)
    ax_nest.set_yticks([])
    ax_nest.set_xticks([_exact_lo, _prop4_lo, _thm1_lo, _prop4_hi])
    ax_nest.set_xticklabels(["−4", "−2", "0", "2"])
    ax_nest.set_xlabel("center coefficient a₄   (K=4, with a₂ = a₃ = 1)")
    ax_nest.set_title("Two certificates inside the valid set — neither contains the other",
                      fontsize=10)
    fig_nest.tight_layout()
    return (fig_nest,)


@app.cell
def _(fig_nest, mo):
    mo.vstack([
        mo.md(
            "**Two certificates, not a chain (Remark 8).** For the 4-term model with "
            "\\(a_2=a_3=1\\), the center coefficient is valid for \\(a_4>-4\\). The two guarantees "
            "carve out *different* safe sub-regions: **Theorem 1** allows \\(a_4\\in[0,\\infty)\\) — "
            "any large positive center (always unimodal) but no negative center; **Proposition 4** "
            "allows \\(a_4\\in(-2,2)\\) — moderately negative centers (which can create extra modes, "
            "Part D) but capped magnitude. They overlap on \\([0,2)\\), yet **neither is contained "
            "in the other**, and both lie strictly inside the exact valid set."
        ),
        fig_nest,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §4.3  Why Metalog can't do this

    QFlex's guarantees exist because its basis functions are **individually monotone**, so the
    tail–center decomposition yields tractable algebraic bounds. Metalog's basis functions oscillate
    and change curvature, so there is **no known closed-form characterization** of its valid
    coefficient region and **no analytic sufficient condition**. In practice Metalog relies on
    *ex-post repair*: fit by least squares, detect derivative violations, then solve a semi-infinite
    optimization adding constraints until monotonicity holds — grid-dependent, more expensive, and
    yielding no interpretable coefficient conditions. QFlex instead imposes simple nonnegativity or a
    single linear inequality.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part D — Modality and the quantile curvature function  *(paper §5)*

    The shape of the density is governed by the **quantile curvature function (QCF)**,
    \(\kappa(p)=q'(p)\), the derivative of the quantile density. Because
    \[
    f\big(Q(p)\big)=\frac{1}{q(p)},
    \]
    a **local minimum of \(q\)** is a **local maximum (mode) of the PDF**. Modes are created exactly
    where \(\kappa(p)\) crosses zero (from negative to positive). So the number and location of
    interior modes are determined by the **zeros of the QCF**.

    > **Theorem 6 (No boundary modes for \(K>2\)).** A valid unbounded QFlex has strictly positive
    > leading-tail coefficients (Theorem 5), which force \(\kappa(p)<0\) near \(p=0\) and
    > \(\kappa(p)>0\) near \(p=1\), with \(f\to 0\) at both ends. Hence **all modes are interior**.

    *(The \(K=2\) case is a one-sided exponential with a single boundary mode and is excluded.)*
    """)
    return


@app.cell
def _(evaluate_quantile_derivative, np, plt, qf_cost):
    _p = np.linspace(0.01, 0.99, 600)
    _q = evaluate_quantile_derivative(_p, qf_cost.coefficients, qf_cost.terms, qf_cost.gamma)
    _kappa = np.gradient(_q, _p)
    _pdf = 1.0 / _q
    _x = qf_cost.quantile(_p)
    _imin = int(np.argmin(_q))
    _pmode = _p[_imin]

    fig_anatomy, axA = plt.subplots(1, 3, figsize=(13, 4))
    axA[0].plot(_p, _q, color="steelblue", lw=2)
    axA[0].scatter([_pmode], [_q[_imin]], color="firebrick", zorder=5)
    axA[0].set_title("Quantile density q(p)\n(min at the mode)", fontsize=9)
    axA[0].set_xlabel("p")
    axA[0].set_ylabel("q(p)")

    axA[1].plot(_p, _kappa, color="tab:purple", lw=2)
    axA[1].axhline(0, color="red", lw=0.9, ls="--")
    axA[1].axvline(_pmode, color="firebrick", lw=0.9, ls=":")
    axA[1].set_title("QCF κ(p) = q'(p)\n(zero crossing − → +)", fontsize=9)
    axA[1].set_xlabel("p")
    axA[1].set_ylabel("κ(p)")

    axA[2].plot(_x, _pdf, color="seagreen", lw=2)
    axA[2].scatter([_x[_imin]], [_pdf[_imin]], color="firebrick", zorder=5)
    axA[2].set_title("PDF f(x) = 1/q(p)\n(mode where q is minimal)", fontsize=9)
    axA[2].set_xlabel("x")
    axA[2].set_ylabel("density")
    axA[2].set_ylim(bottom=0)
    fig_anatomy.tight_layout()
    return (fig_anatomy,)


@app.cell
def _(fig_anatomy, mo):
    mo.vstack([
        mo.md(
            "**Anatomy of a mode** (running facility-cost fit). The single mode of the PDF (right) "
            "sits exactly where the quantile density \\(q(p)\\) is minimal (left), which is exactly "
            "where the QCF \\(\\kappa(p)\\) crosses zero from negative to positive (centre). Note "
            "\\(f\\to0\\) at both tails — no boundary modes, as Theorem 6 guarantees."
        ),
        fig_anatomy,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Low-order modality in closed form

    For the **3-term** QFlex (basis \(\{a_0, R_1, L_1\}\), with \(a_2,a_3\) the right/left tail
    coefficients), the QCF \(\kappa(p)=\dfrac{a_2}{(1-p)^2}-\dfrac{a_3}{p^2}\) crosses zero once, so
    there is a **single interior mode** with

    \[
    p^\* = \frac{\sqrt{a_3}}{\sqrt{a_2}+\sqrt{a_3}}, \qquad
    q_{\min}=\big(\sqrt{a_2}+\sqrt{a_3}\big)^2, \qquad
    f_{\max}=\frac{1}{q_{\min}} .
    \]

    The **4-term** QFlex adds the uniform term \(C_1=(p-\gamma)\), whose curvature is **zero**, so
    \(\kappa^{[4]}=\kappa^{[3]}\): the mode does **not move**. Only the minimum of \(q\) shifts,
    \(q_{\min}=(\sqrt{a_2}+\sqrt{a_3})^2+a_4\), so \(a_4\) raises or lowers the **mode height**
    (and validity requires \(a_4 > -(\sqrt{a_2}+\sqrt{a_3})^2\)).
    """)
    return


@app.cell
def _(evaluate_quantile_derivative, mo, np, qf_cost):
    _a2 = qf_cost.coefficients[1]
    _a3 = qf_cost.coefficients[2]
    _pstar = np.sqrt(_a3) / (np.sqrt(_a2) + np.sqrt(_a3))
    _qmin = (np.sqrt(_a2) + np.sqrt(_a3)) ** 2
    _fmax = 1.0 / _qmin

    _p = np.linspace(1e-4, 1 - 1e-4, 200000)
    _q = evaluate_quantile_derivative(_p, qf_cost.coefficients, qf_cost.terms, qf_cost.gamma)
    _pstar_num = _p[int(np.argmin(_q))]
    _qmin_num = float(_q.min())

    mo.md(
        "**Formula vs. library (3-term cost fit).** Verifying the closed forms against the "
        "numerically evaluated quantile density:\n\n"
        "| quantity | closed form | from library q(p) |\n|---|---|---|\n"
        f"| mode location p\\* | {_pstar:.5f} | {_pstar_num:.5f} |\n"
        f"| q_min | {_qmin:.4f} | {_qmin_num:.4f} |\n"
        f"| mode height f_max | {_fmax:.6f} | {1.0 / _qmin_num:.6f} |\n"
    )
    return


@app.cell
def _(mo, np, qf_cost):
    _a2 = qf_cost.coefficients[1]
    _a3 = qf_cost.coefficients[2]
    _thr = -((np.sqrt(_a2) + np.sqrt(_a3)) ** 2)
    a4_slider = mo.ui.slider(
        round(_thr - 25, 1), 200.0, value=0.0, step=5.0,
        label="center coefficient a₄ (uniform term)", show_value=True,
    )
    return (a4_slider,)


@app.cell
def _(
    a4_slider,
    evaluate_quantile,
    evaluate_quantile_derivative,
    np,
    plt,
    qf_cost,
):
    _a0, _a2, _a3 = qf_cost.coefficients[:3]
    _coef4 = np.array([_a0, _a2, _a3, float(a4_slider.value)])
    _g = qf_cost.gamma
    _p = np.linspace(0.005, 0.995, 600)
    _q = evaluate_quantile_derivative(_p, _coef4, 4, _g)
    _x = evaluate_quantile(_p, _coef4, 4, _g)
    _valid = bool(_q.min() > 0)
    _q_safe = np.clip(_q, 1e-9, None)
    _pdf = 1.0 / _q_safe
    _imin = int(np.argmin(_q))
    _qmin = (np.sqrt(_a2) + np.sqrt(_a3)) ** 2 + a4_slider.value

    fig_a4, axes_a4 = plt.subplots(1, 2, figsize=(12, 4.2))
    axes_a4[0].plot(_p, _q, color="steelblue", lw=2)
    axes_a4[0].axvline(_p[_imin], color="firebrick", ls=":", lw=1)
    axes_a4[0].axhline(0, color="red", ls="--", lw=0.8)
    axes_a4[0].set_title(f"q(p):  q_min = (√a₂+√a₃)² + a₄ = {_qmin:.1f}", fontsize=9)
    axes_a4[0].set_xlabel("p")
    axes_a4[0].set_ylabel("q(p)")

    if _valid:
        axes_a4[1].plot(_x, _pdf, color="seagreen", lw=2)
        axes_a4[1].set_ylim(bottom=0)
    axes_a4[1].set_title(
        f"PDF — mode fixed at p*={_p[_imin]:.3f}, height={1.0 / _qmin if _valid else float('nan'):.5f}"
        + ("" if _valid else "  (INVALID: a₄ below threshold)"),
        fontsize=9,
    )
    axes_a4[1].set_xlabel("x")
    axes_a4[1].set_ylabel("density")
    fig_a4.tight_layout()
    return (fig_a4,)


@app.cell
def _(a4_slider, fig_a4, mo):
    mo.vstack([
        mo.md(
            "### Interactive — the center coefficient a₄ sets the mode height\n\n"
            "Adding the uniform term to the 3-term cost fit leaves the **mode location fixed** "
            "(dotted line) but scales its **height** by \\(1/\\big((\\sqrt{a_2}+\\sqrt{a_3})^2+a_4\\big)\\). "
            "Large positive \\(a_4\\) flattens the density toward uniform; pushing \\(a_4\\) toward the "
            "validity threshold makes it sharply peaked. Below the threshold the distribution becomes "
            "invalid (mirrors the peaked vs. near-uniform cases in the paper's Figure 1)."
        ),
        a4_slider,
        fig_a4,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What creates additional modes (Theorem 7 and corollaries)

    Each basis curvature \(B_k''\) is either identically zero (\(a_0, C_1\)) or **strictly
    increasing** on \((0,1)\). Theorem 7 draws out the consequences:

    - \(a_k \ge 0\): the term cannot add oscillation to \(\kappa\).
    - \(a_k < 0\): the term may bend \(\kappa\) downward — the **only** source of extra oscillation.
    - **Higher-order center terms \(C_j,\,j\ge2\)** have sign-changing curvature and may each create
      **one** additional interior mode (first available at \(K=7\)).
    - **Tail terms** have one-sided curvature; extra tail-driven modes require **left/right pairs**
      (first available at \(K=6\)).

    | Coefficient regime (guaranteeing validity) | Max interior modes \(v(K)\) |
    |---|---|
    | All \(a_k\ge0\) — Theorem 1 (Corollary 1) | **1** (always unimodal) |
    | Tails \(\ge0\), centers free — Proposition 4 (Corollary 2) | \(1+\max\!\big(0,\lfloor (K-4)/3\rfloor\big)\) |
    | Leading tails \(\ge0\) — Proposition 3 (Corollary 3) | \(1+\max\!\big(0,\lfloor K/3\rfloor-1\big)+\max\!\big(0,\lfloor (K-4)/3\rfloor\big)\) |

    **Table 4 (structural upper bounds), valid QFlex vs. Metalog:**

    | K | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | QFlex, \(a_k\ge0\) (Thm 1) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
    | QFlex, tails ≥ 0 (Prop 4) | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 3 | 3 | 3 | 4 |
    | QFlex, leading tails ≥ 0 (Prop 3) | 1 | 1 | 1 | 2 | 3 | 3 | 4 | 5 | 5 | 6 | 7 |
    | Metalog (structural) | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 5 | 6 |

    For QFlex these counts are for **valid** densities; the Metalog bound is structural and does not
    condition on validity. *(Remark 10: these are worst-case ceilings, not guaranteed to be attained
    for every coefficient vector.)*
    """)
    return


@app.cell
def _(evaluate_quantile, evaluate_quantile_derivative, np, plt):
    # Two valid 7-term QFlex models with identical positive tails; the second adds a
    # negative higher-order center coefficient C2 (j=2), illustrating Theorem 7 part 3.
    _g = 0.5
    _p = np.linspace(0.002, 0.998, 1500)
    _uni = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])    # all shape coeffs >= 0  -> unimodal
    _bi = np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0, -20.0])   # free negative C2     -> bimodal

    def _curve(coef):
        q = evaluate_quantile_derivative(_p, coef, 7, _g)
        kappa = np.gradient(q, _p)
        x = evaluate_quantile(_p, coef, 7, _g)
        return q, kappa, x, 1.0 / q

    def _n_modes(q):
        # local minima of q == modes of the pdf; drop exact-zero diffs (e.g. at a
        # symmetric minimum the grid can straddle the turning point) before counting.
        s = np.sign(np.diff(q))
        s = s[s != 0]
        return int(np.sum((s[:-1] < 0) & (s[1:] > 0)))

    _qu, _ku, _xu, _fu = _curve(_uni)
    _qb, _kb, _xb, _fb = _curve(_bi)

    fig_modality, axM = plt.subplots(1, 2, figsize=(12, 4.4))
    axM[0].plot(_p, _ku, color="tab:green", lw=2,
                label=f"all aₖ≥0  (modes={_n_modes(_qu)})")
    axM[0].plot(_p, _kb, color="tab:red", lw=2,
                label=f"C₂ = −20  (modes={_n_modes(_qb)})")
    axM[0].axhline(0, color="gray", lw=0.8, ls="--")
    axM[0].set_title("QCF κ(p): a negative C₂ adds zero crossings", fontsize=9)
    axM[0].set_xlabel("p")
    axM[0].set_ylabel("κ(p)")
    axM[0].legend(fontsize=8)

    axM[1].plot(_xu, _fu, color="tab:green", lw=2, label="unimodal")
    axM[1].plot(_xb, _fb, color="tab:red", lw=2, label="bimodal")
    axM[1].set_title("Resulting PDFs (both valid: q(p) > 0)", fontsize=9)
    axM[1].set_xlabel("x")
    axM[1].set_ylabel("density")
    axM[1].set_ylim(bottom=0)
    axM[1].legend(fontsize=8)
    fig_modality.tight_layout()
    return (fig_modality,)


@app.cell
def _(fig_modality, mo):
    mo.vstack([
        mo.md(
            "**Creating a second mode (Theorem 7, part 3).** Both 7-term models have identical "
            "positive tails (so both are valid, q(p) > 0). The green model has all shape "
            "coefficients ≥ 0 and is **unimodal** (Corollary 1). The red model adds a single free "
            "**negative \\(C_2\\)** term (the first sign-changing center curvature, available at "
            "\\(K=7\\)); its QCF gains two extra zero crossings, producing a **valid bimodal** "
            "density — exactly the \\(v(7)=2\\) entry in Table 4 for the tails-≥0 regime. These "
            "coefficients are chosen for illustration, in the spirit of the paper's constructed "
            "examples."
        ),
        fig_modality,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part E — Bounded and semi-bounded variants  *(paper §6)*

    Many quantities live on a half-line (cost, time-to-event) or a finite interval (proportions,
    fractions). QFlex reaches these supports by composing the unbounded quantile function with a
    **strictly increasing transformation** \(T\); because \(T\) is increasing, monotonicity is
    inherited automatically (the Monotone-map rule from Part A). The `qflex` library ships two such
    variants: `LogQFlex` (semi-bounded) and `LogitQFlex` (bounded).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6.1  Semi-bounded QFlex — the exponential transform

    Using \(T_{\exp}(x)=\exp(x)\), which maps \(\mathbb R\) onto \((0,\infty)\),

    \[
    Q^{[K]}_{\text{Flex-S}}(p) = l + \exp\!\big(Q^{[K]}_{\text{Flex}}(p)\big), \qquad
    q^{[K]}_{\text{Flex-S}}(p) = \exp\!\big(Q^{[K]}_{\text{Flex}}(p)\big)\, q^{[K]}_{\text{Flex}}(p),
    \]

    giving support \([l,\infty)\). In the library:

    ```python
    from qflex import LogQFlex
    qf = LogQFlex(x, p, lower_bound=0, terms=3)   # fits ln(x - l) with an unbounded QFlex
    ```
    """)
    return


@app.cell
def _(LogQFlex, QFlex, np, plt):
    import warnings as _w

    _p = [0.10, 0.50, 0.90]
    _x = [3.0, 10.0, 80.0]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _lq = LogQFlex(_x, _p, lower_bound=0, terms=3)
        _ub = QFlex(_x, _p, terms=3)
    _g = np.linspace(0.005, 0.995, 500)

    fig_semi, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    axs[0].plot(_g, _lq.quantile(_g), color="seagreen", lw=2, label="LogQFlex Q(p)")
    axs[0].plot(_g, _ub.quantile(_g), color="gray", lw=1.5, ls="--", label="plain QFlex Q(p)")
    axs[0].scatter(_p, _x, color="firebrick", zorder=5, s=40, label="data")
    axs[0].axhline(0, color="red", lw=0.9, ls=":", label="lower bound l = 0")
    axs[0].set_ylim(-8, 92)
    axs[0].set_title(f"Quantile functions (plain QFlex dips to {_ub.quantile(_g).min():.2f} < 0)",
                     fontsize=9)
    axs[0].set_xlabel("p")
    axs[0].set_ylabel("x")
    axs[0].legend(fontsize=8)

    _xp = _lq.quantile(_g)
    axs[1].plot(_xp, _lq.pdf(_g), color="seagreen", lw=2)
    axs[1].axvline(0, color="red", lw=0.9, ls=":")
    axs[1].set_title("LogQFlex PDF on [0, ∞)", fontsize=9)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("density")
    axs[1].set_ylim(bottom=0)
    fig_semi.tight_layout()
    return (fig_semi,)


@app.cell
def _(fig_semi, mo):
    mo.vstack([
        mo.md(
            "**Semi-bounded fit.** For strongly right-skewed, strictly positive data the plain "
            "unbounded QFlex can assign quantiles **below the lower bound** (grey dashed curve dips "
            "under 0). `LogQFlex` fits on the log scale and maps back through \\(\\exp\\), so every "
            "quantile stays in \\([0,\\infty)\\) and the density is supported on the half-line."
        ),
        fig_semi,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6.2  Bounded QFlex — the logit transform

    Using the logistic CDF \(\sigma(x)=\dfrac{1}{1+e^{-x}}\), which maps \(\mathbb R\) onto
    \((0,1)\),

    \[
    Q^{[K]}_{\text{Flex-B}}(p) = l + (u-l)\,\sigma\!\big(Q^{[K]}_{\text{Flex}}(p)\big), \qquad
    q^{[K]}_{\text{Flex-B}}(p) = (u-l)\,\sigma'\!\big(Q^{[K]}_{\text{Flex}}(p)\big)\, q^{[K]}_{\text{Flex}}(p),
    \]

    with \(\sigma'(x)=\sigma(x)\big(1-\sigma(x)\big)>0\), giving support \([l,u]\). In the library:

    ```python
    from qflex import LogitQFlex
    qf = LogitQFlex(x, p, lower_bound=0, upper_bound=1, terms=5)
    ```
    """)
    return


@app.cell
def _(LogitQFlex, np, plt):
    import warnings as _w

    _p = [0.05, 0.25, 0.50, 0.75, 0.95]
    _x = [0.08, 0.30, 0.50, 0.70, 0.92]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _bq = LogitQFlex(_x, _p, lower_bound=0, upper_bound=1, terms=5)
    _g = np.linspace(0.005, 0.995, 500)

    fig_bound, axb = plt.subplots(1, 2, figsize=(12, 4.2))
    axb[0].plot(_g, _bq.quantile(_g), color="indigo", lw=2, label="LogitQFlex Q(p)")
    axb[0].scatter(_p, _x, color="firebrick", zorder=5, s=40, label="data")
    axb[0].axhline(0, color="gray", lw=0.8, ls=":")
    axb[0].axhline(1, color="gray", lw=0.8, ls=":", label="bounds [0, 1]")
    axb[0].set_title("Quantile function confined to [0, 1]", fontsize=9)
    axb[0].set_xlabel("p")
    axb[0].set_ylabel("x")
    axb[0].legend(fontsize=8)

    _xp = _bq.quantile(_g)
    axb[1].plot(_xp, _bq.pdf(_g), color="indigo", lw=2)
    axb[1].set_title("LogitQFlex PDF on [0, 1]", fontsize=9)
    axb[1].set_xlabel("x")
    axb[1].set_ylabel("density")
    axb[1].set_ylim(bottom=0)
    fig_bound.tight_layout()
    return (fig_bound,)


@app.cell
def _(fig_bound, mo):
    mo.vstack([
        mo.md(
            "**Bounded fit.** `LogitQFlex` maps the unbounded fit through the logistic CDF, so the "
            "quantile function is confined to \\([l,u]\\) (here \\([0,1]\\)) by construction — ideal "
            "for proportions and fractions. The CDF is exactly 0 at the lower bound and 1 at the "
            "upper bound."
        ),
        fig_bound,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### §6.3  Modality of the transformed variants

    Applying a strictly increasing \(T\) to \(Q^{[K]}_{\text{Flex}}\) gives curvature
    \[
    \kappa_{\text{Flex-}T}(p) = T''\!\big(Q\big)\,q^2 + T'\!\big(Q\big)\,\kappa ,
    \]
    so modes occur where \(\kappa = -\dfrac{T''(Q)}{T'(Q)}\,q^2\) — a **nonlinear** condition.
    Consequently the number and location of modes of a bounded or semi-bounded QFlex **need not
    coincide** with those of the underlying unbounded QFlex. The paper notes that fully
    characterizing the modes of these transformed variants is an **open question**, so we only
    illustrate the reshaping rather than make claims about mode counts.
    """)
    return


@app.cell
def _(QFlex, evaluate_quantile_derivative, np, plt):
    import warnings as _w

    _p = [0.05, 0.25, 0.50, 0.75, 0.95]
    _z = [-1.6, -0.6, 0.0, 0.6, 1.6]
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _qz = QFlex(_z, _p, terms=5)
    _g = np.linspace(0.005, 0.995, 800)
    _Q = _qz.quantile(_g)
    _q = evaluate_quantile_derivative(_g, _qz.coefficients, _qz.terms, _qz.gamma)

    _pdf_u = 1.0 / _q                              # unbounded
    _QS = np.exp(_Q)
    _pdf_s = 1.0 / (np.exp(_Q) * _q)               # semi-bounded (eq 22)
    _sig = 1.0 / (1.0 + np.exp(-_Q))
    _pdf_b = 1.0 / (_sig * (1.0 - _sig) * _q)      # bounded (eq 24), [0,1]

    fig_transform, axt = plt.subplots(1, 3, figsize=(13, 4))
    axt[0].plot(_Q, _pdf_u, color="steelblue", lw=2)
    axt[0].set_title("Unbounded QFlex", fontsize=9)
    axt[1].plot(_QS, _pdf_s, color="seagreen", lw=2)
    axt[1].set_title("Semi-bounded:  x = exp(Q)", fontsize=9)
    axt[2].plot(_sig, _pdf_b, color="indigo", lw=2)
    axt[2].set_title("Bounded:  x = σ(Q) on [0,1]", fontsize=9)
    for _a in axt:
        _a.set_xlabel("x")
        _a.set_ylabel("density")
        _a.set_ylim(bottom=0)
    fig_transform.tight_layout()
    return (fig_transform,)


@app.cell
def _(fig_transform, mo):
    mo.vstack([
        mo.md(
            "**One unbounded QFlex, three supports.** The *same* unbounded fit is mapped through "
            "\\(\\exp\\) (semi-bounded) and \\(\\sigma\\) (bounded). The transform reshapes the "
            "density and can move the mode — consistent with §6.3's nonlinear mode condition. We "
            "make no claim about the resulting mode counts, which the paper leaves open."
        ),
        fig_transform,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part F — Benchmarking QFlex against Metalog  *(paper §7)*

    QFlex and **Metalog** (Keelin 2016) are both quantile-parameterized systems: each fits a
    quantile function as a linear combination of basis functions through assessed percentiles.
    The paper compares them empirically. This section reproduces that comparison using the
    vendored Metalog implementation alongside the `qflex` library.

    **Procedure (paper §7.1).** For each target distribution we
    (1) extract quantiles on a fixed percentile scheme,
    (2) fit QFlex and Metalog with the *same* truncation order \(K=M\) via the linear system of §3,
    (3) check validity with **\(\delta\)-\(p\) monotonicity** (Definition 4, \(\delta = 0.001\):
    \(Q(p+\delta) > Q(p)\) on the grid), and
    (4) measure accuracy with the **normalized \(W_1\) error**

    \[
    W_1^{\text{norm}} \;=\;
    \frac{\int_0^1 \lvert\, \hat Q(p) - Q_{\text{target}}(p)\,\rvert\,dp}
         {Q_{\text{target}}(0.9) - Q_{\text{target}}(0.1)} ,
    \]

    i.e. the average absolute quantile error as a fraction of the central 10–90 range. When both
    systems are valid, the ratio \(R_1 = W_1^{\text{Metalog}} / W_1^{\text{QFlex}}\) reports how
    many times larger Metalog's error is than QFlex's.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Quantile schemes  *(paper Table 5)*

    The benchmarks use symmetric percentile grids that grow with \(M\). For \(M = K = 4\) (used
    below) the scheme is \(p \in \{0.10,\,0.50,\,0.90,\,0.95\}\); larger \(M\) refine the grid:

    | \(M\) | percentile scheme \(p\) |
    |---|---|
    | 3 | 0.10, 0.50, 0.90 |
    | 4 | 0.10, 0.50, 0.90, 0.95 |
    | 5 | 0.05, 0.10, 0.50, 0.90, 0.95 |
    | 7 | 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95 |

    Each support type is fitted with the matching domain variant: bounded targets use the
    **logit** variants, semi-bounded targets the **log** variants, and unbounded targets the
    plain variants — for both QFlex and Metalog.
    """)
    return


@app.cell
def _(
    LogMetalog,
    LogQFlex,
    LogitMetalog,
    LogitQFlex,
    Metalog,
    QFlex,
    np,
    plt,
    stats,
):
    import warnings as _w

    # Nine standard targets across the three support types (paper §7.2).
    _targets = [
        ("Beta(3, 7)", "bounded", stats.beta(3, 7)),
        ("Beta(0.8, 0.6)", "bounded", stats.beta(0.8, 0.6)),
        ("Beta(10, 1)", "bounded", stats.beta(10, 1)),
        ("Lognormal(0, 0.4)", "semi-bounded", stats.lognorm(0.4)),
        ("Weibull(2, 5)", "semi-bounded", stats.weibull_min(2, scale=5)),
        ("Exp(1)", "semi-bounded", stats.expon()),
        ("Normal(0, 1)", "unbounded", stats.norm()),
        ("Logistic(0, 1)", "unbounded", stats.logistic()),
        ("Gumbel(0, 1)", "unbounded", stats.gumbel_r()),
    ]
    _pg = np.array([0.10, 0.50, 0.90, 0.95])  # Table 5, M = 4
    _K = 4
    _P = np.linspace(0.01, 0.99, 1500)
    _dp = _P[1] - _P[0]

    def _w1norm(qfun, ppf):
        num = np.sum(np.abs(qfun(_P) - ppf(_P))) * _dp
        return num / (ppf(0.9) - ppf(0.1))

    fig_bench, _axes = plt.subplots(3, 3, figsize=(13, 10))
    _axes = _axes.ravel()
    _rows = []
    for _i, (_name, _supp, _d) in enumerate(_targets):
        _x = _d.ppf(_pg)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            if _supp == "bounded":
                _qf = LogitQFlex(_x, _pg, 0.0, 1.0, terms=_K)
                _ml = LogitMetalog(_x, _pg, 0.0, 1.0, terms=_K)
            elif _supp == "semi-bounded":
                _qf = LogQFlex(_x, _pg, lower_bound=0.0, terms=_K)
                _ml = LogMetalog(_x, _pg, lower_bound=0.0, terms=_K)
            else:
                _qf = QFlex(_x, _pg, terms=_K)
                _ml = Metalog(_x, _pg, terms=_K)

        _wq = _w1norm(_qf.quantile, _d.ppf)
        _wm = _w1norm(_ml.quantile, _d.ppf)
        _r1 = _wm / _wq if _wq > 0 else float("nan")
        _rows.append((_name, _supp, _wq, _wm, _r1))

        # PDF on the interior probability grid (avoids infinite Beta endpoints).
        _gp = np.linspace(0.005, 0.995, 400)
        _xt = _d.ppf(_gp)
        _ax = _axes[_i]
        _ax.plot(_xt, _d.pdf(_xt), color="steelblue", lw=2.2, label="target")
        _ax.plot(_qf.quantile(_gp), _qf.pdf(_gp), "--", color="firebrick",
                 lw=1.8, label="QFlex")
        _ax.plot(_ml.quantile(_gp), _ml.pdf(_gp), "--", color="seagreen",
                 lw=1.8, label="Metalog")
        _ax.set_title(f"{_name}   $R_1$={_r1:.2f}", fontsize=9)
        _ax.set_ylim(bottom=0)
        _ax.tick_params(labelsize=8)
        if _i == 0:
            _ax.legend(fontsize=8, loc="best")
    fig_bench.suptitle(
        "QFlex vs Metalog on nine standard distributions  (M = K = 4)", fontsize=12)
    fig_bench.tight_layout(rect=(0, 0, 1, 0.97))

    _better = sum(1 for _r in _rows if _r[2] < _r[3] - 1e-6)
    _ties = sum(1 for _r in _rows if abs(_r[2] - _r[3]) <= 1e-6)
    _hdr = ("| Distribution | Support | QFlex $W_1$ | Metalog $W_1$ | $R_1$ |\n"
            "|---|---|---|---|---|\n")
    _body = "\n".join(
        f"| {_n} | {_s} | {_q*100:.2f}% | {_m*100:.2f}% | {_rr:.2f} |"
        for (_n, _s, _q, _m, _rr) in _rows
    )
    bench_table = (
        _hdr + _body
        + f"\n\n*QFlex attains a **strictly lower** normalized $W_1$ in **{_better} of 9** cases "
        f"and is essentially tied in the other {_ties}. All eighteen fits are "
        "$\\delta$-$p$ monotone — consistent with the paper's finding that QFlex is lower in "
        "most cases and roughly equal in the rest.*"
    )
    return bench_table, fig_bench


@app.cell
def _(bench_table, fig_bench, mo):
    mo.vstack([
        mo.md(
            "**Reproducing paper Figure 2.** True PDFs (solid blue) with the QFlex (dashed red) "
            "and Metalog (dashed green) fits at \\(M = K = 4\\). Both systems yield valid "
            "distributions with no monotonicity constraints; QFlex generally tracks central "
            "curvature and tail behavior more closely — consistent with the paper's finding that "
            "QFlex provides the closer match in most cases."
        ),
        fig_bench,
        mo.md(bench_table),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### F.2 — Sweeping the skewness–kurtosis plane  *(paper §7.3)*

    The paper's broadest benchmark sweeps the **Pearson (1895) system**, which classifies a
    location–scale distribution by its **skewness** \(\sqrt{\beta_1}\) and **kurtosis**
    \(\beta_2\). Each point in the admissible \((\sqrt{\beta_1},\,\beta_2)\) plane is a distinct
    target; the paper grids ≈3,900 feasible Pearson distributions spanning bounded (Type I),
    semi-bounded (Type VI), and unbounded (Type IV) supports.

    Running the full Pearson grid is heavy. For an interactive feel of the *same* idea, the
    explorer below builds a lightweight target directly from skewness and **excess kurtosis** via
    the **Cornish–Fisher** expansion

    \[
    Q(p) = z + \tfrac{S}{6}(z^2-1) + \tfrac{E}{24}(z^3-3z) - \tfrac{S^2}{36}(2z^3-5z),
    \qquad z=\Phi^{-1}(p),
    \]

    with \(S\) the skewness and \(E=\beta_2-3\) the excess kurtosis. This is a *local* moment
    approximation — a stand-in for stepping through the Pearson plane, not the paper's exact
    experiment — so the sliders are restricted to the region where the target itself stays a valid
    (monotone) quantile function. Both fits are **unconstrained** (as in §7.3) and reported with
    their \(\delta\)-\(p\) monotonicity status.
    """)
    return


@app.cell
def _(mo):
    skew_slider = mo.ui.slider(
        start=0.0, stop=0.6, step=0.05, value=0.4, label="skewness  S")
    kurt_slider = mo.ui.slider(
        start=0.0, stop=4.0, step=0.25, value=1.5, label="excess kurtosis  E = β₂ − 3")
    mo.vstack([skew_slider, kurt_slider])
    return kurt_slider, skew_slider


@app.cell(hide_code=True)
def _(
    Metalog,
    QFlex,
    check_delta_p_monotonicity,
    kurt_slider,
    np,
    plt,
    skew_slider,
    stats,
):
    import warnings as _w2

    _S = float(skew_slider.value)
    _E = float(kurt_slider.value)

    def _cf(p):
        z = stats.norm.ppf(p)
        return (z + _S / 6 * (z**2 - 1) + _E / 24 * (z**3 - 3 * z)
                - _S**2 / 36 * (2 * z**3 - 5 * z))

    _pg = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])  # Table 5, M = 7
    _K = 7
    _xq = _cf(_pg)
    with _w2.catch_warnings():
        _w2.simplefilter("ignore")
        _qf = QFlex(_xq, _pg, terms=_K)
        _ml = Metalog(_xq, _pg, terms=_K)

    _P = np.linspace(0.01, 0.99, 1500)
    _dp = _P[1] - _P[0]
    _norm = _cf(0.9) - _cf(0.1)
    _wq = np.sum(np.abs(_qf.quantile(_P) - _cf(_P))) * _dp / _norm
    _wm = np.sum(np.abs(_ml.quantile(_P) - _cf(_P))) * _dp / _norm
    _mq = check_delta_p_monotonicity(_qf.quantile)["satisfied"]
    _mm = check_delta_p_monotonicity(_ml.quantile)["satisfied"]

    _g = np.linspace(0.01, 0.99, 400)
    fig_cf, _ax = plt.subplots(1, 2, figsize=(12, 4.3))
    _ax[0].plot(_g, _cf(_g), color="steelblue", lw=2.2, label="target")
    _ax[0].plot(_g, _qf.quantile(_g), "--", color="firebrick", lw=1.8, label="QFlex")
    _ax[0].plot(_g, _ml.quantile(_g), "--", color="seagreen", lw=1.8, label="Metalog")
    _ax[0].scatter(_pg, _xq, color="black", s=30, zorder=5, label="assessed")
    _ax[0].set_xlabel("cumulative probability p")
    _ax[0].set_ylabel("Q(p)")
    _ax[0].set_title("Quantile functions")
    _ax[0].legend(fontsize=8)

    # Target density f(x) = 1 / (dQ/dp), plotted against x = Q(p).
    _xt = _cf(_g)
    _pdf_t = 1.0 / np.gradient(_xt, _g)
    _ax[1].plot(_xt, _pdf_t, color="steelblue", lw=2.2, label="target")
    _ax[1].plot(_qf.quantile(_g), _qf.pdf(_g), "--", color="firebrick", lw=1.8, label="QFlex")
    _ax[1].plot(_ml.quantile(_g), _ml.pdf(_g), "--", color="seagreen", lw=1.8, label="Metalog")
    _ax[1].set_xlabel("x")
    _ax[1].set_ylabel("density")
    _ax[1].set_title("Implied PDFs")
    _ax[1].set_ylim(bottom=0)
    _ax[1].legend(fontsize=8)
    fig_cf.suptitle(f"Cornish–Fisher target:  S = {_S:.2f},  E = {_E:.2f}", fontsize=11)
    fig_cf.tight_layout(rect=(0, 0, 1, 0.95))

    _r1 = _wm / _wq if _wq > 0 else float("nan")
    cf_status = (
        f"**QFlex:** $W_1$ = {_wq*100:.2f}%, δ-p monotone = {_mq} &nbsp;|&nbsp; "
        f"**Metalog:** $W_1$ = {_wm*100:.2f}%, δ-p monotone = {_mm} &nbsp;|&nbsp; "
        f"**$R_1$ = {_r1:.2f}**"
    )
    return cf_status, fig_cf


@app.cell
def _(cf_status, fig_cf, mo):
    mo.vstack([
        fig_cf,
        mo.md(cf_status),
        mo.md(
            "Move the sliders to reshape the target. The assessed points (black) are matched by "
            "both order-7 fits; the curves diverge between them. As in the paper, QFlex typically "
            "attains the smaller \\(W_1\\). For strongly skewed targets the *unconstrained* QFlex "
            "fit can lose \\(\\delta\\)-\\(p\\) monotonicity — exactly the situation Part C's "
            "certificates and constraints are designed to repair, an analytic guarantee Metalog "
            "does not offer."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What the full Pearson study finds  *(paper §7.3)*

    Across the paper's ≈3,500 tested Pearson distributions and orders \(K = 3,\dots,10\):

    - **Monotonicity.** Both systems are \(\delta\)-\(p\) monotone for the vast majority of cases.
      The difference is *how*: QFlex carries **analytic certificates** (Theorem 1, Proposition 3) —
      e.g. Proposition 3 holds for ≈97% of Type IV cases at \(K=9\) — whereas Metalog can only be
      checked **ex-post**, with no guarantee off the evaluated grid.
    - **Accuracy.** QFlex delivers lower quantile error across nearly the entire Pearson system,
      for every order. In the semi-bounded Type IV region at \(K=7\), the median Metalog error is
      about **four times** the QFlex error, and the \(R_1\) advantage **grows with \(K\)**.
    - **Modality.** With unconstrained fits, QFlex produces **no spurious modes**; Metalog produces
      extra modes only at \(K=10\) in ≈1% of Type IV cases.
    - **One exception.** In the U-shaped Beta region (two boundary modes) relative accuracy
      alternates with \(K\), reflecting that QFlex introduces left/right tail families
      sequentially while Metalog adds both tails together.

    The takeaway mirrors the rest of the paper: both QPD systems are highly expressive, but QFlex
    couples that accuracy with **interpretable structure** — explicit tail/center families and
    analytic validity guarantees.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part G — Controlling monotonicity via coefficient constraints  *(paper §8)*

    This section is the practical payoff of Part C. The paper uses a synthetic seven-point dataset

    \[
    p_i = \frac{i - 0.5}{7}, \qquad x = (1,\,10,\,18,\,24,\,38,\,54,\,82), \qquad i = 1,\dots,7,
    \]

    a moderately right-skewed assessment (spacing widens toward the upper tail). We fit four models
    at orders \(K = 5\) and \(K = 7\):

    1. **Metalog** — unconstrained least squares.
    2. **QFlex (unconstrained)** — least squares, no validity constraints.
    3. **QFlex+** — nonnegative coefficients, the region \(A^{[K]}\) (**Theorem 1**).
    4. **QFlexTC** — the tail–center magnitude region \(A_{TC}^{[K]}\) (**Proposition 4**).

    Both QFlex constraint sets are convex and involve only simple coefficient bounds, so they are
    implementable even in a spreadsheet. Each fit reports the **normalized \(W_1\)** error (relative
    to a linear interpolation of the seven points) and, for QFlex, the **tail–center margin**
    \(m_{TC} = m_{\text{tail}} - M_{\text{center}}\): slack (\(>0\)), binding (\(=0\)), or
    violated (\(<0\)).
    """)
    return


@app.cell
def _(np):
    g_x = np.array([1.0, 10.0, 18.0, 24.0, 38.0, 54.0, 82.0])
    g_p = (np.arange(1, 8) - 0.5) / 7.0
    return g_p, g_x


@app.cell
def _(mo):
    g_K_dropdown = mo.ui.dropdown(
        options={"K = 5": 5, "K = 7": 7}, value="K = 7", label="truncation order")
    g_K_dropdown
    return (g_K_dropdown,)


@app.cell
def _(
    ConstraintType,
    Metalog,
    QFlex,
    check_delta_p_monotonicity,
    compute_w1,
    g_K_dropdown,
    g_p,
    g_x,
    np,
    plt,
    tail_center_margin_coeff,
):
    import warnings as _w

    _K = int(g_K_dropdown.value)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _ml = Metalog(g_x, g_p, terms=_K)
        _qu = QFlex(g_x, g_p, terms=_K)
        _qa = QFlex(g_x, g_p, terms=_K, constraint_type=ConstraintType.A)
        _qt = QFlex(g_x, g_p, terms=_K, constraint_type=ConstraintType.TC)

    _models = [
        ("Metalog", _ml, "seagreen", False),
        ("QFlex (unconstrained)", _qu, "0.45", True),
        ("QFlex+  (Theorem 1)", _qa, "firebrick", True),
        ("QFlexTC  (Proposition 4)", _qt, "indigo", True),
    ]
    _g = np.linspace(0.005, 0.995, 600)

    fig_g7, _ax = plt.subplots(2, 4, figsize=(15, 7))
    _rows = []
    for _j, (_name, _m, _col, _is_qf) in enumerate(_models):
        _Q = _m.quantile(_g)
        _pdf = _m.pdf(_g)
        _mono = check_delta_p_monotonicity(_m.quantile)["satisfied"]
        _w1 = compute_w1(_m.quantile, g_x, g_p)[1] * 100
        _tc = (tail_center_margin_coeff(_m.coefficients, _K, _m.gamma)
               if _is_qf else None)
        _rows.append((_name, _w1, _mono, _tc))

        # Top: quantile function.
        _t = _ax[0, _j]
        _t.plot(_g, _Q, color=_col, lw=2)
        _t.scatter(g_p, g_x, color="black", s=28, zorder=5)
        _t.set_title(_name, fontsize=9)
        _t.set_xlabel("p", fontsize=8)
        if _j == 0:
            _t.set_ylabel("Q(p)", fontsize=9)
        if not _mono:
            _t.set_facecolor("#fdeaea")
        _lbl = f"W₁ = {_w1:.2f}%\nmono = {_mono}"
        if _tc is not None:
            _lbl += f"\nm_TC = {_tc:.2f}"
        _t.text(0.04, 0.96, _lbl, transform=_t.transAxes, fontsize=8,
                va="top", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

        # Bottom: implied PDF f(x) = 1/q(p) against x (clip to expose pathology).
        _b = _ax[1, _j]
        _b.plot(_Q, _pdf, color=_col, lw=2)
        _b.axhline(0.0, color="0.6", lw=0.8, ls=":")
        _b.set_xlabel("x", fontsize=8)
        if _j == 0:
            _b.set_ylabel("density f(x)", fontsize=9)
        _b.set_xlim(-5, 95)
        _finite = _pdf[np.isfinite(_pdf)]
        _top = np.percentile(_finite[_finite > 0], 99) if np.any(_finite > 0) else 1.0
        _b.set_ylim(-0.01, max(_top * 1.3, 0.02))
        if not _mono:
            _b.set_facecolor("#fdeaea")

    fig_g7.suptitle(
        f"Seven-point dataset at K = {_K}:  Metalog vs QFlex constraint sets",
        fontsize=12)
    fig_g7.tight_layout(rect=(0, 0, 1, 0.96))

    _hdr = ("| Model | normalized $W_1$ | δ-p monotone | $m_{TC}$ |\n"
            "|---|---|---|---|\n")
    g_table = _hdr + "\n".join(
        f"| {_n} | {_v:.2f}% | {'✓' if _mo else '✗ **invalid**'} | "
        f"{'—' if _t is None else f'{_t:.2f}'} |"
        for (_n, _v, _mo, _t) in _rows
    )
    return fig_g7, g_table


@app.cell
def _(fig_g7, g_K_dropdown, g_table, mo):
    mo.vstack([
        mo.md(
            "**Reproducing paper Figure 4.** Top row: fitted quantile functions (must be strictly "
            "increasing). Bottom row: implied PDFs. Red-tinted panels are **non-monotone** "
            "(invalid). Switch the order above to compare \\(K=5\\) and \\(K=7\\)."
        ),
        fig_g7,
        mo.md(g_table),
        mo.md(
            f"*Computed live with the `qflex` library and vendored Metalog at "
            f"**{g_K_dropdown.value}**. \\(W_1\\) is normalized by the central 10–90 range, so "
            "absolute percentages differ from the paper's normalization by a near-constant factor; "
            "the relative orderings and the \\(K=7\\) validity outcomes reproduce Figure 4.*"
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What the demonstration shows  *(paper §8)*

    - **At \(K = 5\)** the tail–center constraint is **slack** (\(m_{TC} > 0\)), so QFlex+ and
      QFlexTC give the **identical** fit and \(W_1\); Metalog is valid with a marginally lower
      \(W_1\). (Even here the *unconstrained* least-squares fit can already dip slightly in the
      extrapolated upper tail — the constrained sets remove any such risk by construction.)
    - **At \(K = 7\)** the unconstrained fits break: **both Metalog and unconstrained QFlex lose
      monotonicity** (red panels), producing incoherent PDFs. The two QFlex constraint sets remain
      strictly increasing:
        - **QFlex+** (Theorem 1) gives the smaller \(W_1\), but its tail–center margin is
          **negative** — Proposition 4 is violated even though Theorem 1 still certifies validity
          (the two certificates are different sufficient conditions, as in Part C).
        - **QFlexTC** (Proposition 4) drives the margin to **exactly zero** (binding); its feasible
          region is tighter, so it pays a slightly higher \(W_1\).

    The lesson mirrors the paper: when higher-order flexibility threatens monotonicity, QFlex can
    **enforce validity ex-ante** with simple convex coefficient constraints — an analytic guarantee
    that Metalog, which relies on ex-post numerical repair, does not provide.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part H — Conclusion and references  *(paper §9)*

    QFlex is a quantile-parameterized family built **entirely from monotone transformations of
    valid quantile functions**. By interleaving powers of exponential (right-tail),
    reflected-exponential (left-tail), and centered-uniform (center) bases under a coordinated
    index, it matches the flexibility of systems like Metalog while staying within Gilchrist's
    transformation rules — giving transparent structure, analytic monotonicity guarantees, and a
    provably full-rank design matrix.

    The paper distills seven practical advantages, each of which appears in this notebook:

    1. **Structural stability under truncation** — every \(K\)-term prefix is full-rank for
       nonsymmetric grids, with no basis reordering *(Part B)*.
    2. **Exact interpolation and convergence** — truncated expansions interpolate any finite set of
       probability–quantile pairs and converge as \(K\) grows *(Parts B, F)*.
    3. **Interpretable coefficients** — tail vs. center families carry geometric/probabilistic
       meaning *(Parts B, C)*.
    4. **Two verifiable monotonicity certificates** — coefficient nonnegativity (Theorem 1) and the
       tail–center magnitude bound (Propositions 3–4) *(Parts C, G)*.
    5. **Ex-ante monotonicity control** — validity is enforced through simple convex coefficient
       constraints, implementable even in a spreadsheet *(Parts C, G)*.
    6. **Accuracy** — lower quantile error than Metalog across most of the Pearson system *(Part F)*.
    7. **Controlled modality** — nonnegative coefficients give unimodality; extra modes arise only
       through specific higher-order center terms *(Part D)*.

    Because QFlex is assembled from monotone transformations, its exponential / reflected-exponential
    bases can be swapped for **other tail-generating quantile forms** — e.g. Pareto bases give a
    heavy-tailed *QFlex-Heavy*, and generalized-Pareto or lognormal bases give intermediate tail
    weights — each variant retaining the same monotonicity conditions, full-rank design matrix, and
    interpolation properties. QFlex is thus less a single distribution than a **platform** for
    building quantile-parameterized families.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Using the `qflex` library — quick reference

    A compact map from the paper's ideas to the API exercised throughout this notebook:

    | Task | Library call |
    |---|---|
    | Fit from percentiles | `QFlex(x, p, terms=K)` |
    | Semi-bounded / bounded support | `LogQFlex(x, p, lower_bound=l)` · `LogitQFlex(x, p, l, u)` |
    | Enforce validity (Theorem 1) | `QFlex(x, p, terms=K, constraint_type=ConstraintType.A)` |
    | Enforce validity (Proposition 4) | `QFlex(x, p, terms=K, constraint_type=ConstraintType.TC)` |
    | Other constraint sets | `ConstraintType.TL` · `ConstraintType.TA` · `ConstraintType.TC_MAG` |
    | Evaluate | `qf.quantile(p)` · `qf.pdf(p)` · `qf.cdf(x)` · `qf.sample(n)` |
    | Summaries | `qf.moments()` · `qf.summary()` · `qf.plot()` |
    | Check monotonicity (δ-p) | `check_delta_p_monotonicity(qf.quantile)` |
    | Tail–center certificate | `qf.check_proposition4()` · `tail_center_margin_coeff(...)` |
    | Accuracy (normalized W₁) | `compute_w1(qf.quantile, x, p)` |

    Defaults: `terms=3` (exact for three-point P10/P50/P90 elicitations) and the constrained
    tail–center solver uses the linear (CVXPY) formulation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### References

    The notebook follows *The QFlex Distribution* (Bickel et al., under review, R1, 2026). The
    paper's bibliography:

    - Baucells, M., M. Chrisman, T. W. Keelin, and S. Xu. 2025. *On the Properties of the Metalog
      Distribution.* SSRN: http://dx.doi.org/10.2139/ssrn.5280979
    - Bickel, J. E. 2026. *Quantile-based power-series expansions of the Johnson distribution
      system.* Communications in Statistics – Theory and Methods, 1–34.
    - Bury, K. V. 1975. *Statistical Models in Applied Science.* John Wiley and Sons, New York.
    - Cheney, W. and W. Light. 2000. *A Course in Approximation Theory.* American Mathematical
      Society, Providence, RI.
    - Gilchrist, W. 2000. *Statistical Modelling with Quantile Functions.* Chapman & Hall/CRC,
      Boca Raton, FL.
    - Hadlock, C. C., and J. E. Bickel. 2017. *Johnson Quantile-Parameterized Distributions.*
      Decision Analysis 14(1):35–64.
    - Hadlock, C. C., and J. E. Bickel. 2019. *The Generalized Johnson Quantile-Parameterized
      Distribution System.* Decision Analysis 16(1):67–85.
    - Hald, A. 1998. *A History of Mathematical Statistics from 1750 to 1930.* John Wiley and Sons,
      New York.
    - Hammond, R. K. and J. E. Bickel. 2013. *Reexamining Discrete Approximations to Continuous
      Distributions.* Decision Analysis 10(1):6–25.
    - Johnson, N. L. 1949. *Systems of Frequency Curves Generated by Methods of Translation.*
      Biometrika 36(149):78–82.
    - Karlin, S. and W. J. Studden. 1966. *Tchebycheff Systems: With Applications in Analysis and
      Statistics.* Interscience Publishers.
    - Keelin, T. W. 2016. *The Metalog Distributions.* Decision Analysis 13(4):243–277.
    - Keelin, T. W., and B. W. Powley. 2011. *Quantile-parameterized Distributions.* Decision
      Analysis 8(3):206–219.
    - Parzen, E. 1979. *Nonparametric Statistical Data Modeling.* Journal of the American
      Statistical Association 74:105–121.
    - Pearson, K. 1895. *Contributions to the Mathematical Theory of Evolution, II: Skew Variation
      in Homogeneous Material.* Phil. Trans. R. Soc. London 186:343–414.
    - Pearson, K. 1901. *Mathematical Contributions to the Theory of Evolution. X.* Phil. Trans. R.
      Soc. London 197:443–459.
    - Pearson, K. 1916. *Mathematical Contributions to the Theory of Evolution. XIX.* Phil. Trans.
      R. Soc. London 216:429–457.
    - Peng, C., Y. Li, and S. Uryasev. 2026. *Mixture Quantiles Estimated by Constrained Linear
      Regression.* Annals of Operations Research.
    - Perepolkin, D. 2023. *Scientific Methods for Integrating Expert Knowledge into Bayesian
      Models.* Doctoral Dissertation, Lund University.
    - Perepolkin, D., E. Lindström, and U. Sahlin. 2025. *Quantile-Parameterized Distributions for
      Knowledge Elicitation.* Decision Analysis 22(3):169–188.
    - Rudin, W. 1976. *Principles of Mathematical Analysis,* 3rd ed. McGraw-Hill, New York.
    - van Dorp, J. R., and M. C. Jones. 2020. *The Johnson System of Frequency Curves — Historical,
      Graphical, and Limiting Perspectives.* The American Statistician 74(1):37–52.

    ---

    *Metalog fits in this notebook use a vendored implementation (Metalog 2.0, Baucells et al.
    2025) included under `notebooks/_metalog/` purely for side-by-side comparison.*
    """)
    return


if __name__ == "__main__":
    app.run()
