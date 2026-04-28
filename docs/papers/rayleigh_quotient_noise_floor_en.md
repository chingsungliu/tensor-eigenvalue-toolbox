# Rayleigh Quotient Noise Floor

**Numerical Behavior of the NNI Algorithm Under Near-Zero Eigenvector Components**

- **Author:** Ching-Sung Liu
- **Date:** April 2026
- **Version:** v0.1 (research note)

---

## §1 Background and Motivation

### 1.1 Problem Setting

Consider an $m$-order $n$-dimensional nonnegative tensor $\mathcal{A} \in \mathbb{R}_{+}^{n \times n \times \cdots \times n}$ (with $m$ modes). We seek the **largest H-eigenvalue** $\lambda_{\max}$ and the corresponding positive eigenvector $x \in \mathbb{R}_{+}^{n}$ satisfying

$$
\mathcal{A} x^{m-1} = \lambda \cdot x^{[m-1]},
\tag{1.1}
$$

where

- $(\mathcal{A} x^{m-1})_i := \sum_{i_2, \ldots, i_m} a_{i, i_2, \ldots, i_m} \cdot x_{i_2} x_{i_3} \cdots x_{i_m}$ (the tensor-vector product yielding an $n$-dimensional vector);
- $x^{[m-1]} := (x_1^{m-1}, x_2^{m-1}, \ldots, x_n^{m-1})^\top$ (element-wise $(m-1)$-th power).

When $\mathcal{A}$ is a positive M-tensor, Perron-Frobenius-type results guarantee the existence of a unique positive eigenvector associated with the largest eigenvalue. This is the standing assumption throughout this note.

### 1.2 The NNI Algorithm

Newton-Noda Iteration (NNI), introduced for nonnegative tensors in Liu, Guo and Lin (Numer. Math. 2017) and extended in subsequent work, builds on the Newton iteration framework of Noda (1971) for nonnegative matrices. Its key idea is to recast (1.1) as a nonlinear system,

$$
F(x, \lambda) := \mathcal{A} x^{m-1} - \lambda \cdot x^{[m-1]} = 0,
\tag{1.2}
$$

perform Newton iteration on $x$, and update $\lambda$ via a Rayleigh-quotient-style estimate. The full derivation is given in §2.

### 1.3 Motivation of this Note

In the course of porting NNI from MATLAB to Python, the author observed a non-trivial numerical phenomenon: the two implementations remain bit-identical (to machine epsilon) for the first $N$ iterations, but diverge in their stopping behavior afterwards, terminating at different iteration counts. Nonetheless, the final eigenvalue estimates agree within the prescribed tolerance.

A detailed investigation revealed that this divergence is *not* due to implementation-level differences (e.g., sparse vs. dense LU pivot ordering), but rather reflects an **inherent property of the algorithm**: when the eigenvector has a **near-zero component**, the Rayleigh quotient estimate enters a *floating-point-dominated noise floor*. The upper bound of this floor admits an explicit expression in terms of $\varepsilon_{\text{machine}}$ and $\min_i x_i$, and it has practical implications for the choice of stopping tolerance.

The goals of this note are:

1. To document the complete algorithmic architecture of NNI (§2);
2. To quantitatively describe the noise floor phenomenon (§3);
3. To derive the upper bound of the noise floor (§4);
4. To provide practical recommendations for tolerance selection (§5).

---

## §2 The NNI Algorithm: Complete Architecture

### 2.1 From Eigen-Equation to Nonlinear System

Rewriting (1.1) as a residual,

$$
F(x, \lambda) = \mathcal{A} x^{m-1} - \lambda \cdot x^{[m-1]} = 0.
$$

This yields $n$ equations in $n+1$ unknowns (the $n$ components of $x$ plus $\lambda$). To close the system, we impose the normalization $\lVert x \rVert = 1$.

### 2.2 Derivation of the Jacobian

Taking the derivative of $F$ with respect to $x$,

$$
J(x, \lambda) := \frac{\partial F}{\partial x} = \frac{\partial (\mathcal{A} x^{m-1})}{\partial x} - \lambda \cdot \frac{\partial x^{[m-1]}}{\partial x}.
$$

**First term.** Let $B(x) := \partial (\mathcal{A} x^{m-1}) / \partial x$. Differentiating the tensor-vector product entry-wise gives

$$
B(x)_{ij} = (m-1) \sum_{i_3, \ldots, i_m} a_{i, j, i_3, \ldots, i_m} \cdot x_{i_3} \cdots x_{i_m}.
\tag{2.1}
$$

In our implementation, this is computed by the routine `sp_Jaco_Ax(A, x, m)`, which returns an $n \times n$ sparse matrix.

**Second term.** Since $\partial x_i^{m-1} / \partial x_j = (m-1) x_i^{m-2} \cdot \delta_{ij}$, we obtain a diagonal matrix:

$$
\frac{\partial x^{[m-1]}}{\partial x} = (m-1) \cdot \mathrm{diag}(x^{[m-2]}).
\tag{2.2}
$$

Combining,

$$
J(x, \lambda) = B(x) - (m-1) \lambda \cdot \mathrm{diag}(x^{[m-2]}) =: M(x, \lambda).
\tag{2.3}
$$

The matrix $M$ is the linear system operator to be inverted at each NNI iteration.

### 2.3 Newton Update: Full Step

At iteration $k$, given the current estimate $(x^{(k)}, \lambda_U^{(k)})$, the algorithm executes:

**Step 1 (Linear solve):**

$$
(-M(x^{(k)}, \lambda_U^{(k)})) \cdot y^{(k)} = (x^{(k)})^{[m-1]}.
\tag{2.4}
$$

The negative sign arises from rearranging the Newton step $\Delta x = -J^{-1} F$.

**Step 2 (Update and normalize):**

$$
\tilde{x}^{(k+1)} = (m-2) \cdot x^{(k)} + y^{(k)}, \qquad x^{(k+1)} = \frac{\tilde{x}^{(k+1)}}{\lVert \tilde{x}^{(k+1)} \rVert}.
\tag{2.5}
$$

Note that the coefficient $(m-2)$ degenerates to $1$ for $m=3$ (giving $x+y$) and to $0$ for $m=2$ (giving $y$ alone). No special-case handling is required.

**Step 3 (Rayleigh quotient estimate):**

Define $t^{(k+1)} \in \mathbb{R}^n$ as the vector of entry-wise local eigenvalue estimates:

$$
t_i^{(k+1)} := \frac{(\mathcal{A} (x^{(k+1)})^{m-1})_i}{(x_i^{(k+1)})^{m-1}}, \qquad i = 1, \ldots, n.
\tag{2.6}
$$

*Intuition:* if $x^{(k+1)}$ is a true eigenvector, then all $t_i$ coincide and equal $\lambda$.

We then set

$$
\lambda_U^{(k+1)} := \max_i t_i^{(k+1)}, \qquad \lambda_L^{(k+1)} := \min_i t_i^{(k+1)}.
\tag{2.7}
$$

By Collatz-type inequalities (valid when $\mathcal{A}$ is a nonnegative irreducible tensor), $\lambda_U^{(k+1)}$ and $\lambda_L^{(k+1)}$ are upper and lower bounds of the true eigenvalue $\lambda_{\max}$.

**Step 4 (Residual and stopping):**

$$
\mathrm{res}^{(k+1)} := \frac{\lambda_U^{(k+1)} - \lambda_L^{(k+1)}}{\lambda_U^{(k+1)}}.
\tag{2.8}
$$

The algorithm terminates when $\min_{j \leq k+1} \mathrm{res}^{(j)} \leq \mathrm{tol}$ or when $k+1 \geq \mathrm{maxit}$.

### 2.4 Pseudocode

```
Algorithm NNI(A, m, tol, maxit, x_0):
  x ← x_0 / ||x_0||
  compute temp = (A x^{m-1}) ./ x^{[m-1]}
  λ_U ← max(temp);  λ_L ← min(temp)
  k ← 0

  while (λ_U - λ_L)/λ_U > tol and k < maxit:
    k ← k + 1
    B ← sp_Jaco_Ax(A, x, m)
    M ← B - (m-1) λ_U · diag(x^{[m-2]})
    solve (-M) y = x^{[m-1]}
    x_new ← (m-2) x + y
    x ← x_new / ||x_new||
    temp ← (A x^{m-1}) ./ x^{[m-1]}
    λ_U ← max(temp);  λ_L ← min(temp)

  return (λ_U, x, k, λ_L, res_history, λ_U_history)
```

### 2.5 Perron-Frobenius Guarantees

When $\mathcal{A}$ is a positive M-tensor and $x_0 > 0$, Collatz-type theorems ensure:

1. A unique positive eigenvector exists corresponding to $\lambda_{\max}$.
2. The Newton iteration, starting from a positive $x_0$, preserves positivity of $x^{(k)}$ in exact arithmetic.
3. The bounds $\lambda_L^{(k)} \leq \lambda_{\max} \leq \lambda_U^{(k)}$ hold for all $k$, with both sequences converging monotonically to $\lambda_{\max}$.

No active nonnegativity projection is performed by the algorithm: NNI does not enforce $x \geq 0$ explicitly. If the input is not a positive M-tensor, the algorithm may diverge or produce $\lambda_L < 0$; in the latter case $\mathrm{res} > 1$, indicating that the input violates the standing assumption.

---

## §3 The Noise Floor Phenomenon: Observation and Quantification

### 3.1 Experimental Setup

To verify port correctness, we designed a parity test comparing MATLAB and Python implementations under identical inputs, iteration by iteration.

**Test case (Q7):**

- $n = 20$, $m = 3$;
- Random seed $\mathrm{rng}(42)$;
- Diagonal dominance: $d_i \in [1, 11]$ uniformly random;
- Perturbation: sparse random, density 0.02, magnitude 0.01;
- $\mathcal{A} = \mathrm{sp\_tendiag}(d, m) + \mathrm{pert}$;
- $x_0 = \lvert \mathrm{rand}(n) \rvert + 0.1$;
- $\mathrm{tol} = 10^{-12}$, $\mathrm{maxit} = 200$.

MATLAB uses `\` (backslash, sparse LU via UMFPACK). Python uses `scipy.sparse.linalg.spsolve` (also UMFPACK-based).

### 3.2 Observation: Bit-Identical Prefix, Divergent Suffix

**First 20 iterations ($k = 0$ through $k = 19$):**

All per-iteration intermediate quantities ($x^{(k)}$, $w^{(k)} := x^{(k)[m-1]}$, $y^{(k)}$, $\lambda_U^{(k)}$, $\lambda_L^{(k)}$, $\mathrm{res}^{(k)}$) agree between MATLAB and Python to within $10^{-15}$ in absolute error (i.e., machine epsilon).

For instance, the per-iteration maximum absolute error in $x^{(k)}$:

| iter $k$ | $\max_i \lvert x^{(k)}_{i,\text{ML}} - x^{(k)}_{i,\text{PY}} \rvert$ |
|:-:|:-:|
| 1  | $2.9 \times 10^{-16}$ |
| 5  | $4.8 \times 10^{-16}$ |
| 10 | $6.1 \times 10^{-16}$ |
| 15 | $7.5 \times 10^{-16}$ |
| 19 | $7.9 \times 10^{-16}$ |

This is entirely consistent with floating-point round-off accumulation, with no algorithmic discrepancy.

**Starting at iteration 20: divergence in stopping behavior**

| iter $k$ | MATLAB $\mathrm{res}$ | Python $\mathrm{res}$ | Stopping decision |
|:-:|:-:|:-:|:--|
| 20 | $5.2 \times 10^{-11}$ | $5.2 \times 10^{-11}$ | Neither stops |
| 21 | $1.3 \times 10^{-12}$ | $1.1 \times 10^{-14}$ | Python stops; MATLAB continues |
| 22 | $7.7 \times 10^{-12}$ | — | — |
| 23 | $1.9 \times 10^{-12}$ | — | — |
| 24 | $7.5 \times 10^{-12}$ | — | — |
| 25 | $3.2 \times 10^{-12}$ | — | — |
| 26 | $6.9 \times 10^{-13}$ | — | MATLAB stops |

Key observations:

- At iteration 21, $\mathrm{res}$ differs by a factor of 100 between the two, yet at iteration 20 the difference is only 1%;
- A 100-fold amplification within a single iteration is dramatic and demands explanation;
- MATLAB iterations 22–26 exhibit $\mathrm{res}$ that does not decrease monotonically, but oscillates within a band of about $10^{-12}$ — a strong indication that the algorithm has lost its convergent behavior and entered a plateau regime.

### 3.3 Final Results Are Equivalent

Despite stopping at different iterations, the final eigenvalue and eigenvector agree within tolerance:

| Quantity | MATLAB | Python | Difference |
|:--|:--|:--|:--|
| Final $\lambda_U$ | $10.69910094153116$ | $10.69910094153081$ | $3.5 \times 10^{-13}$ |
| Final $\lambda_L$ | $10.69910094152374$ | (same) | — |
| $\lVert x_{\text{ML}} - x_{\text{PY}} \rVert$ | — | — | $4.9 \times 10^{-17}$ |

Since $\lvert \lambda_{\text{ML}} - \lambda_{\text{PY}} \rvert = 3.5 \times 10^{-13} < \mathrm{tol} = 10^{-12}$, the two implementations are equivalent within the prescribed accuracy.

### 3.4 Ruling Out an Alternative Hypothesis

**Hypothesis A:** sparse vs. dense LU pivot divergence. One might speculate that MATLAB's `\` and scipy's `spsolve` — despite both being UMFPACK-based — exhibit different pivot orderings when applied to near-singular systems, causing a divergence in the computed $y^{(k)}$.

**Verification:** If Hypothesis A held, one would expect $y^{(k)}$ to show a pronounced discrepancy starting at iteration 20. Empirically:

| iter $k$ | $\max_i \lvert y^{(k)}_{i,\text{ML}} - y^{(k)}_{i,\text{PY}} \rvert$ | $\lVert y^{(k)}_{\text{ML}} \rVert_\infty$ | Relative error |
|:-:|:-:|:-:|:-:|
| 18 | $4.6 \times 10^{-16}$ | $0.9995$ | $4.6 \times 10^{-16}$ |
| 19 | $1.3 \times 10^{-15}$ | $0.9995$ | $1.3 \times 10^{-15}$ |
| 20 | $9.2 \times 10^{-16}$ | $0.9995$ | $9.2 \times 10^{-16}$ |
| 21 | $1.2 \times 10^{-15}$ | $0.9995$ | $1.2 \times 10^{-15}$ |

At iterations 20 and 21, $y^{(k)}$ remains bit-identical to machine epsilon — Hypothesis A is falsified. The solutions of the linear system are identical between the two implementations.

The divergence must therefore occur in a later step. We identify this step as the element-wise division in (2.6), namely the Rayleigh quotient estimate. This is the subject of §4.

---

## §4 Root Cause: Floating-Point Sensitivity of the Rayleigh Quotient

### 4.1 Pinpointing the Divergence

Recall the computational flow of each NNI iteration (Steps 1–4 in §2.3):

1. $y^{(k)}$ from the linear solve — verified bit-identical (§3.4);
2. $x^{(k+1)}$ from linear combination and normalization — floating-point accumulation is controlled;
3. $t^{(k+1)} = (\mathcal{A} (x^{(k+1)})^{m-1}) \,./\, (x^{(k+1)})^{[m-1]}$ — **suspect**;
4. $\lambda_U^{(k+1)} = \max t^{(k+1)}$, $\lambda_L^{(k+1)} = \min t^{(k+1)}$;
5. $\mathrm{res}^{(k+1)} = (\lambda_U - \lambda_L) / \lambda_U$.

Step 3 involves component-wise division, a classical source of floating-point error amplification.

### 4.2 Floating-Point Error Analysis (Semi-Formal)

Adopt the standard IEEE 754 floating-point model. For any floating-point operation $\mathrm{op}(a, b)$,

$$
\mathrm{fl}(\mathrm{op}(a, b)) = \mathrm{op}(a, b) \cdot (1 + \delta), \qquad \lvert \delta \rvert \leq \varepsilon_{\text{machine}},
\tag{4.1}
$$

where $\varepsilon_{\text{machine}} \approx 2.22 \times 10^{-16}$ in double precision.

**Analysis of Step 3.** Compute $t_i = (\mathcal{A} x^{m-1})_i / x_i^{m-1}$. Let

- $p_i := (\mathcal{A} x^{m-1})_i$ (numerator);
- $q_i := x_i^{m-1}$ (denominator).

**Numerator error.** The computation of $\mathcal{A} x^{m-1}$ involves $O(n^{m-1})$ floating-point multiply-adds. Standard backward error analysis yields

$$
\mathrm{fl}(p_i) = p_i \cdot (1 + \eta_p), \qquad \lvert \eta_p \rvert \leq c_1 \cdot n^{m-1} \cdot \varepsilon_{\text{machine}},
\tag{4.2}
$$

for some $O(1)$ constant $c_1$. For $n = 20$, $m = 3$, we have $n^{m-1} = 400$, giving $\lvert \eta_p \rvert \lesssim 10^{-13}$.

**Denominator error.** Since $q_i = x_i^{m-1}$ involves only a single exponentiation, $\lvert \eta_q \rvert \leq (m-1) \varepsilon_{\text{machine}}$, which is negligible.

**Division's relative error:**

$$
\mathrm{fl}(t_i) = \frac{p_i(1 + \eta_p)}{q_i(1 + \eta_q)} \cdot (1 + \eta_d), \qquad \lvert \eta_d \rvert \leq \varepsilon_{\text{machine}}.
$$

The relative error is approximately $\eta_p + \eta_d - \eta_q \approx \eta_p$ (dominated by the numerator). Thus,

$$
\frac{\lvert \mathrm{fl}(t_i) - t_i \rvert}{\lvert t_i \rvert} \lesssim c_1 \cdot n^{m-1} \cdot \varepsilon_{\text{machine}}.
\tag{4.3}
$$

### 4.3 From Relative Error to the Spread of $t$

**Key observation:** $\lambda_U$ and $\lambda_L$ are the maximum and minimum of $t$, respectively. While max/min do not amplify relative error per se, different implementations (MATLAB vs. Python) may attain their max/min at different indices due to floating-point noise. More precisely:

- Near convergence, $t_i \approx \lambda_{\max}$ for all $i$;
- In fact, $t_i = \lambda_{\max} + \epsilon_i$, where $\epsilon_i \sim O(n^{m-1} \varepsilon_{\text{machine}}) \cdot \lvert \lambda_{\max} \rvert$ represents floating-point noise;
- $\lambda_U = \max t = \lambda_{\max} + \max_i \epsilon_i$;
- $\lambda_L = \min t = \lambda_{\max} + \min_i \epsilon_i$;
- $\lambda_U - \lambda_L = \max_i \epsilon_i - \min_i \epsilon_i \sim O(n^{m-1} \varepsilon_{\text{machine}}) \cdot \lvert \lambda_{\max} \rvert$.

For the Q7 case, this baseline estimate yields $400 \cdot 10^{-16} \cdot 10.7 \approx 4 \cdot 10^{-13}$, on the same order as $\mathrm{tol} = 10^{-12}$. At this stage, the stopping test is already near the floating-point limit.

But this is not the full story. The above analysis implicitly assumes that all $q_i = x_i^{m-1}$ are comparable in magnitude. If any component $x_{i_0}$ is very small, the situation worsens significantly.

### 4.4 Amplification Due to Small Components

Suppose $x_{i_0} = \varepsilon_x$ with $\varepsilon_x \ll 1$. Then $q_{i_0} = \varepsilon_x^{m-1}$ is also small.

Naively, one might expect $t_{i_0}$'s *absolute* error to remain bounded since $\lvert t_{i_0} \rvert \approx \lambda_{\max}$ (the relative error formula (4.3) gives an absolute error of $O(\lambda_{\max} \cdot n^{m-1} \varepsilon_{\text{machine}})$). However, **a subtler issue arises in the numerator**.

The value $p_{i_0} = (\mathcal{A} x^{m-1})_{i_0}$ is itself small: since $t_{i_0} \approx \lambda_{\max}$, we have $p_{i_0} \approx \lambda_{\max} \cdot \varepsilon_x^{m-1}$. But the **numerical computation of $\mathcal{A} x^{m-1}$ does not know that $p_{i_0}$ is intended to be small** — it accumulates rounding errors based on the *global* magnitude of the vector. Specifically,

$$
\mathrm{fl}(p_{i_0}) = p_{i_0} + \xi_{i_0}, \qquad \lvert \xi_{i_0} \rvert \lesssim n^{m-1} \varepsilon_{\text{machine}} \cdot \lVert \mathcal{A} x^{m-1} \rVert_\infty.
$$

Here $\lVert \mathcal{A} x^{m-1} \rVert_\infty$ is determined by the *largest* components, not by $p_{i_0}$'s own magnitude. Hence $\lvert \xi_{i_0} \rvert$ can be of order $n^{m-1} \varepsilon_{\text{machine}} \cdot \lambda_{\max}$, independent of how small $p_{i_0}$ is.

Dividing:

$$
\mathrm{fl}(t_{i_0}) = \frac{p_{i_0} + \xi_{i_0}}{q_{i_0}} = t_{i_0} + \frac{\xi_{i_0}}{\varepsilon_x^{m-1}}.
\tag{4.4}
$$

The amplification factor $1/\varepsilon_x^{m-1}$ is the origin of the noise floor.

### 4.5 Upper Bound of the Noise Floor

Combining §4.2–§4.4, when the eigenvector has a small component $\min_i x_i = \varepsilon_x$, the absolute error of $t_{i_0}$ is bounded by

$$
\lvert \mathrm{fl}(t_{i_0}) - t_{i_0} \rvert \lesssim \frac{n^{m-1} \cdot \lVert \mathcal{A} x^{m-1} \rVert_\infty \cdot \varepsilon_{\text{machine}}}{\varepsilon_x^{m-1}}.
$$

Assuming $\lVert \mathcal{A} x^{m-1} \rVert_\infty \approx \lambda_{\max}$ (reasonable for a normalized eigenvector near convergence) and $\lambda_U \approx \lambda_{\max}$, we obtain the main bound:

$$
\boxed{
\lambda_U - \lambda_L \lesssim \frac{n^{m-1} \cdot \lambda_{\max} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
}
\tag{4.5}
$$

Equivalently, the noise floor of the relative residual is

$$
\boxed{
\mathrm{floor}(\mathrm{res}) \approx \frac{n^{m-1} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
}
\tag{4.6}
$$

### 4.6 Verification on the Q7 Case

**Parameters:**

- $n = 20$, $m = 3$, $n^{m-1} = 400$;
- $\varepsilon_{\text{machine}} \approx 2.2 \times 10^{-16}$;
- $\min_i x_i \approx 4.82 \times 10^{-6}$ (from experimental output);
- $(\min_i x_i)^{m-1} = (4.82 \times 10^{-6})^2 \approx 2.3 \times 10^{-11}$.

Substituting into (4.6),

$$
\mathrm{floor}(\mathrm{res}) \approx \frac{400 \cdot 2.2 \times 10^{-16}}{2.3 \times 10^{-11}} \approx 3.8 \times 10^{-3}.
$$

This is a **worst-case upper bound**. Empirically, $\mathrm{res}$ oscillates in the range $10^{-13}$ to $10^{-11}$ — several orders of magnitude below the bound.

**Why is the bound loose?** The bound assumes that error concentrates entirely on the single small component $x_{i_0}$ *and* that this component determines the max or min of $t$. In practice, errors partially cancel across components, and the max/min are often attained at indices with larger $x_i$, reducing the effective noise floor.

Nevertheless, the **qualitative prediction holds**: with $\mathrm{tol} = 10^{-12}$, empirical floor $\approx 10^{-12}$, and bound $3.8 \times 10^{-3}$, we confirm that *the tolerance lies at the same order as the noise floor*, consistent with the observed random stopping behavior.

### 4.7 Why the Stopping Criterion Becomes Unstable

When $\mathrm{tol}$ falls within the noise floor regime, each iteration's $\mathrm{res}$ fluctuates randomly in that regime (due to subtle differences in floating-point accumulation — operation ordering, LU pivot selection, etc.):

- Some iterations randomly dip below $\mathrm{tol}$ and trigger termination;
- Others remain above $\mathrm{tol}$ and continue.

MATLAB and Python, due to minor differences in their floating-point accumulation, are mutually independent with respect to which iteration first crosses the tolerance threshold — this is precisely the phenomenon observed in §3.2.

*This is not a bug, but an intrinsic behavior of the algorithm at the noise floor.*

---

## §5 Practical Implications for Tolerance Selection

### 5.1 Regime of Failure

By (4.6), the smallest effective tolerance attainable for NNI, given $\mathcal{A}$ and $x$, is approximately

$$
\mathrm{tol}_{\min} \approx \frac{n^{m-1} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
\tag{5.1}
$$

Setting $\mathrm{tol} < \mathrm{tol}_{\min}$ causes the stopping criterion to enter a **lottery regime**, where the stopping iteration is unpredictable.

### 5.2 Tolerance Selection Rules

**Rule of thumb:**

$$
\mathrm{tol} = \max(\mathrm{tol}_{\text{desired}}, \; 10 \cdot \mathrm{tol}_{\min}),
\tag{5.2}
$$

i.e., do not set $\mathrm{tol}$ more than one order of magnitude below the theoretical floor.

**Difficulty:** $\mathrm{tol}_{\min}$ depends on $\min_i x_i$, which is unknown at algorithm start and revealed only during iteration.

**Practical approach 1 (conservative):** Use an a priori estimate. If the eigenvector is known (or assumed) to have roughly uniform distribution, $\min_i x_i \sim 1/\sqrt{n}$, giving $\mathrm{tol}_{\min} \approx n^{m-1} \varepsilon_{\text{machine}} \cdot n^{(m-1)/2}$.

**Practical approach 2 (adaptive):** At each iteration, compute $\mathrm{tol}_{\text{current}} := c \cdot n^{m-1} \varepsilon_{\text{machine}} / (\min_i x^{(k)})^{m-1}$, and terminate when $\mathrm{res} < \mathrm{tol}_{\text{current}}$. Here $c$ is a safety factor (we suggest $c \in [10, 100]$).

### 5.3 Recommendations for Convergence Reporting

If an implementation intends to report convergence quality, we recommend returning:

- $\lambda_U - \lambda_L$ (the actual bracket width);
- $\min_i x^{(k)}$ (to compute the current noise floor);
- A flag `noise_floor_limited := (res < tol_min)`.

This allows the user to recognize when: if `noise_floor_limited == True`, the result has reached the floating-point limit, and further iteration will not improve accuracy.

---

## §6 Conclusion

### 6.1 Contributions of this Note

1. **Complete documentation of the NNI algorithmic architecture** (§2), including Jacobian derivation and the Rayleigh-quotient-based eigenvalue estimate.
2. **Discovery and quantification, from porting practice, of a noise floor phenomenon** (§3), with empirical data.
3. **Derivation of the noise floor upper bound** (4.5)–(4.6), explaining the effect of small eigenvector components on the stopping criterion.
4. **Practical recommendations for adaptive tolerance selection** (§5).

### 6.2 Open Problems

1. **Tighter bound.** (4.5)–(4.6) provide a loose upper bound; empirically, the noise floor is several orders of magnitude smaller. Can one prove a tighter bound, perhaps involving $\lVert x \rVert_{\text{variance}}$ or the dispersion of $\{t_i\}$?
2. **Other Rayleigh quotient forms.** The continuous version $\lambda = x^\top A x / x^\top x$: does it exhibit a similar floor for sparse eigenvectors? Preliminary analysis suggests no: that formulation's denominator is not vanishing with $\min_i x_i$. Element-wise division is the key culprit.
3. **Preconditioning / rescaling.** Can $x$ be rescaled to increase $\min_i x_i$ and avoid the floor? The challenge: such rescaling alters the eigenvalue definition itself and must be designed carefully.

### 6.3 Implications for Port Practice

This phenomenon reveals: **cross-implementation parity (MATLAB vs. Python) should not target bit-identical agreement through the final stopping iteration.** Once the algorithm enters the noise floor, the stopping iteration is a floating-point lottery and is unrelated to algorithmic correctness.

We propose the following design principles for parity testing:

1. **Algorithmic layer (overlap iterations):** Require bit-identity of the iteration trajectory ($x$, $y$, $\lambda$) within the common range;
2. **Final output:** Require equivalence within tolerance ($\lvert \lambda_{\text{ML}} - \lambda_{\text{PY}} \rvert < \mathrm{tol}$);
3. **Tolerate and report:** Allow differences in stopping iteration count, and report noise-floor attainment as a diagnostic signal.

---

**Appendix A:** Full iteration-by-iteration data for the Q7 case (see `matlab_ref/nni/nni_reference.mat`).

**Appendix B:** Python implementation — `python/tensor_utils.py::nni()` (commit `55921bc`).

**Appendix C:** MATLAB reference script — `matlab_ref/nni/generate_nni_reference.m`.
