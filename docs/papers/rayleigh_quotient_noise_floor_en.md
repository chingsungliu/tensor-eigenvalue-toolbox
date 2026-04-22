Rayleigh Quotient Noise Floor
Numerical Behavior of the NNI Algorithm Under Near-Zero Eigenvector Components
Author: Ching-Sung Liu
Date: April 2026
Version: v0.1 (research note)

§1 Background and Motivation
1.1 Problem Setting
Consider an mm
m-order nn
n-dimensional nonnegative tensor A∈R+n×n×⋯×n\mathcal{A} \in \mathbb{R}_{+}^{n \times n \times \cdots \times n}
A∈R+n×n×⋯×n​ (with mm
m modes). We seek the **largest H-eigenvalue** λmax⁡\lambda_{\max}
λmax​ and the corresponding positive eigenvector x∈R+nx \in \mathbb{R}_{+}^{n}
x∈R+n​ satisfying
Axm−1=λ⋅x[m−1],(1.1)\mathcal{A} x^{m-1} = \lambda \cdot x^{[m-1]},
\tag{1.1}Axm−1=λ⋅x[m−1],(1.1)
where

(Axm−1)i:=∑i2,…,imai,i2,…,im⋅xi2xi3⋯xim(\mathcal{A} x^{m-1})_i := \sum_{i_2, \ldots, i_m} a_{i, i_2, \ldots, i_m} \cdot x_{i_2} x_{i_3} \cdots x_{i_m}
(Axm−1)i​:=∑i2​,…,im​​ai,i2​,…,im​​⋅xi2​​xi3​​⋯xim​​ (the tensor-vector product yielding an nn
n-dimensional vector);
x[m−1]:=(x1m−1,x2m−1,…,xnm−1)⊤x^{[m-1]} := (x_1^{m-1}, x_2^{m-1}, \ldots, x_n^{m-1})^\top
x[m−1]:=(x1m−1​,x2m−1​,…,xnm−1​)⊤ (element-wise (m−1)(m-1)
(m−1)-th power).

When A\mathcal{A}
A is a positive M-tensor, Perron-Frobenius-type results guarantee the existence of a unique positive eigenvector associated with the largest eigenvalue. This is the standing assumption throughout this note.
1.2 The NNI Algorithm
NNI (Nonnegative Newton Iteration) is a Newton-type iterative method for solving (1.1), proposed by the author in earlier work (2016, 2020). Its key idea is to recast (1.1) as a nonlinear system,
F(x,λ):=Axm−1−λ⋅x[m−1]=0,(1.2)F(x, \lambda) := \mathcal{A} x^{m-1} - \lambda \cdot x^{[m-1]} = 0,
\tag{1.2}F(x,λ):=Axm−1−λ⋅x[m−1]=0,(1.2)
perform Newton iteration on xx
x, and update λ\lambda
λ via a Rayleigh-quotient-style estimate. The full derivation is given in §2.
1.3 Motivation of this Note
In the course of porting NNI from MATLAB to Python, the author observed a non-trivial numerical phenomenon: the two implementations remain bit-identical (to machine epsilon) for the first NN
N iterations, but diverge in their stopping behavior afterwards, terminating at different iteration counts. Nonetheless, the final eigenvalue estimates agree within the prescribed tolerance.
A detailed investigation revealed that this divergence is *not* due to implementation-level differences (e.g., sparse vs. dense LU pivot ordering), but rather reflects an **inherent property of the algorithm**: when the eigenvector has a **near-zero component**, the Rayleigh quotient estimate enters a *floating-point-dominated noise floor*. The upper bound of this floor admits an explicit expression in terms of εmachine\varepsilon_{\text{machine}}
εmachine​ and min⁡ixi\min_i x_i
mini​xi​, and it has practical implications for the choice of stopping tolerance.
The goals of this note are:

To document the complete algorithmic architecture of NNI (§2);
To quantitatively describe the noise floor phenomenon (§3);
To derive the upper bound of the noise floor (§4);
To provide practical recommendations for tolerance selection (§5).


§2 The NNI Algorithm: Complete Architecture
2.1 From Eigen-Equation to Nonlinear System
Rewriting (1.1) as a residual,
F(x,λ)=Axm−1−λ⋅x[m−1]=0.F(x, \lambda) = \mathcal{A} x^{m-1} - \lambda \cdot x^{[m-1]} = 0.F(x,λ)=Axm−1−λ⋅x[m−1]=0.
This yields nn
n equations in n+1n+1
n+1 unknowns (the nn
n components of xx
x plus λ\lambda
λ). To close the system, we impose the normalization ∥x∥=1\|x\| = 1
∥x∥=1.
2.2 Derivation of the Jacobian
Taking the derivative of FF
F with respect to xx
x,
J(x,λ):=∂F∂x=∂(Axm−1)∂x−λ⋅∂x[m−1]∂x.J(x, \lambda) := \frac{\partial F}{\partial x} = \frac{\partial (\mathcal{A} x^{m-1})}{\partial x} - \lambda \cdot \frac{\partial x^{[m-1]}}{\partial x}.J(x,λ):=∂x∂F​=∂x∂(Axm−1)​−λ⋅∂x∂x[m−1]​.
First term. Let B(x):=∂(Axm−1)/∂xB(x) := \partial (\mathcal{A} x^{m-1}) / \partial x
B(x):=∂(Axm−1)/∂x. Differentiating the tensor-vector product entry-wise gives
B(x)ij=(m−1)∑i3,…,imai,j,i3,…,im⋅xi3⋯xim.(2.1)B(x)_{ij} = (m-1) \sum_{i_3, \ldots, i_m} a_{i, j, i_3, \ldots, i_m} \cdot x_{i_3} \cdots x_{i_m}.
\tag{2.1}B(x)ij​=(m−1)i3​,…,im​∑​ai,j,i3​,…,im​​⋅xi3​​⋯xim​​.(2.1)
In our implementation, this is computed by the routine sp_Jaco_Ax(A, x, m), which returns an n×nn \times n
n×n sparse matrix.
Second term. Since ∂xim−1/∂xj=(m−1)xim−2⋅δij\partial x_i^{m-1} / \partial x_j = (m-1) x_i^{m-2} \cdot \delta_{ij}
∂xim−1​/∂xj​=(m−1)xim−2​⋅δij​, we obtain a diagonal matrix:
∂x[m−1]∂x=(m−1)⋅diag(x[m−2]).(2.2)\frac{\partial x^{[m-1]}}{\partial x} = (m-1) \cdot \mathrm{diag}(x^{[m-2]}).
\tag{2.2}∂x∂x[m−1]​=(m−1)⋅diag(x[m−2]).(2.2)
Combining,
J(x,λ)=B(x)−(m−1)λ⋅diag(x[m−2])=:M(x,λ).(2.3)J(x, \lambda) = B(x) - (m-1) \lambda \cdot \mathrm{diag}(x^{[m-2]}) =: M(x, \lambda).
\tag{2.3}J(x,λ)=B(x)−(m−1)λ⋅diag(x[m−2])=:M(x,λ).(2.3)
The matrix MM
M is the linear system operator to be inverted at each NNI iteration.
2.3 Newton Update: Full Step
At iteration kk
k, given the current estimate (x(k),λU(k))(x^{(k)}, \lambda_U^{(k)})
(x(k),λU(k)​), the algorithm executes:
Step 1 (Linear solve):
(−M(x(k),λU(k)))⋅y(k)=(x(k))[m−1].(2.4)(-M(x^{(k)}, \lambda_U^{(k)})) \cdot y^{(k)} = (x^{(k)})^{[m-1]}.
\tag{2.4}(−M(x(k),λU(k)​))⋅y(k)=(x(k))[m−1].(2.4)
The negative sign arises from rearranging the Newton step Δx=−J−1F\Delta x = -J^{-1} F
Δx=−J−1F.
Step 2 (Update and normalize):
x~(k+1)=(m−2)⋅x(k)+y(k),x(k+1)=x~(k+1)∥x~(k+1)∥.(2.5)\tilde{x}^{(k+1)} = (m-2) \cdot x^{(k)} + y^{(k)}, \quad x^{(k+1)} = \frac{\tilde{x}^{(k+1)}}{\|\tilde{x}^{(k+1)}\|}.
\tag{2.5}x~(k+1)=(m−2)⋅x(k)+y(k),x(k+1)=∥x~(k+1)∥x~(k+1)​.(2.5)
Note that the coefficient (m−2)(m-2)
(m−2) degenerates to 11
1 for m=3m=3
m=3 (giving x+yx+y
x+y) and to 00
0 for m=2m=2
m=2 (giving yy
y alone). No special-case handling is required.
Step 3 (Rayleigh quotient estimate):
Define t(k+1)∈Rnt^{(k+1)} \in \mathbb{R}^n
t(k+1)∈Rn as the vector of entry-wise local eigenvalue estimates:
ti(k+1):=(A(x(k+1))m−1)i(xi(k+1))m−1,i=1,…,n.(2.6)t_i^{(k+1)} := \frac{(\mathcal{A} (x^{(k+1)})^{m-1})_i}{(x_i^{(k+1)})^{m-1}}, \quad i = 1, \ldots, n.
\tag{2.6}ti(k+1)​:=(xi(k+1)​)m−1(A(x(k+1))m−1)i​​,i=1,…,n.(2.6)
Intuition: if x(k+1)x^{(k+1)}
x(k+1) is a true eigenvector, then all tit_i
ti​ coincide and equal λ\lambda
λ.
We then set
λU(k+1):=max⁡iti(k+1),λL(k+1):=min⁡iti(k+1).(2.7)\lambda_U^{(k+1)} := \max_i t_i^{(k+1)}, \quad \lambda_L^{(k+1)} := \min_i t_i^{(k+1)}.
\tag{2.7}λU(k+1)​:=imax​ti(k+1)​,λL(k+1)​:=imin​ti(k+1)​.(2.7)
By Collatz-type inequalities (valid when A\mathcal{A}
A is a nonnegative irreducible tensor), λU(k+1)\lambda_U^{(k+1)}
λU(k+1)​ and λL(k+1)\lambda_L^{(k+1)}
λL(k+1)​ are upper and lower bounds of the true eigenvalue λmax⁡\lambda_{\max}
λmax​.
Step 4 (Residual and stopping):
res(k+1):=λU(k+1)−λL(k+1)λU(k+1).(2.8)\mathrm{res}^{(k+1)} := \frac{\lambda_U^{(k+1)} - \lambda_L^{(k+1)}}{\lambda_U^{(k+1)}}.
\tag{2.8}res(k+1):=λU(k+1)​λU(k+1)​−λL(k+1)​​.(2.8)
The algorithm terminates when min⁡j≤k+1res(j)≤tol\min_{j \leq k+1} \mathrm{res}^{(j)} \leq \mathrm{tol}
minj≤k+1​res(j)≤tol or when k+1≥maxitk+1 \geq \mathrm{maxit}
k+1≥maxit.
2.4 Pseudocode
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
2.5 Perron-Frobenius Guarantees
When A\mathcal{A}
A is a positive M-tensor and x0>0x_0 > 0
x0​>0, Collatz-type theorems ensure:

A unique positive eigenvector exists corresponding to λmax⁡\lambda_{\max}
λmax​.
The Newton iteration, starting from a positive x0x_0
x0​, preserves positivity of x(k)x^{(k)}
x(k) in exact arithmetic.
The bounds λL(k)≤λmax⁡≤λU(k)\lambda_L^{(k)} \leq \lambda_{\max} \leq \lambda_U^{(k)}
λL(k)​≤λmax​≤λU(k)​ hold for all kk
k, with both sequences converging monotonically to λmax⁡\lambda_{\max}
λmax​.

No active nonnegativity projection is performed by the algorithm: NNI does not enforce x≥0x \geq 0
x≥0 explicitly. If the input is not a positive M-tensor, the algorithm may diverge or produce λL<0\lambda_L < 0
λL​<0; in the latter case res>1\mathrm{res} > 1
res>1, indicating that the input violates the standing assumption.

§3 The Noise Floor Phenomenon: Observation and Quantification
3.1 Experimental Setup
To verify port correctness, we designed a parity test comparing MATLAB and Python implementations under identical inputs, iteration by iteration.
Test case (Q7):

n=20n = 20
n=20, m=3m = 3
m=3;
Random seed rng(42)\text{rng}(42)
rng(42);
Diagonal dominance: di∈[1,11]d_i \in [1, 11]
di​∈[1,11] uniformly random;
Perturbation: sparse random, density 0.02, magnitude 0.01;
A=sp_tendiag(d,m)+pert\mathcal{A} = \mathrm{sp\_tendiag}(d, m) + \mathrm{pert}
A=sp_tendiag(d,m)+pert;
x0=∣rand(n)∣+0.1x_0 = |\text{rand}(n)| + 0.1
x0​=∣rand(n)∣+0.1;
tol=10−12\mathrm{tol} = 10^{-12}
tol=10−12, maxit=200\mathrm{maxit} = 200
maxit=200.

MATLAB uses \ (backslash, sparse LU via UMFPACK). Python uses scipy.sparse.linalg.spsolve (also UMFPACK-based).
3.2 Observation: Bit-Identical Prefix, Divergent Suffix
First 20 iterations (k=0k = 0
k=0 through k=19k = 19
k=19):
All per-iteration intermediate quantities (x(k)x^{(k)}
x(k), w(k):=x(k)[m−1]w^{(k)} := x^{(k)[m-1]}
w(k):=x(k)[m−1], y(k)y^{(k)}
y(k), λU(k)\lambda_U^{(k)}
λU(k)​, λL(k)\lambda_L^{(k)}
λL(k)​, res(k)\mathrm{res}^{(k)}
res(k)) agree between MATLAB and Python to within 10−1510^{-15}
10−15 in absolute error (i.e., machine epsilon).
For instance, the per-iteration maximum absolute error in x(k)x^{(k)}
x(k):
| iter kk
k | max⁡i∣xi,ML(k)−xi,PY(k)∣\max_i |x^{(k)}_{i,\text{ML}} - x^{(k)}_{i,\text{PY}}|
maxi​∣xi,ML(k)​−xi,PY(k)​∣ |
|:-:|:-:|
| 1 | 2.9×10−162.9 \times 10^{-16}
2.9×10−16 |
| 5 | 4.8×10−164.8 \times 10^{-16}
4.8×10−16 |
| 10 | 6.1×10−166.1 \times 10^{-16}
6.1×10−16 |
| 15 | 7.5×10−167.5 \times 10^{-16}
7.5×10−16 |
| 19 | 7.9×10−167.9 \times 10^{-16}
7.9×10−16 |
This is entirely consistent with floating-point round-off accumulation, with no algorithmic discrepancy.
Starting at iteration 20: divergence in stopping behavior
iter kk
kMATLAB res\mathrm{res}
resPython res\mathrm{res}
resStopping decision205.2×10−115.2 \times 10^{-11}
5.2×10−115.2×10−115.2 \times 10^{-11}
5.2×10−11Neither stops211.3×10−121.3 \times 10^{-12}
1.3×10−121.1×10−141.1 \times 10^{-14}
1.1×10−14Python stops; MATLAB continues227.7×10−127.7 \times 10^{-12}
7.7×10−12——231.9×10−121.9 \times 10^{-12}
1.9×10−12——247.5×10−127.5 \times 10^{-12}
7.5×10−12——253.2×10−123.2 \times 10^{-12}
3.2×10−12——266.9×10−136.9 \times 10^{-13}
6.9×10−13—MATLAB stops
Key observations:

At iteration 21, res\mathrm{res}
res differs by a factor of 100 between the two, yet at iteration 20 the difference is only 1%;
A 100-fold amplification within a single iteration is dramatic and demands explanation;
MATLAB iterations 22-26 exhibit res\mathrm{res}
res that does not decrease monotonically, but oscillates within a band of about 10−1210^{-12}
10−12—a strong indication that the algorithm has lost its convergent behavior and entered a plateau regime.

3.3 Final Results Are Equivalent
Despite stopping at different iterations, the final eigenvalue and eigenvector agree within tolerance:
QuantityMATLABPythonDifferenceFinal λU\lambda_U
λU​10.6991009415311610.69910094153116
10.6991009415311610.6991009415308110.69910094153081
10.699100941530813.5×10−133.5 \times 10^{-13}
3.5×10−13Final λL\lambda_L
λL​10.6991009415237410.69910094152374
10.69910094152374(same)—∥xML−xPY∥\|x_{\text{ML}} - x_{\text{PY}}\|
∥xML​−xPY​∥——4.9×10−174.9 \times 10^{-17}
4.9×10−17
Since ∣λML−λPY∣=3.5×10−13<tol=10−12|\lambda_{\text{ML}} - \lambda_{\text{PY}}| = 3.5 \times 10^{-13} < \mathrm{tol} = 10^{-12}
∣λML​−λPY​∣=3.5×10−13<tol=10−12, the two implementations are equivalent within the prescribed accuracy.
3.4 Ruling Out an Alternative Hypothesis
Hypothesis A: sparse vs. dense LU pivot divergence. One might speculate that MATLAB's \ and scipy's spsolve—despite both being UMFPACK-based—exhibit different pivot orderings when applied to near-singular systems, causing a divergence in the computed y(k)y^{(k)}
y(k).
Verification: If Hypothesis A held, one would expect y(k)y^{(k)}
y(k) to show a pronounced discrepancy starting at iteration 20. Empirically:
| iter kk
k | max⁡i∣yi,ML(k)−yi,PY(k)∣\max_i |y^{(k)}_{i,\text{ML}} - y^{(k)}_{i,\text{PY}}|
maxi​∣yi,ML(k)​−yi,PY(k)​∣ | ∥yML(k)∥∞\|y^{(k)}_{\text{ML}}\|_\infty
∥yML(k)​∥∞​ | Relative error |
|:-:|:-:|:-:|:-:|
| 18 | 4.6×10−164.6 \times 10^{-16}
4.6×10−16 | 0.99950.9995
0.9995 | 4.6×10−164.6 \times 10^{-16}
4.6×10−16 |
| 19 | 1.3×10−151.3 \times 10^{-15}
1.3×10−15 | 0.99950.9995
0.9995 | 1.3×10−151.3 \times 10^{-15}
1.3×10−15 |
| 20 | 9.2×10−169.2 \times 10^{-16}
9.2×10−16 | 0.99950.9995
0.9995 | 9.2×10−169.2 \times 10^{-16}
9.2×10−16 |
| 21 | 1.2×10−151.2 \times 10^{-15}
1.2×10−15 | 0.99950.9995
0.9995 | 1.2×10−151.2 \times 10^{-15}
1.2×10−15 |
At iterations 20 and 21, y(k)y^{(k)}
y(k) remains bit-identical to machine epsilon—Hypothesis A is falsified. The solutions of the linear system are identical between the two implementations.
The divergence must therefore occur in a later step. We identify this step as the element-wise division in (2.6), namely the Rayleigh quotient estimate. This is the subject of §4.

§4 Root Cause: Floating-Point Sensitivity of the Rayleigh Quotient
4.1 Pinpointing the Divergence
Recall the computational flow of each NNI iteration (Steps 1-4 in §2.3):

y(k)y^{(k)}
y(k) from the linear solve — verified bit-identical (§3.4);
x(k+1)x^{(k+1)}
x(k+1) from linear combination and normalization — floating-point accumulation is controlled;
t(k+1)=(A(x(k+1))m−1) ./ (x(k+1))[m−1]t^{(k+1)} = (\mathcal{A} (x^{(k+1)})^{m-1}) \, ./\, (x^{(k+1)})^{[m-1]}
t(k+1)=(A(x(k+1))m−1)./(x(k+1))[m−1] — suspect;
λU(k+1)=max⁡t(k+1)\lambda_U^{(k+1)} = \max t^{(k+1)}
λU(k+1)​=maxt(k+1), λL(k+1)=min⁡t(k+1)\lambda_L^{(k+1)} = \min t^{(k+1)}
λL(k+1)​=mint(k+1);
res(k+1)=(λU−λL)/λU\mathrm{res}^{(k+1)} = (\lambda_U - \lambda_L) / \lambda_U
res(k+1)=(λU​−λL​)/λU​.

Step 3 involves component-wise division, a classical source of floating-point error amplification.
4.2 Floating-Point Error Analysis (Semi-Formal)
Adopt the standard IEEE 754 floating-point model. For any floating-point operation op(a,b)\mathrm{op}(a, b)
op(a,b),
fl(op(a,b))=op(a,b)⋅(1+δ),∣δ∣≤εmachine,(4.1)\mathrm{fl}(\mathrm{op}(a, b)) = \mathrm{op}(a, b) \cdot (1 + \delta), \quad |\delta| \leq \varepsilon_{\text{machine}},
\tag{4.1}fl(op(a,b))=op(a,b)⋅(1+δ),∣δ∣≤εmachine​,(4.1)
where εmachine≈2.22×10−16\varepsilon_{\text{machine}} \approx 2.22 \times 10^{-16}
εmachine​≈2.22×10−16 in double precision.
Analysis of Step 3. Compute ti=(Axm−1)i/xim−1t_i = (\mathcal{A} x^{m-1})_i / x_i^{m-1}
ti​=(Axm−1)i​/xim−1​. Let

pi:=(Axm−1)ip_i := (\mathcal{A} x^{m-1})_i
pi​:=(Axm−1)i​ (numerator);
qi:=xim−1q_i := x_i^{m-1}
qi​:=xim−1​ (denominator).

Numerator error. The computation of Axm−1\mathcal{A} x^{m-1}
Axm−1 involves O(nm−1)O(n^{m-1})
O(nm−1) floating-point multiply-adds. Standard backward error analysis yields
fl(pi)=pi⋅(1+ηp),∣ηp∣≤c1⋅nm−1⋅εmachine,(4.2)\mathrm{fl}(p_i) = p_i \cdot (1 + \eta_p), \quad |\eta_p| \leq c_1 \cdot n^{m-1} \cdot \varepsilon_{\text{machine}},
\tag{4.2}fl(pi​)=pi​⋅(1+ηp​),∣ηp​∣≤c1​⋅nm−1⋅εmachine​,(4.2)
for some O(1)O(1)
O(1) constant c1c_1
c1​. For n=20n = 20
n=20, m=3m = 3
m=3, we have nm−1=400n^{m-1} = 400
nm−1=400, giving ∣ηp∣≲10−13|\eta_p| \lesssim 10^{-13}
∣ηp​∣≲10−13.
Denominator error. Since qi=xim−1q_i = x_i^{m-1}
qi​=xim−1​ involves only a single exponentiation, ∣ηq∣≤(m−1)εmachine|\eta_q| \leq (m-1) \varepsilon_{\text{machine}}
∣ηq​∣≤(m−1)εmachine​, which is negligible.
Division's relative error:
fl(ti)=pi(1+ηp)qi(1+ηq)⋅(1+ηd),∣ηd∣≤εmachine.\mathrm{fl}(t_i) = \frac{p_i(1 + \eta_p)}{q_i(1 + \eta_q)} \cdot (1 + \eta_d), \quad |\eta_d| \leq \varepsilon_{\text{machine}}.fl(ti​)=qi​(1+ηq​)pi​(1+ηp​)​⋅(1+ηd​),∣ηd​∣≤εmachine​.
The relative error is approximately ηp+ηd−ηq≈ηp\eta_p + \eta_d - \eta_q \approx \eta_p
ηp​+ηd​−ηq​≈ηp​ (dominated by the numerator). Thus,
∣fl(ti)−ti∣∣ti∣≲c1⋅nm−1⋅εmachine.(4.3)\frac{|\mathrm{fl}(t_i) - t_i|}{|t_i|} \lesssim c_1 \cdot n^{m-1} \cdot \varepsilon_{\text{machine}}.
\tag{4.3}∣ti​∣∣fl(ti​)−ti​∣​≲c1​⋅nm−1⋅εmachine​.(4.3)
4.3 From Relative Error to the Spread of tt
t
Key observation: λU\lambda_U
λU​ and λL\lambda_L
λL​ are the maximum and minimum of tt
t, respectively. While max/min do not amplify relative error per se, different implementations (MATLAB vs. Python) may attain their max/min at different indices due to floating-point noise. More precisely:

Near convergence, ti≈λmax⁡t_i \approx \lambda_{\max}
ti​≈λmax​ for all ii
i;
In fact, ti=λmax⁡+ϵit_i = \lambda_{\max} + \epsilon_i
ti​=λmax​+ϵi​, where ϵi∼O(nm−1εmachine)⋅∣λmax⁡∣\epsilon_i \sim O(n^{m-1} \varepsilon_{\text{machine}}) \cdot |\lambda_{\max}|
ϵi​∼O(nm−1εmachine​)⋅∣λmax​∣ represents floating-point noise;
λU=max⁡t=λmax⁡+max⁡iϵi\lambda_U = \max t = \lambda_{\max} + \max_i \epsilon_i
λU​=maxt=λmax​+maxi​ϵi​;
λL=min⁡t=λmax⁡+min⁡iϵi\lambda_L = \min t = \lambda_{\max} + \min_i \epsilon_i
λL​=mint=λmax​+mini​ϵi​;
λU−λL=max⁡iϵi−min⁡iϵi∼O(nm−1εmachine)⋅∣λmax⁡∣\lambda_U - \lambda_L = \max_i \epsilon_i - \min_i \epsilon_i \sim O(n^{m-1} \varepsilon_{\text{machine}}) \cdot |\lambda_{\max}|
λU​−λL​=maxi​ϵi​−mini​ϵi​∼O(nm−1εmachine​)⋅∣λmax​∣.

For the Q7 case, this baseline estimate yields 400⋅10−16⋅10.7≈4⋅10−13400 \cdot 10^{-16} \cdot 10.7 \approx 4 \cdot 10^{-13}
400⋅10−16⋅10.7≈4⋅10−13, on the same order as tol=10−12\mathrm{tol} = 10^{-12}
tol=10−12. At this stage, the stopping test is already near the floating-point limit.
But this is not the full story. The above analysis implicitly assumes that all qi=xim−1q_i = x_i^{m-1}
qi​=xim−1​ are comparable in magnitude. If any component xi0x_{i_0}
xi0​​ is very small, the situation worsens significantly.
4.4 Amplification Due to Small Components
Suppose xi0=εxx_{i_0} = \varepsilon_x
xi0​​=εx​ with εx≪1\varepsilon_x \ll 1
εx​≪1. Then qi0=εxm−1q_{i_0} = \varepsilon_x^{m-1}
qi0​​=εxm−1​ is also small.
Naively, one might expect ti0t_{i_0}
ti0​​'s *absolute* error to remain bounded since ∣ti0∣≈λmax⁡|t_{i_0}| \approx \lambda_{\max}
∣ti0​​∣≈λmax​ (the relative error formula (4.3) gives an absolute error of O(λmax⁡⋅nm−1εmachine)O(\lambda_{\max} \cdot n^{m-1} \varepsilon_{\text{machine}})
O(λmax​⋅nm−1εmachine​)). However, **a subtler issue arises in the numerator**.
The value pi0=(Axm−1)i0p_{i_0} = (\mathcal{A} x^{m-1})_{i_0}
pi0​​=(Axm−1)i0​​ is itself small: since ti0≈λmax⁡t_{i_0} \approx \lambda_{\max}
ti0​​≈λmax​, we have pi0≈λmax⁡⋅εxm−1p_{i_0} \approx \lambda_{\max} \cdot \varepsilon_x^{m-1}
pi0​​≈λmax​⋅εxm−1​. But the **numerical computation of Axm−1\mathcal{A} x^{m-1}
Axm−1 does not know that pi0p_{i_0}
pi0​​ is intended to be small**—it accumulates rounding errors based on the *global* magnitude of the vector. Specifically,
fl(pi0)=pi0+ξi0,∣ξi0∣≲nm−1εmachine⋅∥Axm−1∥∞.\mathrm{fl}(p_{i_0}) = p_{i_0} + \xi_{i_0}, \quad |\xi_{i_0}| \lesssim n^{m-1} \varepsilon_{\text{machine}} \cdot \|\mathcal{A} x^{m-1}\|_\infty.fl(pi0​​)=pi0​​+ξi0​​,∣ξi0​​∣≲nm−1εmachine​⋅∥Axm−1∥∞​.
Here ∥Axm−1∥∞\|\mathcal{A} x^{m-1}\|_\infty
∥Axm−1∥∞​ is determined by the *largest* components, not by pi0p_{i_0}
pi0​​'s own magnitude. Hence ∣ξi0∣|\xi_{i_0}|
∣ξi0​​∣ can be of order nm−1εmachine⋅λmax⁡n^{m-1} \varepsilon_{\text{machine}} \cdot \lambda_{\max}
nm−1εmachine​⋅λmax​, independent of how small pi0p_{i_0}
pi0​​ is.
Dividing:
fl(ti0)=pi0+ξi0qi0=ti0+ξi0εxm−1.(4.4)\mathrm{fl}(t_{i_0}) = \frac{p_{i_0} + \xi_{i_0}}{q_{i_0}} = t_{i_0} + \frac{\xi_{i_0}}{\varepsilon_x^{m-1}}.
\tag{4.4}fl(ti0​​)=qi0​​pi0​​+ξi0​​​=ti0​​+εxm−1​ξi0​​​.(4.4)
The amplification factor 1/εxm−11/\varepsilon_x^{m-1}
1/εxm−1​ is the origin of the noise floor.
4.5 Upper Bound of the Noise Floor
Combining §4.2-§4.4, when the eigenvector has a small component min⁡ixi=εx\min_i x_i = \varepsilon_x
mini​xi​=εx​, the absolute error of ti0t_{i_0}
ti0​​ is bounded by
∣fl(ti0)−ti0∣≲nm−1⋅∥Axm−1∥∞⋅εmachineεxm−1.|\mathrm{fl}(t_{i_0}) - t_{i_0}| \lesssim \frac{n^{m-1} \cdot \|\mathcal{A} x^{m-1}\|_\infty \cdot \varepsilon_{\text{machine}}}{\varepsilon_x^{m-1}}.∣fl(ti0​​)−ti0​​∣≲εxm−1​nm−1⋅∥Axm−1∥∞​⋅εmachine​​.
Assuming ∥Axm−1∥∞≈λmax⁡\|\mathcal{A} x^{m-1}\|_\infty \approx \lambda_{\max}
∥Axm−1∥∞​≈λmax​ (reasonable for a normalized eigenvector near convergence) and λU≈λmax⁡\lambda_U \approx \lambda_{\max}
λU​≈λmax​, we obtain the main bound:
λU−λL≲nm−1⋅λmax⁡⋅εmachine(min⁡ixi)m−1.(4.5)\boxed{
\lambda_U - \lambda_L \lesssim \frac{n^{m-1} \cdot \lambda_{\max} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
}
\tag{4.5}λU​−λL​≲(mini​xi​)m−1nm−1⋅λmax​⋅εmachine​​.​(4.5)
Equivalently, the noise floor of the relative residual is
floor(res)≈nm−1⋅εmachine(min⁡ixi)m−1.(4.6)\boxed{
\mathrm{floor}(\mathrm{res}) \approx \frac{n^{m-1} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
}
\tag{4.6}floor(res)≈(mini​xi​)m−1nm−1⋅εmachine​​.​(4.6)
4.6 Verification on the Q7 Case
Parameters:

n=20n = 20
n=20, m=3m = 3
m=3, nm−1=400n^{m-1} = 400
nm−1=400;
εmachine≈2.2×10−16\varepsilon_{\text{machine}} \approx 2.2 \times 10^{-16}
εmachine​≈2.2×10−16;
min⁡ixi≈4.82×10−6\min_i x_i \approx 4.82 \times 10^{-6}
mini​xi​≈4.82×10−6 (from experimental output);
(min⁡ixi)m−1=(4.82×10−6)2≈2.3×10−11(\min_i x_i)^{m-1} = (4.82 \times 10^{-6})^2 \approx 2.3 \times 10^{-11}
(mini​xi​)m−1=(4.82×10−6)2≈2.3×10−11.

Substituting into (4.6),
floor(res)≈400⋅2.2×10−162.3×10−11≈3.8×10−3.\mathrm{floor}(\mathrm{res}) \approx \frac{400 \cdot 2.2 \times 10^{-16}}{2.3 \times 10^{-11}} \approx 3.8 \times 10^{-3}.floor(res)≈2.3×10−11400⋅2.2×10−16​≈3.8×10−3.
This is a worst-case upper bound. Empirically, res\mathrm{res}
res oscillates in the range 10−1310^{-13}
10−13 to 10−1110^{-11}
10−11—several orders of magnitude below the bound.
Why is the bound loose? The bound assumes that error concentrates entirely on the single small component xi0x_{i_0}
xi0​​ *and* that this component determines the max or min of tt
t. In practice, errors partially cancel across components, and the max/min are often attained at indices with larger xix_i
xi​, reducing the effective noise floor.
Nevertheless, the **qualitative prediction holds**: with tol=10−12\mathrm{tol} = 10^{-12}
tol=10−12, empirical floor ≈10−12\approx 10^{-12}
≈10−12, and bound 3.8×10−33.8 \times 10^{-3}
3.8×10−3, we confirm that *the tolerance lies at the same order as the noise floor*, consistent with the observed random stopping behavior.
4.7 Why the Stopping Criterion Becomes Unstable
When tol\mathrm{tol}
tol falls within the noise floor regime, each iteration's res\mathrm{res}
res fluctuates randomly in that regime (due to subtle differences in floating-point accumulation—operation ordering, LU pivot selection, etc.):

Some iterations randomly dip below tol\mathrm{tol}
tol and trigger termination;
Others remain above tol\mathrm{tol}
tol and continue.

MATLAB and Python, due to minor differences in their floating-point accumulation, are mutually independent with respect to which iteration first crosses the tolerance threshold—this is precisely the phenomenon observed in §3.2.
This is not a bug, but an intrinsic behavior of the algorithm at the noise floor.

§5 Practical Implications for Tolerance Selection
5.1 Regime of Failure
By (4.6), the smallest effective tolerance attainable for NNI, given A\mathcal{A}
A and xx
x, is approximately
tolmin⁡≈nm−1⋅εmachine(min⁡ixi)m−1.(5.1)\mathrm{tol}_{\min} \approx \frac{n^{m-1} \cdot \varepsilon_{\text{machine}}}{(\min_i x_i)^{m-1}}.
\tag{5.1}tolmin​≈(mini​xi​)m−1nm−1⋅εmachine​​.(5.1)
Setting tol<tolmin⁡\mathrm{tol} < \mathrm{tol}_{\min}
tol<tolmin​ causes the stopping criterion to enter a lottery regime, where the stopping iteration is unpredictable.
5.2 Tolerance Selection Rules
Rule of thumb:
tol=max⁡(toldesired,  10⋅tolmin⁡),(5.2)\mathrm{tol} = \max(\mathrm{tol}_{\text{desired}}, \; 10 \cdot \mathrm{tol}_{\min}),
\tag{5.2}tol=max(toldesired​,10⋅tolmin​),(5.2)
i.e., do not set tol\mathrm{tol}
tol more than one order of magnitude below the theoretical floor.
Difficulty: tolmin⁡\mathrm{tol}_{\min}
tolmin​ depends on min⁡ixi\min_i x_i
mini​xi​, which is unknown at algorithm start and revealed only during iteration.
**Practical approach 1 (conservative)**: Use an a priori estimate. If the eigenvector is known (or assumed) to have roughly uniform distribution, min⁡ixi∼1/n\min_i x_i \sim 1/\sqrt{n}
mini​xi​∼1/n​, giving tolmin⁡≈nm−1εmachine⋅n(m−1)/2\mathrm{tol}_{\min} \approx n^{m-1} \varepsilon_{\text{machine}} \cdot n^{(m-1)/2}
tolmin​≈nm−1εmachine​⋅n(m−1)/2.
**Practical approach 2 (adaptive)**: At each iteration, compute tolcurrent:=c⋅nm−1εmachine/(min⁡ix(k))m−1\mathrm{tol}_{\text{current}} := c \cdot n^{m-1} \varepsilon_{\text{machine}} / (\min_i x^{(k)})^{m-1}
tolcurrent​:=c⋅nm−1εmachine​/(mini​x(k))m−1, and terminate when res<tolcurrent\mathrm{res} < \mathrm{tol}_{\text{current}}
res<tolcurrent​. Here cc
c is a safety factor (we suggest c∈[10,100]c \in [10, 100]
c∈[10,100]).
5.3 Recommendations for Convergence Reporting
If an implementation intends to report convergence quality, we recommend returning:

λU−λL\lambda_U - \lambda_L
λU​−λL​ (the actual bracket width);
min⁡ix(k)\min_i x^{(k)}
mini​x(k) (to compute the current noise floor);
A flag noise_floor_limited := (res < tol_min).

This allows the user to recognize when: If noise_floor_limited == True, the result has reached the floating-point limit, and further iteration will not improve accuracy.

§6 Conclusion
6.1 Contributions of this Note

Complete documentation of the NNI algorithmic architecture (§2), including Jacobian derivation and the Rayleigh-quotient-based eigenvalue estimate.
Discovery and quantification, from porting practice, of a noise floor phenomenon (§3), with empirical data.
Derivation of the noise floor upper bound (4.5)-(4.6), explaining the effect of small eigenvector components on the stopping criterion.
Practical recommendations for adaptive tolerance selection (§5).

6.2 Open Problems

Tighter bound. (4.5)-(4.6) provide a loose upper bound; empirically, the noise floor is several orders of magnitude smaller. Can one prove a tighter bound, perhaps involving ∥x∥variance\|x\|_{\text{variance}}
∥x∥variance​ or the dispersion of {ti}\{t_i\}
{ti​}?
Other Rayleigh quotient forms. The continuous version λ=x⊤Ax/x⊤x\lambda = x^\top A x / x^\top x
λ=x⊤Ax/x⊤x: does it exhibit a similar floor for sparse eigenvectors? Preliminary analysis suggests no: that formulation's denominator is not vanishing with min⁡ixi\min_i x_i
mini​xi​. Element-wise division is the key culprit.
Preconditioning / rescaling. Can xx
x be rescaled to increase min⁡ixi\min_i x_i
mini​xi​ and avoid the floor? The challenge: such rescaling alters the eigenvalue definition itself and must be designed carefully.

6.3 Implications for Port Practice
This phenomenon reveals: Cross-implementation parity (MATLAB vs. Python) should not target bit-identical agreement through the final stopping iteration. Once the algorithm enters the noise floor, the stopping iteration is a floating-point lottery and is unrelated to algorithmic correctness.
We propose the following design principles for parity testing:

Algorithmic layer (overlap iterations): Require bit-identity of the iteration trajectory (x,y,λx, y, \lambda
x,y,λ) within the common range;
Final output: Require equivalence within tolerance (∣λML−λPY∣<tol|\lambda_{\text{ML}} - \lambda_{\text{PY}}| < \mathrm{tol}
∣λML​−λPY​∣<tol);
Tolerate and report: Allow differences in stopping iteration count, and report noise-floor attainment as a diagnostic signal.


Appendix A: Full iteration-by-iteration data for the Q7 case (see matlab_ref/nni/nni_reference.mat).
Appendix B: Python implementation — python/tensor_utils.py::nni() (commit 55921bc).
Appendix C: MATLAB reference script — matlab_ref/nni/generate_nni_reference.m.