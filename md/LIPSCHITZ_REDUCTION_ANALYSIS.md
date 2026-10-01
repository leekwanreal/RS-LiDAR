# Mathematical Analysis: Lipschitz Regularization and the Smoothing vs. Bias Trade-off in Gaussian-Smoothed Reward Models

---

## 1. Mathematical Framework & Problem Setting

Let $(\mathbb{R}^D, \mathcal{B}(\mathbb{R}^D), \gamma_\sigma)$ denote the Euclidean probability space endowed with the isotropic Gaussian measure $\gamma_\sigma \triangleq \mathcal{N}(0, \sigma^2 \mathbf{I}_D)$ for smoothing scale $\sigma > 0$, whose Radon-Nikodym derivative with respect to the Lebesgue measure is the Gaussian mollifier:
$$\phi_\sigma(\mathbf{z}) \triangleq \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left( -\frac{\|\mathbf{z}\|_2^2}{2\sigma^2} \right)$$

We study a reward function $R: \mathbb{R}^D \to \mathbb{R}$ satisfying two fundamental analytical properties:
1. **Global Lipschitz Regularity**: $R$ is globally Lipschitz continuous with constant $L_{\text{before}} < \infty$:
   $$|R(\mathbf{x}_1) - R(\mathbf{x}_2)| \le L_{\text{before}} \|\mathbf{x}_1 - \mathbf{x}_2\|_2, \quad \forall \mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^D$$
   Equivalently, $R$ belongs to the Sobolev space $W^{1, \infty}(\mathbb{R}^D)$.
2. **Bounded Scalar Amplitude**: $R$ has bounded range with essential supremum and infimum:
   $$\Delta R \triangleq \operatorname{ess\,sup}_{\mathbf{x} \in \mathbb{R}^D} R(\mathbf{x}) - \operatorname{ess\,inf}_{\mathbf{x} \in \mathbb{R}^D} R(\mathbf{x}) < \infty$$
   *(Note: $\Delta R$ denotes exclusively the scalar amplitude range, never an operator).*
3. **Smoothness Regularity**: Where higher-order expansions are analyzed, $R \in C^4(\mathbb{R}^D)$ with globally bounded Hessian in operator (spectral) norm:
   $$\|\nabla^2 R(\mathbf{x})\|_2 \le H < \infty, \quad \forall \mathbf{x} \in \mathbb{R}^D$$

The **Gaussian smoothed surrogate** $R_\sigma: \mathbb{R}^D \to \mathbb{R}$ is defined via the expectation:
$$R_\sigma(\mathbf{x}) \triangleq \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u})] = (R * \phi_\sigma)(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

We denote by $L_{\text{after}}$ the Lipschitz constant of $R_\sigma$:
$$\|\nabla R_\sigma(\mathbf{x})\|_2 \le L_{\text{after}}, \quad \forall \mathbf{x} \in \mathbb{R}^D$$

---

## 2. Rigorous Derivation of the Lipschitz Upper Bound

### Theorem 1 (Dual Lipschitz Bound)
For any smoothing parameter $\sigma > 0$, the smoothed reward surrogate $R_\sigma \in C^\infty(\mathbb{R}^D)$ is globally Lipschitz continuous with constant $L_{\text{after}}$ satisfying:

$$\boxed{L_{\text{after}} \le \min\left( L_{\text{before}}, \, \frac{\Delta R}{\sigma \sqrt{2\pi}} \right)}$$

---

### Rigorous Proof of Bound 1: $L_{\text{after}} \le L_{\text{before}}$ via Weak Derivatives and Sobolev Mollification

Since $R \in W^{1, \infty}(\mathbb{R}^D)$, $R$ possesses a weak gradient $\nabla R = (\partial_1 R, \dots, \partial_D R)^\top \in L^\infty(\mathbb{R}^D; \mathbb{R}^D)$. By Rademacher's theorem, $R$ is differentiable Lebesgue-almost everywhere, and its weak gradient coincides a.e. with its classical gradient, with essential supremum:
$$\|\nabla R\|_{L^\infty(\mathbb{R}^D, \ell_2)} \triangleq \operatorname{ess\,sup}_{\mathbf{y} \in \mathbb{R}^D} \|\nabla R(\mathbf{y})\|_2 = L_{\text{before}}$$

Because $\phi_\sigma \in \mathcal{S}(\mathbb{R}^D)$ (the Schwartz class of rapidly decreasing smooth functions), the convolution $R_\sigma = R * \phi_\sigma$ is infinitely differentiable ($C^\infty$). 

By the fundamental differentiation property of distributional convolutions, the classical derivative of the mollified function equals the convolution of the weak derivative with the mollifier:
$$\partial_j R_\sigma(\mathbf{x}) = \partial_j (R * \phi_\sigma)(\mathbf{x}) = ((\partial_j R) * \phi_\sigma)(\mathbf{x}), \quad \forall j \in \{1, \dots, D\}$$

Assembling the gradient vector field:
$$\nabla R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} \nabla R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

Taking the Euclidean norm $\|\cdot\|_2$ and applying the continuous Minkowski integral inequality (or Jensen's inequality with respect to the probability measure $\phi_\sigma(\mathbf{z}) d\mathbf{z}$):
$$\|\nabla R_\sigma(\mathbf{x})\|_2 = \left\| \int_{\mathbb{R}^D} \nabla R(\mathbf{x} - \mathbf{z}) \phi_\sigma(\mathbf{z}) \, d\mathbf{z} \right\|_2 \le \int_{\mathbb{R}^D} \|\nabla R(\mathbf{x} - \mathbf{z})\|_2 \phi_\sigma(\mathbf{z}) \, d\mathbf{z}$$

Since $\|\nabla R(\mathbf{x} - \mathbf{z})\|_2 \le \|\nabla R\|_{L^\infty} = L_{\text{before}}$ holds for almost every $\mathbf{z} \in \mathbb{R}^D$:
$$\|\nabla R_\sigma(\mathbf{x})\|_2 \le L_{\text{before}} \int_{\mathbb{R}^D} \phi_\sigma(\mathbf{z}) \, d\mathbf{z} = L_{\text{before}} \cdot 1 = L_{\text{before}}$$

Thus:
$$\boxed{L_{\text{after}} \le L_{\text{before}}}$$

---

### Rigorous Proof of Bound 2: $L_{\text{after}} \le \frac{\Delta R}{\sigma \sqrt{2\pi}}$ via Kernel Differentiation & Isotropic Projection

By the Leibniz integral rule (valid since $\phi_\sigma$ is smooth and $R \in L^\infty$):
$$\nabla R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{y}) \nabla_\mathbf{x} \phi_\sigma(\mathbf{x} - \mathbf{y}) \, d\mathbf{y}$$

Computing the kernel gradient directly:
$$\nabla_\mathbf{x} \phi_\sigma(\mathbf{x} - \mathbf{y}) = -\frac{\mathbf{x} - \mathbf{y}}{\sigma^2} \phi_\sigma(\mathbf{x} - \mathbf{y})$$

Substituting $\mathbf{y} = \mathbf{x} + \sigma \mathbf{u}$ where $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$ yields the Stein score formulation:
$$\nabla R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u}) \, \mathbf{u}]$$

Let $c \triangleq \frac{1}{2} (\operatorname{ess\,sup} R + \operatorname{ess\,inf} R)$ be the midpoint of the essential range. Since $\mathbb{E}_{\mathbf{u}}[\mathbf{u}] = \mathbf{0}$, $\mathbb{E}_{\mathbf{u}}[c \, \mathbf{u}] = \mathbf{0}$, we obtain zero-mean centering:
$$\nabla R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [(R(\mathbf{x} + \sigma \mathbf{u}) - c) \, \mathbf{u}]$$
where $|R(\mathbf{x} + \sigma \mathbf{u}) - c| \le \frac{\Delta R}{2}$ almost surely.

By the dual representation of the Euclidean norm on the unit sphere $\mathbb{S}^{D-1} \triangleq \{ \mathbf{v} \in \mathbb{R}^D : \|\mathbf{v}\|_2 = 1 \}$:
$$\|\nabla R_\sigma(\mathbf{x})\|_2 = \sup_{\mathbf{v} \in \mathbb{S}^{D-1}} |\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})|$$

For any fixed unit vector $\mathbf{v} \in \mathbb{S}^{D-1}$, rotational invariance of the multivariate standard normal dictates that the linear projection is a standard scalar Gaussian:
$$Z \triangleq \mathbf{v}^\top \mathbf{u} \sim \mathcal{N}(0, \|\mathbf{v}\|_2^2) = \mathcal{N}(0, 1)$$

Applying the triangle inequality for integrals:
$$|\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})| = \frac{1}{\sigma} \left| \mathbb{E}_{\mathbf{u}} \left[ (R(\mathbf{x} + \sigma \mathbf{u}) - c) (\mathbf{v}^\top \mathbf{u}) \right] \right| \le \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}} \left[ |R(\mathbf{x} + \sigma \mathbf{u}) - c| \cdot |\mathbf{v}^\top \mathbf{u}| \right]$$
$$\le \frac{\Delta R}{2\sigma} \mathbb{E}_{Z \sim \mathcal{N}(0, 1)} [|Z|]$$

Evaluating the first absolute moment of a standard normal random variable:
$$\mathbb{E}[|Z|] = \int_{-\infty}^\infty |z| \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \, dz = \frac{2}{\sqrt{2\pi}} \int_0^\infty z e^{-z^2/2} \, dz = \sqrt{\frac{2}{\pi}}$$

Substituting back:
$$|\mathbf{v}^\top \nabla R_\sigma(\mathbf{x})| \le \frac{\Delta R}{2\sigma} \sqrt{\frac{2}{\pi}} = \frac{\Delta R}{\sigma \sqrt{2\pi}}$$

Taking the supremum over $\mathbf{v} \in \mathbb{S}^{D-1}$ establishes:
$$\boxed{L_{\text{after}} \le \frac{\Delta R}{\sigma \sqrt{2\pi}}}$$

Combining both bounds concludes the proof of Theorem 1. $\blacksquare$

---

## 3. The Tipping Point & The Ratio $\frac{L_{\text{before}}}{L_{\text{after}}}$

The two bounds represent an intersection between a horizontal line and an inverse hyperbolic curve:
$$y_1(\sigma) = L_{\text{before}}, \qquad y_2(\sigma) = \frac{\Delta R}{\sigma \sqrt{2\pi}}$$

Setting $y_1(\sigma^*) = y_2(\sigma^*)$ yields the **tipping point**:
$$\boxed{\sigma^* \triangleq \frac{\Delta R}{L_{\text{before}} \sqrt{2\pi}}}$$

The ratio of the Lipschitz constant before smoothing to after smoothing is therefore:

$$\boxed{\frac{L_{\text{before}}}{L_{\text{after}}} \ge \max\left( 1, \, \frac{\sigma}{\sigma^*} \right) = \begin{cases} 
1 & \text{if } \sigma \le \sigma^* = \dfrac{\Delta R}{L_{\text{before}} \sqrt{2\pi}} \quad (\text{regime dominated by } L_{\text{before}}) \\[12pt]
\dfrac{\sqrt{2\pi} \cdot L_{\text{before}} \cdot \sigma}{\Delta R} & \text{if } \sigma > \sigma^* = \dfrac{\Delta R}{L_{\text{before}} \sqrt{2\pi}} \quad (\text{linear reduction in } \sigma)
\end{cases}}$$

---

## 4. Multidimensional Taylor Expansion & Rigorous Bias Derivation

While larger $\sigma$ linearly reduces $L_{\text{after}}$, it perturbs the spatial evaluation points, introducing an approximation error:
$$\text{Bias}(\mathbf{x}) \triangleq |R_\sigma(\mathbf{x}) - R(\mathbf{x})|$$

### Step 1: Multidimensional Taylor Expansion at the Fixed Point $\mathbf{x}$
Assume $R \in C^4(\mathbb{R}^D)$. We perform a Taylor expansion of $R(\mathbf{x} + \sigma \mathbf{u})$ around the **fixed point $\mathbf{x}$** up to the **third order**:

$$R(\mathbf{x} + \sigma \mathbf{u}) = R(\mathbf{x}) + \sigma \nabla R(\mathbf{x})^\top \mathbf{u} + \frac{\sigma^2}{2} \mathbf{u}^\top \nabla^2 R(\mathbf{x}) \mathbf{u} + \frac{\sigma^3}{6} \nabla^3 R(\mathbf{x})[\mathbf{u}, \mathbf{u}, \mathbf{u}] + \mathcal{R}_3(\mathbf{x}, \sigma \mathbf{u})$$

where:
- $\nabla R(\mathbf{x}) \in \mathbb{R}^D$ is the gradient vector evaluated at $\mathbf{x}$.
- $\nabla^2 R(\mathbf{x}) \in \mathbb{R}^{D \times D}$ is the symmetric Hessian matrix evaluated at $\mathbf{x}$.
- $\nabla^3 R(\mathbf{x}) \in \mathcal{T}_3(\mathbb{R}^D)$ is the 3rd-order symmetric derivative tensor evaluated strictly at the deterministic expansion point $\mathbf{x}$.
- $\mathcal{R}_3(\mathbf{x}, \sigma \mathbf{u})$ is the 4th-order integral remainder:
  $$\mathcal{R}_3(\mathbf{x}, \sigma \mathbf{u}) = \frac{\sigma^4}{6} \int_0^1 (1 - t)^3 \nabla^4 R(\mathbf{x} + t \sigma \mathbf{u})[\mathbf{u}, \mathbf{u}, \mathbf{u}, \mathbf{u}] \, dt$$

---

### Step 2: Expectation and Elimination of Odd Terms

Taking the expectation $\mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)}[\cdot]$ term by term:

$$R_\sigma(\mathbf{x}) = R(\mathbf{x}) + \sigma \nabla R(\mathbf{x})^\top \mathbb{E}[\mathbf{u}] + \frac{\sigma^2}{2} \mathbb{E}\left[\mathbf{u}^\top \nabla^2 R(\mathbf{x}) \mathbf{u}\right] + \frac{\sigma^3}{6} \mathbb{E}\left[\nabla^3 R(\mathbf{x})[\mathbf{u}, \mathbf{u}, \mathbf{u}]\right] + \mathbb{E}[\mathcal{R}_3(\mathbf{x}, \sigma \mathbf{u})]$$

#### 1. First-Order Term Vanishes:
Because $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$ has zero mean ($\mathbb{E}[\mathbf{u}] = \mathbf{0}$):
$$\sigma \nabla R(\mathbf{x})^\top \mathbb{E}[\mathbf{u}] = \sigma \nabla R(\mathbf{x})^\top \mathbf{0} = 0$$

#### 2. Second-Order Term: Trace of the Hessian $\operatorname{tr}(\nabla^2 R(\mathbf{x}))$:
Using the cyclic trace property $\mathbf{u}^\top \mathbf{A} \mathbf{u} = \operatorname{tr}(\mathbf{A} \mathbf{u} \mathbf{u}^\top)$ and the linearity of expectation and trace:
$$\mathbb{E}\left[\mathbf{u}^\top \nabla^2 R(\mathbf{x}) \mathbf{u}\right] = \operatorname{tr}\left( \nabla^2 R(\mathbf{x}) \, \mathbb{E}[\mathbf{u} \mathbf{u}^\top] \right)$$
Since $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$, its covariance is the identity matrix $\mathbb{E}[\mathbf{u} \mathbf{u}^\top] = \mathbf{I}_D$:
$$\mathbb{E}\left[\mathbf{u}^\top \nabla^2 R(\mathbf{x}) \mathbf{u}\right] = \operatorname{tr}\left( \nabla^2 R(\mathbf{x}) \, \mathbf{I}_D \right) = \operatorname{tr}\left( \nabla^2 R(\mathbf{x}) \right)$$

#### 3. Third-Order Term Legally Vanishes by Gaussian Central Symmetry:
Because $\nabla^3 R(\mathbf{x})$ is evaluated at the **fixed, non-random point $\mathbf{x}$**, it can be factored out of the expectation:
$$\mathbb{E}\left[\nabla^3 R(\mathbf{x})[\mathbf{u}, \mathbf{u}, \mathbf{u}]\right] = \sum_{i=1}^D \sum_{j=1}^D \sum_{k=1}^D \frac{\partial^3 R}{\partial x_i \partial x_j \partial x_k}(\mathbf{x}) \, \mathbb{E}[u_i u_j u_k]$$

Since the standard multivariate Gaussian measure $\gamma_1$ is centrally symmetric (invariant under reflection $\mathbf{u} \mapsto -\mathbf{u}$):
$$\mathbb{E}[u_i u_j u_k] = 0, \quad \forall i, j, k \in \{1, \dots, D\}$$
Consequently:
$$\frac{\sigma^3}{6} \mathbb{E}\left[\nabla^3 R(\mathbf{x})[\mathbf{u}, \mathbf{u}, \mathbf{u}]\right] = 0$$

#### 4. Fourth-Order Remainder Bound:
Assuming $\|\nabla^4 R\|_{L^\infty} \le M_4$:
$$\left| \mathbb{E}[\mathcal{R}_3(\mathbf{x}, \sigma \mathbf{u})] \right| \le \frac{\sigma^4 M_4}{24} \mathbb{E}[\|\mathbf{u}\|_2^4] = \frac{\sigma^4 M_4}{24} D(D + 2) = \mathcal{O}(\sigma^4 D^2)$$

#### Assembling the Bias:
$$R_\sigma(\mathbf{x}) - R(\mathbf{x}) = \frac{\sigma^2}{2} \operatorname{tr}\left(\nabla^2 R(\mathbf{x})\right) + \mathcal{O}(\sigma^4)$$

$$\boxed{\text{Bias}(\mathbf{x}) = |R_\sigma(\mathbf{x}) - R(\mathbf{x})| = \frac{\sigma^2}{2} \left|\operatorname{tr}\left(\nabla^2 R(\mathbf{x})\right)\right| + \mathcal{O}(\sigma^4)}$$

#### Global Worst-Case Upper Bound:
Let $\lambda_i(\nabla^2 R(\mathbf{x}))$ be the eigenvalues of the Hessian. Because $\|\nabla^2 R(\mathbf{x})\|_2 \le H$:
$$|\lambda_i| \le \|\nabla^2 R(\mathbf{x})\|_2 \le H, \quad \forall i \in \{1, \dots, D\}$$
$$\left| \operatorname{tr}\left(\nabla^2 R(\mathbf{x})\right) \right| = \left| \sum_{i=1}^D \lambda_i \right| \le \sum_{i=1}^D |\lambda_i| \le D \cdot H$$

Hence, the worst-case bias over the entire domain satisfies:

$$\boxed{\sup_{\mathbf{x} \in \mathbb{R}^D} \text{Bias}(\mathbf{x}) \le \frac{D H}{2} \sigma^2}$$

---

## 5. The Minimax Optimization Trade-off

To find the optimal operational noise scale $\sigma_{\text{opt}}$, we formulate a Pareto regularized cost functional balancing the Lipschitz instability penalty $\mathcal{L}(\sigma)$ against the distortion bias $\mathcal{B}(\sigma)$:

$$\mathcal{J}(\sigma) \triangleq \alpha \mathcal{L}(\sigma) + \beta \mathcal{B}(\sigma) = \alpha \left( \frac{\Delta R}{\sigma \sqrt{2\pi}} \right) + \beta \left( \frac{D H}{2} \sigma^2 \right)$$

where $\alpha > 0$ and $\beta > 0$ are application-specific trade-off hyperparameters.

Defining positive constants:
$$C_1 \triangleq \frac{\alpha \Delta R}{\sqrt{2\pi}} > 0, \qquad C_2 \triangleq \frac{\beta D H}{2} > 0$$

The objective on $\sigma \in (0, \infty)$ is:
$$\mathcal{J}(\sigma) = \frac{C_1}{\sigma} + C_2 \sigma^2$$

### First-Order Necessary Condition:
$$\frac{d\mathcal{J}}{d\sigma} = -\frac{C_1}{\sigma^2} + 2 C_2 \sigma = 0 \iff 2 C_2 \sigma^3 = C_1 \iff \sigma^3 = \frac{C_1}{2 C_2}$$

Solving for $\sigma_{\text{opt}}$:
$$\sigma_{\text{opt}} = \left( \frac{C_1}{2 C_2} \right)^{1/3} = \left( \frac{\frac{\alpha \Delta R}{\sqrt{2\pi}}}{2 \left(\frac{\beta D H}{2}\right)} \right)^{1/3} = \left( \frac{\alpha \Delta R}{\beta \sqrt{2\pi} \cdot D \cdot H} \right)^{1/3}$$

Setting $\alpha = \beta = 1$:

$$\boxed{\sigma_{\text{opt}} = \left( \frac{\Delta R}{\sqrt{2\pi} \cdot D \cdot H} \right)^{1/3} = \mathcal{O}\left( \left( \frac{\Delta R}{D \cdot H} \right)^{1/3} \right)}$$

### Second-Order Sufficient Condition (Strict Convexity):
$$\frac{d^2 \mathcal{J}}{d\sigma^2} = \frac{2 C_1}{\sigma^3} + 2 C_2$$
For all $\sigma > 0$, since $C_1 > 0$ and $C_2 > 0$, we have $\frac{d^2 \mathcal{J}}{d\sigma^2} > 0$. Thus, $\mathcal{J}(\sigma)$ is strictly convex on $(0, \infty)$, confirming that $\sigma_{\text{opt}}$ is the unique global minimum. $\blacksquare$

---

## 6. The Curse of Dimensionality: Why Latent-Space Smoothing is Necessary

The appearance of the dimension $D$ in the denominator of the optimal smoothing scale:
$$\sigma_{\text{opt}} \propto D^{-1/3}$$
reveals a fundamental structural bottleneck in high-dimensional Gaussian smoothing.

### 6.1. The Mechanism of the Dimensional Bottleneck
The trade-off arises from the dimensional scaling mismatch between gradient regularization and bias accumulation:
1. **The Lipschitz bound is Dimension-Free**:
   $$L_{\text{after}} \le \frac{\Delta R}{\sigma \sqrt{2\pi}} = \mathcal{O}(D^0 \cdot \sigma^{-1})$$
   Testing along a 1D unit vector $\mathbf{v}$ projects the $D$-dimensional isotropic Gaussian onto a single univariate standard normal $Z \sim \mathcal{N}(0, 1)$, making the gradient bound completely independent of $D$.
2. **The Bias Scales Linearly with Dimension**:
   $$\text{Bias}(\mathbf{x}) \approx \frac{\sigma^2}{2} \operatorname{tr}\left(\nabla^2 R(\mathbf{x})\right) = \frac{\sigma^2}{2} \sum_{i=1}^D \lambda_i = \mathcal{O}(D \cdot H \cdot \sigma^2)$$
   Gaussian perturbation diffuses mass isotropically across all $D$ orthogonal coordinate axes. The curvature errors from all $D$ directions accumulate additively in the trace of the Hessian.

### 6.2. Concrete Quantitative Comparison: Pixel Space vs. Latent Space

| Parameter / Space | Pixel Space ($1024 \times 1024 \times 3$) | SDXL Latent Space ($128 \times 128 \times 4$) | Bottleneck Latent Space ($64 \times 64 \times 4$) |
| :--- | :---: | :---: | :---: |
| **Dimension $D$** | **$3,145,728$** | **$65,536$** | **$16,384$** |
| **Dimensional Ratio $\frac{D_{\text{pixel}}}{D}$** | $1\times$ | **$48\times$ smaller** | **$192\times$ smaller** |
| **Optimal Scale $\sigma_{\text{opt}} \propto D^{-1/3}$** | $\sigma_{\text{pixel}}$ | **$3.63 \times \sigma_{\text{pixel}}$** | **$5.77 \times \sigma_{\text{pixel}}$** |
| **Accumulated Bias at fixed $\sigma$** | $3.15 \times 10^6 \cdot \frac{H \sigma^2}{2}$ | **$48\times$ smaller** | **$192\times$ smaller** |

### 6.3. Theoretical & Algorithmic Implication
- **In Pixel Space ($D > 3 \times 10^6$)**: The overwhelming accumulation of trace errors forces $\sigma_{\text{opt}} \to 0$. At such infinitesimal noise scales, $\sigma < \sigma^*$, meaning Gaussian smoothing operates in the un-damped regime ($L_{\text{after}} \approx L_{\text{before}}$). Any attempt to increase $\sigma$ to achieve meaningful Lipschitz damping causes catastrophic semantic distortion (bias explosion).
- **In Latent Space ($D \le 65,536$)**: The compact dimensional manifold suppresses the trace accumulation by several orders of magnitude. This allows operating at substantial noise scales $\sigma \in [0.5, 1.0]$ where $\sigma \gg \sigma^*$, achieving massive Lipschitz reduction ($\frac{L_{\text{before}}}{L_{\text{after}}} \ge 50\times - 100\times$) while keeping the approximation bias within a strictly controlled error budget.
