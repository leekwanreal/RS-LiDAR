# Mathematical Analysis: Lipschitz Regularization via Gaussian-Smoothed Reward Models

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

### Analytical Summary:
- **Strict Regularity Guarantee**: For any noise level $\sigma > 0$, Gaussian smoothing guarantees that $L_{\text{after}}$ never exceeds the original Lipschitz constant ($L_{\text{after}} \le L_{\text{before}}$).
- **Active Lipschitz Damping**: As soon as $\sigma > \sigma^*$, the upper bound on the gradient magnitude scales strictly as $\mathcal{O}\left(\frac{1}{\sigma}\right)$, damping sharp gradient spikes, eliminating non-Lipschitz instabilities, and smoothing out spurious high-frequency local maxima in the reward landscape.
