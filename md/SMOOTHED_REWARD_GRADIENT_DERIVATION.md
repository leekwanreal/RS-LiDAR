# Mathematical Proof: Dimension-Free Lipschitz Bound of Smoothed Reward

---

## 1. Theorem Statement

### Theorem (Dimension-Free Lipschitz Bound)
Let $R: \mathbb{R}^D \to \mathbb{R}$ be any bounded reward function (possibly discontinuous and non-differentiable) such that:
$$R(\mathbf{x}) \in [R_{\min}, R_{\max}], \quad \forall \mathbf{x} \in \mathbb{R}^D$$
with finite range $\Delta R \triangleq R_{\max} - R_{\min} < \infty$.

For any smoothing radius $\sigma > 0$, define the **Gaussian smoothed surrogate**:
$$R_\sigma(\mathbf{x}) \triangleq \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u})]$$

Then, $R_\sigma(\mathbf{x})$ is smooth and globally Lipschitz continuous on $\mathbb{R}^D$. Its gradient satisfies the **dimension-free** bound:

$$\boxed{\|\nabla_\mathbf{x} R_\sigma(\mathbf{x})\|_2 \le \frac{\Delta R}{\sigma \sqrt{2\pi}} = L_\sigma, \quad \forall \mathbf{x} \in \mathbb{R}^D}$$

In particular, the Lipschitz constant $L_\sigma$ is strictly independent of the ambient dimension $D$.

---

## 2. Rigorous Step-by-Step Proof

The proof proceeds through six structured steps:
1. Formulation as a Gaussian convolution via substitution (change of variables).
2. Differentiation under the integral sign (Leibniz rule).
3. Score function form via inverse substitution.
4. Variance reduction and centering via baseline invariance.
5. Isotropic 1D projection and Gaussian first absolute moment.
6. Duality supremum to establish the dimension-free bound.

---

### Step 1: Integral Formulation & Substitution (Change of Variables)

The vector $\mathbf{u} \in \mathbb{R}^D$ follows a standard isotropic Gaussian distribution $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$, whose probability density function is:
$$p(\mathbf{u}) = \frac{1}{(2\pi)^{D/2}} \exp\left( -\frac{1}{2} \|\mathbf{u}\|_2^2 \right)$$

By definition of the expectation:
$$R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{x} + \sigma \mathbf{u}) \, p(\mathbf{u}) \, d\mathbf{u} = \int_{\mathbb{R}^D} R(\mathbf{x} + \sigma \mathbf{u}) \frac{1}{(2\pi)^{D/2}} \exp\left( -\frac{1}{2} \|\mathbf{u}\|_2^2 \right) d\mathbf{u}$$

#### Change of Variables:
Define the bijective spatial mapping:
$$\mathbf{z} = \mathbf{x} + \sigma \mathbf{u} \iff \mathbf{u} = \frac{\mathbf{z} - \mathbf{x}}{\sigma}$$

The Jacobian matrix of this transformation with respect to $\mathbf{u}$ is:
$$\mathbf{J}_{\mathbf{z}}(\mathbf{u}) = \frac{\partial \mathbf{z}}{\partial \mathbf{u}} = \sigma \mathbf{I}_D$$

The Jacobian determinant is:
$$\det(\mathbf{J}_{\mathbf{z}}) = \det(\sigma \mathbf{I}_D) = \sigma^D$$

Hence, the differential volume element transforms according to:
$$d\mathbf{z} = \sigma^D d\mathbf{u} \implies d\mathbf{u} = \frac{1}{\sigma^D} d\mathbf{z}$$

Substitute $\mathbf{u} = \frac{\mathbf{z} - \mathbf{x}}{\sigma}$ and $d\mathbf{u} = \sigma^{-D} d\mathbf{z}$ into the integral:
$$R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{z}) \frac{1}{(2\pi)^{D/2}} \exp\left( -\frac{\|\mathbf{z} - \mathbf{x}\|_2^2}{2\sigma^2} \right) \frac{1}{\sigma^D} d\mathbf{z}$$

Combining the normalizers yields the Gaussian kernel $\phi_\sigma(\mathbf{a}) \triangleq \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left( -\frac{\|\mathbf{a}\|_2^2}{2\sigma^2} \right)$:

$$R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{z}) \phi_\sigma(\mathbf{z} - \mathbf{x}) \, d\mathbf{z} = (R * \phi_\sigma)(\mathbf{x})$$

> **Key Consequence**: 
> In this convolution form, $\mathbf{x}$ appears exclusively in the smooth mollifier $\phi_\sigma(\mathbf{z} - \mathbf{x})$, completely decoupling the differentiation from the regularity of $R(\cdot)$. Thus, $R_\sigma$ is smooth even if $R$ is discontinuous or a black-box evaluation.

---

### Step 2: Differentiating Under the Integral Sign (Leibniz Rule)

Since $R$ is bounded ($|R(\mathbf{z})| \le \max(|R_{\min}|, |R_{\max}|)$) and $\phi_\sigma(\mathbf{z} - \mathbf{x})$ is smooth with exponentially decaying tails, the dominated convergence theorem allows swapping differentiation and integration:

$$\nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{z}) \, \nabla_\mathbf{x} \phi_\sigma(\mathbf{z} - \mathbf{x}) \, d\mathbf{z}$$

#### Evaluating $\nabla_\mathbf{x} \phi_\sigma(\mathbf{z} - \mathbf{x})$:
Write the squared Euclidean norm explicitly:
$$\|\mathbf{z} - \mathbf{x}\|_2^2 = \sum_{j=1}^D (z_j - x_j)^2$$

Differentiating component-wise with respect to $x_k$:
$$\frac{\partial}{\partial x_k} \|\mathbf{z} - \mathbf{x}\|_2^2 = 2(z_k - x_k)(-1) = -2(z_k - x_k)$$
In vector notation:
$$\nabla_\mathbf{x} \|\mathbf{z} - \mathbf{x}\|_2^2 = -2(\mathbf{z} - \mathbf{x})$$

By the multivariate chain rule:
$$\nabla_\mathbf{x} \phi_\sigma(\mathbf{z} - \mathbf{x}) = \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left( -\frac{\|\mathbf{z} - \mathbf{x}\|_2^2}{2\sigma^2} \right) \cdot \nabla_\mathbf{x} \left( -\frac{\|\mathbf{z} - \mathbf{x}\|_2^2}{2\sigma^2} \right)$$
$$= \phi_\sigma(\mathbf{z} - \mathbf{x}) \cdot \left( -\frac{-2(\mathbf{z} - \mathbf{x})}{2\sigma^2} \right) = \frac{\mathbf{z} - \mathbf{x}}{\sigma^2} \phi_\sigma(\mathbf{z} - \mathbf{x})$$

Substituting this back into the integral:
$$\nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{z}) \left( \frac{\mathbf{z} - \mathbf{x}}{\sigma^2} \right) \phi_\sigma(\mathbf{z} - \mathbf{x}) \, d\mathbf{z}$$

---

### Step 3: Inverse Substitution & Stein Score Form

We now map back to the standard Gaussian perturbation $\mathbf{u} \in \mathbb{R}^D$ using:
$$\mathbf{z} = \mathbf{x} + \sigma \mathbf{u} \implies \mathbf{z} - \mathbf{x} = \sigma \mathbf{u}$$
$$d\mathbf{z} = \sigma^D d\mathbf{u}$$

The volume-weighted Gaussian density converts back as:
$$\phi_\sigma(\mathbf{z} - \mathbf{x}) \, d\mathbf{z} = \frac{1}{(2\pi \sigma^2)^{D/2}} \exp\left( -\frac{\|\sigma \mathbf{u}\|_2^2}{2\sigma^2} \right) \sigma^D d\mathbf{u} = \frac{1}{(2\pi)^{D/2}} \exp\left( -\frac{1}{2} \|\mathbf{u}\|_2^2 \right) d\mathbf{u} = p(\mathbf{u}) d\mathbf{u}$$

The vector factor transforms as:
$$\frac{\mathbf{z} - \mathbf{x}}{\sigma^2} = \frac{\sigma \mathbf{u}}{\sigma^2} = \frac{\mathbf{u}}{\sigma}$$

Plugging these back into the integral:
$$\nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \int_{\mathbb{R}^D} R(\mathbf{x} + \sigma \mathbf{u}) \left( \frac{\mathbf{u}}{\sigma} \right) p(\mathbf{u}) \, d\mathbf{u} = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [R(\mathbf{x} + \sigma \mathbf{u}) \, \mathbf{u}]$$

---

### Step 4: Zero-Mean Centering (Baseline Invariance)

Since $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$, its mean vector is zero:
$$\mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [\mathbf{u}] = \mathbf{0}$$

Consequently, for any scalar constant $c \in \mathbb{R}$ independent of $\mathbf{u}$:
$$\mathbb{E}_{\mathbf{u}} [c \, \mathbf{u}] = c \, \mathbb{E}_{\mathbf{u}}[\mathbf{u}] = \mathbf{0}$$

Subtracting $c \, \mathbf{u}$ inside the expectation leaves the gradient completely unchanged:
$$\nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [(R(\mathbf{x} + \sigma \mathbf{u}) - c) \, \mathbf{u}]$$

#### Optimal Midpoint Centering:
To minimize the maximum deviation of the reward factor, choose $c$ as the **midpoint** of the range $[R_{\min}, R_{\max}]$:
$$c \triangleq \frac{R_{\max} + R_{\min}}{2}$$

For any evaluation $R(\cdot) \in [R_{\min}, R_{\max}]$, the centered reward satisfies:
$$-\frac{R_{\max} - R_{\min}}{2} \le R(\cdot) - c \le \frac{R_{\max} - R_{\min}}{2}$$
$$\implies |R(\mathbf{x} + \sigma \mathbf{u}) - c| \le \frac{\Delta R}{2} \quad$$

---

### Step 5: 1D Isotropic Projection

To bound the Euclidean norm $\|\nabla_\mathbf{x} R_\sigma(\mathbf{x})\|_2$, we utilize the dual representation of the $\ell_2$-norm on the unit sphere $\mathbb{S}^{D-1} \triangleq \{ \mathbf{v} \in \mathbb{R}^D : \|\mathbf{v}\|_2 = 1 \}$:

$$\|\nabla_\mathbf{x} R_\sigma(\mathbf{x})\|_2 = \sup_{\mathbf{v} \in \mathbb{S}^{D-1}} \big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big|$$

Let $\mathbf{v} \in \mathbb{S}^{D-1}$ be an arbitrary fixed unit vector. Projecting the gradient along $\mathbf{v}$:
$$\mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}} \left[ (R(\mathbf{x} + \sigma \mathbf{u}) - c) \, (\mathbf{v}^\top \mathbf{u}) \right]$$

#### Rotational Invariance (Isotropy) of Multivariate Gaussian:
A fundamental property of $\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)$ is that any linear combination $\mathbf{v}^\top \mathbf{u}$ is a univariate Gaussian random variable:
$$Z \triangleq \mathbf{v}^\top \mathbf{u} \sim \mathcal{N}\left(0, \, \|\mathbf{v}\|_2^2\right)$$
Since $\mathbf{v} \in \mathbb{S}^{D-1}$ is a unit vector ($\|\mathbf{v}\|_2 = 1$):
$$Z \sim \mathcal{N}(0, 1)$$

#### Bounding the Directional Derivative via $|\mathbb{E}[W]| \le \mathbb{E}[|W|]$:
Define the scalar random variable $W \in \mathbb{R}$ as the product of two real scalars:
$$W \triangleq \underbrace{(R(\mathbf{x} + \sigma \mathbf{u}) - c)}_{A \in \mathbb{R}} \cdot \underbrace{(\mathbf{v}^\top \mathbf{u})}_{B \in \mathbb{R}}$$

The projected directional derivative is simply $\mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) = \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}}[W]$.

Taking the absolute value:
$$\big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big| = \frac{1}{\sigma} \big| \mathbb{E}_{\mathbf{u}}[W] \big|$$

We now apply the **fundamental property of expectations**:
$$\big| \mathbb{E}[W] \big| \le \mathbb{E}[|W|]$$

> **Mathematical Justification**:
> 1. **Via Jensen's Inequality**: The absolute value function $g(t) = |t|$ is convex on $\mathbb{R}$. By Jensen's inequality, $g(\mathbb{E}[W]) \le \mathbb{E}[g(W)] \implies |\mathbb{E}[W]| \le \mathbb{E}[|W|]$.
> 2. **Via Triangle Inequality for Integrals**: In integral form with density $p(\mathbf{u}) \ge 0$, this is simply $\left| \int W(\mathbf{u}) p(\mathbf{u}) d\mathbf{u} \right| \le \int |W(\mathbf{u})| p(\mathbf{u}) d\mathbf{u}$.

Since the absolute value of the product of two real numbers is the product of their absolute values ($|W| = |A \cdot B| = |A| \cdot |B|$):
$$|W| = |R(\mathbf{x} + \sigma \mathbf{u}) - c| \cdot |\mathbf{v}^\top \mathbf{u}|$$

Therefore:
$$\big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big| \le \frac{1}{\sigma} \mathbb{E}_{\mathbf{u}} \left[ |R(\mathbf{x} + \sigma \mathbf{u}) - c| \cdot |\mathbf{v}^\top \mathbf{u}| \right]$$

#### Monotonicity of Expectation:
Because $|R(\mathbf{x} + \sigma \mathbf{u}) - c| \le \frac{\Delta R}{2}$ everywhere, and $|\mathbf{v}^\top \mathbf{u}| \ge 0$, we have:
$$|R(\mathbf{x} + \sigma \mathbf{u}) - c| \cdot |\mathbf{v}^\top \mathbf{u}| \le \frac{\Delta R}{2} |\mathbf{v}^\top \mathbf{u}|$$

By the monotonicity of the expectation operator ($X \le Y \implies \mathbb{E}[X] \le \mathbb{E}[Y]$):
$$\big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big| \le \frac{\Delta R}{2\sigma} \mathbb{E} [|Z|], \quad \text{where } Z \triangleq \mathbf{v}^\top \mathbf{u} \sim \mathcal{N}(0, 1)$$

---

### Step 6: Calculation of the 1D Gaussian Absolute Moment $\mathbb{E}[|Z|]$

Let $Z \sim \mathcal{N}(0, 1)$. Its probability density is $p_Z(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}$.

Compute the first absolute moment $\mathbb{E}[|Z|]$:
$$\mathbb{E}[|Z|] = \int_{-\infty}^\infty |z| \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2} dz$$

Since the integrand is strictly symmetric (even function):
$$\mathbb{E}[|Z|] = 2 \int_0^\infty z \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2} dz = \frac{2}{\sqrt{2\pi}} \int_0^\infty z e^{-z^2 / 2} dz$$

Perform the substitution:
$$w = \frac{z^2}{2} \implies dw = z dz$$
When $z = 0 \implies w = 0$; as $z \to \infty \implies w \to \infty$.

Thus:
$$\int_0^\infty z e^{-z^2 / 2} dz = \int_0^\infty e^{-w} dw = \left[ -e^{-w} \right]_0^\infty = 0 - (-1) = 1$$

Therefore:
$$\mathbb{E}[|Z|] = \frac{2}{\sqrt{2\pi}} \cdot 1 = \sqrt{\frac{4}{2\pi}} = \sqrt{\frac{2}{\pi}}$$

---

### Step 7: Supremum Bound & Conclusion

Substitute $\mathbb{E}[|Z|] = \sqrt{\frac{2}{\pi}}$ back into the directional projection inequality:

$$\big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big| \le \frac{\Delta R}{2\sigma} \sqrt{\frac{2}{\pi}} = \frac{\Delta R}{\sigma} \cdot \frac{\sqrt{2}}{2\sqrt{\pi}} = \frac{\Delta R}{\sigma \sqrt{2\pi}}$$

Because this bound holds identically for **every** unit vector $\mathbf{v} \in \mathbb{S}^{D-1}$:

$$\|\nabla_\mathbf{x} R_\sigma(\mathbf{x})\|_2 = \sup_{\|\mathbf{v}\|_2 = 1} \big| \mathbf{v}^\top \nabla_\mathbf{x} R_\sigma(\mathbf{x}) \big| \le \frac{\Delta R}{\sigma \sqrt{2\pi}} = L_\sigma$$

Finally, by the Mean Value Theorem on $\mathbb{R}^D$, for any two states $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{R}^D$:
$$|R_\sigma(\mathbf{x}_1) - R_\sigma(\mathbf{x}_2)| \le \left( \sup_{\mathbf{x} \in \mathbb{R}^D} \|\nabla_\mathbf{x} R_\sigma(\mathbf{x})\|_2 \right) \|\mathbf{x}_1 - \mathbf{x}_2\|_2 \le L_\sigma \|\mathbf{x}_1 - \mathbf{x}_2\|_2$$

$$\blacksquare$$

---

## 3. Why the "Dimension-Free" Property is Profound

| Property | Standard Naive Gradient Estimate | Smoothed Reward via Isotropic Projection |
| :--- | :--- | :--- |
| **Bound Dependency** | Scales with $\mathcal{O}(\sqrt{D})$ | **$\mathcal{O}(1)$ (Completely Dimension-Free)** |
| **At Latent Res ($128 \times 128 \times 4$)** | $\sqrt{65,536} = 256\times$ larger | **Invariant to $D = 65,536$** |
| **At Pixel Res ($1024 \times 1024 \times 3$)** | $\sqrt{3,145,728} \approx 1,774\times$ larger | **Invariant to $D = 3,145,728$** |

### Mathematical Mechanism Behind the Dimension Freedom:
1. **Naive Cauchy-Schwarz Bound (Dimension-Dependent)**:
   If one bounded the norm crudely using $\|\mathbb{E}[\mathbf{u}]\|_2 \le \mathbb{E}[\|\mathbf{u}\|_2]$:
   $$\mathbb{E}_{\mathbf{u} \sim \mathcal{N}(0, \mathbf{I}_D)} [\|\mathbf{u}\|_2] \approx \sqrt{D}$$
   which would blow up as $D \to \infty$.
2. **Isotropic 1D Projection Duality (Dimension-Free)**:
   By testing along a 1D unit vector $\mathbf{v}$, rotational symmetry collapses the entire $D$-dimensional multivariate Gaussian onto a **single 1D scalar Gaussian** $Z = \mathbf{v}^\top \mathbf{u} \sim \mathcal{N}(0, 1)$, whose expected absolute magnitude is always identically $\sqrt{2/\pi}$, regardless of whether $D = 1$ or $D = 1,000,000$!

---

## 4. Practical Estimators for RS-LiDAR Sampling

In algorithmic implementations (e.g. Randomized Smoothing Guidance in Diffusion Sampling), the true expectation is approximated via Monte Carlo with $K$ particles:

### 1. Centered Monte Carlo Estimator (Single-Sided)
$$\widehat{\nabla}_\mathbf{x} R_\sigma(\mathbf{x}) = \frac{1}{K \sigma} \sum_{k=1}^K \big( R(\mathbf{x} + \sigma \mathbf{u}_k) - R(\mathbf{x}) \big) \mathbf{u}_k$$

### 2. Antithetic Symmetric Estimator (Variance Reduced)
Pairing $+\mathbf{u}_k$ and $-\mathbf{u}_k$:
$$\widehat{\nabla}_\mathbf{x}^{\text{anti}} R_\sigma(\mathbf{x}) = \frac{1}{2 K \sigma} \sum_{k=1}^K \big( R(\mathbf{x} + \sigma \mathbf{u}_k) - R(\mathbf{x} - \sigma \mathbf{u}_k) \big) \mathbf{u}_k$$
This eliminates odd-order error terms, achieving $\mathcal{O}(\sigma^2)$ bias instead of $\mathcal{O}(\sigma)$.
