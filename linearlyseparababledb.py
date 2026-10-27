import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.neural_network import MLPClassifier

# Set seed for reproducible industrial simulation
np.random.seed(42)

# =====================================================================
# 1. DATA GENERATION
# =====================================================================
# Dataset 1: Linearly Separable (Two isotropic Gaussian Blobs)
X_lin, y_lin = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=42)

# Dataset 2: Non-Linearly Separable (Two interlocking half-moons)
X_nonlin, y_nonlin = make_moons(n_samples=200, noise=0.2, random_state=42)

# =====================================================================
# 2. HELPER FUNCTION TO PLOT DECISION BOUNDARIES
# =====================================================================
def plot_decision_boundary(ax, model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Predict over the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Draw decision boundary contour
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', label='Class 0', s=45)
    ax.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', label='Class 1', s=45)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)

# Setup 2x2 Subplot Matrix
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# =====================================================================
# 3. LINEARLY SEPARABLE CASES (CONVEX LOSS LANDSCAPE)
# =====================================================================
# Graph 1: Wrong Method (Linear MLP stuck in early stopping / poor learning rate)
mlp_lin_wrong = MLPClassifier(hidden_layer_sizes=(), activation='identity', max_iter=5, 
                              learning_rate_init=0.0001, random_state=42)
mlp_lin_wrong.fit(X_lin, y_lin)
plot_decision_boundary(axes[0, 0], mlp_lin_wrong, X_lin, y_lin, 
                       "1. Linearly Separable: WRONG Method\n(Under-trained / Stuck at Suboptimal Local Point)")

# Graph 2: Right Method (Standard Adam / SGD convex optimization)
mlp_lin_right = MLPClassifier(hidden_layer_sizes=(), activation='identity', max_iter=500, 
                              solver='adam', learning_rate_init=0.05, random_state=42)
mlp_lin_right.fit(X_lin, y_lin)
plot_decision_boundary(axes[0, 1], mlp_lin_right, X_lin, y_lin, 
                       "2. Linearly Separable: RIGHT Method\n(Converged to Global Minimum on Convex Hyperplane)")

# =====================================================================
# 4. NON-LINEARLY SEPARABLE CASES (NON-CONVEX LOSS LANDSCAPE)
# =====================================================================
# Graph 3: Wrong Method (Linear single-layer model applied to a non-linear manifold)
mlp_nonlin_wrong = MLPClassifier(hidden_layer_sizes=(), activation='identity', max_iter=500, random_state=42)
mlp_nonlin_wrong.fit(X_nonlin, y_nonlin)
plot_decision_boundary(axes[1, 0], mlp_nonlin_wrong, X_nonlin, y_nonlin, 
                       "3. Non-Linearly Separable: WRONG Method\n(Linear Classifier trapped in High-Bias Local Minimum)")

# Graph 4: Right Method (Deep MLP with Hidden Layers & Non-Linear ReLU Activation)
mlp_nonlin_right = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', 
                                max_iter=1000, random_state=42)
mlp_nonlin_right.fit(X_nonlin, y_nonlin)
plot_decision_boundary(axes[1, 1], mlp_nonlin_right, X_nonlin, y_nonlin, 
                       "4. Non-Linearly Separable: RIGHT Method\n(Multi-Layer ReLU MLP - Reaches Global Minimum)")

plt.tight_layout()
plt.show()