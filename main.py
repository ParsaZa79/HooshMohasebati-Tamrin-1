import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.colors import ListedColormap
import random

class PerceptronManual:
    def __init__(self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self._rng = random.Random(self.random_state)

    def fit(self, X, y):
        n_features = len(X[0])
        self.w_ = [(self._rng.random() - 0.5) * 0.02 for _ in range(n_features)]
        self.b_ = (self._rng.random() - 0.5) * 0.02
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                for i in range(n_features):
                    self.w_[i] += update * xi[i]
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0 and _ > 0:
                 print(f'   * Hamgerayi dar epoch {_ + 1} rokh dad.')
                 self.errors_.extend([0] * (self.n_iter - len(self.errors_)))
                 break
        if not self.errors_ or self.errors_[-1] != 0:
            print(f'   * Hamgerayi kamel pas az {self.n_iter} epoch rokh nadad (Khataye Nahayi: {self.errors_[-1] if self.errors_ else "N/A"}).')

        return self

    def net_input(self, X):
        z = 0.0
        for i in range(len(self.w_)):
            z += X[i] * self.w_[i]
        z += self.b_
        return z

    def predict(self, X):
        return 1 if self.net_input(X) >= 0.0 else 0

def plot_decision_regions_manual(X, y, classifier, resolution=0.02, title=''):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = min(x[0] for x in X) - 0.5, max(x[0] for x in X) + 0.5
    x2_min, x2_max = min(x[1] for x in X) - 0.5, max(x[1] for x in X) + 0.5

    xx1 = []
    curr = x1_min
    while curr < x1_max:
        xx1.append(curr)
        curr += resolution
    xx2 = []
    curr = x2_min
    while curr < x2_max:
        xx2.append(curr)
        curr += resolution

    Z = []
    grid_points = []
    for i2 in xx2:
        row_preds = []
        for i1 in xx1:
            pred = classifier.predict([i1, i2])
            row_preds.append(pred)
            grid_points.append([i1, i2])
        Z.append(row_preds)

    plt.figure(figsize=(8, 6))

    X_grid = np.array(grid_points)
    Z_flat = np.array([item for sublist in Z for item in sublist])
    plt.scatter(X_grid[:, 0], X_grid[:, 1], c=Z_flat, cmap=cmap, alpha=0.3, s=10)

    y_int = [int(label) for label in y]
    unique_labels = sorted(list(set(y_int)))

    X_col0 = [item[0] for item in X]
    X_col1 = [item[1] for item in X]

    for idx, cl in enumerate(unique_labels):
        x0_class = [X_col0[i] for i, label in enumerate(y_int) if label == cl]
        x1_class = [X_col1[i] for i, label in enumerate(y_int) if label == cl]
        plt.scatter(x=x0_class,
                    y=x1_class,
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

def plot_errors(errors, title='', n_iter=None):
    plt.figure(figsize=(8, 6))
    if n_iter:
        plt.plot(range(1, n_iter + 1), errors[:n_iter], marker='o')
        plt.xticks(range(1, n_iter + 1))
        if n_iter > 20:
            plt.xticks(range(0, n_iter + 1, max(1, n_iter // 10)))
    else:
        n_iter_actual = len(errors)
        plt.plot(range(1, n_iter_actual + 1), errors, marker='o')
        plt.xticks(range(1, n_iter_actual + 1))
        if n_iter_actual > 20:
             plt.xticks(range(0, n_iter_actual + 1, max(1, n_iter_actual // 10)))

    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.grid(True)
    plt.tight_layout()

print("--- Bakhsh Aval: Darvazeh-haye Manteghi (Bedune Ketabkhane) ---")

# Logic gate sections moved to gates_part.py
if __name__ == "__main__":
    try:
        from gates_part import run_all_gates
        run_all_gates()
    except ImportError:
        print("[Warning] gates_part.py not found. Logic gates section is skipped.")

    # --- Diabetes section moved to diabetes_part.py ---
    try:
        from diabetes_part import run_diabetes_section
        run_diabetes_section()
    except ImportError:
        print("[Warning] diabetes_part.py not found. Diabetes section is skipped.")
