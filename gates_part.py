import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import ListedColormap

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

# Logic gate functions
def run_and_gate():
    print("\n[1] Amuzesh Perceptron baraye Darvazeh AND")
    X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_and = [0, 0, 0, 1]
    ppn_and = PerceptronManual(eta=0.1, n_iter=10, random_state=1)
    ppn_and.fit(X_and, y_and)
    print(f"   Vazn-haye Nahayi: {ppn_and.w_}")
    print(f"   Bayas Nahayi: {ppn_and.b_}")
    plot_decision_regions_manual(X_and, y_and, classifier=ppn_and, title='Decision Boundary for AND Gate')
    plot_errors(ppn_and.errors_, title='Training Error for AND Gate', n_iter=ppn_and.n_iter)
    plt.show()
    return X_and, y_and, ppn_and

def run_or_gate():
    print("\n[2] Amuzesh Perceptron baraye Darvazeh OR")
    X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_or = [0, 1, 1, 1]
    ppn_or = PerceptronManual(eta=0.1, n_iter=10, random_state=1)
    ppn_or.fit(X_or, y_or)
    print(f"   Vazn-haye Nahayi: {ppn_or.w_}")
    print(f"   Bayas Nahayi: {ppn_or.b_}")
    plot_decision_regions_manual(X_or, y_or, classifier=ppn_or, title='Decision Boundary for OR Gate')
    plot_errors(ppn_or.errors_, title='Training Error for OR Gate', n_iter=ppn_or.n_iter)
    plt.show()
    return X_or, y_or, ppn_or

def run_xor_gate():
    print("\n[3] Amuzesh Perceptron baraye Darvazeh XOR")
    X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_xor = [0, 1, 1, 0]
    ppn_xor = PerceptronManual(eta=0.1, n_iter=50, random_state=1)
    ppn_xor.fit(X_xor, y_xor)
    print(f"   Vazn-haye Nahayi (XOR): {ppn_xor.w_}")
    print(f"   Bayas Nahayi (XOR): {ppn_xor.b_}")
    plot_decision_regions_manual(X_xor, y_xor, classifier=ppn_xor, title='Decision Boundary for XOR Gate (Perceptron struggles)')
    plot_errors(ppn_xor.errors_, title='Training Error for XOR Gate', n_iter=ppn_xor.n_iter)
    plt.show()
    print("   Tavvajo: Perceptron tak-laye nemitavanad dade-haye XOR ra ke khatti jodapazir nistand, be dorosti tabaghebandi konad.")
    print("   Namudare khata baraye XOR be sefr nemiresad.")
    return X_xor, y_xor, ppn_xor

def run_gate_analysis(X_and, y_and):
    print("\n[4] Tahlil Asar Nerkh Yadgiri va Tekrar baraye Darvazeh AND")
    etas = [0.01, 0.1, 1.0]
    n_iters_analysis = [5, 15, 30]

    for eta_val in etas:
        for n_iter_val in n_iters_analysis:
            print(f"   --- Amuzesh AND ba eta={eta_val}, n_iter={n_iter_val} ---")
            ppn_analysis = PerceptronManual(eta=eta_val, n_iter=n_iter_val, random_state=1)
            ppn_analysis.fit(X_and, y_and)
            plot_errors(ppn_analysis.errors_,
                        title=f'AND Error: eta={eta_val}, n_iter={n_iter_val}',
                        n_iter=n_iter_val)
            plt.show()
    
    print("   Tahlil: Nerkh Yadgiri paeen-tar (0.01) momken ast be tekrare bishtari baraye hamgerayi niaz dashte bashad.")
    print("   Nerkh Yadgiri balatar (1.0) momken ast sari'tar hamgera shavad vali gahi baes be nowsan mishavad.")
    print("   Tedade Tekrar kam (5) momken ast baraye hamgerayi kafi nabashad, dar hali ke tedade bishtar (15, 30) mamulan kafi ast.")

def run_all_gates():
    print("--- Bakhsh Aval: Darvazeh-haye Manteghi (Bedune Ketabkhane) ---")
    X_and, y_and, _ = run_and_gate()
    run_or_gate()
    run_xor_gate()
    run_gate_analysis(X_and, y_and) 