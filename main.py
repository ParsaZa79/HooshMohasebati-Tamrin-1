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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class PerceptronNumPy:
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self._rng = np.random.RandomState(self.random_state)

    def fit(self, X, y):
        n_features = X.shape[1]
        self.w_ = self._rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.b_ = np.float_(0.)
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        if not self.errors_ or self.errors_[-1] != 0:
             print(f'   * Hamgerayi kamel pas az {self.n_iter} epoch rokh nadad (Khataye Nahayi: {self.errors_[-1] if self.errors_ else "N/A"}).')
        else:
             print(f'   * Hamgerayi dar epoch zood-hengam rokh dad.')

        return self

    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions_numpy(X, y, classifier, feature_indices=(0, 1), feature_names=None, resolution=0.02, title=''):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    X_plot = X[:, feature_indices]

    x1_min, x1_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    x2_min, x2_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    print("   *Tavvajo: Marz-e Tasmim faghat bar asase do vizhegi entekhab shode rasm mishavad.")
    ppn_2d = PerceptronNumPy(eta=classifier.eta, n_iter=classifier.n_iter, random_state=classifier.random_state)
    ppn_2d.fit(X_plot, y)

    Z = ppn_2d.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X_plot[y == cl, 0],
                    y=X_plot[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

    plt.title(title)
    if feature_names:
        plt.xlabel(f'{feature_names[feature_indices[0]]} [standardized]')
        plt.ylabel(f'{feature_names[feature_indices[1]]} [standardized]')
    else:
        plt.xlabel(f'Feature {feature_indices[0]+1} [standardized]')
        plt.ylabel(f'Feature {feature_indices[1]+1} [standardized]')
    plt.legend(loc='upper left')
    plt.grid(True)

print("\n\n--- Bakhsh Dovom: Majmue Dade Diabetes (ba NumPy va Ketabkhane-haye Komaki) ---")

try:
    df = pl.read_csv('diabetes.csv')
    print("[1] Barghozari Dade-haye Diabetes movafaghiyat-amiz bud.")
    print(f"   Tedade Nemune-ha: {df.height}, Tedade Vizhegi-ha: {df.width - 1}")

    X_diabetes = df.drop('Outcome').to_numpy()
    y_diabetes = df['Outcome'].to_numpy()
    feature_names_diabetes = df.drop('Outcome').columns

    X_train, y_train = X_diabetes, y_diabetes
    X_test, y_test = X_diabetes, y_diabetes

    print("[2] Standard Sazi-ye Vizhegi-ha...")
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

except FileNotFoundError:
    print("Khatta: File 'diabetes.csv' yaft nashod. Lotfan file ra kenar script gharar dahid ya masire sahih ra moshakhas konid.")
    exit()
except Exception as e:
    print(f"Khataye pishbini nashode dar barghozari ya pardazesh dade: {e}")
    exit()

print("[3] Amuzesh Model Perceptron roye Dade Diabetes...")
ppn_diabetes = PerceptronNumPy(eta=0.001, n_iter=100, random_state=1)
ppn_diabetes.fit(X_train_std, y_train)

y_pred_train = ppn_diabetes.predict(X_train_std)

accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"\n[4] Daghighat Model roye kole dade (dade amuzeshi): {accuracy_train:.4f}")

print("[5] Rasme Namudare Khataye Amuzesh...")
plot_errors(ppn_diabetes.errors_, title='Training Error for Diabetes Dataset', n_iter=ppn_diabetes.n_iter)
plt.show()

print("[6] Rasme Marz-e Tasmim (bar asase Glucose va BMI)...")
try:
    glucose_idx = feature_names_diabetes.index('Glucose')
    bmi_idx = feature_names_diabetes.index('BMI')
    plot_decision_regions_numpy(X_train_std, y_train, classifier=ppn_diabetes,
                                feature_indices=(glucose_idx, bmi_idx),
                                feature_names=feature_names_diabetes,
                                title='Decision Boundary (Glucose vs BMI)')
    plt.show()
except ValueError:
    print("   Khatta: Name vizhegi 'Glucose' ya 'BMI' dar soton-ha yaft nashod.")
except IndexError:
     print("   Khatta: Andis-e vizhegi-haye Glucose/BMI kharej az mahdude ast.")

print("\n[7] Tahlil Asar Nerkh Yadgiri va Tekrar baraye Diabetes...")
etas_diabetes = [0.0001, 0.001, 0.01]
n_iters_diabetes = [50, 100, 200]

results_analysis = {}

for eta_val in etas_diabetes:
    for n_iter_val in n_iters_diabetes:
        print(f"   --- Amuzesh Diabetes ba eta={eta_val}, n_iter={n_iter_val} ---")
        ppn_d_analysis = PerceptronNumPy(eta=eta_val, n_iter=n_iter_val, random_state=1)
        ppn_d_analysis.fit(X_train_std, y_train)
        acc = accuracy_score(y_train, ppn_d_analysis.predict(X_train_std))
        results_analysis[(eta_val, n_iter_val)] = (acc, ppn_d_analysis.errors_[-1] if ppn_d_analysis.errors_ else -1)
        print(f"      Daghighat Nahayi: {acc:.4f}, Khataye epoch akhar: {results_analysis[(eta_val, n_iter_val)][1]}")

print("\n   Kholase Tahlil:")
for params, (acc, final_err) in results_analysis.items():
    print(f"   Eta={params[0]}, Iter={params[1]} -> Accuracy: {acc:.4f}, Final Epoch Errors: {final_err}")

print("\n   Tahlil Kolli:")
print("   - Dade Diabetes ehtemalan be tore kamel khatti jodapazir nist, banabarin Perceptron momken ast be daghighat 100% naresad va namudare khata sefr nashavad.")
print("   - Nerkh Yadgiri (eta) bayad ba daghighat entekhab shavad. Meghdar haye besyar koochak momken ast hamgerayi ra kond konand.")
print("   - Meghdar haye eta bozorgtar momken ast baes be nowsan khata va adam-e hamgerayi shavand.")
print("   - Tedade Tekrar-ha (n_iter) bayad kafi bashad ta model forsat yadgiri dashte bashad, ama tedade ziadi lazeman daghighat ra behبود nemidahad (momken ast khata dar yek sath baghi bemanad).")
print("   - Standard sazi-ye dade-ha (mesle StandardScaler) baraye amalkard-e monaseb Perceptron roye dade-haye vagheyi hayati ast.")

print("\n--- Payan-e Proje ---")
