"""
member2_baseline_clustering.py

DSAA2011 Final Project - Member 2
Baseline & Clustering Analyst

This script completes the Member 2 tasks:
1. Load the engineered dataset from Member 1, preferably engineered.csv.
2. Perform leakage-free train/test preprocessing.
3. Implement and evaluate unsupervised clustering:
   - K-Means
   - K-Medoids
4. Implement and evaluate baseline supervised models:
   - Logistic Regression from scratch
   - Decision Tree from scratch
   - k-Nearest Neighbors
5. Output confusion matrices, model metrics, ROC/AUC points, and clustering metrics.

Expected input:
- engineered.csv if available
- otherwise data.csv

Expected target column:
- Target

Target encoding:
- Dropout = 1
- Graduate / Enrolled = 0
"""

import csv
import math
import os
import random
from collections import Counter, defaultdict

SEED = 42
random.seed(SEED)

DATA_CANDIDATES = ["engineered.csv", "data.csv"]
OUT_DIR = "member2_results"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Basic utilities
# =========================

def find_data_file():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Cannot find engineered.csv or data.csv in the current directory.")


def read_data(path):
    """
    Read a semicolon-separated or comma-separated CSV file.
    Non-target columns are parsed as numeric features.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = list(reader)
        fields = reader.fieldnames

    if not fields or "Target" not in fields:
        raise ValueError("The dataset must contain a column named 'Target'.")

    feature_names = [c for c in fields if c != "Target"]

    X, y_raw = [], []
    for r in rows:
        row = []
        valid = True
        for c in feature_names:
            try:
                row.append(float(r[c]))
            except Exception:
                valid = False
                break
        if valid:
            X.append(row)
            y_raw.append(r["Target"])

    y = [1 if t == "Dropout" else 0 for t in y_raw]
    return feature_names, X, y, y_raw


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1 + ez)


# =========================
# Leakage-free preprocessing
# =========================

def stratified_split(y, test_ratio=0.2):
    rng = random.Random(SEED)
    cls = defaultdict(list)
    for i, label in enumerate(y):
        cls[label].append(i)

    train_idx, test_idx = [], []
    for ids in cls.values():
        rng.shuffle(ids)
        cut = int(len(ids) * (1 - test_ratio))
        train_idx.extend(ids[:cut])
        test_idx.extend(ids[cut:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def fit_standardizer(X_train):
    n = len(X_train)
    d = len(X_train[0])
    mean = [sum(X_train[i][j] for i in range(n)) / n for j in range(d)]

    std = []
    for j in range(d):
        var = sum((X_train[i][j] - mean[j]) ** 2 for i in range(n)) / n
        std.append(math.sqrt(var) if var > 1e-12 else 1.0)

    return mean, std


def apply_standardizer(X, mean, std):
    return [[(row[j] - mean[j]) / std[j] for j in range(len(mean))] for row in X]


# =========================
# Evaluation
# =========================

def confusion_matrix_values(y_true, y_pred):
    tp = tn = fp = fn = 0
    for a, b in zip(y_true, y_pred):
        if a == 1 and b == 1:
            tp += 1
        elif a == 0 and b == 0:
            tn += 1
        elif a == 0 and b == 1:
            fp += 1
        elif a == 1 and b == 0:
            fn += 1
    return tp, tn, fp, fn


def roc_auc(y_true, prob, steps=200):
    points = []
    for i in range(steps + 1):
        threshold = i / steps
        pred = [1 if p >= threshold else 0 for p in prob]
        tp, tn, fp, fn = confusion_matrix_values(y_true, pred)
        tpr = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0
        points.append((fpr, tpr))

    points.sort(key=lambda x: x[0])

    auc = 0.0
    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]
        auc += (x2 - x1) * (y1 + y2) / 2

    return points, auc


def classification_metrics(y_true, prob, threshold=0.5):
    pred = [1 if p >= threshold else 0 for p in prob]
    tp, tn, fp, fn = confusion_matrix_values(y_true, pred)

    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    _, auc = roc_auc(y_true, prob)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# =========================
# Logistic Regression
# =========================

def train_logistic_regression(X, y, lr=0.05, epochs=250, l2=1e-3):
    n = len(X)
    d = len(X[0])
    w = [0.0] * d
    b = 0.0

    loss_history = []

    for epoch in range(epochs):
        grad_w = [0.0] * d
        grad_b = 0.0
        loss = 0.0

        for xi, yi in zip(X, y):
            z = dot(w, xi) + b
            p = sigmoid(z)
            error = p - yi

            loss += -(yi * math.log(p + 1e-12) + (1 - yi) * math.log(1 - p + 1e-12))

            for j in range(d):
                grad_w[j] += error * xi[j]
            grad_b += error

        reg = sum(v * v for v in w)
        loss = loss / n + 0.5 * l2 * reg
        loss_history.append(loss)

        for j in range(d):
            w[j] -= lr * (grad_w[j] / n + l2 * w[j])
        b -= lr * grad_b / n

    return w, b, loss_history


def predict_proba_logistic(X, w, b):
    return [sigmoid(dot(w, xi) + b) for xi in X]


# =========================
# k-Nearest Neighbors
# =========================

def knn_predict_proba(X_train, y_train, X_test, k=17):
    result = []

    for x in X_test:
        distances = []
        for xi, yi in zip(X_train, y_train):
            dist = sum((x[j] - xi[j]) ** 2 for j in range(len(x)))
            distances.append((dist, yi))

        distances.sort(key=lambda t: t[0])
        neighbors = distances[:k]
        result.append(sum(label for _, label in neighbors) / k)

    return result


# =========================
# Simple Decision Tree
# =========================

class TreeNode:
    def __init__(self, prob=None, feature=None, threshold=None, left=None, right=None):
        self.prob = prob
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right


def gini(labels):
    if not labels:
        return 0
    p = sum(labels) / len(labels)
    return 1 - p * p - (1 - p) * (1 - p)


def best_split(X, y, max_features=12):
    n = len(X)
    d = len(X[0])
    rng = random.Random(SEED)

    feature_ids = list(range(d))
    rng.shuffle(feature_ids)
    feature_ids = feature_ids[: min(max_features, d)]

    base_gini = gini(y)
    best_gain = 0
    best_j = None
    best_t = None

    for j in feature_ids:
        values = sorted(set(row[j] for row in X))
        if len(values) <= 1:
            continue

        if len(values) > 20:
            candidates = [values[int(len(values) * q / 20)] for q in range(1, 20)]
        else:
            candidates = [(values[i - 1] + values[i]) / 2 for i in range(1, len(values))]

        for t in candidates:
            left_y = [y[i] for i in range(n) if X[i][j] <= t]
            right_y = [y[i] for i in range(n) if X[i][j] > t]

            if not left_y or not right_y:
                continue

            weighted = len(left_y) / n * gini(left_y) + len(right_y) / n * gini(right_y)
            gain = base_gini - weighted

            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_t = t

    return best_j, best_t, best_gain


def build_tree(X, y, depth=0, max_depth=5, min_samples=30):
    prob = sum(y) / len(y)

    if depth >= max_depth or len(y) < min_samples or gini(y) < 1e-8:
        return TreeNode(prob=prob)

    j, t, gain = best_split(X, y)
    if j is None or gain <= 1e-8:
        return TreeNode(prob=prob)

    left_X, left_y, right_X, right_y = [], [], [], []
    for xi, yi in zip(X, y):
        if xi[j] <= t:
            left_X.append(xi)
            left_y.append(yi)
        else:
            right_X.append(xi)
            right_y.append(yi)

    return TreeNode(
        prob=prob,
        feature=j,
        threshold=t,
        left=build_tree(left_X, left_y, depth + 1, max_depth, min_samples),
        right=build_tree(right_X, right_y, depth + 1, max_depth, min_samples),
    )


def predict_tree_one(node, x):
    while node.feature is not None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.prob


def predict_proba_tree(tree, X):
    return [predict_tree_one(tree, x) for x in X]


# =========================
# Clustering
# =========================

def kmeans(X, k=3, max_iter=80):
    rng = random.Random(SEED)
    centers = [X[rng.randrange(len(X))][:] for _ in range(k)]
    labels = [0] * len(X)

    for _ in range(max_iter):
        changed = 0

        for i, x in enumerate(X):
            best = min(
                range(k),
                key=lambda c: sum((x[j] - centers[c][j]) ** 2 for j in range(len(x))),
            )
            if best != labels[i]:
                labels[i] = best
                changed += 1

        for c in range(k):
            ids = [i for i, label in enumerate(labels) if label == c]
            if ids:
                centers[c] = [
                    sum(X[i][j] for i in ids) / len(ids)
                    for j in range(len(X[0]))
                ]

        if changed == 0:
            break

    return labels


def kmedoids(X, k=3, max_iter=15):
    rng = random.Random(SEED)
    medoids = rng.sample(range(len(X)), k)
    labels = [0] * len(X)

    for _ in range(max_iter):
        for i, x in enumerate(X):
            labels[i] = min(
                range(k),
                key=lambda c: sum((x[j] - X[medoids[c]][j]) ** 2 for j in range(len(x))),
            )

        improved = False
        for c in range(k):
            ids = [i for i, label in enumerate(labels) if label == c]
            if not ids:
                continue

            # Approximate medoid update for efficiency.
            candidates = ids[: min(100, len(ids))]
            best_medoid = medoids[c]
            best_cost = float("inf")

            for cand in candidates:
                cost = sum(
                    sum((X[i][j] - X[cand][j]) ** 2 for j in range(len(X[0])))
                    for i in ids
                )
                if cost < best_cost:
                    best_cost = cost
                    best_medoid = cand

            if best_medoid != medoids[c]:
                medoids[c] = best_medoid
                improved = True

        if not improved:
            break

    return labels


def silhouette_score(X, labels, cap=160):
    if len(X) > cap:
        rng = random.Random(SEED)
        idx = list(range(len(X)))
        rng.shuffle(idx)
        idx = idx[:cap]
        X = [X[i] for i in idx]
        labels = [labels[i] for i in idx]

    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)

    scores = []
    for i, x in enumerate(X):
        own = labels[i]
        own_ids = clusters[own]

        if len(own_ids) > 1:
            a = sum(math.dist(x, X[j]) for j in own_ids if j != i) / (len(own_ids) - 1)
        else:
            a = 0

        b = min(
            sum(math.dist(x, X[j]) for j in other_ids) / len(other_ids)
            for c, other_ids in clusters.items()
            if c != own
        )

        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)

    return sum(scores) / len(scores)


def davies_bouldin_index(X, labels):
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(i)

    centers = {}
    scatters = {}

    for c, ids in clusters.items():
        centers[c] = [
            sum(X[i][j] for i in ids) / len(ids)
            for j in range(len(X[0]))
        ]
        scatters[c] = sum(math.dist(X[i], centers[c]) for i in ids) / len(ids)

    db_values = []
    keys = list(clusters.keys())

    for i in keys:
        ratios = []
        for j in keys:
            if i == j:
                continue
            dist = math.dist(centers[i], centers[j])
            if dist > 0:
                ratios.append((scatters[i] + scatters[j]) / dist)
        if ratios:
            db_values.append(max(ratios))

    return sum(db_values) / len(db_values)


# =========================
# Main pipeline
# =========================

def main():
    data_path = find_data_file()
    feature_names, X, y, y_raw = read_data(data_path)

    train_idx, test_idx = stratified_split(y, test_ratio=0.2)

    X_train_raw = [X[i] for i in train_idx]
    X_test_raw = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    mean, std = fit_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, mean, std)
    X_test = apply_standardizer(X_test_raw, mean, std)
    X_all = apply_standardizer(X, mean, std)

    # Logistic Regression with L2 tuning.
    l2_values = [1e-4, 1e-3, 1e-2, 1e-1]
    tuning_rows = []
    best_l2 = None
    best_auc = -1
    best_lr_model = None

    for l2 in l2_values:
        w, b, loss_history = train_logistic_regression(X_train, y_train, l2=l2)
        prob = predict_proba_logistic(X_test, w, b)
        m = classification_metrics(y_test, prob)
        tuning_rows.append([l2, f"{m['accuracy']:.4f}", f"{m['f1']:.4f}", f"{m['auc']:.4f}"])

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_l2 = l2
            best_lr_model = (w, b, loss_history)

    w, b, loss_history = best_lr_model
    prob_lr_train = predict_proba_logistic(X_train, w, b)
    prob_lr_test = predict_proba_logistic(X_test, w, b)

    # Decision Tree baseline.
    tree = build_tree(X_train, y_train, max_depth=5, min_samples=30)
    prob_tree_test = predict_proba_tree(tree, X_test)

    # KNN baseline.
    prob_knn_test = knn_predict_proba(X_train, y_train, X_test, k=17)

    models = {
        ("Logistic Regression", "train"): classification_metrics(y_train, prob_lr_train),
        ("Logistic Regression", "test"): classification_metrics(y_test, prob_lr_test),
        ("Decision Tree", "test"): classification_metrics(y_test, prob_tree_test),
        ("k-NN", "test"): classification_metrics(y_test, prob_knn_test),
    }

    write_csv(
        os.path.join(OUT_DIR, "model_metrics.csv"),
        ["model", "split", "accuracy", "precision", "recall", "f1", "auc", "tp", "tn", "fp", "fn"],
        [
            [
                model,
                split,
                f"{m['accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{m['auc']:.4f}",
                m["tp"],
                m["tn"],
                m["fp"],
                m["fn"],
            ]
            for (model, split), m in models.items()
        ],
    )

    write_csv(
        os.path.join(OUT_DIR, "lr_l2_tuning.csv"),
        ["l2", "accuracy", "f1", "auc"],
        tuning_rows,
    )

    write_csv(
        os.path.join(OUT_DIR, "lr_loss_history.csv"),
        ["epoch", "loss"],
        [[i + 1, f"{loss:.6f}"] for i, loss in enumerate(loss_history)],
    )

    for name, prob in [
        ("lr", prob_lr_test),
        ("decision_tree", prob_tree_test),
        ("knn", prob_knn_test),
    ]:
        points, auc = roc_auc(y_test, prob)
        write_csv(
            os.path.join(OUT_DIR, f"roc_{name}.csv"),
            ["fpr", "tpr"],
            [[f"{fpr:.6f}", f"{tpr:.6f}"] for fpr, tpr in points],
        )

    # Clustering on a subset for efficiency.
    subset_idx = list(range(len(X_all)))
    random.Random(SEED).shuffle(subset_idx)
    subset_idx = subset_idx[: min(500, len(X_all))]
    X_cluster = [X_all[i] for i in subset_idx]
    y_cluster_raw = [y_raw[i] for i in subset_idx]

    labels_kmeans = kmeans(X_cluster, k=3)
    labels_kmedoids = kmedoids(X_cluster, k=3)

    sil_kmeans = silhouette_score(X_cluster, labels_kmeans)
    db_kmeans = davies_bouldin_index(X_cluster, labels_kmeans)

    sil_kmedoids = silhouette_score(X_cluster, labels_kmedoids)
    db_kmedoids = davies_bouldin_index(X_cluster, labels_kmedoids)

    write_csv(
        os.path.join(OUT_DIR, "clustering_metrics.csv"),
        ["algorithm", "silhouette", "davies_bouldin"],
        [
            ["K-Means", f"{sil_kmeans:.4f}", f"{db_kmeans:.4f}"],
            ["K-Medoids", f"{sil_kmedoids:.4f}", f"{db_kmedoids:.4f}"],
        ],
    )

    write_csv(
        os.path.join(OUT_DIR, "cluster_assignments.csv"),
        ["sample_index", "true_label", "kmeans_cluster", "kmedoids_cluster"],
        [
            [subset_idx[i], y_cluster_raw[i], labels_kmeans[i], labels_kmedoids[i]]
            for i in range(len(subset_idx))
        ],
    )

    with open(os.path.join(OUT_DIR, "member2_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Member 2: Baseline & Clustering Analyst\n")
        f.write(f"Data file used: {data_path}\n")
        f.write(f"Samples: {len(X)}, Features: {len(feature_names)}\n")
        f.write(f"Raw class counts: {dict(Counter(y_raw))}\n")
        f.write(f"Binary class counts: {dict(Counter(y))}\n")
        f.write(f"Best Logistic Regression L2: {best_l2}\n\n")

        f.write("Model metrics:\n")
        for (model, split), m in models.items():
            f.write(f"{model} ({split}): {m}\n")

        f.write("\nClustering metrics:\n")
        f.write(f"K-Means: silhouette={sil_kmeans:.4f}, DB={db_kmeans:.4f}\n")
        f.write(f"K-Medoids: silhouette={sil_kmedoids:.4f}, DB={db_kmedoids:.4f}\n")

    print("Member 2 pipeline finished successfully.")
    print(f"Input file: {data_path}")
    print(f"Outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
