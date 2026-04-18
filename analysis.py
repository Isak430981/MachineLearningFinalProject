import csv
import math
import os
import random
from collections import Counter, defaultdict

SEED = 42
random.seed(SEED)

DATA_PATH = "data.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def read_dataset(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        fields = reader.fieldnames
    feature_names = [c for c in fields if c != "Target"]
    X = [[float(r[c]) for c in feature_names] for r in rows]
    y_text = [r["Target"] for r in rows]
    y_bin = [1 if t == "Dropout" else 0 for t in y_text]
    return feature_names, X, y_text, y_bin


def standardize(X):
    n, d = len(X), len(X[0])
    means = [sum(X[i][j] for i in range(n)) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((X[i][j] - means[j]) ** 2 for i in range(n)) / n
        stds.append(math.sqrt(var) if var > 1e-12 else 1.0)
    Z = [[(X[i][j] - means[j]) / stds[j] for j in range(d)] for i in range(n)]
    return Z


def stratified_split(y_text, test_ratio=0.2):
    rng = random.Random(SEED)
    cls = defaultdict(list)
    for i, c in enumerate(y_text):
        cls[c].append(i)
    tr, te = [], []
    for ids in cls.values():
        rng.shuffle(ids)
        cut = int(len(ids) * (1 - test_ratio))
        tr.extend(ids[:cut])
        te.extend(ids[cut:])
    rng.shuffle(tr)
    rng.shuffle(te)
    return tr, te


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1 / (1 + ez)
    ez = math.exp(z)
    return ez / (1 + ez)


def train_logreg(X, y, lr=0.05, epochs=260, l2=1e-3):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for xi, yi in zip(X, y):
            p = sigmoid(dot(w, xi) + b)
            e = p - yi
            for j in range(d):
                gw[j] += e * xi[j]
            gb += e
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def predict_lr_proba(X, w, b):
    return [sigmoid(dot(w, xi) + b) for xi in X]


def knn_proba(X_train, y_train, X_eval, k=17):
    probs = []
    d = len(X_train[0])
    for xe in X_eval:
        dists = []
        for xt, yt in zip(X_train, y_train):
            ds = 0.0
            for j in range(d):
                v = xe[j] - xt[j]
                ds += v * v
            dists.append((ds, yt))
        dists.sort(key=lambda t: t[0])
        probs.append(sum(y for _, y in dists[:k]) / k)
    return probs


def confusion(y_true, y_pred):
    tp = tn = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def roc_curve_auc(y_true, probs, steps=200):
    pts = []
    for i in range(steps + 1):
        t = i / steps
        pred = [1 if p >= t else 0 for p in probs]
        tp, tn, fp, fn = confusion(y_true, pred)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        pts.append((fpr, tpr))
    pts.sort(key=lambda x: x[0])
    auc = 0.0
    for i in range(1, len(pts)):
        x1, y1 = pts[i - 1]
        x2, y2 = pts[i]
        auc += (x2 - x1) * (y1 + y2) / 2
    return pts, auc


def evaluate(y_true, probs):
    pred = [1 if p >= 0.5 else 0 for p in probs]
    tp, tn, fp, fn = confusion(y_true, pred)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    _, auc = roc_curve_auc(y_true, probs)
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


def kmeans(X, k=3, max_iter=60):
    rng = random.Random(SEED)
    centers = [X[rng.randrange(len(X))][:] for _ in range(k)]
    labels = [0] * len(X)
    d = len(X[0])
    for _ in range(max_iter):
        changed = 0
        for i, x in enumerate(X):
            best = min(range(k), key=lambda c: sum((x[j] - centers[c][j]) ** 2 for j in range(d)))
            if best != labels[i]:
                labels[i] = best
                changed += 1
        for c in range(k):
            ids = [i for i, l in enumerate(labels) if l == c]
            if ids:
                centers[c] = [sum(X[i][j] for i in ids) / len(ids) for j in range(d)]
        if changed == 0:
            break
    return labels


def kmedoids(X, k=3, max_iter=20):
    rng = random.Random(SEED)
    medoids = rng.sample(range(len(X)), k)
    labels = [0] * len(X)
    d = len(X[0])
    for _ in range(max_iter):
        for i, x in enumerate(X):
            labels[i] = min(range(k), key=lambda c: sum((x[j] - X[medoids[c]][j]) ** 2 for j in range(d)))
        improved = False
        for c in range(k):
            ids = [i for i, l in enumerate(labels) if l == c]
            if not ids:
                continue
            best = medoids[c]
            best_cost = float("inf")
            # capped candidate set for speed
            for cand in ids[: min(30, len(ids))]:
                xc = X[cand]
                cost = 0.0
                for i in ids:
                    xi = X[i]
                    for j in range(d):
                        dv = xi[j] - xc[j]
                        cost += dv * dv
                if cost < best_cost:
                    best_cost = cost
                    best = cand
            if best != medoids[c]:
                medoids[c] = best
                improved = True
        if not improved:
            break
    return labels


def silhouette_score(X, labels, cap=180):
    if len(X) > cap:
        idx = list(range(len(X)))
        random.Random(SEED).shuffle(idx)
        idx = idx[:cap]
        X = [X[i] for i in idx]
        labels = [labels[i] for i in idx]

    clusters = defaultdict(list)
    for i, l in enumerate(labels):
        clusters[l].append(i)

    vals = []
    for i, x in enumerate(X):
        own = labels[i]
        own_ids = clusters[own]
        if len(own_ids) > 1:
            a = sum(math.dist(x, X[j]) for j in own_ids if j != i) / (len(own_ids) - 1)
        else:
            a = 0.0
        b = min(sum(math.dist(x, X[j]) for j in ids) / len(ids) for c, ids in clusters.items() if c != own)
        vals.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(vals) / len(vals)


def davies_bouldin_score(X, labels):
    clusters = defaultdict(list)
    for i, l in enumerate(labels):
        clusters[l].append(i)
    d = len(X[0])
    centers = {
        c: [sum(X[i][j] for i in ids) / len(ids) for j in range(d)]
        for c, ids in clusters.items()
    }
    scat = {
        c: sum(math.dist(X[i], centers[c]) for i in ids) / len(ids)
        for c, ids in clusters.items()
    }

    keys = list(clusters.keys())
    r_vals = []
    for i in keys:
        max_r = 0.0
        for j in keys:
            if i == j:
                continue
            dij = math.dist(centers[i], centers[j])
            if dij == 0:
                continue
            rij = (scat[i] + scat[j]) / dij
            if rij > max_r:
                max_r = rij
        r_vals.append(max_r)
    return sum(r_vals) / len(r_vals)


def pca2(X):
    n, d = len(X), len(X[0])
    C = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            v = sum(X[r][i] * X[r][j] for r in range(n)) / n
            C[i][j] = C[j][i] = v

    W = [row[:] for row in C]
    comps = []
    for _ in range(2):
        vec = [random.random() - 0.5 for _ in range(d)]
        for _ in range(120):
            nvec = [sum(W[i][j] * vec[j] for j in range(d)) for i in range(d)]
            norm = math.sqrt(sum(v * v for v in nvec)) or 1.0
            vec = [v / norm for v in nvec]
        lam = sum(vec[i] * sum(W[i][j] * vec[j] for j in range(d)) for i in range(d))
        comps.append(vec)
        for i in range(d):
            for j in range(d):
                W[i][j] -= lam * vec[i] * vec[j]

    return [[dot(x, comps[0]), dot(x, comps[1])] for x in X]


def _hbeta(dist_row, beta):
    p = [math.exp(-d * beta) for d in dist_row]
    s = sum(p)
    if s <= 1e-12:
        p = [1.0 / len(dist_row)] * len(dist_row)
        s = 1.0
    p = [v / s for v in p]
    h = -sum(v * math.log(v + 1e-12) for v in p)
    return h, p


def tsne_2d(X, perplexity=20.0, iterations=80, learning_rate=120.0):
    n, d = len(X), len(X[0])
    # pairwise squared distances
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi = X[i]
        for j in range(i + 1, n):
            xj = X[j]
            ds = 0.0
            for k in range(d):
                dv = xi[k] - xj[k]
                ds += dv * dv
            D[i][j] = D[j][i] = ds

    # conditional probabilities with binary search on beta
    log_u = math.log(perplexity)
    P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        beta = 1.0
        betamin, betamax = None, None
        dist_row = [D[i][j] for j in range(n) if j != i]
        for _ in range(45):
            h, this_p = _hbeta(dist_row, beta)
            hdiff = h - log_u
            if abs(hdiff) < 1e-5:
                break
            if hdiff > 0:
                betamin = beta
                beta = beta * 2 if betamax is None else (beta + betamax) / 2
            else:
                betamax = beta
                beta = beta / 2 if betamin is None else (beta + betamin) / 2
        row_vals = this_p
        idx = 0
        for j in range(n):
            if i == j:
                continue
            P[i][j] = row_vals[idx]
            idx += 1

    # symmetrize
    for i in range(n):
        for j in range(n):
            P[i][j] = (P[i][j] + P[j][i]) / (2 * n)

    # optimize Y
    rng = random.Random(SEED)
    Y = [[(rng.random() - 0.5) * 1e-4, (rng.random() - 0.5) * 1e-4] for _ in range(n)]
    gains = [[1.0, 1.0] for _ in range(n)]
    y_inc = [[0.0, 0.0] for _ in range(n)]

    for it in range(iterations):
        Qnum = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dx = Y[i][0] - Y[j][0]
                dy = Y[i][1] - Y[j][1]
                q = 1.0 / (1.0 + dx * dx + dy * dy)
                Qnum[i][j] = Qnum[j][i] = q

        qsum = sum(sum(row) for row in Qnum) - sum(Qnum[i][i] for i in range(n))
        qsum = max(qsum, 1e-12)

        grads = [[0.0, 0.0] for _ in range(n)]
        exaggeration = 4.0 if it < 100 else 1.0

        for i in range(n):
            gx = gy = 0.0
            for j in range(n):
                if i == j:
                    continue
                qij = Qnum[i][j] / qsum
                mult = 4.0 * (exaggeration * P[i][j] - qij) * Qnum[i][j]
                gx += mult * (Y[i][0] - Y[j][0])
                gy += mult * (Y[i][1] - Y[j][1])
            grads[i][0] = gx
            grads[i][1] = gy

        momentum = 0.5 if it < 120 else 0.8
        for i in range(n):
            for m in range(2):
                gains[i][m] = (gains[i][m] + 0.2) if (grads[i][m] > 0) != (y_inc[i][m] > 0) else (gains[i][m] * 0.8)
                if gains[i][m] < 0.01:
                    gains[i][m] = 0.01
                y_inc[i][m] = momentum * y_inc[i][m] - learning_rate * gains[i][m] * grads[i][m]
                Y[i][m] += y_inc[i][m]

        # re-center
        meanx = sum(y[0] for y in Y) / n
        meany = sum(y[1] for y in Y) / n
        for i in range(n):
            Y[i][0] -= meanx
            Y[i][1] -= meany

    return Y


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def cross_validate_lr(X, y, l2_values, folds=3, sample_cap=900):
    if len(X) > sample_cap:
        idx0=list(range(len(X))); random.Random(SEED).shuffle(idx0); idx0=idx0[:sample_cap];
        X=[X[i] for i in idx0]; y=[y[i] for i in idx0]
    idx = list(range(len(X)))
    random.Random(SEED).shuffle(idx)
    fold_size = len(idx) // folds
    out = []
    for l2 in l2_values:
        scores = []
        for f in range(folds):
            va = idx[f * fold_size : (f + 1) * fold_size]
            va_set = set(va)
            tr = [i for i in idx if i not in va_set]
            w, b = train_logreg([X[i] for i in tr], [y[i] for i in tr], l2=l2, epochs=90, lr=0.06)
            probs = predict_lr_proba([X[i] for i in va], w, b)
            scores.append(evaluate([y[i] for i in va], probs)["f1"])
        out.append((l2, sum(scores) / len(scores)))
    return out


def cross_validate_knn(X, y, k_values, folds=4, train_cap=700):
    idx = list(range(len(X)))
    random.Random(SEED).shuffle(idx)
    fold_size = len(idx) // folds
    out = []
    for k in k_values:
        scores = []
        for f in range(folds):
            va = idx[f * fold_size : (f + 1) * fold_size]
            va_set = set(va)
            tr = [i for i in idx if i not in va_set]
            random.Random(SEED + f + k).shuffle(tr)
            tr = tr[: min(train_cap, len(tr))]
            probs = knn_proba([X[i] for i in tr], [y[i] for i in tr], [X[i] for i in va], k=k)
            scores.append(evaluate([y[i] for i in va], probs)["f1"])
        out.append((k, sum(scores) / len(scores)))
    return out


def main():
    features, X, y_text, y_bin = read_dataset(DATA_PATH)
    Z = standardize(X)

    tr_idx, te_idx = stratified_split(y_text, test_ratio=0.2)
    Xtr = [Z[i] for i in tr_idx]
    ytr = [y_bin[i] for i in tr_idx]
    Xte = [Z[i] for i in te_idx]
    yte = [y_bin[i] for i in te_idx]

    # LR
    w, b = train_logreg(Xtr, ytr)
    p_lr_tr = predict_lr_proba(Xtr, w, b)
    p_lr_te = predict_lr_proba(Xte, w, b)
    p_lr_all = predict_lr_proba(Z, w, b)

    # kNN (full train for test; sample-based for full/train for runtime)
    knn_test_train_ids=list(range(len(Xtr))); random.Random(SEED).shuffle(knn_test_train_ids); knn_test_train_ids=knn_test_train_ids[:300]
    Xtr_knn_test=[Xtr[i] for i in knn_test_train_ids]; ytr_knn_test=[ytr[i] for i in knn_test_train_ids]
    p_knn_te = knn_proba(Xtr_knn_test, ytr_knn_test, Xte, k=17)
    knn_train_sample_ids = list(range(len(Xtr)))
    random.Random(SEED).shuffle(knn_train_sample_ids)
    knn_train_sample_ids = knn_train_sample_ids[:300]
    Xtr_small = [Xtr[i] for i in knn_train_sample_ids]
    ytr_small = [ytr[i] for i in knn_train_sample_ids]
    p_knn_tr_small = knn_proba(Xtr_small, ytr_small, Xtr_small, k=17)

    all_ids = list(range(len(Z)))
    random.Random(SEED).shuffle(all_ids)
    all_ids = all_ids[:350]
    Xall_small = [Z[i] for i in all_ids]
    yall_small = [y_bin[i] for i in all_ids]
    p_knn_all_small = knn_proba(Xtr_small, ytr_small, Xall_small, k=17)

    evals = {
        ("LR", "train"): evaluate(ytr, p_lr_tr),
        ("LR", "test"): evaluate(yte, p_lr_te),
        ("LR", "all"): evaluate(y_bin, p_lr_all),
        ("KNN", "train_sampled"): evaluate(ytr_small, p_knn_tr_small),
        ("KNN", "test"): evaluate(yte, p_knn_te),
        ("KNN", "all_sampled"): evaluate(yall_small, p_knn_all_small),
    }

    # clustering on subset
    subset_ids = list(range(len(Z)))
    random.Random(SEED).shuffle(subset_ids)
    subset_ids = subset_ids[:180]
    Xc = [Z[i] for i in subset_ids]
    yc = [y_text[i] for i in subset_ids]

    labels_km = kmeans(Xc, k=3)
    labels_md = kmedoids(Xc, k=3)
    cmetrics = [
        ("kmeans", silhouette_score(Xc, labels_km), davies_bouldin_score(Xc, labels_km)),
        ("kmedoids", silhouette_score(Xc, labels_md), davies_bouldin_score(Xc, labels_md)),
    ]

    # 2D embeddings
    pca_xy = pca2(Xc)
    tsne_ids = subset_ids[:80]
    Xts = [Z[i] for i in tsne_ids]
    yts = [y_text[i] for i in tsne_ids]
    tsne_xy = tsne_2d(Xts, perplexity=15.0, iterations=40, learning_rate=100.0)

    # Validation
    lr_cv = cross_validate_lr(Xtr, ytr, l2_values=[1e-4, 1e-3, 1e-2, 1e-1], folds=3, sample_cap=900)
    knn_cv = cross_validate_knn(Xtr[:500], ytr[:500], k_values=[11, 17, 23], folds=2, train_cap=220)

    # Save outputs
    write_csv(
        os.path.join(OUT_DIR, "model_metrics.csv"),
        ["model", "split", "accuracy", "precision", "recall", "f1", "auc", "tp", "tn", "fp", "fn"],
        [
            [m, s,
             f"{v['accuracy']:.4f}", f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1']:.4f}", f"{v['auc']:.4f}",
             v["tp"], v["tn"], v["fp"], v["fn"]]
            for (m, s), v in evals.items()
        ],
    )

    write_csv(
        os.path.join(OUT_DIR, "clustering_metrics.csv"),
        ["algorithm", "silhouette", "davies_bouldin"],
        [[a, f"{s:.4f}", f"{d:.4f}"] for a, s, d in cmetrics],
    )

    write_csv(
        os.path.join(OUT_DIR, "embedding_points.csv"),
        ["x", "y", "label", "kmeans", "kmedoids", "embed_type"],
        [[f"{pca_xy[i][0]:.6f}", f"{pca_xy[i][1]:.6f}", yc[i], labels_km[i], labels_md[i], "pca"] for i in range(len(pca_xy))]
        + [[f"{tsne_xy[i][0]:.6f}", f"{tsne_xy[i][1]:.6f}", yts[i], "", "", "tsne"] for i in range(len(tsne_xy))],
    )

    for name, probs in [("lr", p_lr_te), ("knn", p_knn_te)]:
        roc_pts, _ = roc_curve_auc(yte, probs)
        write_csv(
            os.path.join(OUT_DIR, f"roc_{name}.csv"),
            ["fpr", "tpr"],
            [[f"{x:.6f}", f"{y:.6f}"] for x, y in roc_pts],
        )

    write_csv(
        os.path.join(OUT_DIR, "validation_metrics.csv"),
        ["model", "hyperparameter", "value", "cv_f1"],
        [["LR", "l2", str(v), f"{s:.4f}"] for v, s in lr_cv]
        + [["KNN", "k", str(v), f"{s:.4f}"] for v, s in knn_cv],
    )

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(f"samples={len(X)},features={len(features)},class_counts={dict(Counter(y_text))}\n")
        f.write("model_splits=" + ",".join([f"{m}:{s}" for m, s in evals.keys()]) + "\n")
        f.write("clustering=" + str([(a, round(s, 4), round(d, 4)) for a, s, d in cmetrics]) + "\n")
        f.write("lr_cv=" + str([(v, round(s, 4)) for v, s in lr_cv]) + "\n")
        f.write("knn_cv=" + str([(v, round(s, 4)) for v, s in knn_cv]) + "\n")

    print("done")


if __name__ == "__main__":
    main()
