import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

protected_attrs = ["race", "gender", "socioeconomic_status"]
y = train_df["admitted"]
X = train_df.drop(["admitted"] + protected_attrs, axis=1)
X_prot = train_df[protected_attrs]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
base_clf = Pipeline(
    [
        ("pre", preprocessor),
        (
            "lgb",
            LGBMClassifier(
                n_estimators=200, learning_rate=0.05, random_state=42, n_jobs=-1
            ),
        ),
    ]
)


def compute_equalized_odds_gap(y_true, y_pred, prot_df):
    gaps = []
    for attr in protected_attrs:
        tprs, fprs = [], []
        for grp in prot_df[attr].unique():
            mask = prot_df[attr] == grp
            yt, yp = y_true[mask], y_pred[mask]
            pos = yt == 1
            neg = yt == 0
            if pos.sum() > 0:
                tprs.append((yp[pos] == 1).sum() / pos.sum())
            if neg.sum() > 0:
                fprs.append((yp[neg] == 1).sum() / neg.sum())
        gap = max(
            (max(tprs) - min(tprs)) if len(tprs) > 1 else 0,
            (max(fprs) - min(fprs)) if len(fprs) > 1 else 0,
        )
        gaps.append(gap)
    return np.mean(gaps)


def composite_metric(y_true, y_pred, prot_df):
    acc = accuracy_score(y_true, y_pred)
    gap = compute_equalized_odds_gap(y_true, y_pred, prot_df)
    return acc - gap


def compute_reweights(prot_df, y_series):
    grp = prot_df.astype(str).agg("_".join, axis=1)
    df = pd.DataFrame({"grp": grp, "y": y_series})
    N = len(df)
    p_grp = df.grp.value_counts() / N
    p_y = df.y.value_counts() / N
    p_joint = df.groupby(["grp", "y"]).size() / N
    weights = df.apply(
        lambda row: (p_grp[row.grp] * p_y[row.y]) / p_joint.loc[(row.grp, row.y)],
        axis=1,
    )
    return weights.values


# Cross-validation with calibration and threshold tuning
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
composite_scores = []
thresholds = np.linspace(0.2, 0.8, 31)

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    prot_tr = X_prot.iloc[train_idx]
    prot_val = X_prot.iloc[val_idx]
    w_tr = compute_reweights(prot_tr, y_tr)
    # Fit base pipeline with weights
    base_clf.fit(X_tr, y_tr, lgb__sample_weight=w_tr)
    # Calibrate probabilities isotonic on train fold
    calib = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
    calib.fit(X_tr, y_tr)
    proba_tr = calib.predict_proba(X_tr)[:, 1]
    best_thr, best_score = 0.5, -np.inf
    for thr in thresholds:
        y_tr_pred = (proba_tr >= thr).astype(int)
        score = composite_metric(y_tr, y_tr_pred, prot_tr)
        if score > best_score:
            best_score, best_thr = score, thr
    proba_val = calib.predict_proba(X_val)[:, 1]
    y_val_pred = (proba_val >= best_thr).astype(int)
    composite_scores.append(composite_metric(y_val, y_val_pred, prot_val))

mean_composite = np.mean(composite_scores)
print(f"Composite metric: {mean_composite:.4f}")

# Retrain on full data and calibrate
w_full = compute_reweights(X_prot, y)
base_clf.fit(X, y, lgb__sample_weight=w_full)
calib_full = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
calib_full.fit(X, y)
proba_full = calib_full.predict_proba(X)[:, 1]
best_thr_full, best_score_full = 0.5, -np.inf
for thr in thresholds:
    y_full_pred = (proba_full >= thr).astype(int)
    score = composite_metric(y, y_full_pred, X_prot)
    if score > best_score_full:
        best_score_full, best_thr_full = score, thr

# Predict test set
X_test = test_df.drop(
    columns=[c for c in protected_attrs if c in test_df.columns], errors="ignore"
)
proba_test = calib_full.predict_proba(X_test)[:, 1]
preds = (proba_test >= best_thr_full).astype(int)
submission = pd.DataFrame({"admitted": preds})
submission.to_csv("./working/submission.csv", index=False)
