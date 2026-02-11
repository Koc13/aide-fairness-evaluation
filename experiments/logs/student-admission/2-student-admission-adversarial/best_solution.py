import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
input_dir = "./input"
output_dir = "./working"
os.makedirs(output_dir, exist_ok=True)
protected_cols = ["gender", "race", "socioeconomic_status", "first_generation"]
batch_size = 128
n_epochs = 10
adv_steps = 1
lambda_adv = 0.1
lr_pred = 1e-3
lr_adv = 1e-3
weight_decay = 1e-4
dropout_rate = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_df = pd.read_csv(f"{input_dir}/train.csv")
test_df = pd.read_csv(f"{input_dir}/test.csv")
target_col = "admitted" if "admitted" in train_df.columns else train_df.columns[-1]

# Encode protected attributes
label_encoders = {}
for col in protected_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    label_encoders[col] = le

# Train/validation split
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df[target_col], random_state=42
)

# Feature preprocessing
all_cols = [c for c in train_df.columns if c not in protected_cols + [target_col, "id"]]
categorical_cols = [c for c in all_cols if train_df[c].dtype == "object"]
numeric_cols = [c for c in all_cols if c not in categorical_cols]
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)
X_train = preprocessor.fit_transform(train_df[all_cols])
X_val = preprocessor.transform(val_df[all_cols])
X_test = preprocessor.transform(test_df[all_cols])
y_train = train_df[target_col].values.astype(int)
y_val = val_df[target_col].values.astype(int)
prot_train = train_df[protected_cols].values
prot_val = val_df[protected_cols].values

# Compute class weights for BCE loss
neg, pos = np.bincount(y_train)
pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)


# Dataset and DataLoader
class AdvDataset(Dataset):
    def __init__(self, X, y, prot):
        arr = X.toarray() if hasattr(X, "toarray") else X
        self.X = torch.tensor(arr, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.prot = torch.tensor(prot, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.prot[idx]


train_loader = DataLoader(
    AdvDataset(X_train, y_train, prot_train), batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    AdvDataset(X_val, y_val, prot_val), batch_size=batch_size, shuffle=False
)
test_X_tensor = torch.tensor(
    X_test.toarray() if hasattr(X_test, "toarray") else X_test, dtype=torch.float32
)

# Model definitions
input_dim = X_train.shape[1]
hidden_dim1 = 128
hidden_dim2 = 64
num_classes = [len(label_encoders[c].classes_) for c in protected_cols]


class Predictor(nn.Module):
    def __init__(self, in_dim, h1, h2, drop_p):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(drop_p)
        self.fc2 = nn.Linear(h1, h2)
        self.dropout2 = nn.Dropout(drop_p)
        self.fc3 = nn.Linear(h2, 1)

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h1d = self.dropout1(h1)
        h2 = self.relu(self.fc2(h1d))
        h2d = self.dropout2(h2)
        logit = self.fc3(h2d)
        return logit, h2

    def get_hidden(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return h2


class Adversary(nn.Module):
    def __init__(self, hid_dim, n_classes_list):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(hid_dim, n) for n in n_classes_list])

    def forward(self, h):
        return [head(h) for head in self.heads]


predictor = Predictor(input_dim, hidden_dim1, hidden_dim2, dropout_rate).to(device)
adversary = Adversary(hidden_dim2, num_classes).to(device)

bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
ce_loss = nn.CrossEntropyLoss()
pred_optimizer = optim.Adam(
    predictor.parameters(), lr=lr_pred, weight_decay=weight_decay
)
adv_optimizer = optim.Adam(adversary.parameters(), lr=lr_adv, weight_decay=weight_decay)

# Training loop
for epoch in range(n_epochs):
    predictor.train()
    adversary.train()
    for X_batch, y_batch, prot_batch in train_loader:
        X_batch, y_batch, prot_batch = (
            X_batch.to(device),
            y_batch.to(device),
            prot_batch.to(device),
        )
        # Adversary update
        for p in predictor.parameters():
            p.requires_grad = False
        for p in adversary.parameters():
            p.requires_grad = True
        hidden = predictor.get_hidden(X_batch).detach()
        for _ in range(adv_steps):
            adv_optimizer.zero_grad()
            adv_logits = adversary(hidden)
            loss_adv = sum(
                ce_loss(adv_logits[i], prot_batch[:, i])
                for i in range(len(protected_cols))
            )
            loss_adv.backward()
            adv_optimizer.step()
        # Predictor update
        for p in predictor.parameters():
            p.requires_grad = True
        for p in adversary.parameters():
            p.requires_grad = False
        pred_optimizer.zero_grad()
        logits, hid = predictor(X_batch)
        loss_pred = bce_loss(logits.squeeze(), y_batch)
        adv_logits_for_pred = adversary(hid)
        loss_adv_pred = sum(
            ce_loss(adv_logits_for_pred[i], prot_batch[:, i])
            for i in range(len(protected_cols))
        )
        (loss_pred - lambda_adv * loss_adv_pred).backward()
        pred_optimizer.step()
        for p in adversary.parameters():
            p.requires_grad = True

# Validation evaluation
predictor.eval()
all_preds, all_truth = [], []
with torch.no_grad():
    for X_batch, y_batch, _ in val_loader:
        X_batch = X_batch.to(device)
        logits, _ = predictor(X_batch)
        preds = (torch.sigmoid(logits.squeeze()) > 0.5).cpu().numpy().astype(int)
        all_preds.append(preds)
        all_truth.append(y_batch.numpy().astype(int))
all_preds = np.concatenate(all_preds)
all_truth = np.concatenate(all_truth)
bal_acc = balanced_accuracy_score(all_truth, all_preds)
print("Validation Balanced Accuracy:", bal_acc)

# Test set predictions and save submission
predictor.eval()
with torch.no_grad():
    test_logits, _ = predictor(test_X_tensor.to(device))
    test_preds = (torch.sigmoid(test_logits.squeeze()) > 0.5).cpu().numpy().astype(int)
if "id" in test_df.columns:
    sub = pd.DataFrame({"id": test_df["id"], target_col: test_preds})
else:
    sub = pd.DataFrame({target_col: test_preds})
sub.to_csv(f"{output_dir}/submission.csv", index=False)
