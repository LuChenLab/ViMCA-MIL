# MIL_bootstrap_multi_trait_260714.py
# Single-trait MIL training script for single-GPU multi-process parallel running

import os
import re
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import trange

warnings.filterwarnings("ignore")

# ===============================
# 0. Argument parsing
# ===============================
parser = argparse.ArgumentParser()
parser.add_argument("--trait", type=str, required=True, help="Trait name to run")
parser.add_argument("--geno", type=str, default="../data/clinical_phenotype_analysis/MIL/prepeare/geno_feature.csv", help="Mutation feature matrix (long format)")
parser.add_argument("--pheno", type=str, default="../data/clinical_phenotype_analysis/MIL/prepeare/nor_pheno_by_Age_Gender_CCI.csv", help="Normalized clinical phenotype matrix")
parser.add_argument("--output_dir", type=str, default="./multi_traits_results")
parser.add_argument("--n_boot", type=int, default=100)
parser.add_argument("--n_full", type=int, default=10)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--base_seed", type=int, default=42)
parser.add_argument("--r_threshold", type=float, default=0.0)
parser.add_argument("--save_phase1_model", type=int, default=0, help="1: save phase1 model weights; 0: do not save")
parser.add_argument("--save_phase1_attention", type=int, default=1)
parser.add_argument("--save_phase1_pred", type=int, default=1)
parser.add_argument("--save_phase2_model", type=int, default=1)
parser.add_argument("--save_phase2_attention", type=int, default=1)
parser.add_argument("--save_phase2_pred", type=int, default=1)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = args.base_seed

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR = args.output_dir
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
WEIGHT_DIR = os.path.join(OUTPUT_DIR, "model_weights")
ATTN_DIR = os.path.join(OUTPUT_DIR, "attention_each_run")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary_each_trait")

for d in [OUTPUT_DIR, PRED_DIR, WEIGHT_DIR, ATTN_DIR, SUMMARY_DIR]:
    os.makedirs(d, exist_ok=True)

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_name(x):
    return re.sub(r"[^A-Za-z0-9_.%+-]+", "_", str(x))

trait = args.trait
trait_safe = safe_name(trait)

print(f"Trait: {trait}")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_DIR}")
print(f"N_BOOT={args.n_boot}, N_FULL={args.n_full}, EPOCHS={args.epochs}")
print(f"SAVE_PHASE1_MODEL={args.save_phase1_model}")

# ===============================
# 1. Data loading
# ===============================
GENO_PATH = args.geno
PHENO_PATH = args.pheno

geno = pd.read_csv(GENO_PATH)
pheno = pd.read_csv(PHENO_PATH)

FEATURE_COLS = [
    "phastCons", "SIFT_Probability", "PROVEAN_Score", "phyloP",
    "selection_coefficient", "relative_fitness", "evo_idx", "Protein_Func",
    "Molecular_weight", "Theoretical_PI", "Extinction_coefficients",
    "Aliphatic_index", "grand_average_of_hydropathicity",
    "delta_DDG_Env", "delta_DDG_Int"
]

if trait not in pheno.columns:
    raise ValueError(f"Trait {trait} not found in phenotype file.")

# ===============================
# 2. Standardization + sample cache
# ===============================
scaler = StandardScaler()
geno_scaled = geno.copy()
geno_scaled[FEATURE_COLS] = scaler.fit_transform(geno_scaled[FEATURE_COLS].fillna(0))

sample_cache = {}
for sid, g in geno_scaled.groupby("sample_id"):
    X = np.nan_to_num(g[FEATURE_COLS].values.astype(np.float32))
    mut_ids = g["mutation_id"].values.tolist()
    sample_cache[sid] = {
        "X": torch.tensor(X, dtype=torch.float32),
        "mut_ids": mut_ids
    }

# ===============================
# 3. Dataset class
# ===============================
class MutationMILDataset(Dataset):
    def __init__(self, pheno_df, target_col, sample_ids, sample_cache):
        pheno_df = pheno_df.dropna(subset=[target_col])
        pheno_df = pheno_df[pheno_df["sample_id"].isin(sample_ids)]

        self.y_dict = pheno_df.set_index("sample_id")[target_col].to_dict()
        self.samples = [
            s for s in pheno_df["sample_id"].values
            if s in sample_cache and s in self.y_dict
        ]
        self.sample_cache = sample_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid = self.samples[idx]
        X = self.sample_cache[sid]["X"]
        y = np.float32(self.y_dict[sid])
        mut_ids = self.sample_cache[sid]["mut_ids"]
        return X, torch.tensor(y), sid, mut_ids

# ===============================
# 4. MIL model
# ===============================
class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.attn_V = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        self.attn_U = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        self.attn_w = nn.Linear(hidden_dim // 2, 1)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, X):
        H = self.encoder(X)
        A = self.attn_w(self.attn_V(H) * self.attn_U(H))
        A = torch.softmax(A, dim=0)
        M = torch.sum(A * H, dim=0)
        y = self.regressor(M).squeeze()
        return y, A.squeeze(1)

# ===============================
# 5. Core training and evaluation functions
# ===============================
def train_and_eval(pheno_df, target_trait, train_ids, val_ids, seed, return_model=False):
    set_seed(seed)

    train_dataset = MutationMILDataset(
        pheno_df=pheno_df,
        target_col=target_trait,
        sample_ids=train_ids,
        sample_cache=sample_cache
    )

    if len(train_dataset) < 5:
        return None, {}, [], None

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    val_loader, val_dataset = None, None
    if len(val_ids) > 0:
        val_dataset = MutationMILDataset(
            pheno_df=pheno_df,
            target_col=target_trait,
            sample_ids=val_ids,
            sample_cache=sample_cache
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0
            )

    model = GatedAttentionMIL(len(FEATURE_COLS)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=1e-2
    )
    criterion = nn.MSELoss()

    # ---- Training Loop ----
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        for X, y, _, _ in train_loader:
            X = X[0].to(DEVICE)
            y = y.to(DEVICE)

            pred, _ = model(X)
            loss = criterion(pred.view(-1), y.view(-1))
            loss.backward()

        optimizer.step()

    def parse_sid(sid):
        if isinstance(sid, (list, tuple)):
            return sid[0]
        return sid

    def parse_mut_ids(mut_ids):
        curr_muts = mut_ids
        if isinstance(curr_muts, tuple):
            curr_muts = list(curr_muts)
        if isinstance(curr_muts, list) and len(curr_muts) > 0 and isinstance(curr_muts[0], tuple):
            curr_muts = [m[0] for m in curr_muts]
        return curr_muts

    def evaluate(loader, dataset_name):
        model.eval()

        y_true = []
        y_pred = []
        pred_rows = []
        total_loss = 0.0

        extract_attn = dataset_name == "Train"
        mut_stats = defaultdict(list)

        with torch.no_grad():
            for X, y, sid, mut_ids in loader:
                X = X[0].to(DEVICE)
                y = y.to(DEVICE)

                pred, attn = model(X)
                loss = criterion(pred.view(-1), y.view(-1))
                total_loss += loss.item()

                sid_clean = parse_sid(sid)
                y_true_val = float(y.item())
                y_pred_val = float(pred.item())

                y_true.append(y_true_val)
                y_pred.append(y_pred_val)

                pred_rows.append({
                    "sample_id": sid_clean,
                    "trait": target_trait,
                    "dataset": dataset_name,
                    "y_true": y_true_val,
                    "y_pred": y_pred_val,
                    "abs_error": abs(y_true_val - y_pred_val),
                    "squared_error": (y_true_val - y_pred_val) ** 2
                })

                if extract_attn:
                    curr_muts = parse_mut_ids(mut_ids)
                    curr_bag_size = len(curr_muts)
                    attn_vals = attn.cpu().numpy()

                    for m, w in zip(curr_muts, attn_vals):
                        mut_stats[m].append(float(w) * curr_bag_size)

        avg_loss = total_loss / len(loader)

        pearson_r = 0.0
        pearson_p = np.nan
        if len(y_true) > 2 and np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)

        return avg_loss, pearson_r, pearson_p, mut_stats, pred_rows

    # ---- Train evaluation ----
    t_loss, t_pr, t_p, mut_stats, train_pred_rows = evaluate(
        DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0),
        "Train"
    )

    mean_attn = {m: np.mean(v) for m, v in mut_stats.items()}

    metrics = {
        "Train_Loss": t_loss,
        "Train_Pearson_R": t_pr,
        "Train_Pearson_P": t_p,
        "Train_N": len(train_dataset)
    }

    all_pred_rows = train_pred_rows

    # ---- Val evaluation ----
    if val_loader:
        v_loss, v_pr, v_p, _, val_pred_rows = evaluate(val_loader, "Val")
        metrics.update({
            "Val_Loss": v_loss,
            "Val_Pearson_R": v_pr,
            "Val_Pearson_P": v_p,
            "Val_N": len(val_dataset)
        })
        all_pred_rows.extend(val_pred_rows)
    else:
        metrics.update({
            "Val_Loss": np.nan,
            "Val_Pearson_R": np.nan,
            "Val_Pearson_P": np.nan,
            "Val_N": 0
        })

    if return_model:
        return mean_attn, metrics, all_pred_rows, model

    return mean_attn, metrics, all_pred_rows, None

# ===============================
# 6. Main: single trait
# ===============================
valid_pheno = pheno.dropna(subset=[trait])

if len(valid_pheno) < 10:
    print(f"Skipping {trait}: N={len(valid_pheno)}")
    pd.DataFrame([{
        "Trait": trait,
        "Status": "Skipped_low_N",
        "N_samples": len(valid_pheno)
    }]).to_csv(f"{SUMMARY_DIR}/Summary_{trait_safe}.csv", index=False)
    raise SystemExit

all_samples = valid_pheno["sample_id"].unique()
print(f"N samples for {trait}: {len(all_samples)}")

# ==========================================
# Phase 1: Bootstrap Stability Evaluation
# ==========================================
print(f"  > Phase 1: Stability Check (N={args.n_boot})")

phase1_metrics = []
phase1_pred_all = []

for b in trange(args.n_boot, desc=f"{trait_safe} Phase1"):
    seed_b = BASE_SEED + b

    n_train = int(0.8 * len(all_samples))
    train_samples = np.random.choice(
        all_samples,
        size=n_train,
        replace=False
    )
    val_samples = np.setdiff1d(all_samples, train_samples)

    attn_b, m, pred_rows, model = train_and_eval(
        pheno_df=valid_pheno,
        target_trait=trait,
        train_ids=train_samples,
        val_ids=val_samples,
        seed=seed_b,
        return_model=True
    )

    if m:
        m.update({
            "trait": trait,
            "run_type": "phase1_bootstrap",
            "run_index": b,
            "seed": seed_b,
            "n_train_input": len(train_samples),
            "n_val_input": len(val_samples)
        })
        phase1_metrics.append(m)

        if args.save_phase1_pred:
            for r in pred_rows:
                r.update({
                    "run_type": "phase1_bootstrap",
                    "run_index": b,
                    "seed": seed_b
                })
                phase1_pred_all.append(r)

        if args.save_phase1_model:
            torch.save({
                "trait": trait,
                "run_type": "phase1_bootstrap",
                "run_index": b,
                "seed": seed_b,
                "model_state_dict": model.state_dict(),
                "feature_cols": FEATURE_COLS,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "train_sample_ids": list(train_samples),
                "val_sample_ids": list(val_samples),
                "metrics": m
            }, f"{WEIGHT_DIR}/{trait_safe}_phase1_run{b:03d}_model.pt")

        if args.save_phase1_attention and attn_b is not None:
            attn_rows_b = [
                {
                    "trait": trait,
                    "run_type": "phase1_bootstrap",
                    "run_index": b,
                    "seed": seed_b,
                    "mutation_id": mut,
                    "attention": val
                }
                for mut, val in attn_b.items()
            ]

            if len(attn_rows_b) > 0:
                pd.DataFrame(attn_rows_b).sort_values(
                    "attention",
                    ascending=False
                ).to_csv(
                    f"{ATTN_DIR}/{trait_safe}_phase1_run{b:03d}_attention.csv",
                    index=False
                )

if not phase1_metrics:
    print(f"Failed: no successful phase1 run for {trait}")
    pd.DataFrame([{
        "Trait": trait,
        "Status": "Failed_phase1",
        "N_samples": len(all_samples)
    }]).to_csv(f"{SUMMARY_DIR}/Summary_{trait_safe}.csv", index=False)
    raise SystemExit

df_p1 = pd.DataFrame(phase1_metrics)
df_p1.to_csv(f"{OUTPUT_DIR}/{trait_safe}_phase1_fit_metrics.csv", index=False)

if args.save_phase1_pred and len(phase1_pred_all) > 0:
    pd.DataFrame(phase1_pred_all).to_csv(
        f"{PRED_DIR}/{trait_safe}_phase1_predictions.csv",
        index=False
    )

avg_val_pr = df_p1["Val_Pearson_R"].mean()
print(f"  Avg Val Pearson R: {avg_val_pr:.4f}")

if avg_val_pr < args.r_threshold:
    print(f"  [SKIP] Unstable model: Pearson R < {args.r_threshold}")
    pd.DataFrame([{
        "Trait": trait,
        "Status": "Skipped_unstable",
        "Phase1_Val_Pearson_R": avg_val_pr,
        "N_samples": len(all_samples),
        "N_phase1_runs": len(phase1_metrics)
    }]).to_csv(f"{SUMMARY_DIR}/Summary_{trait_safe}.csv", index=False)
    raise SystemExit

# ==========================================
# Phase 2: Full Dataset Discovery
# ==========================================
print(f"  > Phase 2: Full Training Discovery (Repeats={args.n_full})")

final_attentions = defaultdict(list)
phase2_fits = []
phase2_pred_all = []

for i in trange(args.n_full, desc=f"{trait_safe} Phase2"):
    seed_i = BASE_SEED + 1000 + i

    attn, m, pred_rows, model = train_and_eval(
        pheno_df=valid_pheno,
        target_trait=trait,
        train_ids=all_samples,
        val_ids=[],
        seed=seed_i,
        return_model=True
    )

    phase2_fits.append({
        "trait": trait,
        "run_type": "phase2_full_fit",
        "run_index": i,
        "seed": seed_i,
        "Fit_Pearson_R": m["Train_Pearson_R"],
        "Fit_Pearson_P": m["Train_Pearson_P"],
        "Fit_Loss": m["Train_Loss"],
        "Fit_N": m["Train_N"]
    })

    if args.save_phase2_pred:
        for r in pred_rows:
            r.update({
                "run_type": "phase2_full_fit",
                "run_index": i,
                "seed": seed_i
            })
            phase2_pred_all.append(r)

    if args.save_phase2_attention:
        attn_rows_i = []
        for mut, val in attn.items():
            final_attentions[mut].append(val)
            attn_rows_i.append({
                "trait": trait,
                "run_type": "phase2_full_fit",
                "run_index": i,
                "seed": seed_i,
                "mutation_id": mut,
                "attention": val
            })

        pd.DataFrame(attn_rows_i).sort_values(
            "attention",
            ascending=False
        ).to_csv(
            f"{ATTN_DIR}/{trait_safe}_phase2_run{i:02d}_attention.csv",
            index=False
        )
    else:
        for mut, val in attn.items():
            final_attentions[mut].append(val)

    if args.save_phase2_model:
        torch.save({
            "trait": trait,
            "run_type": "phase2_full_fit",
            "run_index": i,
            "seed": seed_i,
            "model_state_dict": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "train_sample_ids": list(all_samples),
            "metrics": m
        }, f"{WEIGHT_DIR}/{trait_safe}_phase2_run{i:02d}_model.pt")

df_p2 = pd.DataFrame(phase2_fits)
avg_fit_pr = df_p2["Fit_Pearson_R"].mean()
print(f"  Full Fit Pearson R: {avg_fit_pr:.4f}")

df_p2.to_csv(f"{OUTPUT_DIR}/{trait_safe}_full_fit_metrics.csv", index=False)

if args.save_phase2_pred and len(phase2_pred_all) > 0:
    pd.DataFrame(phase2_pred_all).to_csv(
        f"{PRED_DIR}/{trait_safe}_phase2_full_fit_predictions.csv",
        index=False
    )

# Aggregate mutation importance
mut_rows = []
for mut, vals in final_attentions.items():
    vals = np.array(vals)
    mut_rows.append({
        "mutation_id": mut,
        "mean_attention": np.mean(vals),
        "std_attention": np.std(vals),
        "median_attention": np.median(vals),
        "min_attention": np.min(vals),
        "max_attention": np.max(vals),
        "n_repeats": len(vals)
    })

df_mut = pd.DataFrame(mut_rows).sort_values(
    "mean_attention",
    ascending=False
)
df_mut.to_csv(
    f"{OUTPUT_DIR}/Result_{trait_safe}_Importance.csv",
    index=False
)

pd.DataFrame([{
    "Trait": trait,
    "Status": "Success",
    "Phase1_Val_Pearson_R": avg_val_pr,
    "Phase2_Fit_Pearson_R": avg_fit_pr,
    "N_samples": len(all_samples),
    "N_phase1_runs": len(phase1_metrics),
    "N_phase2_runs": args.n_full
}]).to_csv(
    f"{SUMMARY_DIR}/Summary_{trait_safe}.csv",
    index=False
)

print(f"Done: {trait}")
