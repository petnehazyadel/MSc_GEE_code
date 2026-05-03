import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# =========================================================
# PARAMÉTEREK
# =========================================================
SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 10
LR = 5e-4
RANDOM_SEED = 42

DATE_COL = "Idopont"
RAW_COL = "Nyers"
TARGET_COL = "Korrigalt"

TRAIN_START = "2020-01-01"
TRAIN_END   = "2024-12-31"

PRED_FILE = "prediktorok.xlsx"
PRED_DATE_COL = "datum"

SHEET_CSAP   = "pred_csap"
SHEET_HOM    = "pred_hom"
SHEET_SZELSB = "pred_szelseb"
SHEET_SZELIR = "pred_szelir"

CSAP_COL   = "csapadek"
HOM_COL    = "homerseklet"
SZELSB_COL = "szelsebesseg"
SZELIR_COL = "szelirany"

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# =========================================================
# SEGÉDFÜGGVÉNYEK
# =========================================================
def read_daily_series(sheet, value_col):
    """
    Napi prediktor beolvasása a PRED_FILE adott munkalapjáról.
    Visszaad: pandas Series, DateTimeIndex-szel (napi), duplikált napok átlagolva.
    """
    dfp = pd.read_excel(PRED_FILE, sheet_name=sheet)
    dfp[PRED_DATE_COL] = pd.to_datetime(dfp[PRED_DATE_COL], errors="coerce")

    dfp = dfp.dropna(subset=[PRED_DATE_COL]).copy()
    dfp = dfp.sort_values(PRED_DATE_COL)
    dfp = dfp[[PRED_DATE_COL, value_col]].copy()

    # Duplikált dátumok kezelése: átlag
    s = dfp.groupby(PRED_DATE_COL)[value_col].mean()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def bimonth(m):
    if m in [1, 2]:
        return "Jan-Feb"
    elif m in [3, 4]:
        return "Mar-Apr"
    elif m in [5, 6]:
        return "May-Jun"
    elif m in [7, 8]:
        return "Jul-Aug"
    elif m in [9, 10]:
        return "Sep-Oct"
    else:
        return "Nov-Dec"

def fit_standardizer(df, cols):
    mu = df[cols].mean()
    sd = df[cols].std().replace(0, 1.0)
    return mu, sd

def apply_standardizer(df, cols, mu, sd):
    out = df.copy()
    out[cols] = (out[cols] - mu) / sd
    return out

# =========================================================
# DATASET
# =========================================================
class CorrectionDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_col].values.astype(np.float32)

    def __len__(self):
        return len(self.y) - SEQ_LEN

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx:idx+SEQ_LEN]),
            torch.tensor(self.y[idx+SEQ_LEN])
        )

# =========================================================
# LSTM MODELL
# =========================================================
class LSTMCorrection(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =========================================================
# 1) PREDIKTOROK (NAPI → dátum szerinti illesztés)
# =========================================================
csap_s   = read_daily_series(SHEET_CSAP, CSAP_COL)
hom_s    = read_daily_series(SHEET_HOM, HOM_COL)
szelsb_s = read_daily_series(SHEET_SZELSB, SZELSB_COL)
szelir_s = read_daily_series(SHEET_SZELIR, SZELIR_COL)

# =========================================================
# 2) TANÍTÓ ADATOK (2020–2024)
# =========================================================
df_train = pd.read_excel("uj_vizhozam_tanito.xlsx")
df_train[DATE_COL] = pd.to_datetime(df_train[DATE_COL])
df_train = df_train.set_index(DATE_COL).sort_index()
df_train = df_train.loc[TRAIN_START:TRAIN_END]

idx = pd.date_range(df_train.index.min(), df_train.index.max(), freq="D")
df_train = df_train.reindex(idx)

df_train[RAW_COL] = df_train[RAW_COL].interpolate("time").ffill().bfill()
df_train[TARGET_COL] = df_train[TARGET_COL].interpolate("time").ffill().bfill()

# Prediktorok dátum szerinti hozzárendelése (join)
df_train[CSAP_COL]   = csap_s.reindex(df_train.index)
df_train[HOM_COL]    = hom_s.reindex(df_train.index)
df_train[SZELSB_COL] = szelsb_s.reindex(df_train.index)
df_train[SZELIR_COL] = szelir_s.reindex(df_train.index)

# Hiányok kezelése a prediktorokban
# (napi prediktoroknál ez a legkevesebb "érdemi változtatás": időbeli interpoláció + szélek kitöltése)
pred_cols = [CSAP_COL, HOM_COL, SZELSB_COL, SZELIR_COL]
df_train[pred_cols] = df_train[pred_cols].interpolate("time").ffill().bfill()

feature_cols = [RAW_COL, CSAP_COL, HOM_COL, SZELSB_COL, SZELIR_COL]

mu, sd = fit_standardizer(df_train, feature_cols)
df_train_scaled = apply_standardizer(df_train, feature_cols, mu, sd)

# =========================================================
# 3) LSTM TANÍTÁS
# =========================================================
dataset = CorrectionDataset(df_train_scaled, feature_cols, TARGET_COL)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LSTMCorrection(len(feature_cols))
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

model.train()
for e in range(EPOCHS):
    for X, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {e+1}/{EPOCHS} Loss: {loss.item():.4f}")

# =========================================================
# 4) LSTM VISSZAJÓSLÁS A TANÍTÓ IDŐSZAKRA
# =========================================================
model.eval()
preds_train = [np.nan]*SEQ_LEN
with torch.no_grad():
    Xall = df_train_scaled[feature_cols].values.astype(np.float32)
    for i in range(len(df_train_scaled)-SEQ_LEN):
        preds_train.append(model(torch.tensor(Xall[i:i+SEQ_LEN]).unsqueeze(0)).item())

df_train["Korrigalt_LSTM"] = preds_train

# =========================================================
# 5) KÉT-HAVI KOEFFICIENSEK (LSTM-ALAPÚ)
# =========================================================
df_train["bimonth"] = df_train.index.month.map(bimonth)
df_train["k_lstm"] = df_train["Korrigalt_LSTM"] / df_train[RAW_COL]
df_train["k_lstm"] = df_train["k_lstm"].replace([np.inf, -np.inf], np.nan)

bimonth_coeff = df_train.groupby("bimonth")["k_lstm"].mean()

print("\nLSTM + prediktor alapú két-havi koefficiensek:")
print(bimonth_coeff)

# =========================================================
# 6) ALKALMAZÁS 2018–2019-RE (PREDIKTOR NÉLKÜL)
# =========================================================
df_fix = pd.read_excel("uj_vizhozam_uj.xlsx")
df_fix[DATE_COL] = pd.to_datetime(df_fix[DATE_COL])
df_fix = df_fix.set_index(DATE_COL).sort_index()
df_fix = df_fix.loc["2018-01-01":"2019-12-31"]

idx_fix = pd.date_range(df_fix.index.min(), df_fix.index.max(), freq="D")
df_fix = df_fix.reindex(idx_fix)
df_fix[RAW_COL] = df_fix[RAW_COL].interpolate("time").ffill().bfill()

df_fix["bimonth"] = df_fix.index.month.map(bimonth)
df_fix["Korrigalt_pred"] = df_fix[RAW_COL] * df_fix["bimonth"].map(bimonth_coeff)

out = df_fix.reset_index().rename(columns={"index": DATE_COL})
out.to_excel("vizhozam_2018_2019_LSTM_2havi_koeff.xlsx", index=False)

print("\nKimenet elkészült: vizhozam_2018_2019_LSTM_2havi_koeff.xlsx")

import numpy as np
import pandas as pd
import torch

def eval_mse(model, df_scaled, feature_cols, target_col, seq_len=30):
    model.eval()
    X = df_scaled[feature_cols].values.astype(np.float32)
    y = df_scaled[target_col].values.astype(np.float32)

    preds = []
    trues = []
    with torch.no_grad():
        for i in range(len(df_scaled) - seq_len):
            X_seq = torch.tensor(X[i:i+seq_len]).unsqueeze(0)
            preds.append(model(X_seq).item())
            trues.append(y[i+seq_len])
    preds = np.array(preds)
    trues = np.array(trues)
    return np.mean((preds - trues)**2)

def permutation_importance(model, df_scaled, feature_cols, target_col, seq_len=30, n_repeats=5, seed=0):
    rng = np.random.default_rng(seed)
    base = eval_mse(model, df_scaled, feature_cols, target_col, seq_len)
    out = {}
    for col in feature_cols:
        deltas = []
        for _ in range(n_repeats):
            df_perm = df_scaled.copy()
            df_perm[col] = rng.permutation(df_perm[col].values)  # időbeli szétkeverés
            mse_perm = eval_mse(model, df_perm, feature_cols, target_col, seq_len)
            deltas.append(mse_perm - base)
        out[col] = (np.mean(deltas), np.std(deltas))
    res = pd.DataFrame(out, index=["mean_delta_mse", "std_delta_mse"]).T
    res = res.sort_values("mean_delta_mse", ascending=False)
    return base, res

base_mse, pimps = permutation_importance(model, df_train_scaled, feature_cols, TARGET_COL, seq_len=SEQ_LEN)
print("Base MSE:", base_mse)
print(pimps)

