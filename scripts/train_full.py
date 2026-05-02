# scripts/train_full.py
# Script complet — repart de zéro et sauvegarde tout

import sys
sys.path.append('/home/jack/Dev/candles-bot/scripts')
from smc_features import *

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              f1_score)
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = '/home/jack/Dev/candles-bot/models'
DATA_DIR   = '/home/jack/Dev/candles-bot/data'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
print(f"Device : {device}")

# ── FEATURES ───────────────────────────────────────────────────
FEATURES = [
    'RSI_score_1H','RSI_7_1H','RSI_14_1H','RSI_21_1H',
    'RSI_score_4H','RSI_7_4H','RSI_14_4H',
    'RSI_score_1D','RSI_7_1D','RSI_14_1D',
    'ssl_distance_1H','bsl_distance_1H',
    'near_ssl_1H','near_bsl_1H',
    'ob_bullish_1H','ob_bearish_1H',
    'fvg_bullish_1H','fvg_bearish_1H',
    'choch_bullish_1H','choch_bearish_1H',
    'liq_sweep_bull_1H','liq_sweep_bear_1H',
    'ssl_distance_4H','bsl_distance_4H',
    'ob_bullish_4H','ob_bearish_4H',
    'choch_bullish_4H','choch_bearish_4H',
    'liq_sweep_bull_4H','liq_sweep_bear_4H',
    'ssl_distance_1D','bsl_distance_1D',
    'ob_bullish_1D','ob_bearish_1D',
    'choch_bullish_1D','liq_sweep_bull_1D',
    'CVD_norm_1H','div_bull_cvd_1H','div_bear_cvd_1H',
    'CVD_norm_4H','div_bull_cvd_4H','div_bear_cvd_4H',
    'CVD_norm_1D','div_bull_cvd_1D','div_bear_cvd_1D',
    'ATR_norm_1H','high_vol_1H','low_vol_1H',
    'ATR_norm_4H','high_vol_4H',
    'ATR_norm_1D','high_vol_1D',
    'kz_multiplier_1H','in_kill_zone_1H',
    'kz_london_1H','kz_ny_1H',
]
SEQ_LEN = 60
PIP     = 0.0001

# ── STEP 1 : DONNÉES ───────────────────────────────────────────
print("\n[1/6] Téléchargement données EUR/USD...")
df_1d_raw = yf.download("EURUSD=X", start="2020-01-01",
                         interval='1d', progress=False)
df_1d_raw.columns = [col[0] for col in df_1d_raw.columns]
df_4h_raw = yf.download("EURUSD=X", period="729d",
                         interval='4h', progress=False)
df_4h_raw.columns = [col[0] for col in df_4h_raw.columns]
df_1h_raw = yf.download("EURUSD=X", period="729d",
                         interval='1h', progress=False)
df_1h_raw.columns = [col[0] for col in df_1h_raw.columns]

print(f"  Daily:{len(df_1d_raw)} 4H:{len(df_4h_raw)} "
      f"1H:{len(df_1h_raw)}")

# ── STEP 2 : PIPELINE ──────────────────────────────────────────
print("\n[2/6] Pipeline SMC...")
df_1d  = build_pipeline(df_1d_raw, '1D')
df_4h  = build_pipeline(df_4h_raw, '4H')
df_1h  = build_pipeline(df_1h_raw, '1H')
df_mtf = align_timeframes(df_1d, df_4h, df_1h)
print(f"  MTF fusionné : {df_mtf.shape}")

# Features disponibles
features = [f for f in FEATURES if f in df_mtf.columns]
print(f"  Features     : {len(features)}")

# ── STEP 3 : LABELS ────────────────────────────────────────────
print("\n[3/6] Création labels...")

# Labels contexte
def create_context_labels_v2(df, swing_period=20):
    df    = df.copy()
    n     = len(df)
    close = df['Close'].values
    high  = df['High'].values if 'High' in df.columns \
            else df['Close'].values
    low   = df['Low'].values  if 'Low'  in df.columns \
            else df['Close'].values

    # Trend
    trend = np.zeros(n, dtype=int)
    for i in range(swing_period * 2, n):
        wh = high[i-swing_period:i]
        wl = low[i-swing_period:i]
        ph = high[i-swing_period*2:i-swing_period]
        pl = low[i-swing_period*2:i-swing_period]
        if wh.max() > ph.max() and wl.min() > pl.min():
            trend[i] =  1
        elif wh.max() < ph.max() and wl.min() < pl.min():
            trend[i] = -1
    df['trend_label'] = trend

    # AMD binaire
    amd     = np.zeros(n, dtype=int)
    atr_col = 'ATR_norm_1H' if 'ATR_norm_1H' in df.columns \
              else 'ATR_norm'
    lsb     = 'liq_sweep_bull_1H' \
              if 'liq_sweep_bull_1H' in df.columns \
              else 'liq_sweep_bull'
    lss     = 'liq_sweep_bear_1H' \
              if 'liq_sweep_bear_1H' in df.columns \
              else 'liq_sweep_bear'
    if atr_col in df.columns:
        atr      = df[atr_col].values
        atr_roll = pd.Series(atr).rolling(50).mean().values
        for i in range(50, n):
            if np.isnan(atr_roll[i]):
                continue
            is_hv   = atr[i] > 1.3 * atr_roll[i]
            is_sw   = (df[lsb].iloc[i]==1 or
                      df[lss].iloc[i]==1) \
                      if lsb in df.columns else False
            mom     = abs(close[i]-close[i-3]) / \
                      close[i-3] if i >= 3 else 0
            if is_sw or (is_hv and mom > 0.001):
                amd[i] = 1
    df['amd_label'] = amd
    return df

df_ctx = create_context_labels_v2(df_mtf)

# Label Smart RR
def create_smart_label(df, horizon=6,
                        pip_target=15, pip_stop=10):
    df    = df.copy()
    close = df['Close'].values
    n     = len(close)
    label = np.zeros(n, dtype=int)
    for i in range(n - horizon):
        entry  = close[i]
        target = entry + pip_target * PIP
        stop   = entry - pip_stop   * PIP
        for j in range(i+1, min(i+horizon+1, n)):
            if close[j] >= target:
                label[i] = 1
                break
            if close[j] <= stop:
                break
    df['Label'] = label
    return df

df_full = create_smart_label(df_ctx)
df_ml   = df_full[
    features + ['Label','trend_label','amd_label']
].dropna()

hausse = df_ml['Label'].sum()
total  = len(df_ml)
print(f"  Total  : {total}")
print(f"  Signal : {hausse} ({hausse/total*100:.1f}%)")

# ── STEP 4 : SÉQUENCES ─────────────────────────────────────────
print("\n[4/6] Préparation séquences...")

split     = int(len(df_ml) * 0.8)
df_tr     = df_ml.iloc[:split]
df_te     = df_ml.iloc[split:]

scaler    = RobustScaler()
X_tr_sc   = scaler.fit_transform(df_tr[features])
X_te_sc   = scaler.transform(df_te[features])

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_tr_seq, y_tr_seq = create_sequences(
    X_tr_sc, df_tr['Label'].values, SEQ_LEN)
X_te_seq, y_te_seq = create_sequences(
    X_te_sc, df_te['Label'].values, SEQ_LEN)

y_tr_trend = df_tr['trend_label'].values[SEQ_LEN:] + 1
y_tr_amd   = df_tr['amd_label'].values[SEQ_LEN:]
y_te_trend = df_te['trend_label'].values[SEQ_LEN:] + 1
y_te_amd   = df_te['amd_label'].values[SEQ_LEN:]

X_tr = torch.FloatTensor(X_tr_seq).to(device)
X_te = torch.FloatTensor(X_te_seq).to(device)
y_tr = torch.FloatTensor(y_tr_seq).to(device)
y_te = torch.FloatTensor(y_te_seq).to(device)

y_tr_t = torch.LongTensor(y_tr_trend).to(device)
y_tr_a = torch.LongTensor(y_tr_amd).to(device)
y_te_t = torch.LongTensor(y_te_trend).to(device)
y_te_a = torch.LongTensor(y_te_amd).to(device)

n_neg  = (y_tr_seq == 0).sum()
n_pos  = (y_tr_seq == 1).sum()
pos_w  = torch.tensor([n_neg/n_pos]).to(device)

n_acc  = (y_tr_amd == 0).sum()
n_act  = (y_tr_amd == 1).sum()
w_amd  = torch.FloatTensor(
    [1.0, n_acc/max(n_act,1)]).to(device)

print(f"  X_train : {X_tr.shape}")
print(f"  X_test  : {X_te.shape}")
print(f"  Pos weight  : {pos_w.item():.2f}")
print(f"  AMD weight  : {w_amd[1].item():.1f}x")

# ── STEP 5 : ARCHITECTURES ─────────────────────────────────────
print("\n[5/6] Architectures...")

class AttentionLayer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(h, h//2), nn.Tanh(),
            nn.Linear(h//2, 1))
    def forward(self, x):
        w = torch.softmax(self.att(x), dim=1)
        return (w * x).sum(dim=1), w

class BiLSTMContext_v2(nn.Module):
    def __init__(self, input_size, hidden=64,
                 dropout=0.3, embed_size=32):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden, num_layers=2,
            batch_first=True, bidirectional=True,
            dropout=dropout)
        self.norm         = nn.LayerNorm(hidden*2)
        self.dropout      = nn.Dropout(dropout)
        self.trend_head   = nn.Linear(hidden*2, 3)
        self.amd_head     = nn.Linear(hidden*2, 2)
        self.context_head = nn.Sequential(
            nn.Linear(hidden*2, embed_size), nn.Tanh())
    def forward(self, x):
        o, _  = self.bilstm(x)
        o     = self.norm(o)
        o     = self.dropout(o)
        last  = o[:, -1, :]
        return (self.trend_head(last),
                self.amd_head(last),
                self.context_head(last))

class TradingSystemMTF(nn.Module):
    def __init__(self, input_size,
                 ctx_embed=32, h1=64, h2=32,
                 dropout=0.3):
        super().__init__()
        self.model1 = BiLSTMContext_v2(
            input_size, hidden=64,
            dropout=dropout, embed_size=ctx_embed)
        inp2 = input_size + ctx_embed + 3 + 2
        self.lstm1    = nn.LSTM(inp2, h1, batch_first=True)
        self.lstm2    = nn.LSTM(h1,  h2, batch_first=True)
        self.att      = AttentionLayer(h2)
        self.norm1    = nn.LayerNorm(h1)
        self.norm2    = nn.LayerNorm(h2)
        self.dropout  = nn.Dropout(dropout)
        self.fc1      = nn.Linear(h2, 32)
        self.fc2      = nn.Linear(32, 1)
        self.relu     = nn.ReLU()

    def forward(self, x, freeze_m1=False):
        if freeze_m1:
            with torch.no_grad():
                tl, al, ctx = self.model1(x)
        else:
            tl, al, ctx = self.model1(x)
        ts  = torch.softmax(tl, dim=1)
        as_ = torch.softmax(al, dim=1)
        sl  = x.shape[1]
        xe  = torch.cat([
            x,
            ctx.unsqueeze(1).expand(-1,sl,-1),
            ts.unsqueeze(1).expand(-1,sl,-1),
            as_.unsqueeze(1).expand(-1,sl,-1)
        ], dim=2)
        o1,_ = self.lstm1(xe)
        o1   = self.norm1(o1)
        o1   = self.dropout(o1)
        o2,_ = self.lstm2(o1)
        o2   = self.norm2(o2)
        o2   = self.dropout(o2)
        c,_  = self.att(o2)
        out  = self.relu(self.fc1(c))
        out  = self.dropout(out)
        return self.fc2(out).squeeze()

# Instancier
n_feat = len(features)
system = TradingSystemMTF(n_feat).to(device)
print(f"  Paramètres : "
      f"{sum(p.numel() for p in system.parameters()):,}")

# ── STEP 6 : ENTRAÎNEMENT ──────────────────────────────────────
print("\n[6/6] Entraînement...")

# Phase A — Pré-entraîner M1 seul
print("\n  Phase A : Pré-entraînement BiLSTM M1...")
ce_trend = nn.CrossEntropyLoss()
ce_amd   = nn.CrossEntropyLoss(weight=w_amd)
opt_m1   = torch.optim.AdamW(
    system.model1.parameters(),
    lr=0.0003, weight_decay=1e-3)

ds_m1 = TensorDataset(X_tr, y_tr_t, y_tr_a)
dl_m1 = DataLoader(ds_m1, batch_size=64, shuffle=False)

best_m1_score = 0
best_m1_state = None
patience_m1   = 20
no_imp_m1     = 0

for epoch in range(80):
    system.model1.train()
    for Xb, yt, ya in dl_m1:
        opt_m1.zero_grad()
        tl, al, _ = system.model1(Xb)
        loss = (ce_trend(tl, yt)*0.6 +
                ce_amd(al, ya)*0.4)
        loss.backward()
        nn.utils.clip_grad_norm_(
            system.model1.parameters(), 0.5)
        opt_m1.step()

    system.model1.eval()
    with torch.no_grad():
        tl, al, _ = system.model1(X_te)
        at = (tl.argmax(1)==y_te_t).float().mean().item()
        ap = al.argmax(1).cpu().numpy()
        at2= y_te_a.cpu().numpy()
        mask = at2 == 1
        ar = (ap[mask]==1).mean() if mask.sum()>0 else 0
        score = at*0.5 + ar*0.5

    if score > best_m1_score:
        best_m1_score = score
        best_m1_state = {
            k:v.clone()
            for k,v in system.model1.state_dict().items()}
        no_imp_m1 = 0
    else:
        no_imp_m1 += 1

    if (epoch+1) % 20 == 0:
        print(f"    Epoch {epoch+1:3d} | "
              f"Trend:{at*100:.1f}% | "
              f"AMD recall:{ar*100:.1f}% | "
              f"Score:{score:.3f}")

    if no_imp_m1 >= patience_m1:
        print(f"    Early stop epoch {epoch+1}")
        break

system.model1.load_state_dict(best_m1_state)
print(f"  M1 score : {best_m1_score:.3f}")

# Sauvegarder M1
torch.save(best_m1_state,
    f'{MODELS_DIR}/bilstm_context.pt')
print(f"  M1 sauvegardé !")

# Phase B — Warmup M2 (M1 gelé)
print("\n  Phase B : Warmup M2...")
for p in system.model1.parameters():
    p.requires_grad = False

crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
opt_m2= torch.optim.AdamW(
    filter(lambda p: p.requires_grad,
           system.parameters()),
    lr=0.0003, weight_decay=3e-3)

ds_sys = TensorDataset(X_tr, y_tr)
dl_sys = DataLoader(ds_sys, batch_size=32, shuffle=False)

for epoch in range(5):
    system.train()
    for Xb, yb in dl_sys:
        opt_m2.zero_grad()
        out  = system(Xb, freeze_m1=True)
        loss = crit(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(
            system.parameters(), 0.5)
        opt_m2.step()
    print(f"    Warmup {epoch+1}/5")

# Phase C — Fine-tuning conjoint
print("\n  Phase C : Fine-tuning conjoint...")
for p in system.model1.parameters():
    p.requires_grad = True

opt_ft = torch.optim.AdamW(
    system.parameters(),
    lr=6e-5, weight_decay=3e-3)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_ft, T_max=100, eta_min=1e-7)

best_f1    = 0
best_state = None
patience   = 30
no_imp     = 0

for epoch in range(100):
    system.train()
    for Xb, yb in dl_sys:
        opt_ft.zero_grad()
        out  = system(Xb)
        loss = crit(out, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(
            system.parameters(), 0.5)
        opt_ft.step()
    sched.step()

    system.eval()
    with torch.no_grad():
        vout  = system(X_te)
        vprob = torch.sigmoid(vout).cpu().numpy()
        vpred = (vprob > 0.5).astype(int)
        vtrue = y_te.cpu().numpy().astype(int)
        f1    = f1_score(vtrue, vpred,
                          pos_label=1, zero_division=0)
        acc   = accuracy_score(vtrue, vpred)

    if f1 > best_f1:
        best_f1    = f1
        best_state = {
            k:v.clone()
            for k,v in system.state_dict().items()}
        no_imp = 0
    else:
        no_imp += 1

    if (epoch+1) % 20 == 0:
        print(f"    Epoch {epoch+1:3d} | "
              f"Acc:{acc*100:.1f}% | "
              f"F1:{f1:.3f} | "
              f"Best:{best_f1:.3f}")

    if no_imp >= patience:
        print(f"    Early stop epoch {epoch+1}")
        break

system.load_state_dict(best_state)
print(f"\n  Meilleur F1 : {best_f1:.3f}")

# ── ÉVALUATION FINALE ──────────────────────────────────────────
system.eval()
with torch.no_grad():
    vout  = system(X_te)
    proba = torch.sigmoid(vout).cpu().numpy()
    pred  = (proba > 0.5).astype(int)
    true  = y_te.cpu().numpy().astype(int)

print("\n" + "=" * 55)
print("   RÉSULTATS FINAUX")
print("=" * 55)
print(classification_report(
    true, pred,
    target_names=['Pas signal','Signal']))

print("\nAnalyse seuils :")
for s in [0.50, 0.55, 0.60, 0.65, 0.70]:
    mask = proba > s
    n    = mask.sum()
    if n > 15:
        prec = true[mask].mean()
        print(f"  {s:.2f} → {n:5d} signaux "
              f"({n/len(proba)*100:.1f}%) "
              f"precision:{prec*100:.1f}%")

# ── SAUVEGARDE FINALE ──────────────────────────────────────────
torch.save(best_state,
    f'{MODELS_DIR}/trading_system.pt')

with open(f'{MODELS_DIR}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open(f'{MODELS_DIR}/features.pkl', 'wb') as f:
    pickle.dump(features, f)

print(f"\nModèles sauvegardés dans {MODELS_DIR}/")
print("  trading_system.pt")
print("  bilstm_context.pt")
print("  scaler.pkl")
print("  features.pkl")
print("\nENTRAÎNEMENT TERMINÉ !")