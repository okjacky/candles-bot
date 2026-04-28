# scripts/smc_features.py
# Fonctions SMC centralisées — EUR/USD Multi-Timeframe

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import ta
from ta.momentum   import RSIIndicator
from ta.volatility import AverageTrueRange

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines   import Line2D


# ── CONFIG GRAPHIQUE ───────────────────────────────────────────
def set_dark_theme():
    plt.rcParams.update({
        'figure.facecolor' : '#0d1117',
        'axes.facecolor'   : '#0d1117',
        'axes.edgecolor'   : '#30363d',
        'axes.labelcolor'  : '#8b949e',
        'text.color'       : '#e6edf3',
        'xtick.color'      : '#8b949e',
        'ytick.color'      : '#8b949e',
        'grid.color'       : '#21262d',
        'grid.linewidth'   : 0.5,
        'figure.dpi'       : 120,
    })


# ── RSI MULTI-PERIODE ──────────────────────────────────────────
def add_rsi_ribbon(df):
    df = df.copy()

    def calc_rsi(series, window):
        delta    = series.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=window-1, adjust=False).mean()
        avg_loss = loss.ewm(com=window-1, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    df['RSI_7']  = calc_rsi(df['Close'], 7)
    df['RSI_14'] = calc_rsi(df['Close'], 14)
    df['RSI_21'] = calc_rsi(df['Close'], 21)

    def score(row):
        s  = 1 if row['RSI_7']  > row['RSI_14'] else -1
        s += 1 if row['RSI_14'] > row['RSI_21'] else -1
        s += 1 if row['RSI_7']  > row['RSI_21'] else -1
        return s

    df['RSI_score']  = df.apply(score, axis=1)
    mapping = {
         3: 'BULL_MAX',  2: 'BULL',  1: 'BULL_WEAK',
        -1: 'BEAR_WEAK',-2: 'BEAR', -3: 'BEAR_MAX',
         0: 'NEUTRAL'
    }
    df['RSI_signal'] = df['RSI_score'].map(
        mapping).fillna('NEUTRAL')
    return df


# ── ATR MANUEL ─────────────────────────────────────────────────
def calc_atr(df, window=14):
    high  = df['High']
    low   = df['Low']
    close = df['Close']
    tr1   = high - low
    tr2   = (high - close.shift()).abs()
    tr3   = (low  - close.shift()).abs()
    tr    = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=window-1, adjust=False).mean()


# ── SMC FEATURES ───────────────────────────────────────────────
def add_smc_features(df, swing_period=20):
    df = df.copy()

    # SSL / BSL
    df['swing_low']    = df['Low'].rolling(swing_period).min()
    df['swing_high']   = df['High'].rolling(swing_period).max()
    df['ssl_distance'] = (
        (df['Close'] - df['swing_low']) / df['Close'])
    df['bsl_distance'] = (
        (df['swing_high'] - df['Close']) / df['Close'])
    df['near_ssl']     = (df['ssl_distance'] < 0.005).astype(int)
    df['near_bsl']     = (df['bsl_distance'] < 0.005).astype(int)

    # Order Blocks
    vol_mean  = df['Volume'].rolling(20).mean()
    ret_std   = df['Close'].pct_change().rolling(20).std()
    body_size = abs(df['Close'] - df['Open']) / df['Open']

    df['ob_bullish'] = (
        (df['Close'] > df['Open']) &
        (body_size   > 2 * ret_std) &
        (df['Volume']> 1.5 * vol_mean)
    ).astype(int)
    df['ob_bearish'] = (
        (df['Close'] < df['Open']) &
        (body_size   > 2 * ret_std) &
        (df['Volume']> 1.5 * vol_mean)
    ).astype(int)

    # FVG
    df['fvg_bullish'] = (
        df['Low'] > df['High'].shift(2)).astype(int)
    df['fvg_bearish'] = (
        df['High'] < df['Low'].shift(2)).astype(int)

    # CHoCH
    prev_high = df['High'].rolling(10).max().shift(1)
    prev_low  = df['Low'].rolling(10).min().shift(1)
    df['choch_bullish'] = (
        (df['Close'] > prev_high) &
        (df['Close'].shift(1) <= prev_high.shift(1))
    ).astype(int)
    df['choch_bearish'] = (
        (df['Close'] < prev_low) &
        (df['Close'].shift(1) >= prev_low.shift(1))
    ).astype(int)

    # Liquidity Sweep
    df['liq_sweep_bull'] = (
        (df['Low']   < df['swing_low'].shift(1)) &
        (df['Close'] > df['swing_low'].shift(1))
    ).astype(int)
    df['liq_sweep_bear'] = (
        (df['High']  > df['swing_high'].shift(1)) &
        (df['Close'] < df['swing_high'].shift(1))
    ).astype(int)

    return df.dropna()


# ── CONFIRMATION : ATR + CVD + DIVERGENCES ────────────────────
def add_confirmation_features(df):
    df = df.copy()

    # ATR
    df['ATR']       = calc_atr(df, window=14)
    df['ATR_norm']  = df['ATR'] / df['Close']
    atr_mean        = df['ATR_norm'].rolling(50).mean()
    df['high_vol']  = (
        df['ATR_norm'] > 1.5 * atr_mean).astype(int)
    df['low_vol']   = (
        df['ATR_norm'] < 0.7 * atr_mean).astype(int)
    df['sl_dynamic']= df['Close'] - 1.5 * df['ATR']
    df['tp_dynamic']= df['Close'] + 3.0 * df['ATR']

    # CVD proxy
    vol = df['Volume'].copy()
    if (vol == 0).mean() > 0.5:
        vol = df['ATR'] * df['Close'] * 1000

    hl             = (df['High'] - df['Low']).replace(0, np.nan)
    df['delta']    = vol * (
        (df['Close'] - df['Low']) -
        (df['High']  - df['Close'])
    ) / hl
    df['CVD']      = df['delta'].cumsum()
    std_cvd        = df['CVD'].rolling(50).std().replace(0, np.nan)
    df['CVD_norm'] = df['CVD'] / std_cvd

    # Divergences
    lb = 5
    df['div_bear_rsi'] = (
        (df['Close']  > df['Close'].shift(lb)) &
        (df['RSI_14'] < df['RSI_14'].shift(lb))
    ).astype(int)
    df['div_bull_rsi'] = (
        (df['Close']  < df['Close'].shift(lb)) &
        (df['RSI_14'] > df['RSI_14'].shift(lb))
    ).astype(int)
    df['div_bear_cvd'] = (
        (df['Close'] > df['Close'].shift(lb)) &
        (df['CVD']   < df['CVD'].shift(lb))
    ).astype(int)
    df['div_bull_cvd'] = (
        (df['Close'] < df['Close'].shift(lb)) &
        (df['CVD']   > df['CVD'].shift(lb))
    ).astype(int)

    df = df[df['ATR'].notna()]
    df = df[df['CVD_norm'].notna()]
    return df


# ── KILL ZONES FOREX ───────────────────────────────────────────
def add_kill_zones_forex(df):
    df = df.copy()
    try:
        hour = df.index.hour
        df['kz_asian']      = (
            (hour >= 0)  & (hour < 3)).astype(int)
        df['kz_london']     = (
            (hour >= 7)  & (hour < 10)).astype(int)
        df['kz_ny']         = (
            (hour >= 13) & (hour < 16)).astype(int)
        df['kz_london_cls'] = (
            (hour >= 15) & (hour < 17)).astype(int)
        df['kz_ny_cls']     = (
            (hour >= 20) & (hour < 22)).astype(int)
        df['in_kill_zone']  = (
            df['kz_london'] | df['kz_ny']).astype(int)
        df['kz_multiplier'] = 1.0
        df.loc[df['kz_london']==1,     'kz_multiplier'] = 3.0
        df.loc[df['kz_ny']==1,         'kz_multiplier'] = 2.5
        df.loc[df['kz_london_cls']==1, 'kz_multiplier'] = 2.0
        df.loc[df['kz_ny_cls']==1,     'kz_multiplier'] = 1.5
        df.loc[df['kz_asian']==1,      'kz_multiplier'] = 0.5
        df['session'] = 'offmarket'
        df.loc[df['kz_asian']==1,      'session'] = 'asian'
        df.loc[df['kz_london']==1,     'session'] = 'london'
        df.loc[df['kz_ny']==1,         'session'] = 'newyork'
        df.loc[df['kz_london_cls']==1, 'session'] = 'london_close'
        df.loc[df['kz_ny_cls']==1,     'session'] = 'ny_close'
    except AttributeError:
        df['kz_london']    = 0
        df['kz_ny']        = 0
        df['in_kill_zone'] = 0
        df['kz_multiplier']= 1.0
        df['session']      = 'daily'
    return df


# ── ZONES OB / FVG DYNAMIQUES ─────────────────────────────────
def compute_zones(df, max_len_strong=20, max_len_weak=5):
    zones    = []
    vol_mean = df['Volume'].rolling(20).mean()
    atr      = df['ATR']

    for i in range(2, len(df)):
        row      = df.iloc[i]
        vol_r    = (row['Volume'] / vol_mean.iloc[i]
                    if vol_mean.iloc[i] > 0 else 1)
        body     = abs(row['Close'] - row['Open'])
        atr_val  = atr.iloc[i]
        strength = vol_r * (body / atr_val if atr_val > 0 else 1)
        zone_len = int(np.clip(
            max_len_weak + (max_len_strong - max_len_weak)
            * min(strength / 3, 1),
            max_len_weak, max_len_strong))

        configs = []
        if row['ob_bullish'] == 1:
            configs.append(
                ('OB↑','#26a69a', row['High'], row['Open'], False))
        if row['ob_bearish'] == 1:
            configs.append(
                ('OB↓','#ef5350', row['Open'], row['Low'],  True))
        if row['fvg_bullish'] == 1 and i >= 2:
            top = row['Low']
            bot = df.iloc[i-2]['High']
            if top > bot:
                gap = (top-bot)/atr_val if atr_val > 0 else 1
                zl  = int(np.clip(
                    max_len_weak + (max_len_strong-max_len_weak)
                    * min(gap, 1),
                    max_len_weak, max_len_strong))
                configs.append(('FVG↑','#26a69a', top, bot, False))
                zone_len = zl
        if row['fvg_bearish'] == 1 and i >= 2:
            top = df.iloc[i-2]['Low']
            bot = row['High']
            if top > bot:
                gap = (top-bot)/atr_val if atr_val > 0 else 1
                zl  = int(np.clip(
                    max_len_weak + (max_len_strong-max_len_weak)
                    * min(gap, 1),
                    max_len_weak, max_len_strong))
                configs.append(('FVG↓','#ef5350', top, bot, True))
                zone_len = zl

        for ztype, color, top, bot, is_bear in configs:
            end_idx = i + zone_len
            for j in range(i+1, min(i+zone_len+1, len(df))):
                if is_bear and df.iloc[j]['High'] >= top:
                    end_idx = j
                    break
                elif not is_bear and df.iloc[j]['Low'] <= bot:
                    end_idx = j
                    break
            zones.append({
                'type'     : ztype,
                'start'    : i,
                'end'      : end_idx,
                'top'      : top,
                'bottom'   : bot,
                'color'    : color,
                'strength' : strength,
                'mitigated': end_idx < i + zone_len
            })
    return zones


# ── PIPELINE COMPLET ───────────────────────────────────────────
def build_pipeline(df_raw, timeframe, symbol='EURUSD'):
    print(f"\nPipeline {timeframe}...")
    df = df_raw.copy()

    df = add_kill_zones_forex(df)
    print(f"  Kill Zones   : {df.shape}")

    df = add_rsi_ribbon(df)
    print(f"  RSI Ribbon   : {df.shape}")

    df = add_smc_features(df, swing_period=20)
    print(f"  SMC Features : {df.shape}")

    df = add_confirmation_features(df)
    print(f"  ATR + CVD    : {df.shape}")

    exclude = ['Open','High','Low','Close','Volume']
    rename  = {
        col: f"{col}_{timeframe}"
        for col in df.columns
        if col not in exclude
    }
    df = df.rename(columns=rename)
    print(f"  Shape final  : {df.shape}")
    print(f"  Période      : "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


# ── ALIGNEMENT MULTI-TIMEFRAME ─────────────────────────────────
def align_timeframes(df_1d, df_4h, df_1h):
    print("Alignement des timeframes...")

    df_1d.index = pd.to_datetime(df_1d.index, utc=True)
    df_4h.index = pd.to_datetime(df_4h.index, utc=True)
    df_1h.index = pd.to_datetime(df_1h.index, utc=True)

    cols_1d = [c for c in df_1d.columns
               if c not in ['Open','High','Low','Close','Volume']]
    cols_4h = [c for c in df_4h.columns
               if c not in ['Open','High','Low','Close','Volume']]

    df_1d_ff = df_1d[cols_1d].reindex(
        df_1h.index, method='ffill')
    df_4h_ff = df_4h[cols_4h].reindex(
        df_1h.index, method='ffill')

    df_mtf = pd.concat([df_1h, df_1d_ff, df_4h_ff], axis=1)
    df_mtf = df_mtf.dropna()

    print(f"Shape fusionné : {df_mtf.shape}")
    print(f"Période        : "
          f"{df_mtf.index[0].date()} → "
          f"{df_mtf.index[-1].date()}")
    return df_mtf


# ── GRAPHIQUE SMC COMPLET ──────────────────────────────────────
def plot_smc_chart(df, zones, title, n_bars=90,
                   show_killzones=False, figsize=(20, 14)):

    df_plot = df.tail(n_bars).copy().reset_index(drop=False)
    n       = len(df_plot)
    offset  = len(df) - n_bars

    visible = [
        z for z in zones
        if z['end'] >= offset and z['start'] <= offset + n_bars
    ]

    hr   = ([5, 1.8, 1.8, 1.5, 1.2]
            if show_killzones else [5, 1.8, 1.8, 1.5])
    fig  = plt.figure(figsize=figsize)
    gs   = gridspec.GridSpec(
        len(hr), 1, height_ratios=hr, hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(len(hr))]
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]

    # Bougies
    for i, row in df_plot.iterrows():
        c = '#26a69a' if row['Close'] >= row['Open'] \
            else '#ef5350'
        ax1.bar(i, abs(row['Close'] - row['Open']),
                bottom=min(row['Open'], row['Close']),
                color=c, width=0.7, alpha=0.9, zorder=2)
        ax1.plot([i, i], [row['Low'], row['High']],
                 color=c, linewidth=0.8, zorder=2)

    # SSL / BSL
    ax1.plot(range(n), df_plot['swing_low'].values,
             color='#ef5350', linewidth=1.0,
             linestyle='--', alpha=0.6, label='SSL')
    ax1.plot(range(n), df_plot['swing_high'].values,
             color='#26a69a', linewidth=1.0,
             linestyle='--', alpha=0.6, label='BSL')

    # Zones OB / FVG
    legend_zones = {}
    for z in visible:
        x_start = max(0,   z['start'] - offset)
        x_end   = min(n-1, z['end']   - offset)
        if x_start >= x_end:
            continue
        a_fill = 0.07 if z['mitigated'] else 0.18
        a_edge = 0.3  if z['mitigated'] else 0.8
        lstyle = ':'  if z['mitigated'] else '-'

        rect = Rectangle(
            (x_start, z['bottom']),
            x_end - x_start,
            z['top'] - z['bottom'],
            linewidth=0.8,
            edgecolor=z['color'],
            facecolor=z['color'],
            alpha=a_fill,
            linestyle=lstyle,
            zorder=1)
        ax1.add_patch(rect)
        ax1.plot([x_start, x_end],
                 [z['top'],    z['top']],
                 color=z['color'], linewidth=0.8,
                 linestyle=lstyle, alpha=a_edge, zorder=3)
        ax1.plot([x_start, x_end],
                 [z['bottom'], z['bottom']],
                 color=z['color'], linewidth=0.8,
                 linestyle=lstyle, alpha=a_edge, zorder=3)
        status = '✓' if z['mitigated'] else ''
        ax1.text(x_end + 0.3,
                 (z['top'] + z['bottom']) / 2,
                 f"{z['type']}{status}",
                 fontsize=6, color=z['color'],
                 alpha=a_edge, va='center', zorder=4)
        key = z['type'] + (
            '_mit' if z['mitigated'] else '_act')
        if key not in legend_zones:
            lbl = (f"{z['type']} "
                   f"{'mitigé' if z['mitigated'] else 'actif'}")
            legend_zones[key] = mpatches.Patch(
                color=z['color'], alpha=0.4, label=lbl)

    # CHoCH
    for i, row in df_plot.iterrows():
        if row.get('choch_bullish', 0) == 1:
            ax1.annotate('CHoCH↑',
                xy=(i, row['High']),
                xytext=(i, row['High'] * 1.004),
                fontsize=6, color='#26a69a',
                ha='center', va='bottom',
                arrowprops=dict(
                    arrowstyle='->', color='#26a69a', lw=1.2),
                zorder=5)
        if row.get('choch_bearish', 0) == 1:
            ax1.annotate('CHoCH↓',
                xy=(i, row['Low']),
                xytext=(i, row['Low'] * 0.996),
                fontsize=6, color='#ef5350',
                ha='center', va='top',
                arrowprops=dict(
                    arrowstyle='->', color='#ef5350', lw=1.2),
                zorder=5)

    # Liquidity Sweeps
    for i, row in df_plot.iterrows():
        if row.get('liq_sweep_bull', 0) == 1:
            ax1.scatter(i, row['Low'] * 0.999,
                       marker='*', s=100,
                       color='#26a69a', zorder=6)
        if row.get('liq_sweep_bear', 0) == 1:
            ax1.scatter(i, row['High'] * 1.001,
                       marker='*', s=100,
                       color='#ef5350', zorder=6)

    base_legend = [
        Line2D([0],[0], color='#ef5350', linewidth=1,
               linestyle='--', label='SSL'),
        Line2D([0],[0], color='#26a69a', linewidth=1,
               linestyle='--', label='BSL'),
        Line2D([0],[0], marker='*', color='w',
               markerfacecolor='#26a69a',
               markersize=7, label='Sweep↑'),
        Line2D([0],[0], marker='*', color='w',
               markerfacecolor='#ef5350',
               markersize=7, label='Sweep↓'),
    ]
    ax1.legend(
        handles=base_legend + list(legend_zones.values()),
        loc='upper left', fontsize=6.5, ncol=4,
        facecolor='#161b22', edgecolor='#30363d',
        framealpha=0.85)
    ax1.set_title(title, fontsize=11, pad=8)
    ax1.set_ylabel('Prix', fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, p: f'{v:.5f}'))

    # RSI Ribbon
    rsi7  = df_plot['RSI_7'].values
    rsi14 = df_plot['RSI_14'].values
    rsi21 = df_plot['RSI_21'].values
    ax2.plot(range(n), rsi7,  color='#26a69a',
             linewidth=1.0, label='RSI 7',  alpha=0.9)
    ax2.plot(range(n), rsi14, color='#EF9F27',
             linewidth=1.0, label='RSI 14', alpha=0.9)
    ax2.plot(range(n), rsi21, color='#ef5350',
             linewidth=1.0, label='RSI 21', alpha=0.9)
    ax2.fill_between(range(n), rsi7, rsi21,
                     where=(rsi7 > rsi21),
                     alpha=0.12, color='#26a69a')
    ax2.fill_between(range(n), rsi7, rsi21,
                     where=(rsi7 < rsi21),
                     alpha=0.12, color='#ef5350')
    scores = df_plot['RSI_score'].values
    for i, s in enumerate(scores):
        if   s ==  3:
            ax2.axvspan(i-0.5, i+0.5,
                       alpha=0.12, color='#26a69a')
        elif s == -3:
            ax2.axvspan(i-0.5, i+0.5,
                       alpha=0.12, color='#ef5350')
    for lvl, col in [
            (70,'#ef5350'),(50,'#8b949e'),(30,'#26a69a')]:
        ax2.axhline(lvl, color=col, linewidth=0.6,
                   linestyle=':', alpha=0.5)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', fontsize=8)
    ax2.legend(loc='upper left', fontsize=6.5, ncol=3,
               facecolor='#161b22', edgecolor='#30363d',
               framealpha=0.85)
    ax2.grid(True, alpha=0.2)

    # CVD
    cvd = df_plot['CVD_norm'].values
    ax3.plot(range(n), cvd, color='#b87af5',
             linewidth=1.0, label='CVD Proxy')
    ax3.fill_between(range(n), cvd, 0,
                     where=(cvd > 0),
                     alpha=0.15, color='#26a69a')
    ax3.fill_between(range(n), cvd, 0,
                     where=(cvd < 0),
                     alpha=0.15, color='#ef5350')
    for i, row in df_plot.iterrows():
        if row.get('div_bull_cvd', 0) == 1:
            ax3.scatter(i, cvd[i], marker='^',
                       s=50, color='#26a69a', zorder=5)
        if row.get('div_bear_cvd', 0) == 1:
            ax3.scatter(i, cvd[i], marker='v',
                       s=50, color='#ef5350', zorder=5)
    ax3.axhline(0, color='#8b949e',
                linewidth=0.6, alpha=0.5)
    ax3.set_ylabel('CVD', fontsize=8)
    ax3.legend(loc='upper left', fontsize=6.5,
               facecolor='#161b22', edgecolor='#30363d',
               framealpha=0.85)
    ax3.grid(True, alpha=0.2)

    # ATR
    atr = df_plot['ATR_norm'].values * 100
    ax4.fill_between(range(n), atr, 0,
                     color='#EF9F27', alpha=0.35)
    ax4.plot(range(n), atr, color='#EF9F27',
             linewidth=0.8,
             label=f'ATR% (moy:{np.mean(atr):.3f}%)')
    ax4.axhline(np.mean(atr), color='#EF9F27',
                linewidth=0.8, linestyle='--', alpha=0.6)
    for i, row in df_plot.iterrows():
        if row.get('high_vol', 0) == 1:
            ax4.axvspan(i-0.5, i+0.5,
                       alpha=0.12, color='#ef5350')
    ax4.set_ylabel('ATR%', fontsize=8)
    ax4.legend(loc='upper left', fontsize=6.5,
               facecolor='#161b22', edgecolor='#30363d',
               framealpha=0.85)
    ax4.grid(True, alpha=0.2)

    # Kill Zones panel 5
    if show_killzones and len(axes) == 5:
        ax5  = axes[4]
        kz   = (df_plot['kz_multiplier'].values
                if 'kz_multiplier' in df_plot.columns
                else np.ones(n))
        kz_colors = []
        for v in kz:
            if   v >= 3.0: kz_colors.append('#26a69a')
            elif v >= 2.5: kz_colors.append('#EF9F27')
            elif v >= 2.0: kz_colors.append('#b87af5')
            elif v <= 0.8: kz_colors.append('#30363d')
            else:          kz_colors.append('#8b949e')
        ax5.bar(range(n), kz, color=kz_colors,
                alpha=0.75, width=0.8)
        ax5.axhline(1.0, color='#8b949e',
                   linewidth=0.6, linestyle='--')
        ax5.set_ylim(0, 3.5)
        ax5.set_ylabel('KZ ×', fontsize=8)
        ax5.grid(True, alpha=0.2)
        kz_leg = [
            mpatches.Patch(color='#26a69a',
                           label='London ×3.0'),
            mpatches.Patch(color='#EF9F27',
                           label='NY Open ×2.5'),
            mpatches.Patch(color='#b87af5',
                           label='London Cls ×2.0'),
            mpatches.Patch(color='#30363d',
                           label='Asian ×0.5'),
        ]
        ax5.legend(handles=kz_leg, loc='upper right',
                  fontsize=6, ncol=4,
                  facecolor='#161b22', edgecolor='#30363d')
        for i, v in enumerate(kz):
            if v >= 2.5:
                ax1.axvspan(i-0.5, i+0.5,
                           alpha=0.04, color='#26a69a',
                           zorder=0)

    # Dates
    last_ax  = axes[-1]
    step     = max(1, n // 12)
    ticks    = list(range(0, n, step))
    date_col = ('Date' if 'Date' in df_plot.columns
                else df_plot.columns[0])
    labels   = []
    for i in ticks:
        try:
            d   = pd.Timestamp(df_plot[date_col].iloc[i])
            fmt = ('%d/%m %Hh'
                   if hasattr(d, 'hour') and d.hour != 0
                   else '%d/%m')
            labels.append(d.strftime(fmt))
        except:
            labels.append(str(i))
    last_ax.set_xticks(ticks)
    last_ax.set_xticklabels(
        labels, rotation=45, fontsize=7)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    safe  = (title.replace('/', '-')
                   .replace(' ', '_')
                   .replace('—', '-'))
    plt.savefig(
        f"/home/jack/Dev/candles-bot/models/{safe}.png",
        bbox_inches='tight',
        facecolor='#0d1117', dpi=150)
    plt.show()
    print(f"Sauvegardé : {title}")
