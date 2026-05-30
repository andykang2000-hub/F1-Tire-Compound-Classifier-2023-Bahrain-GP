"""
F1 Tire Compound Classifier — 2023 Bahrain GP Race
3-Class Classification (SOFT / MEDIUM / HARD) · 5 Models Compared
+ Confidence Intervals · Stint Life Predictor

Author : Yoon
Data   : FastF1 (https://github.com/theOehrly/Fast-F1)
Output : outputs/bahrain_tyre_classifier_full.png
         outputs/bahrain_tyre_classifier_confidence.png
         outputs/medium_synth_validation.png

Critical self-evaluation:
- Cause-effect: tire behavior CAUSES telemetry patterns — relationship is valid
- Genuine novelty: rival teams cannot directly observe competitor tire age/compound
- Limitation: trained on one race, generalisation requires multi-race data
- LapNumber excluded as feature (data leak: model would learn race structure, not tire physics)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import fastf1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import ks_2samp

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Cache & session ───────────────────────────────────────────────────────────
os.makedirs('f1_cache', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

fastf1.Cache.enable_cache('f1_cache')
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load(laps=True, telemetry=False, weather=False, messages=False)
laps = session.laps

# ── Feature engineering ───────────────────────────────────────────────────────
laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

valid_compounds = ['SOFT', 'MEDIUM', 'HARD']
df = laps[
    laps['Compound'].isin(valid_compounds) &
    laps['LapTimeSeconds'].notna() &
    laps['TyreLife'].notna() &
    (laps['LapTimeSeconds'] < 120)
].copy()

df['Sector1Seconds'] = df['Sector1Time'].dt.total_seconds()
df['Sector2Seconds'] = df['Sector2Time'].dt.total_seconds()
df['Sector3Seconds'] = df['Sector3Time'].dt.total_seconds()

df['DriverStintBest']  = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].transform('min')
df['DegradationDelta'] = df['LapTimeSeconds'] - df['DriverStintBest']

df = df.sort_values(['Driver', 'Stint', 'LapNumber'])
df['LapTimeDiff'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].diff()

# Features — LapNumber excluded (data leak: encodes race structure not tire physics)
features = [
    'TyreLife',
    'LapTimeSeconds',
    'DegradationDelta',
    'LapTimeDiff',
    'Sector1Seconds',
    'Sector2Seconds',
    'Sector3Seconds',
]

df_clean = df[features + ['Compound', 'LapNumber', 'Driver', 'Stint']].dropna()
df_model = df_clean[df_clean['Compound'] != 'MEDIUM'].copy()

soft_laps = df_model[df_model['Compound'] == 'SOFT']
hard_laps = df_model[df_model['Compound'] == 'HARD']

print(f"Real data — SOFT: {len(soft_laps)}, HARD: {len(hard_laps)}")

# ── Synthetic MEDIUM generation ───────────────────────────────────────────────
np.random.seed(42)

def synthesize_medium(soft_df, hard_df, n=200):
    synth  = {}
    weight = 0.55

    for feat in features:
        soft_mean = soft_df[feat].mean()
        soft_std  = soft_df[feat].std()
        hard_mean = hard_df[feat].mean()
        hard_std  = hard_df[feat].std()
        med_mean  = soft_mean + weight * (hard_mean - soft_mean)
        med_std   = soft_std  + weight * (hard_std  - soft_std)
        synth[feat] = np.random.normal(med_mean, med_std * 0.85, n)

    synth['TyreLife'] = np.random.gamma(shape=4, scale=4, size=n).clip(1, 28)
    synth['DegradationDelta'] = (
        0.05 * synth['TyreLife'] + np.random.normal(0, 0.25, n)
    ).clip(0)

    df_synth = pd.DataFrame(synth)
    df_synth['Compound'] = 'MEDIUM'
    return df_synth

df_medium_synth = synthesize_medium(soft_laps, hard_laps, n=200)
print(f"Synthetic MEDIUM laps generated: {len(df_medium_synth)}")

# ── Validation plot ───────────────────────────────────────────────────────────
BG   = '#ffffff'
TEXT = '#111111'
GRID = '#dddddd'

validate_features = ['TyreLife', 'LapTimeSeconds', 'DegradationDelta', 'Sector2Seconds']

fig_val, axes_val = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
fig_val.suptitle(
    'Synthetic MEDIUM Tyre Data Validation\n'
    '(Does it sit plausibly between SOFT and HARD?)',
    color=TEXT, fontsize=13, y=1.01
)
axes_val = axes_val.flatten()

for ax, feat in zip(axes_val, validate_features):
    ax.set_facecolor(BG)
    soft_vals = soft_laps[feat].dropna()
    hard_vals = hard_laps[feat].dropna()
    med_vals  = df_medium_synth[feat].dropna()

    ax.hist(soft_vals, bins=30, alpha=0.4, color='#E8002D',
            density=True, label='SOFT (real)')
    ax.hist(hard_vals, bins=30, alpha=0.4, color='#333333',
            density=True, label='HARD (real)')
    ax.hist(med_vals,  bins=30, alpha=0.5, color='#FFA500',
            density=True, label='MEDIUM (synthetic)', edgecolor='#FFA500')

    ks_soft, _ = ks_2samp(med_vals, soft_vals)
    ks_hard, _ = ks_2samp(med_vals, hard_vals)

    ax.set_title(f'{feat}\nKS vs SOFT: {ks_soft:.2f} · KS vs HARD: {ks_hard:.2f}',
                 color=TEXT, fontsize=9)
    ax.set_xlabel(feat, color=TEXT, fontsize=8)
    ax.set_ylabel('Density', color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.legend(fontsize=7, facecolor=BG, labelcolor=TEXT)
    ax.grid(axis='y', color=GRID, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)

plt.tight_layout()
plt.savefig('outputs/medium_synth_validation.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: outputs/medium_synth_validation.png")

# ── Combine real + synthetic ──────────────────────────────────────────────────
df_real = df_model[features + ['Compound', 'LapNumber', 'Driver', 'Stint']].dropna()
df_full = pd.concat([
    df_real[features + ['Compound']],
    df_medium_synth[features + ['Compound']]
], ignore_index=True)

X_full = df_full[features]
y_full = df_full['Compound']

print(f"\nFull dataset: {df_full['Compound'].value_counts().to_dict()}")

# ── Define 5 models ───────────────────────────────────────────────────────────
compound_colors = {'SOFT': '#E8002D', 'MEDIUM': '#FFA500', 'HARD': '#444444'}

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced',
                                   random_state=42))]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=7))]),
    'Support Vector Machine': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced',
                    random_state=42))]),
}

# ── 5-fold cross-validation ───────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n── Cross-Validation Results ──────────────────────────────────────")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_full, y_full, cv=cv,
                                scoring='accuracy', n_jobs=-1)
    results[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std(),
                     'scores': cv_scores}
    print(f"{name:25s}  Acc: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

best_name  = max(results, key=lambda k: results[k]['mean'])
best_model = models[best_name]
print(f"\nBest model: {best_name} ({results[best_name]['mean']:.3f})")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Main classifier figure ────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18), facecolor=BG)
fig.suptitle(
    '2023 Bahrain GP — Tire Compound Classifier\n'
    '3-Class (SOFT / MEDIUM / HARD) · 5 Models Compared',
    color=TEXT, fontsize=14, y=0.98
)
gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Model leaderboard
ax_lb = fig.add_subplot(gs[0, :2])
ax_lb.set_facecolor(BG)
sorted_models = sorted(results.items(), key=lambda x: x[1]['mean'], reverse=True)
names  = [r[0] for r in sorted_models]
means  = [r[1]['mean'] for r in sorted_models]
stds   = [r[1]['std']  for r in sorted_models]
colors = ['#1E41FF' if n == best_name else '#aaaaaa' for n in names]

bars = ax_lb.barh(names, means, xerr=stds, color=colors,
                  edgecolor=GRID, linewidth=0.5, capsize=4)
for bar, mean, std in zip(bars, means, stds):
    ax_lb.text(bar.get_width() + 0.005,
               bar.get_y() + bar.get_height() / 2,
               f'{mean:.3f} ± {std:.3f}', va='center', color=TEXT, fontsize=9)
ax_lb.set_xlabel('5-Fold CV Accuracy', color=TEXT, fontsize=9)
ax_lb.set_title('Model Leaderboard (5-Fold Cross-Validation)',
                color=TEXT, fontsize=10)
ax_lb.set_xlim(0, 1.05)
ax_lb.tick_params(colors=TEXT)
ax_lb.set_axisbelow(True)
ax_lb.grid(axis='x', color=GRID, linewidth=0.5, linestyle='--')
ax_lb.invert_yaxis()
for spine in ax_lb.spines.values():
    spine.set_color(GRID)

# Panel 2: CV boxplot
ax_box = fig.add_subplot(gs[0, 2])
ax_box.set_facecolor(BG)
box_data = [results[n]['scores'] for n in names]
bp = ax_box.boxplot(box_data, vert=True, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5))
for patch, name in zip(bp['boxes'], names):
    patch.set_facecolor('#1E41FF' if name == best_name else '#cccccc')
    patch.set_alpha(0.7)
ax_box.set_xticks(range(1, len(names) + 1))
ax_box.set_xticklabels([n.replace(' ', '\n') for n in names],
                        fontsize=6.5, color=TEXT)
ax_box.set_ylabel('Accuracy', color=TEXT, fontsize=9)
ax_box.set_title('CV Score Distribution', color=TEXT, fontsize=10)
ax_box.tick_params(colors=TEXT)
ax_box.set_axisbelow(True)
ax_box.grid(axis='y', color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_box.spines.values():
    spine.set_color(GRID)

# Panel 3: Confusion matrix
ax_cm = fig.add_subplot(gs[1, 0])
labels = ['SOFT', 'MEDIUM', 'HARD']
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels, ax=ax_cm, cbar=False)
ax_cm.set_title(f'Confusion Matrix\n({best_name})', color=TEXT, fontsize=10)
ax_cm.set_xlabel('Predicted', color=TEXT)
ax_cm.set_ylabel('Actual', color=TEXT)
ax_cm.tick_params(colors=TEXT)

# Panel 4: Feature importance
ax_fi = fig.add_subplot(gs[1, 1])
ax_fi.set_facecolor(BG)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'named_steps'):
    clf_step = best_model.named_steps.get('clf')
    importances = (clf_step.feature_importances_
                   if hasattr(clf_step, 'feature_importances_')
                   else np.abs(clf_step.coef_).mean(axis=0)
                   if hasattr(clf_step, 'coef_')
                   else np.ones(len(features)) / len(features))
else:
    importances = np.ones(len(features)) / len(features)

feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})\
            .sort_values('Importance', ascending=True)
bars = ax_fi.barh(feat_df['Feature'], feat_df['Importance'],
                  color='#1E41FF', edgecolor=GRID, linewidth=0.5)
for bar, val in zip(bars, feat_df['Importance']):
    ax_fi.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
               f'{val:.3f}', va='center', color=TEXT, fontsize=8)
ax_fi.set_title(f'Feature Importance\n({best_name})', color=TEXT, fontsize=10)
ax_fi.set_xlabel('Importance Score', color=TEXT, fontsize=9)
ax_fi.tick_params(colors=TEXT)
ax_fi.set_axisbelow(True)
ax_fi.grid(axis='x', color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_fi.spines.values():
    spine.set_color(GRID)

# Panel 5: Per-driver accuracy
ax_drv = fig.add_subplot(gs[1, 2])
ax_drv.set_facecolor(BG)
driver_accs = []
for drv in df_real['Driver'].unique():
    drv_data = df_real[df_real['Driver'] == drv].dropna(subset=features)
    if len(drv_data) < 5:
        continue
    drv_pred = best_model.predict(drv_data[features])
    acc = np.mean(drv_pred == drv_data['Compound'])
    driver_accs.append({'Driver': drv, 'Accuracy': acc})

drv_df = pd.DataFrame(driver_accs).sort_values('Accuracy', ascending=True)
ax_drv.barh(drv_df['Driver'], drv_df['Accuracy'],
            color='#1E41FF', edgecolor=GRID, linewidth=0.5, alpha=0.8)
ax_drv.axvline(0.79, color='red', linewidth=1, linestyle='--',
               alpha=0.6, label='Baseline (79%)')
ax_drv.set_xlabel('Prediction Accuracy', color=TEXT, fontsize=9)
ax_drv.set_title('Per-Driver Accuracy\n(best model)', color=TEXT, fontsize=10)
ax_drv.set_xlim(0, 1.1)
ax_drv.tick_params(colors=TEXT, labelsize=7)
ax_drv.legend(fontsize=7, facecolor=BG, labelcolor=TEXT)
ax_drv.set_axisbelow(True)
ax_drv.grid(axis='x', color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_drv.spines.values():
    spine.set_color(GRID)

# Panel 6: VER lap-by-lap prediction
ax_lap = fig.add_subplot(gs[2, :])
ax_lap.set_facecolor(BG)
ver_data = df_real[df_real['Driver'] == 'VER'].dropna(subset=features).copy()
ver_data = ver_data.sort_values('LapNumber')
ver_data['Predicted'] = best_model.predict(ver_data[features])
ver_data['Correct']   = ver_data['Compound'] == ver_data['Predicted']

for _, row in ver_data.iterrows():
    ax_lap.axvspan(row['LapNumber'] - 0.5, row['LapNumber'] + 0.5,
                   color=compound_colors.get(row['Compound'], '#aaaaaa'),
                   alpha=0.15, zorder=0)

correct   = ver_data[ver_data['Correct']]
incorrect = ver_data[~ver_data['Correct']]
ax_lap.scatter(correct['LapNumber'],   correct['LapTimeSeconds'],
               color='green', s=25, zorder=3, label='Correct')
ax_lap.scatter(incorrect['LapNumber'], incorrect['LapTimeSeconds'],
               color='red',   s=40, marker='X', zorder=4, label='Wrong')

ax_lap.set_xlabel('Lap Number', color=TEXT, fontsize=9)
ax_lap.set_ylabel('Lap Time (s)', color=TEXT, fontsize=9)
ax_lap.set_title(
    f'VER — Lap-by-Lap Compound Prediction ({best_name})\nBackground = actual compound',
    color=TEXT, fontsize=10)
ax_lap.tick_params(colors=TEXT)
ax_lap.set_axisbelow(True)
ax_lap.grid(axis='y', color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_lap.spines.values():
    spine.set_color(GRID)

handles = [
    mpatches.Patch(color=compound_colors['SOFT'], alpha=0.4, label='SOFT (actual)'),
    mpatches.Patch(color=compound_colors['HARD'], alpha=0.4, label='HARD (actual)'),
    plt.scatter([], [], color='green', s=25,       label='Correct prediction'),
    plt.scatter([], [], color='red',   s=40, marker='X', label='Wrong prediction'),
]
ax_lap.legend(handles=handles, facecolor=BG, labelcolor=TEXT,
              fontsize=8, loc='upper right')

plt.savefig('outputs/bahrain_tyre_classifier_full.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: outputs/bahrain_tyre_classifier_full.png")

# ════════════════════════════════════════════════════════════════════════════════
# PART 2: CONFIDENCE INTERVALS + STINT LIFE PREDICTOR
# Genuine value-add: what rival teams cannot directly observe
# ════════════════════════════════════════════════════════════════════════════════

# Retrain clf on clean features (no LapNumber)
df_model_clean = df_full[df_full['Compound'] != 'MEDIUM'][
    features + ['Compound']
].dropna()

X_c = df_model_clean[features]
y_c = df_model_clean['Compound']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_c, y_c, test_size=0.2, random_state=42, stratify=y_c)

clf = GradientBoostingClassifier(
    n_estimators=200, max_depth=4,
    learning_rate=0.05, random_state=42)
clf.fit(X_train_c, y_train_c)
print(f"\nclf retrained ✓")

# Build all_laps from original df
all_laps = df[
    features + ['Compound', 'LapNumber', 'Driver', 'Stint', 'PitOutTime']
].dropna(subset=features + ['Compound']).copy()
all_laps = all_laps[all_laps['Compound'].isin(['SOFT', 'HARD'])]

proba   = clf.predict_proba(all_laps[features])
classes = clf.classes_

all_laps['PredictedCompound'] = classes[np.argmax(proba, axis=1)]
all_laps['Confidence']        = np.max(proba, axis=1)
all_laps['Correct']           = all_laps['Compound'] == all_laps['PredictedCompound']

print("\n── Confidence Analysis ───────────────────────────────────────────")
bins_conf   = [0.5, 0.7, 0.85, 1.01]
labels_conf = ['Low (50-70%)', 'Medium (70-85%)', 'High (85-100%)']
all_laps['ConfidenceBin'] = pd.cut(all_laps['Confidence'],
                                    bins=bins_conf, labels=labels_conf)
for bin_label in labels_conf:
    mask  = all_laps['ConfidenceBin'] == bin_label
    acc   = all_laps[mask]['Correct'].mean()
    count = mask.sum()
    print(f"{bin_label}: {count} laps, accuracy={acc:.3f}")

# Stint life predictor
stint_features = ['TyreLife', 'DegradationDelta', 'LapTimeDiff', 'LapTimeSeconds']
stints = []
for (driver, stint), group in all_laps.groupby(['Driver', 'Stint']):
    group = group.sort_values('LapNumber')
    if len(group) < 3:
        continue
    compound      = group['Compound'].iloc[0]
    total_laps    = len(group)
    min_tyre_life = group['TyreLife'].min()
    for _, row in group.iterrows():
        remaining = total_laps - (row['TyreLife'] - min_tyre_life)
        stints.append({
            'Driver':           driver,
            'Stint':            stint,
            'Compound':         compound,
            'TyreLife':         row['TyreLife'],
            'LapTimeSeconds':   row['LapTimeSeconds'],
            'DegradationDelta': row['DegradationDelta'],
            'LapTimeDiff':      row['LapTimeDiff'],
            'LapNumber':        row['LapNumber'],
            'PitOutTime':       row['PitOutTime'],
            'StintTotalLaps':   total_laps,
            'RemainingLaps':    max(0, remaining),
        })

stint_df = pd.DataFrame(stints).dropna(subset=stint_features)

stint_models = {}
print("\n── Stint Life Predictor ──────────────────────────────────────────")
for compound in ['SOFT', 'HARD']:
    mask = stint_df['Compound'] == compound
    data = stint_df[mask].dropna(subset=stint_features + ['RemainingLaps'])
    if len(data) < 20:
        continue
    X_s = data[stint_features]
    y_s = data['RemainingLaps']
    reg = GradientBoostingRegressor(
        n_estimators=200, max_depth=3,
        learning_rate=0.05, random_state=42)
    cv_mae = -cross_val_score(reg, X_s, y_s, cv=5,
                               scoring='neg_mean_absolute_error')
    reg.fit(X_s, y_s)
    stint_models[compound] = reg
    print(f"{compound}: CV MAE = {cv_mae.mean():.2f} ± {cv_mae.std():.2f} laps")

# ── Confidence + Stint Life Visualization ─────────────────────────────────────
fig2 = plt.figure(figsize=(22, 18), facecolor=BG)
fig2.suptitle(
    '2023 Bahrain GP — Tire Classifier: Confidence Intervals + Stint Life Predictor\n'
    'Genuine value-add: what rivals cannot directly observe',
    color=TEXT, fontsize=13, y=0.98
)
gs2 = GridSpec(3, 3, figure=fig2, hspace=0.45, wspace=0.35)

# Panel 1: Confidence distribution
ax_conf = fig2.add_subplot(gs2[0, 0])
ax_conf.set_facecolor(BG)
for compound in ['SOFT', 'HARD']:
    mask = all_laps['Compound'] == compound
    ax_conf.hist(all_laps[mask]['Confidence'], bins=20,
                 alpha=0.5, color=compound_colors[compound],
                 label=compound, density=True)
ax_conf.axvline(0.70, color='orange', linewidth=1.5, linestyle='--',
                alpha=0.7, label='Low/Med threshold')
ax_conf.axvline(0.85, color='green',  linewidth=1.5, linestyle='--',
                alpha=0.7, label='Med/High threshold')
ax_conf.set_xlabel('Prediction Confidence', color=TEXT, fontsize=9)
ax_conf.set_ylabel('Density', color=TEXT, fontsize=9)
ax_conf.set_title('Prediction Confidence Distribution\nby Actual Compound',
                  color=TEXT, fontsize=10)
ax_conf.tick_params(colors=TEXT)
ax_conf.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
ax_conf.set_axisbelow(True)
ax_conf.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_conf.spines.values():
    spine.set_color(GRID)

# Panel 2: Calibration plot
ax_acc = fig2.add_subplot(gs2[0, 1])
ax_acc.set_facecolor(BG)
conf_bins_range = np.linspace(0.5, 1.0, 11)
bin_accs, bin_mids = [], []
for i in range(len(conf_bins_range) - 1):
    mask = ((all_laps['Confidence'] >= conf_bins_range[i]) &
             (all_laps['Confidence'] <  conf_bins_range[i+1]))
    if mask.sum() > 5:
        bin_accs.append(all_laps[mask]['Correct'].mean())
        bin_mids.append((conf_bins_range[i] + conf_bins_range[i+1]) / 2)
ax_acc.bar(bin_mids, bin_accs, width=0.045,
           color='#1E41FF', edgecolor=GRID, linewidth=0.5, alpha=0.8)
ax_acc.plot([0.5, 1.0], [0.5, 1.0], color='grey',
            linewidth=1, linestyle='--', alpha=0.5, label='Perfect calibration')
ax_acc.set_xlabel('Model Confidence', color=TEXT, fontsize=9)
ax_acc.set_ylabel('Actual Accuracy', color=TEXT, fontsize=9)
ax_acc.set_title('Calibration Plot\nDoes confidence = actual accuracy?',
                 color=TEXT, fontsize=10)
ax_acc.set_xlim(0.5, 1.05)
ax_acc.set_ylim(0, 1.1)
ax_acc.tick_params(colors=TEXT)
ax_acc.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
ax_acc.set_axisbelow(True)
ax_acc.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_acc.spines.values():
    spine.set_color(GRID)

# Panel 3: Where model struggles
ax_low = fig2.add_subplot(gs2[0, 2])
ax_low.set_facecolor(BG)
low_conf  = all_laps[all_laps['Confidence'] <  0.70]
high_conf = all_laps[all_laps['Confidence'] >= 0.85]
ax_low.scatter(low_conf['TyreLife'],  low_conf['DegradationDelta'],
               color='red',   alpha=0.5, s=20, label='Low confidence (<70%)')
ax_low.scatter(high_conf['TyreLife'], high_conf['DegradationDelta'],
               color='green', alpha=0.3, s=10, label='High confidence (>85%)')
ax_low.set_xlabel('Tyre Life (laps)', color=TEXT, fontsize=9)
ax_low.set_ylabel('Degradation Delta (s)', color=TEXT, fontsize=9)
ax_low.set_title('Where Does Model Struggle?\nLow vs High Confidence Laps',
                 color=TEXT, fontsize=10)
ax_low.tick_params(colors=TEXT)
ax_low.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
ax_low.set_axisbelow(True)
ax_low.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_low.spines.values():
    spine.set_color(GRID)

# Panel 4: SOFT stint life
ax_soft = fig2.add_subplot(gs2[1, 0])
ax_soft.set_facecolor(BG)
if 'SOFT' in stint_models:
    soft_data = stint_df[stint_df['Compound'] == 'SOFT'].copy()
    soft_data['PredictedRemaining'] = stint_models['SOFT']\
        .predict(soft_data[stint_features])
    max_val = max(soft_data['RemainingLaps'].max(),
                  soft_data['PredictedRemaining'].max())
    ax_soft.scatter(soft_data['RemainingLaps'], soft_data['PredictedRemaining'],
                    color=compound_colors['SOFT'], alpha=0.4, s=15)
    ax_soft.plot([0, max_val], [0, max_val], color='grey',
                 linewidth=1.5, linestyle='--', label='Perfect prediction')
    mae_soft = mean_absolute_error(soft_data['RemainingLaps'],
                                   soft_data['PredictedRemaining'])
    ax_soft.set_title(f'SOFT Stint Life Predictor\nMAE = {mae_soft:.1f} laps',
                      color=TEXT, fontsize=10)
    ax_soft.set_xlabel('Actual Remaining Laps', color=TEXT, fontsize=9)
    ax_soft.set_ylabel('Predicted Remaining Laps', color=TEXT, fontsize=9)
    ax_soft.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
    ax_soft.tick_params(colors=TEXT)
    ax_soft.set_axisbelow(True)
    ax_soft.grid(color=GRID, linewidth=0.5, linestyle='--')
    for spine in ax_soft.spines.values():
        spine.set_color(GRID)

# Panel 5: HARD stint life
ax_hard = fig2.add_subplot(gs2[1, 1])
ax_hard.set_facecolor(BG)
if 'HARD' in stint_models:
    hard_data = stint_df[stint_df['Compound'] == 'HARD'].copy()
    hard_data['PredictedRemaining'] = stint_models['HARD']\
        .predict(hard_data[stint_features])
    max_val = max(hard_data['RemainingLaps'].max(),
                  hard_data['PredictedRemaining'].max())
    ax_hard.scatter(hard_data['RemainingLaps'], hard_data['PredictedRemaining'],
                    color=compound_colors['HARD'], alpha=0.4, s=15)
    ax_hard.plot([0, max_val], [0, max_val], color='grey',
                 linewidth=1.5, linestyle='--', label='Perfect prediction')
    mae_hard = mean_absolute_error(hard_data['RemainingLaps'],
                                   hard_data['PredictedRemaining'])
    ax_hard.set_title(f'HARD Stint Life Predictor\nMAE = {mae_hard:.1f} laps',
                      color=TEXT, fontsize=10)
    ax_hard.set_xlabel('Actual Remaining Laps', color=TEXT, fontsize=9)
    ax_hard.set_ylabel('Predicted Remaining Laps', color=TEXT, fontsize=9)
    ax_hard.legend(facecolor=BG, labelcolor=TEXT, fontsize=8)
    ax_hard.tick_params(colors=TEXT)
    ax_hard.set_axisbelow(True)
    ax_hard.grid(color=GRID, linewidth=0.5, linestyle='--')
    for spine in ax_hard.spines.values():
        spine.set_color(GRID)

# Panel 6: VER real-time dashboard
ax_dash = fig2.add_subplot(gs2[1, 2])
ax_dash.set_facecolor(BG)
ver_data = all_laps[all_laps['Driver'] == 'VER'].sort_values('LapNumber').copy()
ver_proba = clf.predict_proba(ver_data[features])
ver_data['PredictedCompound'] = classes[np.argmax(ver_proba, axis=1)]
ver_data['Confidence']        = np.max(ver_proba, axis=1)

ver_remaining = []
for _, row in ver_data.iterrows():
    compound = row['PredictedCompound']
    if compound in stint_models:
        rem = stint_models[compound].predict(
            pd.DataFrame([row[stint_features]]))[0]
        ver_remaining.append(max(0, rem))
    else:
        ver_remaining.append(np.nan)
ver_data['PredictedRemaining'] = ver_remaining

ax_dash.fill_between(ver_data['LapNumber'], ver_data['Confidence'],
                      alpha=0.2, color='#1E41FF')
ax_dash.plot(ver_data['LapNumber'], ver_data['Confidence'],
             color='#1E41FF', linewidth=1.5, label='Prediction confidence')
ax_dash.axhline(0.85, color='green',  linewidth=1, linestyle='--',
                alpha=0.7, label='High (>85%)')
ax_dash.axhline(0.70, color='orange', linewidth=1, linestyle='--',
                alpha=0.7, label='Low (<70%)')
for _, row in ver_data.iterrows():
    ax_dash.axvspan(row['LapNumber'] - 0.5, row['LapNumber'] + 0.5,
                    color=compound_colors.get(row['PredictedCompound'], '#aaaaaa'),
                    alpha=0.08, zorder=0)
ax_dash.set_xlabel('Lap Number', color=TEXT, fontsize=9)
ax_dash.set_ylabel('Prediction Confidence', color=TEXT, fontsize=9)
ax_dash.set_title('VER — Real-time Strategy Dashboard\n'
                  'Background = predicted compound · Line = confidence',
                  color=TEXT, fontsize=10)
ax_dash.set_ylim(0.4, 1.1)
ax_dash.tick_params(colors=TEXT)
ax_dash.legend(facecolor=BG, labelcolor=TEXT, fontsize=7, loc='lower right')
ax_dash.set_axisbelow(True)
ax_dash.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_dash.spines.values():
    spine.set_color(GRID)

# Panel 7: VER remaining laps
ax_rem = fig2.add_subplot(gs2[2, :2])
ax_rem.set_facecolor(BG)
valid_rem = ver_data[ver_data['PredictedRemaining'].notna()]
ax_rem.plot(valid_rem['LapNumber'], valid_rem['PredictedRemaining'],
            color='#1E41FF', linewidth=2, label='Predicted remaining laps')

ver_actual_rem = []
for _, row in ver_data.iterrows():
    stint_laps = ver_data[ver_data['Stint'] == row['Stint']]
    actual_rem = stint_laps['LapNumber'].max() - row['LapNumber']
    ver_actual_rem.append(max(0, actual_rem))
ver_data['ActualRemaining'] = ver_actual_rem

ax_rem.plot(ver_data['LapNumber'], ver_data['ActualRemaining'],
            color='grey', linewidth=1.5, linestyle='--',
            label='Actual remaining laps')

for compound, mae_val in [('SOFT', mae_soft if 'SOFT' in stint_models else 3.5),
                           ('HARD', mae_hard if 'HARD' in stint_models else 5.0)]:
    mask = valid_rem['PredictedCompound'] == compound
    ax_rem.fill_between(
        valid_rem[mask]['LapNumber'],
        (valid_rem[mask]['PredictedRemaining'] - mae_val).clip(0),
        valid_rem[mask]['PredictedRemaining'] + mae_val,
        alpha=0.1, color=compound_colors[compound],
        label=f'{compound} ±{mae_val:.0f} lap uncertainty'
    )

for lap in ver_data[ver_data['PitOutTime'].notna()]['LapNumber']:
    ax_rem.axvline(lap, color='grey', linewidth=1, linestyle=':', alpha=0.7)
    ax_rem.text(lap + 0.3, 25, 'PIT', color='grey', fontsize=7)

ax_rem.set_xlabel('Lap Number', color=TEXT, fontsize=9)
ax_rem.set_ylabel('Remaining Laps in Stint', color=TEXT, fontsize=9)
ax_rem.set_title('VER — Stint Life Predictor\n'
                 'Shaded = uncertainty band · This is what rivals cannot see',
                 color=TEXT, fontsize=10)
ax_rem.tick_params(colors=TEXT)
ax_rem.legend(facecolor=BG, labelcolor=TEXT, fontsize=8, loc='upper right')
ax_rem.set_axisbelow(True)
ax_rem.grid(color=GRID, linewidth=0.5, linestyle='--')
for spine in ax_rem.spines.values():
    spine.set_color(GRID)

# Panel 8: Strategic value summary
ax_sum = fig2.add_subplot(gs2[2, 2])
ax_sum.axis('off')
ax_sum.set_facecolor(BG)
summary_text = """
STRATEGIC VALUE ANALYSIS

What rivals CANNOT directly observe:
✓ Competitor tire compound
✓ Competitor tire age
✓ Laps remaining in stint
✓ Confidence in assessment

How this helps strategy engineers:

1. UNDERCUT TIMING
   If rival has <5 laps remaining
   (high confidence), pit now to
   undercut before they react.

2. OVERCUT OPPORTUNITY
   If rival pits unexpectedly early
   (low confidence = ambiguous tire),
   stay out to overcut.

3. SAFETY CAR WINDOW
   Know which rivals benefit most
   from SC pit window based on
   predicted remaining stint life.

Limitation: trained on one race.
Generalisation requires multi-race
training data.
"""
ax_sum.text(0.05, 0.95, summary_text,
            transform=ax_sum.transAxes,
            fontsize=8, color=TEXT, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5',
                      edgecolor=GRID, alpha=0.8))

plt.savefig('outputs/bahrain_tyre_classifier_confidence.png',
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Saved: outputs/bahrain_tyre_classifier_confidence.png")
