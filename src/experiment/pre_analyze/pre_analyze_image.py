"""画像matchingを1セッションずつ解析し、CI・検定なしの6図と左右差MSD CSVを保存する。
cd src && py -m experiment.pre_analyze.pre_analyze_image [session_dir]
省略時はimage_matching.csvを含む最新セッション。旧rating CSVは対象外。
値は物理輝度ではなく、確定した前景ゲイン（スライダー値0〜1）。
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LAB_ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = LAB_ROOT / 'results/tables/pre-experiment-image-matching'
FIGURE_ROOT = LAB_ROOT / 'results/figures/pre-experiment-image-matching'
CONDITIONS = {
    'DP_BINOCULAR': 'DP binocular',
    'DP_MONO_DOM': 'DP dominant',
    'DP_MONO_NONDOM': 'DP non-dominant',
    'SP_NO_DEFOCUS': 'SP no defocus',
    'SP_DEFOCUS_SIMPLE': 'SP fused',
    'SP_REPRO_LEFT_EYE': 'SP reproduce Left',
    'SP_REPRO_RIGHT_EYE': 'SP reproduce Right',
}
DP_EYES = ('DP_MONO_DOM', 'DP_MONO_NONDOM')
PAIR_COLUMNS = ['Pair_ID', 'Difference_Group', 'Group_Rank']


def resolve_target(argument=None):
    if argument:
        path = Path(argument).expanduser().resolve()
        if path.is_file():
            if path.name != 'image_matching.csv':
                raise ValueError('Specify image_matching.csv, not rating/calibration CSV')
            path = path.parent
    else:
        candidates = list(RESULT_ROOT.glob('*/image_matching.csv'))
        if not candidates:
            raise FileNotFoundError(f'No matching sessions: {RESULT_ROOT}')
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))).parent
    if not (path / 'image_matching.csv').is_file():
        raise FileNotFoundError(path / 'image_matching.csv')
    return path


def load_results(target):
    frame = pd.read_csv(target / 'image_matching.csv', encoding='utf-8-sig', dtype=str)
    required = ['ID', 'Condition', *PAIR_COLUMNS, 'Dominance', 'Match_Status', 'Matched_Gain']
    missing = set(required) - set(frame.columns)
    if frame.empty or missing:
        raise ValueError(f'Empty matching data or missing columns: {sorted(missing)}')
    for column in required:
        if column != 'Matched_Gain':
            frame[column] = frame[column].fillna('').str.strip()
            if frame[column].eq('').any():
                raise ValueError(f'Blank required value: {column}')
    if frame['ID'].nunique() != 1:
        raise ValueError('Analyze one participant/session at a time')
    frame['Dominance'] = frame['Dominance'].str.capitalize()
    if frame['Dominance'].nunique() != 1 or not frame['Dominance'].isin(['Left', 'Right']).all():
        raise ValueError('Dominance must be one consistent Left/Right value')
    if not frame['Condition'].isin(CONDITIONS).all():
        raise ValueError('Unknown matching condition')
    if not set(DP_EYES).issubset(set(frame['Condition'])):
        raise ValueError('Both DP_MONO_DOM and DP_MONO_NONDOM are required for the 3 figures')
    allowed = ['matched', 'above_range', 'below_range', 'cannot_judge']
    if not frame['Match_Status'].isin(allowed).all():
        raise ValueError('Unknown/incomplete Match_Status')
    if frame.duplicated(['Condition', 'Pair_ID']).any():
        raise ValueError('Duplicate Condition/Pair_ID')
    frame['Difference_Group'] = frame['Difference_Group'].str.lower()
    frame['Group_Rank'] = pd.to_numeric(frame['Group_Rank'], errors='raise')
    if not frame['Group_Rank'].isin(range(1, 11)).all():
        raise ValueError('Group_Rank must be integer 1..10')
    metadata = frame[PAIR_COLUMNS].drop_duplicates()
    if metadata['Pair_ID'].duplicated().any():
        raise ValueError('Pair metadata differs between conditions')
    if len(metadata) != 20 or set(metadata['Difference_Group']) != {'high', 'low'}:
        raise ValueError('Exactly high 10 + low 10 image pairs are required')
    for _, group in metadata.groupby('Difference_Group'):
        if len(group) != 10 or set(group['Group_Rank']) != set(range(1, 11)):
            raise ValueError('Each group requires ranks 1..10')
    for condition, group in frame.groupby('Condition'):
        if set(group['Pair_ID']) != set(metadata['Pair_ID']):
            raise ValueError(f'Incomplete image pairs: {condition}')
    frame['Gain'] = pd.to_numeric(frame['Matched_Gain'], errors='coerce')
    matched = frame['Match_Status'].eq('matched')
    values = frame.loc[matched, 'Gain']
    if not np.isfinite(values).all() or not values.between(0, 1).all():
        raise ValueError('Matched_Gain must be finite and in [0,1] for matched trials')
    # 範囲外・判断不能は一致値として平均しない。Final_Gainへの代替もしない。
    frame.loc[~matched, 'Gain'] = np.nan
    print('Response counts:\n', pd.crosstab(frame['Condition'], frame['Match_Status']))
    metadata = metadata.assign(_order=metadata['Difference_Group'].map({'high': 0, 'low': 1}))
    metadata = metadata.sort_values(['_order', 'Group_Rank']).drop(columns='_order')
    return frame, metadata


def build_dp_pairs(frame, metadata):
    dp = frame.loc[frame['Condition'].isin(DP_EYES)].copy()
    dominant = frame['Dominance'].iloc[0]
    other = 'Left' if dominant == 'Right' else 'Right'
    dp['Eye'] = np.where(dp['Condition'].eq('DP_MONO_DOM'), dominant, other)
    # 実眼情報があるCSVでは、利き眼から導いた左右対応と照合する。
    if 'Observed_Eye' in dp:
        if not dp['Observed_Eye'].str.strip().str.capitalize().eq(dp['Eye']).all():
            raise ValueError('Observed_Eye disagrees with Condition/Dominance')
    pivot = dp.pivot(index='Pair_ID', columns='Eye', values='Gain')
    paired = metadata.merge(pivot.reindex(columns=['Left', 'Right']),
                            left_on='Pair_ID', right_index=True, how='left', validate='one_to_one')
    paired['Difference'] = paired['Left'] - paired['Right']
    return paired


def save_dp_msd_csv(frame, paired, target):
    """high/low別の平均二乗差mean((Left-Right)**2)。平方根は取らない。"""
    rows = []
    for group_name in ('high', 'low'):
        group = paired.loc[paired['Difference_Group'].eq(group_name)]
        valid = np.isfinite(group[['Left', 'Right']].to_numpy(dtype=float)).all(axis=1)
        differences = (
            group.loc[valid, 'Left'] - group.loc[valid, 'Right']
        ).to_numpy(dtype=float)
        n = len(differences)
        rows.append({
            'ID': frame['ID'].iloc[0],
            'Session': target.name,
            'Difference_Group': group_name,
            'Total_Pairs': len(group),
            'Valid_Pairs': n,
            'Excluded_Pairs': len(group) - n,
            'Mean_Squared_Difference': float(np.mean(differences ** 2)) if n else np.nan,
            'Calculation_Status': 'ok' if n else 'no_valid_pairs',
            'Metric_Definition': 'mean((Left_Gain - Right_Gain)**2)',
            'Unit': 'normalized_gain_squared',
        })
    output = target / 'matching_analysis' / 'dp_eye_mean_squared_difference.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8-sig', na_rep='')
    print(f'Saved high/low mean squared differences: {output}')
    return output


def save_figures(frame, paired, output):
    output.mkdir(parents=True, exist_ok=True)
    identity = f"ID={frame['ID'].iloc[0]} / {len(paired)} image pairs"
    def save(fig, name):
        fig.tight_layout()
        try:
            fig.savefig(output / name, dpi=300, bbox_inches='tight')
        finally:
            plt.close(fig)

    order = [c for c in CONDITIONS if c in set(frame['Condition'])]
    for group, subset, total, filename in (
        ('All', frame, 20, '01_condition_mean_gain.png'),
        ('High', frame.loc[frame['Difference_Group'].eq('high')], 10,
         '01_condition_mean_gain_high.png'),
        ('Low', frame.loc[frame['Difference_Group'].eq('low')], 10,
         '01_condition_mean_gain_low.png'),
    ):
        summary = subset.groupby('Condition')['Gain'].agg(['mean', 'count']).reindex(order)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(order))
        ax.bar(x, summary['mean'], color='steelblue')  # 信頼区間・その他の誤差棒なし。
        for i, row in enumerate(summary.itertuples()):
            text = (f'{row.mean:.3f}\nn={row.count}/{total}' if row.count
                    else f'no matched\nn=0/{total}')
            ax.text(i, row.mean if row.count else 0, text,
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x, [CONDITIONS[c] for c in order], rotation=20, ha='right')
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_ylabel('Mean matched foreground gain (slider 0-1)')
        ax.set_title(f'Condition means / {group} (matched responses only)\n'
                     f"ID={frame['ID'].iloc[0]} / {total} image pairs")
        save(fig, filename)

    valid = paired[['Left', 'Right']].notna().all(axis=1)
    n = int(valid.sum())
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = paired['Difference_Group'].map({'high': 'tab:red', 'low': 'tab:blue'})
    ax.bar(np.arange(len(paired)), paired['Difference'], color=colors)
    for i in np.flatnonzero(~valid.to_numpy()):
        ax.text(i, 0, 'NA', ha='center', va='bottom', color='gray', fontsize=8)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(9.5, color='gray', linestyle=':')
    ax.set_xticks(np.arange(len(paired)), paired['Pair_ID'], rotation=60, ha='right')
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('DP matched gain: Left - Right')
    ax.set_xlabel('Image pair (red=high, blue=low; NA=unmatched)')
    ax.set_title(f'DP eye differences / valid pairs={n}/20\n{identity}')
    save(fig, '02_dp_left_minus_right_gain.png')

    fig, ax = plt.subplots(figsize=(7, 7))
    for group, color in [('high', 'tab:red'), ('low', 'tab:blue')]:
        subset = paired.loc[valid & paired['Difference_Group'].eq(group)]
        ax.scatter(subset['Left'], subset['Right'], color=color, label=group)
        for row in subset.itertuples():
            ax.annotate(str(int(row.Group_Rank)), (row.Left, row.Right),
                        xytext=(4, 4), textcoords='offset points', fontsize=8)
    ax.plot([0, 1], [0, 1], '--', color='black', label='Left = Right')
    ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03),
           xlabel='DP Left-eye matched gain (slider 0-1)',
           ylabel='DP Right-eye matched gain (slider 0-1)',
           title=f'DP eye comparison / valid pairs={n}/20\n{identity}')
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    save(fig, '03_dp_left_vs_right_gain.png')

    # 集約しない生データ。現在の5条件なら横20×縦5の正方形セル。
    pair_order = paired['Pair_ID'].tolist()  # high_01..10, low_01..10
    grid = frame.pivot(index='Condition', columns='Pair_ID', values='Gain')
    grid = grid.reindex(index=order, columns=pair_order)
    statuses = frame.pivot(index='Condition', columns='Pair_ID', values='Match_Status')
    statuses = statuses.reindex(index=order, columns=pair_order)
    values = grid.to_numpy(dtype=float)
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('lightgray')
    fig, ax = plt.subplots(figsize=(16, max(5, len(order) * 0.65 + 2)))
    image = ax.imshow(np.ma.masked_invalid(values), vmin=0, vmax=1,
                      cmap=cmap, aspect='equal', interpolation='nearest')
    ax.set_xticks(np.arange(len(pair_order)), pair_order, rotation=60, ha='right')
    ax.set_yticks(np.arange(len(order)), [CONDITIONS[c] for c in order])
    ax.set_xticks(np.arange(-0.5, len(pair_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.axvline(9.5, color='black', linewidth=2)
    status_labels = {'above_range': 'Above', 'below_range': 'Below',
                     'cannot_judge': '?'}
    for row in range(len(order)):
        for col in range(len(pair_order)):
            value = values[row, col]
            if np.isfinite(value):
                label = f'{value:.3f}'
                # 背景色の明るさに合わせて数値の文字色を切り替える。
                red, green, blue, _ = cmap(value)
                lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = 'black' if lightness > 0.5 else 'white'
            else:
                label = status_labels.get(statuses.iat[row, col], 'NA')
                color = 'black'
            ax.text(col, row, label, ha='center', va='center', fontsize=7, color=color)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02,
                 ticks=np.linspace(0, 1, 6), label='Matched foreground gain (slider 0-1)')
    ax.set_xlabel('Image pair: high 10 | low 10')
    ax.set_ylabel('Condition')
    ax.set_title(f'Raw matching gains / {identity}\n'
                 'Gray: Above=above range, Below=below range, ?=cannot judge, NA=missing')
    save(fig, '04_raw_gain_heatmap.png')
    print(f'Saved 6 figures (no CI, no tests): {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_dir', nargs='?', help='Session directory or image_matching.csv')
    args = parser.parse_args()
    target = resolve_target(args.target_dir)
    frame, metadata = load_results(target)
    paired = build_dp_pairs(frame, metadata)
    save_dp_msd_csv(frame, paired, target)
    save_figures(frame, paired, FIGURE_ROOT / target.name / 'matching_analysis')


if __name__ == '__main__':
    main()
