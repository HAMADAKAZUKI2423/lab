"""画像matchingを1セッションずつ解析し、必要条件を満たす図と左右差MSD CSVを保存する。
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
    'SP_ALPHA_BLEND': 'SP alpha blend',
    'SP_NO_DEFOCUS': 'SP no defocus',
    'SP_DEFOCUS_SIMPLE': 'SP fused',
    'SP_REPRO_LEFT_EYE': 'SP reproduce Left (monocular)',
    'SP_REPRO_RIGHT_EYE': 'SP reproduce Right (monocular)',
    'SP_MODEL_MEAN': 'SP model1 mean',
    'SP_MODEL_WTA': 'SP model2 WTA',
    'SP_MODEL_GAIN': 'SP model3 gain',
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


def load_results(target, direction='all', control='gain'):
    frame = pd.read_csv(target / 'image_matching.csv', encoding='utf-8-sig', dtype=str)
    alpha_rows=frame['Condition'].eq('SP_ALPHA_BLEND')
    frame=frame.loc[alpha_rows if control=='alpha' else ~alpha_rows].copy()
    value_column='Matched_Alpha' if control=='alpha' else 'Matched_Gain'
    if control=='alpha':
        frame['Matched_Gain']=frame[value_column]  # 内部作業列のみ。gain結果へは混在させない。
    keys = ['Condition', 'Pair_ID']
    if direction != 'all':
        if 'Matching_Direction' not in frame:
            raise ValueError('Matching_Direction column is required')
        frame['Matching_Direction'] = frame['Matching_Direction'].fillna('').str.strip().str.lower()
        if direction in {'up', 'down'}:
            frame = frame.loc[frame['Matching_Direction'].eq(direction)].copy()
        elif direction == 'mean':
            keys = keys + ['Matching_Direction']
        else:
            raise ValueError(f'Unknown direction: {direction}')
    required = ['ID', 'Condition', *PAIR_COLUMNS, 'Dominance', 'Observed_Eye',
                'Match_Status', 'Matched_Gain']
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
    dom = frame['Dominance'].iloc[0]
    oth = 'Left' if dom == 'Right' else 'Right'
    frame['Observed_Eye'] = frame['Observed_Eye'].fillna('').str.strip().str.capitalize()
    cond = frame['Condition']
    expected_eye = np.select(
        [
            cond.eq('DP_MONO_DOM'),
            cond.eq('DP_MONO_NONDOM'),
            cond.eq('SP_REPRO_LEFT_EYE'),
            cond.eq('SP_REPRO_RIGHT_EYE'),
        ],
        [dom, oth, 'Left', 'Right'],
        'Both',
    )
    invalid_eye = frame['Observed_Eye'].ne(
        pd.Series(expected_eye, index=frame.index)
    )
    if invalid_eye.any():
        details = frame.loc[
            invalid_eye, ['Condition', 'Observed_Eye']
        ].drop_duplicates().to_dict('records')
        raise ValueError(f'Observed_Eye does not match condition: {details}')
    allowed = ['matched', 'above_range', 'below_range', 'cannot_judge']
    if not frame['Match_Status'].isin(allowed).all():
        raise ValueError('Unknown/incomplete Match_Status')
    if 'Repetition' in frame:
        frame['Repetition']=pd.to_numeric(frame['Repetition'],errors='raise')
        if not (frame['Repetition'].ge(1) & frame['Repetition'].mod(1).eq(0)).all():
            raise ValueError('Invalid repetition index')
        keys=keys+['Repetition']
    if 'Mirrored' in frame:
        keys=keys+['Mirrored']  # 反転試行は同一画像組として後段で平均する。
    if frame.duplicated(keys).any():
        raise ValueError('Duplicate Condition/Pair_ID')
    frame['Difference_Group'] = frame['Difference_Group'].str.lower()
    frame['Group_Rank'] = pd.to_numeric(frame['Group_Rank'], errors='raise')
    frame['Group_Rank'] = frame['Group_Rank'].astype(int)
    metadata = frame[PAIR_COLUMNS].drop_duplicates()
    if metadata['Pair_ID'].duplicated().any():
        raise ValueError('Pair metadata differs between conditions')
    pair_count=len(metadata); per_group=pair_count//2
    if pair_count not in {10,18,20} or set(metadata['Difference_Group']) != {'high','low'}:
        raise ValueError('high/low 5+5 pairs required; legacy 9+9 and 10+10 also supported')
    for _, group in metadata.groupby('Difference_Group'):
        if len(group)!=per_group or set(group['Group_Rank'])!=set(range(1,per_group+1)):
            raise ValueError('Invalid group ranks')
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
    # 左右差解析用: 単眼条件の検証済み観察眼を、反転時に入れ替えた「画像内容の眼」。
    monocular = cond.isin(
        ('DP_MONO_DOM', 'DP_MONO_NONDOM',
         'SP_REPRO_LEFT_EYE', 'SP_REPRO_RIGHT_EYE')
    )
    eye = frame['Observed_Eye'].where(monocular, '')
    mirrored = (frame['Mirrored'].astype(str).str.strip().str.lower().eq('true') if 'Mirrored' in frame
                else pd.Series(False, index=frame.index))
    flip = {'Left': 'Right', 'Right': 'Left'}
    frame['Content_Eye'] = [flip[e] if m and e else e for e, m in zip(eye, mirrored)]
    eyed = frame.loc[frame['Content_Eye'].ne('')]
    # 条件内で反復・反転を平均し、各内容眼で条件を等重みに平均。
    if eyed.empty:
        content = pd.DataFrame(index=pd.Index([], name='Pair_ID'))
    else:
        content = (eyed.groupby(['Pair_ID', 'Condition', 'Content_Eye'])['Gain'].mean()
                   .groupby(['Pair_ID', 'Content_Eye']).mean().unstack())
    if 'Repetition' in frame:
        grouping=['ID','Condition','Pair_ID']
        if direction!='all': grouping+=['Matching_Direction']
        # 有効な一致回答のみを画像組単位で平均。全反復無効ならNaN。
        summary=frame.groupby(grouping,sort=False,dropna=False)['Gain'].agg(['mean','count'])
        frame=frame.drop_duplicates(grouping).set_index(grouping)
        frame['Gain']=summary['mean'];frame['Valid_Repetitions']=summary['count']
        frame['Match_Status']=np.where(frame['Gain'].notna(),'matched','cannot_judge')
        frame=frame.reset_index()
    print(f'[{direction}] Response counts:\n', pd.crosstab(frame['Condition'], frame['Match_Status']))
    if direction == 'mean':
        if not frame['Matching_Direction'].isin(['up', 'down']).all():
            raise ValueError('Matching_Direction must be up/down')
        counts = frame.groupby(['Condition', 'Pair_ID'])['Matching_Direction'].nunique()
        if not counts.eq(2).all():
            raise ValueError('Both up and down trials are required for mean')
        # up/down両方がmatchedの組だけ算術平均。片方欠損はNaN。
        gains = frame.groupby(['Condition', 'Pair_ID'])['Gain'].agg(lambda s: s.mean() if s.notna().all() else np.nan)
        frame = frame.drop_duplicates(['Condition', 'Pair_ID']).set_index(['Condition', 'Pair_ID'])
        frame['Gain'] = gains
        frame['Match_Status'] = np.where(frame['Gain'].notna(), 'matched', 'mean_excluded')
        frame = frame.reset_index()
    frame.attrs['content_gain'] = content
    metadata = metadata.assign(_order=metadata['Difference_Group'].map({'high': 0, 'low': 1}))
    metadata = metadata.sort_values(['_order', 'Group_Rank']).drop(columns='_order')
    return frame, metadata


def build_dp_pairs(frame, metadata):
    # 提示・観察条件をまたぎ、左目基準/右目基準の重畳画像を見たときのゲインを比較(反転考慮)。
    content = frame.attrs['content_gain'].reindex(columns=['Left', 'Right'])
    paired = metadata.merge(content,
                            left_on='Pair_ID', right_index=True, how='left', validate='one_to_one')
    paired['Difference'] = paired['Left'] - paired['Right']
    return paired


def save_dp_msd_csv(frame, paired, target, subdir='matching_analysis'):
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
            'Metric_Definition': 'mean((LeftEyeImage_Gain - RightEyeImage_Gain)**2); content eye, mirror-aware, pooled over conditions',
            'Unit': 'normalized_gain_squared',
        })
    output = target / subdir / 'dp_eye_mean_squared_difference.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding='utf-8-sig', na_rep='')
    print(f'Saved high/low mean squared differences: {output}')
    return output


def save_figures(frame, paired, output):
    output.mkdir(parents=True, exist_ok=True)
    eye_analysis = paired is not None
    eye_filenames = {
        '02_dp_left_minus_right_gain.png',
        '03_dp_left_vs_right_gain.png',
    }
    if not eye_analysis:
        paired = frame[PAIR_COLUMNS].drop_duplicates().copy()
        paired['_order'] = paired['Difference_Group'].map({'high': 0, 'low': 1})
        paired = paired.sort_values(['_order', 'Group_Rank']).drop(columns='_order')
        paired[['Left', 'Right', 'Difference']] = np.nan
        for filename in eye_filenames:
            (output / filename).unlink(missing_ok=True)
    identity = f"ID={frame['ID'].iloc[0]} / {len(paired)} image pairs"
    def save(fig, name):
        if not eye_analysis and name in eye_filenames:
            plt.close(fig)
            return
        fig.tight_layout()
        try:
            fig.savefig(output / name, dpi=300, bbox_inches='tight')
        finally:
            plt.close(fig)

    order = [c for c in CONDITIONS if c in set(frame['Condition'])]
    for group, subset, total, filename in (
        ('All', frame, frame['Pair_ID'].nunique(), '01_condition_mean_gain.png'),
        ('High', frame.loc[frame['Difference_Group'].eq('high')], frame.loc[frame['Difference_Group'].eq('high'),'Pair_ID'].nunique(),
         '01_condition_mean_gain_high.png'),
        ('Low', frame.loc[frame['Difference_Group'].eq('low')], frame.loc[frame['Difference_Group'].eq('low'),'Pair_ID'].nunique(),
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
    ax.axvline((len(paired) / 2) - 0.5, color='gray', linestyle=':')
    ax.set_xticks(np.arange(len(paired)), paired['Pair_ID'], rotation=60, ha='right')
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('Gain: left-eye-based - right-eye-based image')
    ax.set_xlabel('Image pair (red=high, blue=low; NA=unmatched)')
    ax.set_title(f'Eye-based image differences / valid pairs={n}/{len(paired)}\n{identity}')
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
           xlabel='Left-eye-based image gain (slider 0-1)',
           ylabel='Right-eye-based image gain (slider 0-1)',
           title=f'DP eye comparison / valid pairs={n}/{len(paired)}\n{identity}')
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
    if not eye_analysis:
        print('Skipped eye-based difference/scatter figures: no valid Left/Right content-eye pair')
    print(f'Saved applicable figures (no CI, no tests): {output}')


def relation_trials(target):
    """一致回答のみ。反復は後段で画像組単位に平均する。"""
    raw = pd.read_csv(target/'image_matching.csv', dtype=str, encoding='utf-8-sig')
    if raw['ID'].nunique() != 1:
        raise ValueError('Analyze one participant/session at a time')
    raw = raw.loc[raw['Match_Status'].eq('matched')].copy()
    alpha = raw['Condition'].eq('SP_ALPHA_BLEND')
    raw['Value'] = pd.to_numeric(raw['Matched_Gain'].where(~alpha, raw['Matched_Alpha']), errors='coerce')
    if not np.isfinite(raw['Value']).all() or not raw['Value'].between(0, 1).all():
        raise ValueError('Invalid matched gain/alpha')
    return raw


def relation_means(target):
    return relation_trials(target).groupby(
        ['ID', 'Dominance', 'Pair_ID', 'Condition'], as_index=False)['Value'].mean()


def save_relation_analysis(target):
    from scipy.stats import pearsonr
    frame = relation_means(target)
    table = target/'matching_analysis'/'relations'; table.mkdir(parents=True, exist_ok=True)
    output = FIGURE_ROOT/target.name/'matching_analysis'/'relations'
    output.mkdir(parents=True, exist_ok=True)
    wide = frame.pivot(index=['ID','Dominance','Pair_ID'], columns='Condition', values='Value')
    available = set(wide.columns)
    fig, ax = plt.subplots(figsize=(6, 6)); points=[]; statistics=[]
    dominance = frame['Dominance'].str.strip().str.capitalize().unique()
    if len(dominance) != 1 or dominance[0] not in ('Left','Right'):
        raise ValueError('One consistent Left/Right dominance is required')
    for eye, color in [('Right','tab:blue'), ('Left','tab:orange')]:
        dp = 'DP_MONO_DOM' if eye == dominance[0] else 'DP_MONO_NONDOM'
        sp = 'SP_REPRO_RIGHT_EYE' if eye == 'Right' else 'SP_REPRO_LEFT_EYE'
        if not {sp, dp}.issubset(available):
            print(f'Skipped {eye} SP-vs-DP same-eye analysis: required conditions are absent')
            continue
        part = wide.reindex(columns=[sp,dp]).dropna().reset_index()
        part = part.rename(columns={sp:'SP_Gain', dp:'DP_Gain'})
        part['Eye']=eye; part['SP_Condition']=sp; part['DP_Condition']=dp
        points.append(part)
        ax.scatter(part['SP_Gain'], part['DP_Gain'], color=color, label=f'{eye} (n={len(part)})')
        r=p=np.nan; status='insufficient_pairs'
        if len(part)>=3:
            status='constant_values'
            if part['SP_Gain'].nunique()>1 and part['DP_Gain'].nunique()>1:
                r,p=pearsonr(part['SP_Gain'],part['DP_Gain']); status='ok'
        # 回帰直線を当てはめず、固定したy=x（予測DP=SP）に対する決定係数。
        # R2 = 1 - sum((DP-SP)**2) / sum((DP-mean(DP))**2)。負値もそのまま保存。
        identity_r2=np.nan; identity_status='insufficient_pairs'
        if len(part)>=2:
            x=part['SP_Gain'].to_numpy(dtype=float)
            y=part['DP_Gain'].to_numpy(dtype=float)
            sst=float(np.sum((y-y.mean())**2))
            identity_status='constant_DP'
            if sst>0:
                identity_r2=1.0-float(np.sum((y-x)**2))/sst
                identity_status='ok'
        statistics.append(dict(Eye=eye, SP_Condition=sp, DP_Condition=dp,
                               N=len(part), Pearson_r=r, P_value=p, Status=status,
                               R2_y_equals_x=identity_r2, R2_Status=identity_status))
    if points:
        ax.plot([0,1],[0,1], '--', color='gray', linewidth=1)
        ax.set(xlabel='SP reproduced-eye gain', ylabel='DP same-eye gain', xlim=(0,1), ylim=(0,1))
        ax.legend(); fig.tight_layout(); fig.savefig(output/'sp_dp_same_eye.png', dpi=300); plt.close(fig)
        pd.concat(points,ignore_index=True).to_csv(table/'sp_dp_same_eye_points.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame(statistics).to_csv(table/'sp_dp_same_eye_correlations.csv',index=False,encoding='utf-8-sig')
    else:
        plt.close(fig)
        for path in (
            output/'sp_dp_same_eye.png',
            table/'sp_dp_same_eye_points.csv',
            table/'sp_dp_same_eye_correlations.csv',
        ):
            path.unlink(missing_ok=True)
    alpha = frame.loc[frame['Condition'].eq('SP_ALPHA_BLEND'),['ID','Pair_ID','Value']].rename(columns={'Value':'Alpha'})
    gains = frame.loc[~frame['Condition'].eq('SP_ALPHA_BLEND')]
    if alpha.empty or gains.empty:
        for path in (
            output/'alpha_vs_gain.png',
            table/'alpha_vs_gain_points.csv',
            table/'alpha_vs_gain_correlations.csv',
        ):
            path.unlink(missing_ok=True)
        print('Skipped alpha-vs-gain analysis: alpha or gain condition is absent')
        return
    joined = gains.merge(alpha,on=['ID','Pair_ID'],validate='many_to_one')
    joined.to_csv(table/'alpha_vs_gain_points.csv',index=False,encoding='utf-8-sig')
    alpha_statistics=[]
    fig, ax = plt.subplots(figsize=(7,5))
    for condition, part in joined.groupby('Condition'):
        ax.scatter(part['Alpha'],part['Value'],label=CONDITIONS.get(condition,condition))
        r=p=np.nan; status='insufficient_pairs'
        if len(part)>=3:
            status='constant_values'
            if part['Alpha'].nunique()>1 and part['Value'].nunique()>1:
                r,p=pearsonr(part['Alpha'],part['Value']); status='ok'
        alpha_statistics.append(dict(
            Condition=condition, N=len(part), Pearson_r=r, P_value=p, Status=status))
    pd.DataFrame(alpha_statistics).to_csv(
        table/'alpha_vs_gain_correlations.csv',index=False,encoding='utf-8-sig')
    ax.set(xlabel='Matched alpha (SP alpha blend)',ylabel='Matched gain',xlim=(0,1),ylim=(0,1))
    if len(joined): ax.legend(fontsize=8)
    else: ax.text(.5,.5,'No corresponding alpha/gain data',ha='center')
    fig.tight_layout(); fig.savefig(output/'alpha_vs_gain.png',dpi=300); plt.close(fig)


def save_alpha_analysis(target):
    frame,metadata=load_results(target,control='alpha')
    output=FIGURE_ROOT/target.name/'matching_analysis'/'alpha'
    output.mkdir(parents=True,exist_ok=True)
    table=target/'matching_analysis'/'alpha';table.mkdir(parents=True,exist_ok=True)
    frame.rename(columns={'Gain':'Mean_Matched_Alpha'}).drop(
        columns=['Matched_Gain'],errors='ignore').to_csv(
            table/'alpha_pair_means.csv',index=False,encoding='utf-8-sig')
    for group in ('all','high','low'):
        part=frame if group=='all' else frame.loc[frame['Difference_Group'].eq(group)]
        fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(part['Pair_ID'],part['Gain']);ax.set_ylim(0,1)
        ax.set_ylabel('Matched alpha (not foreground gain)')
        ax.set_title(f'Alpha matching / {group} / repetition means')
        ax.tick_params(axis='x',labelrotation=90)
        fig.tight_layout();fig.savefig(output/f'alpha_{group}.png',dpi=300);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_dir', nargs='?', help='Session directory or image_matching.csv')
    args = parser.parse_args()
    target = resolve_target(args.target_dir)
    save_relation_analysis(target)
    columns = pd.read_csv(target / 'image_matching.csv', encoding='utf-8-sig', nrows=0).columns
    conditions=pd.read_csv(target/'image_matching.csv',encoding='utf-8-sig',usecols=['Condition'])['Condition']
    if conditions.eq('SP_ALPHA_BLEND').any(): save_alpha_analysis(target)
    if conditions.eq('SP_ALPHA_BLEND').all(): return
    directions = ('up', 'down', 'mean') if 'Matching_Direction' in columns else ('all',)
    for direction in directions:
        frame, metadata = load_results(target, direction)
        content = frame.attrs.get('content_gain', pd.DataFrame())
        eye_ready = (
            {'Left', 'Right'}.issubset(content.columns)
            and content.reindex(columns=['Left', 'Right']).notna().all(axis=1).any()
        )
        paired = build_dp_pairs(frame, metadata) if eye_ready else None
        name = 'matching_analysis' if direction == 'all' else f'matching_analysis/{direction}'
        if eye_ready:
            save_dp_msd_csv(frame, paired, target, name)
        else:
            stale = target / name / 'dp_eye_mean_squared_difference.csv'
            stale.unlink(missing_ok=True)
            print(f'[{direction}] Skipped DP eye MSD: no valid Left/Right content-eye pair')
        save_figures(frame, paired, FIGURE_ROOT / target.name / name)


if __name__ == '__main__':
    main()
