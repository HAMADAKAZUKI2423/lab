"""selectorのhigh5/low5計10組を読み込む。旧18/20組も対応。"""
from dataclasses import dataclass
from pathlib import Path
import csv

@dataclass(frozen=True)
class SelectedPair:
    pair_id: str; difference_group: str; group_rank: int; global_rank: int
    foreground_texture_class: int; foreground_path: Path; background_path: Path; pair_dir: Path
    right_visibility_score: float|None; left_visibility_score: float|None
    absolute_score_difference: float|None; visibility_level: str=""; visibility_map_l2: float|None=None

def _optional_float(row,key):
    value=row.get(key,'').strip(); return None if value=='' else float(value)

def _resolve_original(selection_dir,pair_dir,row,role):
    saved=sorted(pair_dir.glob(f'{role}_original.*')) if pair_dir.is_dir() else []
    if len(saved)>1: raise ValueError(f'Multiple saved {role} images: {saved}')
    if saved: return saved[0].resolve()
    raw=Path(row[f'{role}_path']).expanduser(); path=raw if raw.is_absolute() else selection_dir/raw
    if not path.is_file(): raise FileNotFoundError(path)
    return path.resolve()

def load_selected_pairs(selection_dir: Path)->list[SelectedPair]:
    selection_dir=selection_dir.expanduser().resolve()
    if not selection_dir.is_dir(): raise FileNotFoundError(selection_dir)
    combined=selection_dir/'selected_pairs.csv'; pairs=[]
    if combined.is_file():
        sources=[(combined,None)]
    else:
        suffix=next((n for n in (5,9,10) if all(
            (selection_dir/f'{g}{n}.csv').is_file() for g in ('high','low'))), None)
        if suffix is None:
            raise FileNotFoundError('selected_pairs.csv または high5.csv / low5.csv が必要です')
        sources=[(selection_dir/f'{g}{suffix}.csv',g) for g in ('high','low')]
    raw_rows=[]
    for csv_path,expected in sources:
        with csv_path.open(newline='',encoding='utf-8-sig') as stream:
            raw_rows.extend(csv.DictReader(stream))
    counts={g:sum(r['difference_group'].strip().lower()==g for r in raw_rows)
            for g in ('high','low')}
    for csv_path,expected in sources:
        if not csv_path.is_file(): raise FileNotFoundError(csv_path)
        with csv_path.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        for row in rows:
            group=row['difference_group'].strip().lower()
            if expected and group!=expected: raise ValueError(f'{csv_path.name}: invalid group')
            group_rank=int(row['group_rank']); suffix=counts.get(group,0)
            pair_dir=selection_dir/f'{group}{suffix}'/f'rank_{group_rank:02d}'
            level=row.get('visibility_level','').strip().lower()
            pairs.append(SelectedPair(
                pair_id=f'{group}_{level+"_" if level else ""}{group_rank:02d}',
                difference_group=group,group_rank=group_rank,global_rank=int(row['rank']),
                foreground_texture_class=int(row.get('foreground_texture_class') or 0),
                foreground_path=_resolve_original(selection_dir,pair_dir,row,'foreground'),
                background_path=_resolve_original(selection_dir,pair_dir,row,'background'),pair_dir=pair_dir.resolve(),
                right_visibility_score=_optional_float(row,'right_visibility_score'),
                left_visibility_score=_optional_float(row,'left_visibility_score'),
                absolute_score_difference=_optional_float(row,'absolute_score_difference'),
                visibility_level=level,visibility_map_l2=_optional_float(row,'visibility_map_l2')))
    pairs.sort(key=lambda x:(0 if x.difference_group=='high' else 1,x.group_rank)); _validate_selected_pairs(pairs)
    return pairs

def _validate_selected_pairs(pairs):
    expected=len(pairs)
    if expected not in {10,18,20} or len({p.pair_id for p in pairs})!=expected:
        raise ValueError('10 unique pairs (high5/low5) required; legacy 18/20 also supported')
    per_group=expected//2
    if {p.difference_group for p in pairs}!={'high','low'}:
        raise ValueError('Both high and low groups are required')
    for group in ('high','low'):
        rows=[p for p in pairs if p.difference_group==group]
        if len(rows)!=per_group or {p.group_rank for p in rows}!=set(range(1,per_group+1)):
            raise ValueError(f'{group}: invalid pair count/ranks')
        if expected==10 and any(p.visibility_level not in {'','mid'} for p in rows):
            raise ValueError(f'{group}: current 5-pair format requires mid visibility')
        if expected==18 and {level:sum(p.visibility_level==level for p in rows) for level in ('low','mid','high')}!={'low':3,'mid':3,'high':3}:
            raise ValueError(f'{group}: legacy 18-pair format requires 3 pairs per visibility level')
    fg={str(p.foreground_path).casefold() for p in pairs}; bg={str(p.background_path).casefold() for p in pairs}
    if len(fg)!=expected or len(bg)!=expected:
        raise ValueError('Foreground and background images must each be unique')
    if any(p.foreground_path==p.background_path for p in pairs):
        raise ValueError('The same image cannot be used for both roles in one pair')
