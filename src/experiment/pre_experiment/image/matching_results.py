"""ratingと混同しないマッチングCSV・設定・調整履歴を原子的に保存する。"""
from pathlib import Path
from dataclasses import asdict
import csv, json, os, random
from .results import RESULT_FIELDS, _trial_metadata, _jsonable

EXTRA_FIELDS=['Task_Mode','Reference_Observed_Eye','Alignment_Eye','Initial_Gain',
 'Final_Gain','Matched_Gain','Original_FG_Y_cd_m2','Reference_Gain',
 'Reference_Target_Y_cd_m2','Reference_Quantized_Y_cd_m2',
 'Test_Target_FG_Y_cd_m2','Test_Quantized_FG_Y_cd_m2','Match_Status',
 'Adjustment_Time_s','Adjustment_Count','Reference_Center_Deg','Test_Center_Deg',
 'Layout_JSON','Luminance_Basis','Gamut_Policy',
 'Gamut_Ratio_At_Zero','Gamut_Ratio_At_One']
MATCHING_FIELDS=[f for f in RESULT_FIELDS if f!='Score']+EXTRA_FIELDS
HISTORY_FIELDS=['ID','Trial_ID','Condition','Pair_ID','Elapsed_s','Requested_Gain','Displayed_Gain','Event']


def atomic_csv(path,fields,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    try:
        with tmp.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            w.writerows({k:r.get(k,'') for k in fields} for r in rows)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    try:
        with tmp.open('w',encoding='utf-8') as f:
            json.dump(_jsonable(value),f,ensure_ascii=False,indent=2,allow_nan=False)
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

def initial_gain_plan(trials,seed):
    """試行順とは独立した乱数で、各条件・high/lowごとに両端を均衡化。"""
    rng=random.Random(seed ^ 0x45A771);groups={};result={}
    for t in trials: groups.setdefault((t.condition_id,t.pair.difference_group),[]).append(t)
    for key,group in sorted(groups.items()):
        values=[0.]* (len(group)//2)+[1.]*(len(group)-len(group)//2);rng.shuffle(values)
        result.update({t.trial_order:v for t,v in zip(group,values)})
    return result

def trial_base(app,trial):
    row=_trial_metadata(trial,dominant_eye=app.participant_dominance.get(),
                       selection_dir=app.selected_pairs_dir,participant_seed=app.participant_seed)
    row.update({'Task_Mode':'matching','Initial_Gain':app.initial_gains[trial.trial_order],
      'Reference_Observed_Eye':row['Observed_Eye'],
      'Reference_Center_Deg':app.session_config.reference_center_deg,
      'Test_Center_Deg':app.session_config.test_center_deg,
      'Reference_Gain':app.session_config.reference_gain})
    return row

def build_matching_row(app,status,elapsed):
    row=trial_base(app,app.current_trial)
    right=app.calib_results['Right'];left=app.calib_results['Left'];cfg=app.session_config
    row.update({'ID':app.participant_id.get(),'Age':app.participant_age.get(),
      'Gender':app.participant_gender.get(),'IPD(mm)':app.participant_ipd.get(),
      'Dominance':app.participant_dominance.get(),'Distance_FG(cm)':cfg.distance_fg_cm,
      'Distance_BG(cm)':cfg.distance_bg_cm,'Visual_Angle(deg)':cfg.visual_angle_deg,
      'PD_Right':right['pd_mean'],'PD_Left':left['pd_mean'],
      'OffsetX_Right':right['offset_x'],'OffsetY_Right':right['offset_y'],
      'OffsetX_Left':left['offset_x'],'OffsetY_Left':left['offset_y'],
      'Alignment_Eye':app.current_alignment_eye,'Final_Gain':app.rendered_gain,
      'Matched_Gain':app.rendered_gain if status=='matched' else '',
      'Match_Status':status,'Adjustment_Time_s':elapsed,'Adjustment_Count':app.adjustment_count,
      'Layout_JSON':json.dumps(app.current_layout_metadata,ensure_ascii=False)})
    row.update(app.matching_prepared.metadata);row.update(app.render_metrics)
    return {k:row.get(k,'') for k in MATCHING_FIELDS}

def save_matching_manifest(app):
    fields=[k for k in MATCHING_FIELDS if k not in {'Match_Status','Matched_Gain','Final_Gain',
            'Adjustment_Time_s','Adjustment_Count'}]
    atomic_csv(app.result_dir/'trial_manifest.csv',fields,[trial_base(app,t) for t in app.trial_list])

def save_matching_session(app):
    cfg=app.session_config
    atomic_csv(app.result_dir/'image_matching.csv',MATCHING_FIELDS,app.results)
    atomic_csv(app.result_dir/'matching_adjustments.csv',HISTORY_FIELDS,app.adjustment_history)
    settings=asdict(cfg)
    settings.update({'task_mode':'matching','condition_enabled':cfg.condition_enabled,
      'condition_order':list(app.condition_order),'condition_count':cfg.condition_count,
      'expected_trial_count':cfg.expected_trial_count,
      'actual_trial_count':sum(r['Match_Status']!='aborted' for r in app.results),
      'matched_trial_count':sum(r['Match_Status']=='matched' for r in app.results),
      'selected_pairs_dir':str(app.selected_pairs_dir),'participant_seed':app.participant_seed,
      'calibration_files':app.calibration_file_info,'preflight_complete':app.preflight_complete,
      'preflight':app.preflight_records,'session_state':app.matching_state,
      'reference_center_deg':cfg.reference_center_deg,'test_center_deg':cfg.test_center_deg,
      'runtime_config':__import__('experiment.pre_experiment.image.config',fromlist=['RUNTIME_CONFIG']).RUNTIME_CONFIG,
      'gamut_policy':'clip_linear_rgb_0_1',
      'luminance_basis':'calibrated_XYZ_increment; black-inclusive absolute luminance not verified'})
    atomic_json(app.result_dir/'used_experiment_config.json',settings)
    return app.result_dir/'image_matching.csv'
