"""ratingと混同しないマッチングCSV・設定・調整履歴を原子的に保存する。"""
from pathlib import Path
from dataclasses import asdict
import csv, json, os, random
import tempfile
from .results import RESULT_FIELDS, _trial_metadata, _jsonable

EXTRA_FIELDS=['Task_Mode','Reference_Observed_Eye','Alignment_Eye','Initial_Gain',
 'Final_Gain','Matched_Gain','Original_FG_Y_cd_m2','Reference_Gain',
 'Reference_Target_Y_cd_m2','Reference_Quantized_Y_cd_m2',
 'Test_Target_FG_Y_cd_m2','Test_Quantized_FG_Y_cd_m2','Match_Status',
 'Adjustment_Time_s','Adjustment_Count','Reference_Center_Deg','Test_Center_Deg',
 'Layout_JSON','Luminance_Basis','Gamut_Policy',
 'Gamut_Ratio_At_Zero','Gamut_Ratio_At_One',
 'Reference_Plane_Mode','Reference_BG_Type','Reference_Noise_Seed',
 'Reference_Noise_Center_Cpd','Reference_Noise_Bandwidth_Octave',
 'Reference_Noise_Mean_Y_cd_m2','Reference_Noise_Modulation',
 'Reference_Reproduction_Method','Reference_OutOfGamut_Ratio',
 'Reference_Total_Target_Y_cd_m2','Reference_Total_Quantized_Y_cd_m2']
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

def save_matching_manifest(app,output_dir=None):
    fields=[k for k in MATCHING_FIELDS if k not in {'Match_Status','Matched_Gain','Final_Gain',
            'Adjustment_Time_s','Adjustment_Count'}]
    directory=app.result_dir if output_dir is None else Path(output_dir)
    atomic_csv(directory/'trial_manifest.csv',fields,[trial_base(app,t) for t in app.trial_list])

def save_matching_session(app):
    cfg=app.session_config
    # 途中の試行数・中断行・不完全な校正から結果ファイルを作らない。
    if app.matching_state!='finished':
        raise ValueError('Only a completed matching session can be saved')
    expected=cfg.expected_trial_count
    if (not app.preflight_complete or len(app.trial_list)!=expected
            or len(app.results)!=expected or app.current_trial_index!=expected):
        raise ValueError('All configured trials must be completed before saving')
    planned=[t.trial_order for t in app.trial_list]
    actual=[r.get('Trial_ID') for r in app.results]
    valid_statuses={'matched','above_range','below_range','cannot_judge'}
    if (len(set(planned))!=expected or actual!=planned
            or any(r.get('Match_Status') not in valid_statuses for r in app.results)):
        raise ValueError('Completed trial IDs/statuses do not match the manifest')
    defocus=app.detailed_defocus_results
    if any(sum(r.get('Eye')==eye for r in defocus)!=cfg.defocus_repetitions for eye in ('Right','Left')):
        raise ValueError('Both eyes must have complete defocus calibration results')
    destination=Path(app.result_dir)
    if destination.parent.resolve()!=cfg.result_root.resolve():
        raise ValueError('Session destination is outside the matching result root')
    def check_destination():
        # 空フォルダ、または先行保存されたdefocus結果だけなら許容する。
        # 本試行結果や他のファイルがあるセッションは上書きしない。
        if destination.is_symlink():
            raise FileExistsError(f'Refusing to overwrite a linked session: {destination}')
        if destination.exists():
            if not destination.is_dir() or any(
                p.name != cfg.defocus_result_filename or not p.is_file() or p.is_symlink()
                for p in destination.iterdir()
            ):
                raise FileExistsError(f'Refusing to overwrite an existing result session: {destination}')
    check_destination()
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
      'gamut_policy':'clip_linear_rgb_0_1','result_save_policy':'complete_session_only',
      'luminance_basis':'calibrated_XYZ_increment; black-inclusive absolute luminance not verified'})
    # 全試行完了後にだけ一時領域へ書き、全ファイル成功後に一括公開する。
    # 一時領域は通常のmatching結果ルートの外。同一filesystem内のrenameを使う。
    cfg.result_root.mkdir(parents=True,exist_ok=True)
    defocus_fields=['ID','Eye','Trial','Spatial_Freq(cpd)','Noise_Seed','Matched_PD(mm)']
    with tempfile.TemporaryDirectory(prefix='.image-matching-',dir=cfg.result_root.parent) as temp:
        staging=Path(temp)/'session';staging.mkdir()
        atomic_csv(staging/'image_matching.csv',MATCHING_FIELDS,app.results)
        atomic_csv(staging/'matching_adjustments.csv',HISTORY_FIELDS,app.adjustment_history)
        atomic_csv(staging/cfg.defocus_result_filename,defocus_fields,defocus)
        save_matching_manifest(app,staging)
        atomic_json(staging/'used_experiment_config.json',settings)
        check_destination()
        backup=Path(temp)/'prior_calibration'
        if destination.exists():
            existing_defocus=destination/cfg.defocus_result_filename
            if existing_defocus.is_file():
                # 先行保存された校正CSVは内容を変えずに引き継ぐ。
                (staging/cfg.defocus_result_filename).write_bytes(existing_defocus.read_bytes())
            os.rename(destination,backup)
        try:
            os.rename(staging,destination)
        except Exception:
            if backup.exists():
                try:
                    os.rename(backup,destination)
                except Exception as restore_error:
                    # 復元にも失敗した場合は一時領域外へ退避して消失を防ぐ。
                    recovery=cfg.result_root/(destination.name+'_defocus_recovery_'+Path(temp).name)
                    os.rename(backup,recovery)
                    raise RuntimeError(f'Calibration backup preserved at: {recovery}') from restore_error
            raise
    return destination/'image_matching.csv' 
