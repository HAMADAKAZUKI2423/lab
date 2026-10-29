"""実測alphaの各反復で画像を再生成し、視認性L2対ゲイン/alphaを保存する。
cd src && py -m experiment.pre_analyze.pre_analyze_image_alpha_visibility [session_dir] --device cuda:0
同じフォルダの更新済みpre_analyze_image.pyが必要。既定CPU、モデルvismlp_norm。
モデル入力は校正済みFG経路の線形RGB成分と実測alpha（合成画像を再入力しない）。
全背景幅を保持し、高さ512pxに統一。前景領域外の視認性は0、全マップのL2を集約。
予測L2を反復ごとに計算後、画像組単位で平均。平均alphaからの再予測はしない。
alpha自体もスコア生成に使うため、alpha対スコアの関係は独立な検証ではない。
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image
from .pre_analyze_image import (resolve_target, relation_trials, relation_means,
                                FIGURE_ROOT, CONDITIONS, plt)


def resize_linear(values, size):
    return np.stack([np.asarray(Image.fromarray(values[...,c].astype('float32')).resize(
        size, Image.Resampling.BILINEAR)) for c in range(3)],axis=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target_dir',nargs='?')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--model',default='vismlp_norm')
    parser.add_argument('--image-size',type=int,default=512,help='全条件共通のマップ高さ')
    parser.add_argument('--display-dir',type=Path,help='移動した校正フォルダ。同一hashが必要')
    args=parser.parse_args()
    if args.image_size<16: parser.error('image-size must be at least 16')
    target=resolve_target(args.target_dir)
    raw=relation_trials(target)
    trials=raw.loc[raw['Condition'].eq('SP_ALPHA_BLEND')].copy()
    if trials.empty: raise ValueError('No valid matched alpha trials')
    settings=json.loads((target/'used_experiment_config.json').read_text(encoding='utf-8'))
    import torch
    from experiment.stimuli import select_fukiage_disparity_pairs as base
    from experiment.common.display_calibration import load_display_calibration
    from experiment.pre_experiment.image.matching_stimuli import (
        prepare_matching_stimulus, encode, calibration_files_metadata)
    from experiment.pre_experiment.image.matching_layout import Box
    display=args.display_dir or Path(settings['display_dir'])
    recorded=settings.get('calibration_files')
    if not recorded or calibration_files_metadata(display)!=recorded:
        raise ValueError('Original calibration files are required (hash mismatch/missing record)')
    calibration=load_display_calibration(display)
    config=SimpleNamespace(**settings)
    base_angle=config.visual_angle_deg*config.background_width_factor
    config.background_visual_angle_width_deg=math.degrees(2*math.atan(
        config.background_width_scale*math.tan(math.radians(base_angle/2))))
    device=torch.device(args.device); model=base.load_visibility_model(args.model,device)
    table=target/'matching_analysis'/'alpha_visibility'; table.mkdir(parents=True,exist_ok=True)
    output=FIGURE_ROOT/target.name/'matching_analysis'/'alpha_visibility'
    output.mkdir(parents=True,exist_ok=True)
    images=output/'blends'; images.mkdir(exist_ok=True)
    maps=table/'maps'; maps.mkdir(exist_ok=True)
    rows=[]; shape=None
    for index,(_,row) in enumerate(trials.iterrows(),1):
        layout={k:Box(**v) for k,v in json.loads(row['Layout_JSON']).items()}
        pair=SimpleNamespace(foreground_path=Path(row['Foreground_Path']),
                             background_path=Path(row['Background_Path']),pair_id=row['Pair_ID'])
        trial=SimpleNamespace(condition_id='SP_ALPHA_BLEND',pair=pair,
                              mirrored=str(row.get('Mirrored','')).strip().lower()=='true')
        prepared=prepare_matching_stimulus(trial,config,layout=layout,calibration=calibration,
            dominant_eye=row['Dominance'],left_pupil_mm=float(row['PD_Left']),
            right_pupil_mm=float(row['PD_Right']),ipd_mm=float(row['IPD(mm)']))
        alpha=float(row['Value'])
        stem=f'trial_{index:04d}'
        # 実験の同じ線形合成・色域クリップ・量子化。PNGは鏡反転前の観察像。
        encode(prepared.raw_test_linear(alpha),calibration.gamma_fg).save(images/f'{stem}.png')
        foreground=prepared.foreground_linear
        background=prepared.background_xyz @ calibration.r_prime_inv.T
        h,w=foreground.shape[:2]
        size=(round(w/h*args.image_size),args.image_size)
        if shape is not None and shape!=size:
            raise ValueError('Different stimulus geometry: raw L2 scores are not comparable')
        shape=size
        fg=resize_linear(foreground,size); bg=resize_linear(background,size)
        mask=np.zeros((h,w),dtype=np.float32); mask[prepared.fg_slice]=1
        mask=np.asarray(Image.fromarray(mask).resize(size,Image.Resampling.NEAREST)).copy()
        mask=mask[...,None]
        # ピラミッドの縮小・拡大で1pxずれないよう、各辺を2**levelの倍数へ補完。
        # 元画像は伸縮せず右・下端を複製。補完部・前景外は集約時に除外する。
        level=int(model.level)
        if not 0 <= level <= 12:
            raise ValueError(f'Unsupported pyramid level: {level}')
        multiple=2**level
        map_h,map_w=mask.shape[:2]
        pad_h=(-map_h)%multiple; pad_w=(-map_w)%multiple
        padding=((0,pad_h),(0,pad_w),(0,0))
        # 推論マスクとL2集約領域は別物。既存selectと同じ全1マスク・一様alphaを使う。
        # 前景外はFG=BGにして、alphaによらず元の背景を保持する（alphaの二重適用なし）。
        clipped_bg=np.clip(bg,0,1)
        model_fg=np.pad(np.where(mask>0,np.clip(fg,0,1),clipped_bg),padding,mode='edge')
        model_bg=np.pad(clipped_bg,padding,mode='edge')
        model_mask=np.ones((*model_bg.shape[:2],1),dtype=np.float32)
        alpha_map=np.full_like(model_mask,alpha)
        with torch.no_grad():
            model.set_inputs_tg_ref_alphamap(
                base.to_tensor(model_fg[...,::-1].copy(),device),
                base.to_tensor(model_bg[...,::-1].copy(),device),
                base.to_tensor(alpha_map,device),
                base.to_tensor(model_mask,device),blend_mode='linear')
            model.compute_weights(); model.compute_visibility_wo_weight()
        visibility=model.norm_vismap.squeeze().detach().cpu().numpy().astype(np.float32)
        expected=model_mask.shape[:2]
        context=f"ID={row['ID']}, pair={row['Pair_ID']}, trial={row.get('Trial_ID','')}, alpha={alpha:g}"
        native_shape=visibility.shape
        if visibility.shape!=expected:
            factor=expected[0]/visibility.shape[0] if visibility.ndim==2 else 0
            if (visibility.ndim!=2 or factor<1 or factor!=int(factor)
                    or expected[1]!=visibility.shape[1]*factor):
                np.save(maps/f'{stem}_invalid_raw.npy',visibility)
                raise ValueError(f'Visibility shape mismatch: actual={visibility.shape}, expected={expected}; {context}')
            # 一部試行でピラミッド上の粗いグリッドが返る。全試行を同じ画素数でL2集約するため双線形で入力解像度へ戻す。
            print(f'WARN: coarse visibility map {visibility.shape}->{expected}; {context}',flush=True)
            visibility=np.asarray(Image.fromarray(visibility.astype(np.float32),mode='F').resize(
                (expected[1],expected[0]),Image.Resampling.BILINEAR),dtype=np.float32)
        visibility=visibility[:map_h,:map_w]
        roi=mask[...,0]>0
        invalid=~np.isfinite(visibility)
        if np.any(invalid & roi):
            diagnostic=maps/f'{stem}_invalid.npz'
            np.savez_compressed(diagnostic,visibility=visibility,roi=roi,
                                foreground=model_fg,background=model_bg,alpha=alpha)
            raise ValueError(
                f'Non-finite visibility in foreground: NaN={np.count_nonzero(np.isnan(visibility)&roi)}, '
                f'Inf={np.count_nonzero(np.isinf(visibility)&roi)}; {context}; diagnostics={diagnostic}')
        # NaN*0もNaNになるため、領域外は乗算ではなく明示的に0へ置換。
        # 領域内のNaN/Infを0埋めして不正なスコアを作ることはしない。
        visibility=np.where(roi,visibility,0.0)
        np.save(maps/f'{stem}.npy',visibility)
        score=float(np.linalg.norm(visibility.astype(np.float64).ravel()))
        rows.append(dict(ID=row['ID'],Pair_ID=row['Pair_ID'],Trial_ID=row.get('Trial_ID',''),
            Repetition=row.get('Repetition',''),Alpha=alpha,Visibility_L2=score,
            Width=size[0],Height=size[1],Native_Map_H=native_shape[0],Native_Map_W=native_shape[1],OutOfGamut_Ratio=prepared.gamut_ratio(alpha),
            Model_Component_Clip_Ratio=float(np.mean(np.any(
                (fg<0)|(fg>1)|(bg<0)|(bg>1),axis=-1))),Blend=str(images/f'{stem}.png')))
    scores=pd.DataFrame(rows)
    scores.to_csv(table/'alpha_visibility_trials.csv',index=False,encoding='utf-8-sig')
    means=scores.groupby(['ID','Pair_ID'],as_index=False).agg(
        Visibility_L2=('Visibility_L2','mean'),Alpha=('Alpha','mean'),Valid_Repetitions=('Alpha','size'))
    means.to_csv(table/'alpha_visibility_pair_means.csv',index=False,encoding='utf-8-sig')
    values=relation_means(target)
    joined=values.merge(means,on=['ID','Pair_ID'],validate='many_to_one')
    joined['Control_Kind']=np.where(joined['Condition'].eq('SP_ALPHA_BLEND'),'alpha','gain')
    joined.to_csv(table/'visibility_vs_response_points.csv',index=False,encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(8,5))
    for condition,part in joined.groupby('Condition'):
        ax.scatter(part['Visibility_L2'],part['Value'],label=CONDITIONS.get(condition,condition),
                   marker='s' if condition=='SP_ALPHA_BLEND' else 'o')
    ax.set(xlabel='Predicted visibility L2 (measured-alpha blend)',ylabel='Matched gain / alpha',ylim=(0,1))
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output/'visibility_vs_gain_alpha.png',dpi=300); plt.close(fig)
    (table/'analysis_settings.json').write_text(json.dumps(dict(model=args.model,
        image_height=args.image_size,map_width=shape[0],aggregation='mean_of_per_trial_L2',
        map_scope='full_background_strip_zero_outside_foreground',model_input='calibrated_linear_RGB_components',
        model_component_gamut_policy='clip_to_0_1_report_ratio',
        inference_mask='all_ones',alpha_map='uniform_measured_alpha',
        outside_foreground_components='FG_equals_BG',invalid_roi_policy='save_diagnostics_and_stop',
        calibration_files=recorded),indent=2),encoding='utf-8')
    print(f'Saved alpha visibility tables: {table}; figures/blends: {output}')


if __name__=='__main__':
    main()