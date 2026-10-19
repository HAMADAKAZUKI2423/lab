"""校正済み前景の線形光量だけを変更。SP背景・referenceは固定する。"""
from dataclasses import dataclass
from pathlib import Path
import hashlib, math
import numpy as np
from PIL import Image
from .conditions import get_condition, resolve_observed_eye
from .stimuli import (_prepare_sources, _color_correct_foreground, _apply_channel_gamma,
                      _blur_eye_background, RESAMPLE)
from .disparity import right_aligned_disparity_px, make_constant_eye_views, fuse_eye_views

CALIBRATION_FILES=('C.csv','T_prime.csv','R_prime.csv','gamma_bg.csv','gamma_fg.csv')

def calibration_files_metadata(directory):
    data={}
    for name in CALIBRATION_FILES:
        p=Path(directory)/name
        if not p.is_file(): raise FileNotFoundError(f"Matching requires measured calibration: {p}")
        if name in ('C.csv','T_prime.csv','R_prime.csv'):
            matrix=np.loadtxt(p,delimiter=',')
            if matrix.shape!=(3,3) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid measured calibration file: {p}')
        data[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    return data

def validate_calibration(cal):
    for name in ('color_matrix','t_prime','r_prime','r_prime_inv'):
        a=np.asarray(getattr(cal,name),float)
        if a.shape!=(3,3) or not np.isfinite(a).all(): raise ValueError(f"Invalid calibration matrix: {name}")
    if not np.allclose(cal.r_prime @ cal.r_prime_inv,np.eye(3),atol=1e-7):
        raise ValueError("R_prime inverse is inconsistent")
    for gamma in (cal.gamma_fg,cal.gamma_bg):
        if gamma is None or any(c not in gamma or not math.isfinite(gamma[c]) or gamma[c]<=0 for c in 'RGB'):
            raise ValueError("Matching requires finite positive R/G/B gamma values")

def decode(image,gamma):
    return _apply_channel_gamma(np.asarray(image,float)/255.,gamma,inverse=False)

def encode(linear,gamma):
    # レーティング同様、色域外の線形RGBを0〜1へクリップしてから逆ガンマ。
    linear=np.asarray(linear,dtype=float)
    if not np.isfinite(linear).all(): raise ValueError("Non-finite display RGB")
    values=_apply_channel_gamma(np.clip(linear,0,1),gamma,inverse=True)
    return Image.fromarray(np.rint(values*255).astype(np.uint8),'RGB')

def mean_y(linear,matrix):
    return float(np.mean((linear @ matrix.T)[...,1]))

@dataclass
class MatchingStimulus:
    foreground_linear: np.ndarray
    background_xyz: np.ndarray | None
    physical_background: Image.Image | None
    reference_image: Image.Image
    reference_gain: float
    calibration: object
    metadata: dict
    tolerance: float
    fg_slice: tuple
    baseline_display_y: float
    reference_background: Image.Image | None = None

    def raw_test_linear(self,gain):
        if not math.isfinite(gain) or not 0<=gain<=1: raise ValueError("Gain must be finite and in [0,1]")
        if self.background_xyz is None: return gain*self.foreground_linear
        return (self.background_xyz+gain*(self.foreground_linear @ self.calibration.r_prime.T)) @ self.calibration.r_prime_inv.T

    def gamut_ratio(self,gain):
        a=self.raw_test_linear(gain)
        if not np.isfinite(a).all(): raise ValueError("Non-finite display RGB")
        return float(np.mean(np.any((a < -self.tolerance)|(a > 1+self.tolerance),axis=-1)))

    def render(self,gain):
        a=self.raw_test_linear(gain)
        ratio=self.gamut_ratio(gain)
        # 色域外は停止理由にしない。クリップ前の比率は結果へ残す。
        image=encode(a,self.calibration.gamma_fg)
        shown=decode(image,self.calibration.gamma_fg)[self.fg_slice]
        metrics={'Displayed_Gain':float(gain),
                 'Test_Target_FG_Y_cd_m2':gain*self.metadata['Original_FG_Y_cd_m2'],
                 'Test_Quantized_FG_Y_cd_m2':mean_y(shown,self.calibration.r_prime)-self.baseline_display_y,
                 'SinglePlane_OutOfGamut_Ratio':ratio}
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),metrics

def prepare_matching_stimulus(trial,config,*,layout,calibration,dominant_eye,left_pupil_mm,right_pupil_mm,ipd_mm):
    validate_calibration(calibration)
    spec=get_condition(trial.condition_id)
    source,physical_bg,sp_bg=_prepare_sources(trial.pair,config)
    source=source.resize(layout['test'].size,RESAMPLE)
    corrected=_color_correct_foreground(source,calibration)
    fg=decode(corrected,calibration.gamma_fg)
    source_out=corrected
    original_y=mean_y(fg,calibration.r_prime)
    if not math.isfinite(original_y) or original_y<=0: raise ValueError("Foreground has no positive calibrated luminance")
    # Reference前景はtestと同じ元画像の校正済み線形光量を50%に固定する。
    if config.reference_gain != 0.5:
        raise ValueError("Image matching reference_gain must be 0.5")
    ref=encode(0.5*fg,calibration.gamma_fg)
    # 条件名や試行順をseedへ含めず、同一ペアでは全条件で同じ静止ノイズ。
    noise_key=f"{config.random_seed}:reference_noise:{trial.pair.pair_id}"
    noise_seed=int.from_bytes(hashlib.sha256(noise_key.encode()).digest()[:4],'big')
    from experiment.common import patterns, geometry, photometry
    noise_width,noise_height=layout['reference_background'].size
    ppd_bg=geometry.get_size_for_visual_angle(
        config.distance_bg_cm,1.0,config.background_pixels_per_cm)
    if ppd_bg/2 <= 4.0*math.sqrt(2.0):
        raise ValueError("Background resolution is insufficient for the reference noise")
    noise=patterns.create_noise_base(
        noise_width,noise_height,ppd_bg,4.0,bandwidth_octave=1.0,
        rng=np.random.default_rng(noise_seed))
    # 輝度領域で帯域制限。平均15 cd/m²、ピーク絶対値で正規化した変調係数1。
    noise_luminance=15.0*(1.0+noise)
    if (not np.isfinite(noise_luminance).all()
            or noise_luminance.min()<calibration.bg_lums.min()
            or noise_luminance.max()>calibration.bg_lums.max()):
        raise ValueError("Reference noise luminance is outside the measured background LUT")
    reference_background=photometry.luminance_to_pil(
        noise_luminance,calibration.bg_lums,calibration.bg_pixels).convert('RGB')
    # 現在のreference二経路の実表示RGBを線形化し、XYZで光学加算。
    # 背景を前景面の同じ角度範囲へ写像。補間はガンマ符号値でなく線形RGBで行う。
    rw,rh=layout['reference_singleplane'].size
    bg_linear=decode(reference_background,calibration.gamma_bg)
    bg_on_fg=np.stack([
        np.asarray(Image.fromarray(bg_linear[...,i].astype(np.float32)).resize(
            (rw,rh),Image.Resampling.BILINEAR),dtype=float)
        for i in range(3)],axis=-1)
    ref_fg_linear=decode(ref,calibration.gamma_fg)
    fw,fh=ref.size
    if fw>rw or fh>rh:
        raise ValueError("Reference foreground does not fit inside its background")
    x=(rw-fw)//2; y=(rh-fh)//2
    reference_layer=np.zeros((rh,rw,3))
    reference_layer[y:y+fh,x:x+fw]=ref_fg_linear
    reference_xyz=(bg_on_fg @ calibration.t_prime.T
                   + reference_layer @ calibration.r_prime.T)
    reference_linear=reference_xyz @ calibration.r_prime_inv.T
    if not np.isfinite(reference_linear).all():
        raise ValueError("Non-finite reference reproduction RGB")
    reference_gamut=float(np.mean(np.any(
        (reference_linear < -config.gamut_tolerance)
        | (reference_linear > 1+config.gamut_tolerance),axis=-1)))
    reference_composite=encode(reference_linear,calibration.gamma_fg)
    if reference_gamut:
        print(f"WARN: reference reproduction out-of-gamut ratio={reference_gamut:.6f}; clipping linear RGB to [0,1].",flush=True)
    meta={'Task_Mode':'matching','Plane_Mode':spec.plane_mode,'Viewing_Mode':spec.viewing_mode,
          'Reference_Observed_Eye':resolve_observed_eye(spec,dominant_eye),
          'Reproduced_Eye':spec.reproduced_eye,'Defocus_Mode':spec.defocus_mode,
          'Disparity_Mode':spec.disparity_mode,'Disparity_Px':0,
          'Original_FG_Y_cd_m2':original_y,'Reference_Gain':config.reference_gain,
          'Reference_Target_Y_cd_m2':config.reference_gain*original_y,
          'Reference_Quantized_Y_cd_m2':mean_y(decode(ref,calibration.gamma_fg),calibration.r_prime),
          'Luminance_Basis':'calibrated_XYZ_increment_not_black_inclusive_measurement',
          'Reference_Center_Deg':config.reference_center_deg,'Test_Center_Deg':config.test_center_deg,
          'Visual_Angle_Deg':config.visual_angle_deg,
          'Gamut_Policy':'clip_linear_rgb_0_1',
          'Reference_Plane_Mode':'single_matrix_reproduction',
          'Reference_Reproduction_Method':'XYZ_sum_Tprime_BG_plus_Rprime_FG_then_inv_Rprime',
          'Reference_OutOfGamut_Ratio':reference_gamut,
          'Reference_Total_Target_Y_cd_m2':float(np.mean(reference_xyz[...,1])),
          'Reference_Total_Quantized_Y_cd_m2':mean_y(
              decode(reference_composite,calibration.gamma_fg),calibration.r_prime),
          'Reference_BG_Type':'static_bandpass_white_noise',
          'Reference_Noise_Seed':noise_seed,
          'Reference_Noise_Center_Cpd':4.0,
          'Reference_Noise_Bandwidth_Octave':1.0,
          'Reference_Noise_Mean_Y_cd_m2':15.0,
          'Reference_Noise_Modulation':1.0}
    background_xyz=None; physical=None; fg_slice=(slice(None),slice(None))
    if spec.plane_mode=='dual':
        physical=physical_bg.resize(layout['background'].size,RESAMPLE)
    else:
        sp_bg=sp_bg.resize(layout['singleplane'].size,RESAMPLE)
        ppd=corrected.width/config.visual_angle_deg
        if spec.disparity_mode=='simple':
            disparity=right_aligned_disparity_px(ipd_mm=ipd_mm,distance_fg_cm=config.distance_fg_cm,
                        distance_bg_cm=config.distance_bg_cm,pixels_per_degree=ppd)
            right,left=make_constant_eye_views(sp_bg,disparity)
            meta['Disparity_Px']=disparity
        else: right,left=sp_bg.copy(),sp_bg.copy()
        if spec.defocus_mode=='matched_simulation':
            right=_blur_eye_background(right,pupil_mm=right_pupil_mm,config=config,pixels_per_degree=ppd)
            left=_blur_eye_background(left,pupil_mm=left_pupil_mm,config=config,pixels_per_degree=ppd)
        if spec.reproduced_eye=='Left': reproduced=left
        elif spec.reproduced_eye=='Right': reproduced=right
        elif spec.reproduced_eye=='Both':
            reproduced=fuse_eye_views(right,left,dominant_eye=dominant_eye,dominant_weight=config.binocular_fusion_dominant_weight)
        else: reproduced=sp_bg
        background_xyz=decode(reproduced,calibration.gamma_bg) @ calibration.t_prime.T
        bw,bh=reproduced.size; fw,fh=corrected.size
        if fw>bw or fh>bh: raise ValueError("Foreground does not fit inside SP background")
        x=(bw-fw)//2; y=(bh-fh)//2
        layer=np.zeros((bh,bw,3)); layer[y:y+fh,x:x+fw]=fg
        fg=layer; fg_slice=(slice(y,y+fh),slice(x,x+fw))
    stim=MatchingStimulus(fg,background_xyz,physical,
                         reference_composite.transpose(Image.Transpose.FLIP_LEFT_RIGHT),config.reference_gain,
                         calibration,meta,config.gamut_tolerance,fg_slice,0.,
                         reference_background=None)
    # 両端の色域外比率を記録する。色域外でも背景・ゲイン上限は変更せず続行。
    low,high=stim.gamut_ratio(0),stim.gamut_ratio(1)
    meta.update({'Gamut_Ratio_At_Zero':low,'Gamut_Ratio_At_One':high})
    if low or high:
        print(f"WARN: {trial.condition_id}/{trial.pair.pair_id}: out-of-gamut at 0/100% ({low:.6f}/{high:.6f}); clipping linear RGB to [0,1]. Background/gain limits unchanged.",flush=True)
    baseline=encode(stim.raw_test_linear(0),calibration.gamma_fg)
    stim.baseline_display_y=mean_y(decode(baseline,calibration.gamma_fg)[fg_slice],calibration.r_prime)
    # 大きな配列を永続化しない。referenceはこの試行を通して不変。
    return stim
