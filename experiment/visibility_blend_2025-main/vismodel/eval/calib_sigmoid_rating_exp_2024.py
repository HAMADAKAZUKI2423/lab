# rating data (2024)に基づいてsigmoidをcalibする

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
#from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import cv2

import csv
import pathlib
import re
import seaborn as sns
from matplotlib import pyplot as plt
from pylab import rcParams
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
import os
from scipy.stats import spearmanr, pearsonr

import random

import sys
import time


import numpy as np
from scipy import fftpack 
from scipy import stats
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.stats import spearmanr, pearsonr
# from sklearn.metrics import r2_score

load_data = False
show_vismap = False

contrast_energy_weighting = False

show_img_with_rating = False

# outpath = "./results/sigmoid_calib/rating_exp2024/"
# os.makedirs(outpath, exist_ok=True)

func_type = 'generalized_sigmoid'

version = "v4"

use_agg_data = True

width = 384
height = 384

model_name = "MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e5_rep1"#"MLP_3visspatialweightshallow_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_ignoreres"#"MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_lp"#"MLP_3visspatialweightshallow_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"#"MLP_2visspatialweight_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"#"MLP_3visspatialweight_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"#
# model_name = "MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e5_rep1"#"MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"#"MLP_3way_ds1_h2_c64_lp2_do_botelow_relu_zeromask_gb_scale_wd1e3"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e4_wosame2"#"MLP_2visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_wd1e4"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_uni_wd1e4"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_wd1e4"#"org_lp2_precise"#"MLP_3wayTE_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb"#"MLP_3wayTE_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_wd1e4_generalized_sigmoid"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_wd1e4_generalized_sigmoid"#"MLP_3way_ds4_h2_c64_lp2_do_botelow_csig2_zero_wd1e4_rawmaskloss"#"MLP_3way_ds4_h2_c64_lp1_do_bote_csig2_zero"
model_name = "org_lptrain_precise_ds1_wd1e5"#
model_name = "MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_nn_wd1e6"
#model_name = "IQA_cvvdp"
# model_name = "IQA_ciede"
# model_name = "IQA_lpips"
#model_name = "alpha"
model_name = "org_lptrain_precise_ds1_trainscale"

data_path = '/Users/taikifukiage/Documents/Projects/Visibility/rating_exp_2024_dataset/visrating2024_results_agg.csv'
img_path = '/Users/taikifukiage/Documents/Projects/Visibility/rating_exp_2024_dataset/stimuli_source/'

outpath = "results/sigmoid_calib2024/"

current_palette = sns.color_palette(n_colors=4)

out_fname = outpath+model_name+f"/agg_df_md_{version}.csv"# out_path + model_name + f'_response_vs_prediction_data_{version}.csv'

root_dir = pathlib.Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from vismodel.utils import load_vismodel

device_name = "cpu"

if model_name != "alpha":
    model = load_vismodel(model_name, device_name, load_param = True)
    model.visualize_weights(showplot=False)
    print(model.get_params())

if os.path.exists(out_fname) and load_data:
    
    data_df = pd.read_csv(out_fname)

else:

    data_df = pd.read_csv(data_path)

    if not use_agg_data:
        data_df = data_df[data_df['trial_type']=='image-slider-response-visibility']
    data_df['fg_id']=data_df['fg_id'].astype(int)
    data_df['bg_id']=data_df['bg_id'].astype(int)

    # data_df = data_df[data_df['fg_id']>0]
    # data_df = data_df[data_df['bg_id']>4]

    # responseとpredictionの散布図を作成（blend_modeごとに異なる色でプロット）
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_df, x='alpha', y='response', hue='fg_id', alpha=0.5)
    plt.ylabel('Response')
    plt.xlabel('Alpha')
    plt.title('Response vs Alpha')
    plt.grid(True)
    plt.show()

    # load all images
    fg_id_list = data_df['fg_id'].unique()
    bg_id_list = data_df['bg_id'].unique()

    fg_imgs = {}
    bg_imgs = {}

    def np_to_tensor(img):
        _img = np.float32(img.transpose([2,0,1]))/255
        return torch.as_tensor(_img).to(device_name)

    for fg_id in fg_id_list:
        img = cv2.imread(img_path + f'fg_{fg_id}.png')
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        _img = np_to_tensor(resized_img)
        fg_imgs[fg_id] = _img.unsqueeze(0)
    for bg_id in bg_id_list:
        img = cv2.imread(img_path + f'bg_{bg_id}.png')
        resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        _img = np_to_tensor(resized_img)
        bg_imgs[bg_id] = _img.unsqueeze(0)

    ref_maskimg = torch.ones_like(_img[0].unsqueeze(0).unsqueeze(0)).to(device_name)

    result_data = []
    for idx, row in tqdm(data_df.iterrows()):
        fg_id = row['fg_id']
        bg_id = row['bg_id']
        alpha = float(row['alpha'])
        response = float(row['response'])/100

        if model_name != "alpha":

            test_fg = fg_imgs[fg_id]
            test_bg = bg_imgs[bg_id]

            model.set_inputs_tg_ref_alphamap(test_fg, test_bg, ref_maskimg*alpha, ref_maskimg, blend_mode='linear')
            # if model.sigmoid_type.startswith('bezier'):
            #     model.max_vis_score = max_vis_score
            model.compute_weights()
            # model.set_target(fg_mod)
            # 2024/7/11 added これがないとtargetを入れ替えてもblend imageが更新されない
            # model.set_alphamap(ref_maskimg*alpha.view(-1,1,1,1), blend_mode=blend_type)

            if contrast_energy_weighting:
                model.compute_spatial_weights()
            
            model.compute_visibility_wo_weight()
            vis_score_raw = model.vis_score

            if show_vismap:
                if model.spatial_weight is not None:
                    fig, ax = plt.subplots(1,4)
                else:
                    fig, ax = plt.subplots(1,3)
                blend = model.blend
                ax[0].imshow(model.vis_map[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=10)
                ax[1].imshow(model.ref[0].detach().cpu().numpy().transpose([1,2,0])[...,[2,1,0]])
                # vismap_raw = F.avg_pool2d(model.vis_map, 32, stride=1, padding=16)
                # vismap = model.visibility_to_norm(vismap_raw)
                # ax[1].imshow(model.norm_vismap[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=1)
                ax[2].imshow(blend[0].detach().cpu().numpy().transpose([1,2,0])[...,[2,1,0]])
                if model.spatial_weight is not None:
                    ax[3].imshow(model.spatial_weight[0,:,:].detach().cpu().numpy(), vmin=0, vmax=1)
                plt.show()

            # vis_rating = model.norm_score#model.visibility_to_norm(vis_level) * 4 + 1 

            data_dict = {
                'fg_id': fg_id,
                'bg_id': bg_id,
                'alpha': alpha,
                'response': response,
                'prediction': vis_score_raw[0].item()
            }
        else:
            data_dict = {
                'fg_id': fg_id,
                'bg_id': bg_id,
                'alpha': alpha,
                'response': response,
                'prediction': alpha
            }

        result_data.append(data_dict)




    def showimg(tensor):
        #tensor: BCHW
        npimg = tensor.data.cpu().numpy()[0].transpose([1,2,0])

        if npimg.shape[2]==1:
            npimg = npimg[:,:,0]

        plt.imshow(npimg)
        plt.show()

    # DataFrameに収集したデータを追加
    data_df = pd.DataFrame(result_data)

    # データをCSVに保存
    os.makedirs(outpath+model_name+'/', exist_ok=True)
    data_df.to_csv(out_fname, index=False)


if show_img_with_rating:
    
    for idx, trial in data_df.iterrows():
        fg_id = int(trial["fg_id"])
        bg_id = int(trial["bg_id"])
        alpha = float(trial['alpha'])
        response = float(trial['response'])
        fg_img = cv2.imread(img_path + f'fg_{fg_id}.png')
        bg_img = cv2.imread(img_path + f'bg_{bg_id}.png')

        blendimg = alpha * fg_img + (1-alpha)*bg_img
        blendimg = blendimg[:,:,::-1]
        plt.imshow(blendimg/255)
        plt.title(f'Response: {response}')
        plt.show()


# responseとpredictionの散布図を作成（blend_modeごとに異なる色でプロット）
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_df, x='prediction', y='response', hue='fg_id', style='bg_id', alpha=0.5, palette=current_palette)
plt.ylabel('Response')
plt.xlabel('Prediction')
plt.title('Response vs Prediction')
plt.grid(True)
plt.savefig(outpath+model_name+'/pred_vs_resp.png')
plt.show()

current_palette = sns.color_palette(n_colors=7)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_df, x='prediction', y='response', hue='bg_id', style='fg_id', alpha=0.5, palette=current_palette)
plt.ylabel('Response')
plt.xlabel('Prediction')
plt.title('Response vs Prediction')
plt.grid(True)
plt.savefig(outpath+model_name+'/pred_vs_resp_2.png')
plt.show()

# ピアソン相関係数とスピアマン順位相関係数を計算
overall_pearson_corr, _ = pearsonr(data_df['response'], data_df['prediction'])
overall_spearman_corr, _ = spearmanr(data_df['response'], data_df['prediction'])

# 決定係数を計算
overall_mean_response = data_df['response'].mean()
ss_total = ((data_df['response'] - overall_mean_response) ** 2).sum()
ss_residual = ((data_df['response'] - data_df['prediction']) ** 2).sum()
overall_r2 = 1 - (ss_residual / ss_total)

print(f'Overall Pearson Correlation: {overall_pearson_corr:.3f}')
print(f'Overall Spearman Correlation: {overall_spearman_corr:.3f}')
print(f'Overall R^2 (Coefficient of Determination): {overall_r2:.3f}')

# blend_modeごとに相関係数と決定係数を計算
for fg_id in data_df['fg_id'].unique():
    subset = data_df[data_df['fg_id'] == fg_id]
    pearson_corr, _ = pearsonr(subset['response'], subset['prediction'])
    spearman_corr, _ = spearmanr(subset['response'], subset['prediction'])
    mean_response = subset['response'].mean()
    ss_total = ((subset['response'] - mean_response) ** 2).sum()
    ss_residual = ((subset['response'] - subset['prediction']) ** 2).sum()
    r2 = 1 - (ss_residual / ss_total)
    print(f'fg_id: {fg_id}')
    print(f'  Pearson Correlation: {pearson_corr:.3f}')
    print(f'  Spearman Correlation: {spearman_corr:.3f}')
    print(f'  R^2 (Coefficient of Determination): {r2:.3f}')




def custom_sigmoid(vis, sig_shift, sig_scale, min_val=1, max_val=5):
    c = -np.exp(sig_shift) # force negative
    y = c*(1-c) / (c-np.exp( -sig_scale * vis )) + c
    return y*(max_val-min_val)+min_val

def custom_sigmoid_v2(vis, sig_a, sig_b, sig_c, min_val=1, max_val=5):
    # no upper bound but forced to cross the origin
    a = np.log(np.exp(sig_a) + 1) # force_positive
    b = sig_b
    c = -np.log(np.exp(sig_c) + 1) # force negative

    y = (-c*(1+np.exp(b))) / (1+np.exp(-a*vis+b)) + c
    return y*(max_val-min_val)+min_val


# def generalized_sigmoid(X, A, B, v):
#     # X=0, Y= 1
#     # A + (5-A)/((1+Q)**(1/v)) = 1
#     # (5-A)/(1-A) = (1+Q)**(1/v)
#     # 1+Q = ((5-A)/(1-A))**v
#     print(f"A: {A}")
#     print(f"v: {v}")
#     Q = ((5-A)/(1-A))**v - 1
    
#     Y = A + (5-A)/((1+Q*np.exp(-B*X))**(1/v))
#     return Y

def generalized_sigmoid(X, param_A, param_B, param_v, min_val=0, max_val=1):
    # X=0, Y= 0
    # A + (5-A)/((1+Q)**(1/v)) = 0
    # (5-A)/(-A) = (1+Q)**(1/v)
    # 1+Q = ((5-A)/(-A))**v
    A = np.log(np.exp(param_A) + 1) # force_positive
    B = np.log(np.exp(param_B) + 1) # force_positive
    v = np.log(np.exp(param_v) + 1) # force_positive
    print(f"A: {A}")
    print(f"v: {v}")
    Q = ((1+A)/A)**v - 1
    
    Y = -A + (1+A)/((1+Q*np.exp(-B*X))**(1/v))
    return Y*(max_val-min_val)+min_val

def save_scatter(target_vis_val, model_vis_val, path):
    plt.scatter(target_vis_val,model_vis_val)
    plt.xlabel("Experiment Visibility Value")
    plt.ylabel("Model Visibility Value")
    plt.savefig(path+'visibility_scatter.png')
    #plt.show()
    plt.clf()

from scipy.optimize import curve_fit

def fitting(vis_scores, vis_val_stack, path, name = '', type='bezier'):

    vis_val = vis_val_stack[:,0]
    # vis_max_val = vis_val_stack[:,1]

    if type == 'custom_sigmoid':
        popt =  [1.0,1.0]
        popt, pcov = curve_fit(custom_sigmoid, vis_val, vis_scores, p0=popt)
        print(popt)

        pred = custom_sigmoid(vis_val, *popt)

    elif type=='custom_sigmoid_v2':
        popt =  [0.0,0.0,0]
        popt, pcov = curve_fit(custom_sigmoid_v2, vis_val, vis_scores, p0=popt)
        print(popt)

        pred = custom_sigmoid_v2(vis_val, *popt)
    elif type=='generalized_sigmoid':
        
        popt = [-0.06604367,  2.58078744, 22.72348767]#[1,1,1]#[0.0,0.5,1]#[0.0,0.5,1]
        popt, pcov = curve_fit(generalized_sigmoid, vis_val, vis_scores, p0=popt)
        print(popt)

        pred = generalized_sigmoid(vis_val, *popt)
    

    rmse = (((pred-vis_scores)**2).mean())**0.5
    print(rmse)
    print()

    plt.scatter(pred, vis_scores)
    plt.xlabel('Prediction')
    plt.ylabel('Subjective Score')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(path+'sigmoid_'+name+type+'_prediction.png')
    plt.show()
    plt.clf()

    # pearson and spearman correlation
    from scipy.stats import pearsonr, spearmanr
    pcorr, _ = pearsonr(vis_scores, pred)
    print('Pearsons correlation: %.3f' % pcorr)

    scorr, _ = spearmanr(vis_scores, pred)
    print('Spearmans correlation: %.3f' % scorr)

    rpcorr, _ = pearsonr(vis_scores, vis_val)
    print('raw Pearsons correlation: %.3f' % rpcorr)

    rscorr, _ = spearmanr(vis_scores, vis_val)
    print('raw Spearmans correlation: %.3f' % rscorr)

    
    xdata = np.linspace(0, np.max(vis_val), 100)
    if type == 'custom_sigmoid':
        plt.plot(vis_val, vis_scores, 'o', label='data')
        plt.plot(xdata, custom_sigmoid(xdata, *popt), 'r-',label='fit')
    elif type=='custom_sigmoid_v2':
        plt.plot(vis_val, vis_scores, 'o', label='data')
        plt.plot(xdata, custom_sigmoid_v2(xdata, *popt), 'r-',label='fit')
    elif type=='generalized_sigmoid':
        plt.plot(vis_val, vis_scores, 'o', label='data')
        plt.plot(xdata, generalized_sigmoid(xdata, *popt), 'r-',label='fit')
    
        # plt.plot(xdata, bezier_curve(np.column_stack((xdata, np.ones_like(xdata) * max_val)), *popt), 'r-', label='fit')

    plt.xlabel("visibility Value")
    plt.ylabel("Experiment Scores")
    plt.savefig(path+'sigmoid_'+name+type+'_fitting.png')
    plt.show()
    plt.clf()

    return popt, rmse, pcorr, scorr, rpcorr, rscorr

if model_name != "alpha":
    # vismodels_path = "./mlp/vismodels/"
    # model = load_vismodel(model_name, vismodels_path, device_name)
    model.eval()

model_vis_val = data_df['prediction'].values
vis_scores = data_df['response'].values


vis_stack = np.stack([model_vis_val], axis=1)


print(f"fitting: {model_name}")
popt, rmse, pcorr, scorr, rpcorr, rscorr = fitting(vis_scores, vis_stack, outpath+model_name+'/', type=func_type)
f = open(outpath+model_name+'/'+'result.txt', 'w')
f.write(f"fitting: {model_name}\n")
f.write(f"popt: {popt}\n")
f.write(f"rmse: {rmse}\n")
f.write(f"pcorr: {pcorr}\n")
f.write(f"scorr: {scorr}\n")
f.write(f"raw pcorr: {rpcorr}\n")
f.write(f"raw scorr: {rscorr}\n")
f.close()

import json

vismodels_path = "/Users/taikifukiage/Documents/Projects/Visibility/Visibility_Blending_Kobayashi_fork/vismodel/"

vismodel_presets = open(f"{vismodels_path}vismodel_configs.json", 'r')
vismodel_presets = json.load(vismodel_presets)

if model_name == "alpha":
    vismodel_presets = {}
else:
    vismodel_presets = vismodel_presets[model_name]

    if model.sigmoid_type=='linear':
        del model.vis_slope


    # modelを保存
    new_model_name = model_name+'_'+func_type+'.pth'
    torch.save(model.state_dict(), outpath+model_name+'/' + new_model_name)
    vismodel_presets['path'] = 'sigfit2024/'+new_model_name

vismodel_presets['sigmoid_type'] = func_type
vismodel_presets['sigmoid_param'] = popt.tolist()  # Convert numpy array to list to make it JSON serializable

# save vismodel_presets as json
with open(outpath+model_name+'/'+"vismodel_info_"+func_type+".json", 'w') as f:
    json.dump(vismodel_presets, f, indent=4)
