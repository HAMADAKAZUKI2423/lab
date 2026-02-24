import csv
import numpy as np
import pathlib
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import spearmanr, pearsonr, norm
# from util_calib import recalc_visibility
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from vismodel.calib.utils import recalc_visibility
# from ..calib.utils import recalc_visibility
import torch
# from util_blend_mlp import load_vismodel
from vismodel.utils import load_vismodel

import pandas as pd
#import pyiqa



fit_vis = True


path_to_data = '/home/Dataset/TAP_usertest1_data/aggregated_data.csv'
# path_to_data = 'vismodel/eval/aggregated_data_userstudy2021.csv'
stimulus_path = '/home/Dataset/TAP_usertest1_data/stimuli/'


device_name = 'cuda:0'
if not torch.cuda.is_available():
    device_name = "cpu"


force_compute_vis = False
calib_data = 'alpha'#'global_sd3_int_cf_0001_ave_sm1'#"local_sd3_int_cf_0001_ave_sm1"#
func_type = 'generalized_sigmoid'#'custom_sigmoid'#'custom_sigmoid_v2'#'bezier_v2'#
models_list = ["MLP_3visfusion_ds4_h2_c64_lp1_do_botelow_aa_relu_zeromask_gb_scale_wd1e5"]#["MLP_3visspatialweightshallow_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"]
#["MLP_3visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e4_rep4"]#["MLP_2visfusion_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale"]#["MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e4_wosame2"]#["org_lptrain_precise_ds1_wd1e5"]#["IQA_cvvdp"]#["IQA_ciede2000"]#["IQA_ciede"]#["IQA_lpips"]#["IQA_dists"]#["IQA_nlpd"]#["MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_scale_wd1e4"]#["org_lp2_precise"]#["MLP_3wayTE_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb"]#["MLP_3way_ds4_h2_c64_lp2_do_botelow_aa_relu_zeromask_gb_wd1e4"]#["MLP_3way_ds4_h2_c64_lp2_do_botelow_csig2_zero_wd1e4_rawmaskloss"]
load_param = True

init_param = [-0.6051653548036195,
        1.8023380888688512,
        14.83366136940297]
# [-0.06604367,  2.58078744, 22.72348767]#[1,1,1]#[0.0,0.5,1]

outpath = "results/sigmoid_calib/" + calib_data + "/"
os.makedirs(outpath, exist_ok=True)


agg_df = pd.read_csv(path_to_data)


def generalized_sigmoid(X, param_A, param_B, param_v, min_val=1, max_val=5):
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

def fitting(vis_scores, vis_val_stack, path, name = '', type='bezier'):

    vis_val = vis_val_stack[:,0]
    # vis_max_val = vis_val_stack[:,1]

    if type=='generalized_sigmoid':
        
        popt =  init_param
        popt, pcov = curve_fit(generalized_sigmoid, vis_val, vis_scores, p0=popt)
        print(popt)

        pred = generalized_sigmoid(vis_val, *popt)
    

    rmse = (((pred-vis_scores)**2).mean())**0.5
    print(rmse)
    print()

    plt.scatter(pred, vis_scores)
    plt.xlabel('Prediction')
    plt.ylabel('Subjective Score')
    plt.xlim(1, 5)
    plt.ylim(1, 5)
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
    if type=='generalized_sigmoid':
        plt.plot(vis_val, vis_scores, 'o', label='data')
        plt.plot(xdata, generalized_sigmoid(xdata, *popt), 'r-',label='fit')
    
    plt.xlabel("visibility Value")
    plt.ylabel("Experiment Scores")
    plt.savefig(path+'sigmoid_'+name+type+'_fitting.png')
    plt.show()
    plt.clf()

    return popt, rmse, pcorr, scorr, rpcorr, rscorr

if fit_vis:

    # agg_df_md = agg_df[agg_df.model=='local_sd3_int_cf_0001_ave_sm1']
    agg_df_md = agg_df[agg_df.model==calib_data]
    print(agg_df_md.columns)

    from scipy.optimize import curve_fit

    # if compare_wocalib:
    #     model_wocalib = load_vismodel("Fukiage_2023", "./vismodels/", device_name)
    #     model_wocalib.eval()
    #     model_wocalib_vis_val, vis_scores, target_vis_val = recalc_visibility(model_wocalib, agg_df_md, stimulus_path, device_name)
    #     save_scatter(target_vis_val, model_wocalib_vis_val, outpath)
    #     popt, rmse, pcorr, scorr, rpcorr, rscorr = fitting(vis_scores, target_vis_val, outpath)
    #     wocalib_popt, wocalib_rmse, wocalib_pcorr, wocalib_scorr, wocalib_rpcorr, wocalib_rscorr = fitting(vis_scores, model_wocalib_vis_val, outpath, name = 'wocalib_')
    #     f = open(outpath+'result.txt', 'w')
    #     f.write(f"fitting: expData\n")
    #     f.write(f"popt: {popt}\n")
    #     f.write(f"rmse: {rmse}\n\n")
    #     f.write(f"fitting: woCalib Model\n")
    #     f.write(f"popt: {wocalib_popt}\n")
    #     f.write(f"rmse: {wocalib_rmse}\n")
    #     f.write(f"pcorr: {wocalib_pcorr}\n")
    #     f.write(f"scorr: {wocalib_scorr}\n")
    #     f.close()

    for model_name in models_list:

        # vismodels_path = "./mlp/vismodels/"
        model = load_vismodel(model_name, device_name, load_param)
        model.eval()

        if os.path.exists(outpath+model_name+'/'+'agg_df_md.csv') and not force_compute_vis:
            
            agg_df_md = pd.read_csv(outpath+model_name+'/'+'agg_df_md.csv')
            model_vis_val = agg_df_md['model_vis_val'].values
            vis_scores = agg_df_md['vis_scores'].values
            target_vis_val = agg_df_md['target_vis_val'].values
            # max_vis_val = agg_df_md['max_vis_val'].values
        else:
            os.makedirs(outpath+model_name+'/', exist_ok=True)
            
            if False:# model.sigmoid_type == 'bezier':
                model_vis_val, vis_scores, target_vis_val, max_vis_val = recalc_visibility(model, agg_df_md, stimulus_path, device_name, show_img=True)      
                agg_df_md = pd.DataFrame({'model_vis_val': model_vis_val, 'vis_scores': vis_scores, 'target_vis_val': target_vis_val, 'max_vis_val':max_vis_val})

            else:
                model_vis_val, vis_scores, target_vis_val = recalc_visibility(model, agg_df_md, stimulus_path, device_name, show_img=True)      
                agg_df_md = pd.DataFrame({'model_vis_val': model_vis_val, 'vis_scores': vis_scores, 'target_vis_val': target_vis_val})

            # dataframeを保存
            agg_df_md.to_csv(outpath+model_name+'/'+'agg_df_md.csv', index=False)

        # model_vis_valにinfやnanがある場合は，そのデータを削除し，そのインデックスを取得
        nan_index = np.where(np.isnan(model_vis_val))[0]
        inf_index = np.where(np.isinf(model_vis_val))[0]
        naninf_index = np.unique(np.concatenate([nan_index, inf_index]))
        model_vis_val = np.delete(model_vis_val, naninf_index)
        vis_scores = np.delete(vis_scores, naninf_index)
        target_vis_val = np.delete(target_vis_val, naninf_index)
        # max_vis_val = np.delete(max_vis_val, naninf_index)

        vis_stack = np.stack([model_vis_val], axis=1)
        
        # if compare_wocalib:
        #     plt.scatter(model_wocalib_vis_val,model_vis_val)
        #     plt.xlabel("Initial Model Visibility Value")
        #     plt.ylabel("Calibrate New Model Visibility Value")
        #     plt.savefig(outpath+model_name+'/'+'visibility_difference_scatter.png')
        #     #plt.show()
        #     plt.clf()

        save_scatter(target_vis_val, model_vis_val, outpath+model_name+'/')
        
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

        vismodel_presets = open(f"vismodel/vismodel_configs.json", 'r')
        vismodel_presets = json.load(vismodel_presets)
        vismodel_presets = vismodel_presets[model_name]

        if model.sigmoid_type=='linear':
            del model.vis_slope
        
        
        # modelを保存
        new_model_name = model_name+'_'+func_type+'.pth'
        torch.save(model.state_dict(), outpath+model_name+'/' + new_model_name)
        
        vismodel_presets['sigmoid_type'] = func_type
        vismodel_presets['sigmoid_param'] = popt.tolist()  # Convert numpy array to list to make it JSON serializable
        vismodel_presets['path'] = 'sigfit/'+new_model_name
        # save vismodel_presets as json
        with open(outpath+model_name+'/'+"vismodel_info_"+func_type+".json", 'w') as f:
            json.dump(vismodel_presets, f, indent=4)

    
    