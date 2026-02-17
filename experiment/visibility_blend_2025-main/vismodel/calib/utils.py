from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import detect_anomaly
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import numpy as np
import sys
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import cv2
import os
from tqdm import tqdm
from vismodel.supermodels.visModel import VisModel
from vismodel.vismodel_mlp import VisModel_MLP
from .data import VisDataset

debug_mode = False

#%% Loss function
def dataload_deivce(dataset, device: torch.device):
    dataset['ref_fg'] = dataset['ref_fg'].to(device)
    dataset['ref_bg'] = dataset['ref_bg'].to(device)
    dataset['test_fg'] = dataset['test_fg'].to(device)
    dataset['test_bg'] = dataset['test_bg'].to(device)
    dataset['ref_mask'] = dataset['ref_mask'].to(device)
    dataset['test_mask'] = dataset['test_mask'].to(device)
    dataset['ref_alpha'] = dataset['ref_alpha'].to(device)
    dataset['test_alpha'] = dataset['test_alpha'].to(device)
    #batch_dict['response_std'] = torch.cat(batch_dict['response_std'],dim=0)#.to(device=cuda)
    #batch_dict['weight'] = torch.cat(batch_dict['weight'],dim=0)#.to(device=cuda)
    #batch_dict['dist'] = torch.cat(batch_dict['dist'],dim=0)#.to(device=cuda)

def scale_transform(dataset):
    # random data augmentation
    if np.random.rand() > 0.5:
        ref_size = 256 + 32 * np.random.randint(-2,3)
        test_size = 256 + 32 * np.random.randint(-2,3)
        dataset['ref_fg'] = F.interpolate(dataset['ref_fg'], size=(ref_size, ref_size), mode='bilinear')
        dataset['ref_bg'] = F.interpolate(dataset['ref_bg'], size=(ref_size, ref_size), mode='bilinear')
        dataset['ref_mask'] = F.interpolate(dataset['ref_mask'], size=(ref_size, ref_size), mode='bilinear')

        dataset['test_fg'] = F.interpolate(dataset['test_fg'], size=(test_size, test_size), mode='bilinear')
        dataset['test_bg'] = F.interpolate(dataset['test_bg'], size=(test_size, test_size), mode='bilinear')
        dataset['test_mask'] = F.interpolate(dataset['test_mask'], size=(test_size, test_size), mode='bilinear')

def maskloss(model, use_dispersion_loss, use_mask_loss):
    _mask_data = model.dilated_mask_data

    index_mask = _mask_data < 1e-3
    binary_mask = torch.zeros_like(_mask_data)
    binary_mask[index_mask] = 1

    mask_loss = 0.0

    mask_loss_map = torch.sum(torch.abs(model.vis_map * binary_mask), dim=(1,2,3))/torch.sum(binary_mask, dim=(1,2,3))
    
    if use_mask_loss:
        mask_loss += mask_loss_map.mean()

        # mask_loss = mask_loss * model.mask_loss_weight

    if use_dispersion_loss:
        inv_mask_loss_map = torch.sum(torch.abs(model.vis_map * (1-binary_mask)), dim=(1,2,3))/torch.sum((1-binary_mask), dim=(1,2,3))
        diff = torch.abs(mask_loss_map - inv_mask_loss_map)
        diff_loss = torch.where(diff < 0.1, torch.exp(-10.0*diff)*2, torch.zeros_like(diff))

        mask_loss += diff_loss.mean()#* model.mask_loss_weight
    
    print("mask loss", mask_loss.item())
    
    return mask_loss * model.mask_loss_weight

def compute_loss(model: VisModel,
                dataset: dict[str,torch.Tensor],
                show_data: bool = False,
                result_df: pd.DataFrame | None = None,
                opt_func: str ='mle',
                precise_mode: bool = False,
                device: torch.device | None =None,
                use_ref_weight: bool = False,
                use_dispersion_loss: bool = False,
                use_mask_loss: bool = False,
                use_zero_loss: bool = False,
                ) -> torch.Tensor:
    
    stable_mode = True

    ref_target = dataset['ref_fg'].clone()
    ref_ref = dataset['ref_bg'].clone()
    test_target = dataset['test_fg'].clone()
    test_ref = dataset['test_bg'].clone()
    ref_alpha =  dataset['ref_mask'] * dataset['ref_alpha'].view(-1,1,1,1)
    test_alpha =  dataset['test_mask'] * dataset['test_alpha'].view(-1,1,1,1)

    # if model.running_std and model.training:
    #     #model.compute_std_running([dataset['test_bg'].clone(), dataset['test_fg'].clone()])
    #     model.compute_std_running([dataset['test_bg']])

    # reference image
    model.set_inputs_tg_ref_alphamap(ref_target, ref_ref, ref_alpha, dataset['ref_mask'])
    model.compute_visibility()
    ref_vis = model.vis_score

    mask_loss = 0.0
    if (use_dispersion_loss or use_mask_loss) and model.mask_loss_weight > 0:
        ref_mask_loss = maskloss(model, use_dispersion_loss, use_mask_loss)
        mask_loss += ref_mask_loss

        # use_zero_loss = True#とりあえずmask lossと併用

    if use_zero_loss and model.mask_loss_weight > 0:
        model.set_inputs_tg_ref_alphamap(ref_target, ref_ref, ref_alpha*0, dataset['ref_mask'])
        model.compute_visibility()
        zero_loss = torch.abs(model.raw_score).mean()
        print("zero loss", zero_loss.item())
        mask_loss += zero_loss * model.mask_loss_weight
    
    
        model.set_inputs_tg_ref_alphamap(test_target, test_ref, test_alpha*0, dataset['test_mask'])
        model.compute_visibility()
        zero_loss = torch.abs(model.raw_score).mean()
        print("zero loss", zero_loss.item())
        mask_loss += zero_loss * model.mask_loss_weight
        # mask_loss += torch.abs(model.raw_score).mean() * model.mask_loss_weight

    #マスクのgaussian pyramid生成処理
    # model.generate_maskPyr(dataset['ref_mask'])
    

    # if model.use_upperbound:
    #     #ref_stimulus, ref_fgcomp = model.generate_stimulus_batch(dataset['ref_fg'].clone(),dataset['ref_bg'].clone(), torch.ones_like(dataset['ref_alpha']),maskimg=dataset['ref_mask'], get_clean_fg=False)
    #     if model.clean_bound:
    #         #model.compute_vis_vector(torch.ones_like(dataset['ref_bg'])*0.5, dataset['ref_fg'].clone(), dataset['ref_mask'] )
    #         model.compute_vis_vector(torch.ones_like(dataset['ref_bg']), dataset['ref_fg'].clone(), torch.ones_like(dataset['ref_mask']) )
    #     else:
    #         model.compute_vis_vector(dataset['ref_bg'].clone(), dataset['ref_fg'].clone(), dataset['ref_mask'] )
    #     #グレー背景で計算する場合は以下
    #     #model.compute_vis_vector(torch.ones_like(dataset['ref_bg'])*0.5, dataset['ref_fg'].clone(), dataset['ref_mask'] )

    # #ref_stimulus, ref_fgcomp = model.generate_stimulus_batch(dataset['ref_fg'].clone(),dataset['ref_bg'].clone(), dataset['ref_alpha'],maskimg=dataset['ref_mask'])
    # ref_vis = model.compare(dataset['ref_bg'].clone(), dataset['ref_fg'].clone(), dataset['ref_mask'] * dataset['ref_alpha'].view(-1,1,1,1))
    # #ref_vis = model.compare(dataset['ref'], dataset['ref_bg'], dataset['ref_fg'], dataset['ref_DC'])
    
    # test image

    #マスクのgaussian pyramid生成処理
    model.set_inputs_tg_ref_alphamap(test_target, test_ref, test_alpha, dataset['test_mask'])
    model.compute_visibility()
    test_vis = model.vis_score

    if (use_dispersion_loss or use_mask_loss) and model.mask_loss_weight > 0:
        test_mask_loss = maskloss(model, use_dispersion_loss, use_mask_loss)
        mask_loss += test_mask_loss

    # if (not use_ref_weight) and (dataset['condition'][0]=='same2'):# 参照画像と共通の重みを使う場合はここをスキップする
        
    #     if model.use_upperbound:
    #         if model.clean_bound:
    #             #model.compute_vis_vector(torch.ones_like(dataset['test_bg'])*0.5, dataset['test_fg'].clone(), dataset['test_mask'] )
    #             model.compute_vis_vector(torch.ones_like(dataset['test_bg']), dataset['test_fg'].clone(), torch.ones_like(dataset['test_mask']) )
    #         else:
    #             #test_stimulus, test_fgcomp = model.generate_stimulus_batch(dataset['test_fg'].clone(),dataset['test_bg'].clone(), torch.ones_like(dataset['test_alpha']),maskimg=dataset['test_mask'], get_clean_fg=False)
    #             model.compute_vis_vector(dataset['test_bg'].clone(), dataset['test_fg'].clone(), dataset['test_mask'] )
    #         #グレー背景で計算する場合は以下
    #         #model.compute_vis_vector(torch.ones_like(dataset['test_bg'])*0.5, dataset['test_fg'].clone(), dataset['test_mask'] )
    
    #test_stimulus, test_fgcomp = model.generate_stimulus_batch(dataset['test_fg'].clone(),dataset['test_bg'].clone(), dataset['test_alpha'],maskimg=dataset['test_mask'])
    # test_vis = model.compare(dataset['test_bg'].clone(), dataset['test_fg'].clone(), dataset['test_mask'] * dataset['test_alpha'].view(-1,1,1,1))
    
    if opt_func == 'mle':
        
        if precise_mode:
            num_negatives_per_example = 33
        else:
            num_negatives_per_example = 20#12#25#15
    
        #num_negative_per_example = 4#subbatch数が3の場合、合計num_negative_per_subbatch*3個のsampleを評価可能
        
        #評価数が少ない場合、最初にbisection searchで最適化付近を探索しておいた方がよいかも

        positive_loss = torch.mean((test_vis - ref_vis)**2.)
        subbatch_idx = np.ones((ref_target.shape[0]),dtype=np.int32)*-1
        num_example_list = []
        
        #compute negative loss
        test_stack = []
        test_stack.append(test_vis)

        for i in range(ref_target.shape[0]):
            #current_idx = dataset['unique_index'][i]
            if subbatch_idx[i]<0:
                subbatch_idx[i] = i
            num_subbatch = 1
            #sub_stack = []
            for j in range(ref_target.shape[0]):
                if dataset['ref_fg_label'][i]==dataset['ref_fg_label'][j] and dataset['test_fg_label'][i]==dataset['test_fg_label'][j] and dataset['test_bg_label'][i]==dataset['test_bg_label'][j]:
                    if dataset['unique_index'][i] != dataset['unique_index'][j]:
                        if subbatch_idx[j]<0:
                            subbatch_idx[j] = i
                        num_subbatch += 1
                        #negative example
                        #(ref_vis-test_vis(neg))^2 - [logit(pos)-logit(neg)]^2
                        #sub_stack.append(test_vis[j])

            num_example_list.append(num_subbatch)
            #negative_stack.append(torch.stack(sub_stack))
        
        #negative_stack = torch.stack(negative_stack, dim=0)    #[B,negative sample]
        
        #ここから先、同一subbatch内のexmapleは並んで格納されているという前提で書いている(subbatchはshuffleされないので正しいはず)
        alpha_list = np.arange(0.0,1.01,0.01)
        alpha_batch_list = []
        for i in np.unique(subbatch_idx):
            num_example = num_example_list[i]

            #すでに評価済みのalpha値を除外(めんどうなので省略)
            #tmp_alpha_list = alpha_list[alpha_list!=dataset['test_alpha'][i]]
            #for idx in range(i+1,i+num_example):
            #    evaluated_alpha = dataset['test_alpha'][idx]
            #    tmp_alpha_list = tmp_alpha_list[tmp_alpha_list!=evaluated_alpha]
            num_negatives = num_negatives_per_example * num_example
            samples = torch.tensor(np.random.choice(alpha_list,size=num_negatives,replace=False),dtype=torch.float32,device=device)
            
            #num_negatives_per_example = num_negatives//num_example
            
            for j in range(num_example):
                alpha_batch_list.append(samples[j*num_negatives_per_example:(j+1)*num_negatives_per_example])
        alpha_batch = torch.stack(alpha_batch_list,dim=0)#[B,num_negatives_per_example]


        for i in range(num_negatives_per_example):
            #test_stimulus, test_fgcomp = model.generate_stimulus_batch(dataset['test_fg'].clone(),dataset['test_bg'].clone(), alpha_batch[:,i],maskimg=dataset['test_mask'])
            #test_vis_tmp = model.compare(test_stimulus, test_fgcomp)
            # test_vis_tmp = model.compare(dataset['test_bg'].clone(), dataset['test_fg'].clone(), dataset['test_mask'] * alpha_batch[:,i].view(-1,1,1,1))

            model.set_alphamap(dataset['test_mask'] * alpha_batch[:,i].view(-1,1,1,1))
            # model.set_inputs_tg_ref_alphamap(test_target, test_ref,  dataset['test_mask'] * alpha_batch[:,i].view(-1,1,1,1), dataset['test_mask'])
            model.compute_visibility_wo_weight()
            test_vis_tmp = model.vis_score

            test_stack.append(test_vis_tmp)
            #negative_stack = torch.cat([negative_stack,test_vis.unsqueeze(1)],dim=1)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #test_stackにexample毎に1(response) + num_negatives_per_example分の評価結果
        test_stack = torch.stack(test_stack,dim=1)#[B,example]

        #example毎の全てのtest評価データを集める(最後はbatchごとまとめて足すから、こんなこと無駄では？->ref_visはexample毎に異なるので無駄ではない!)
        negative_stack = []
        num_negative_list = []
        for i in range(ref_target.shape[0]):
            sub_stack = []
            sub_stack.append(test_stack[i,:])
            for j in range(ref_target.shape[0]):
                if dataset['ref_fg_label'][i]==dataset['ref_fg_label'][j] and dataset['test_fg_label'][i]==dataset['test_fg_label'][j] and dataset['test_bg_label'][i]==dataset['test_bg_label'][j]:
                    if dataset['unique_index'][i] != dataset['unique_index'][j]:
                        #negative example
                        sub_stack.append(test_stack[j,:])

            num_example_list.append(num_subbatch)
            negative_stack.append(torch.cat(sub_stack,dim=0))
            num_negative_list.append(negative_stack[-1].shape[0])

        
        #conditionでsubbatchのexample数が異なるとここでエラーが生じるはず
        #paddingして長さを揃えないといけない
        negative_stack = torch.stack(negative_stack,dim=0)
        #test_stack = torch.cat([test_stack,negative_stack],dim=1)
        num_negative_tensor = torch.tensor(num_negative_list,dtype=torch.float32,device=device)
        #print(num_negative_list)

        if stable_mode:
            if debug_mode:
                if torch.isnan(negative_stack).any():
                    print("nan detected in negative stack")
                    print(negative_stack)
                if torch.isnan(ref_vis).any():
                    print("nan detected in ref")
                    print(ref_vis)
                if torch.isinf(negative_stack).any():
                    print("inf detected in negative stack")
                    print(negative_stack)
                if torch.isinf(ref_vis).any():
                    print("inf detected in ref")
                    print(ref_vis)
                if torch.isinf((ref_vis.unsqueeze(1)-negative_stack)**2.0).any():
                    print("inf detected in ref-negative_stack")
                    print(ref_vis)
                    print(negative_stack)

            logsumexp = torch.logsumexp(-(ref_vis.unsqueeze(1)-negative_stack)**2.0,dim=1) + torch.log(101.0/num_negative_tensor)
            loss = positive_loss + torch.mean(logsumexp)
            #log(c*e^(A)+c*e^(B))=log(c*(e^A+e^B))=log(c)+log(e^A+e^B)
        else:
            negative_loss = torch.sum(torch.exp(-(ref_vis.unsqueeze(1)-negative_stack)**2.0)*(101.0/num_negative_tensor.unsqueeze(1)),dim=1)
            #print(ref_vis.data, negative_loss)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()

            loss = positive_loss + torch.mean(torch.log(negative_loss))

    elif opt_func == 'mle_precise':

        positive_loss = torch.mean((test_vis - ref_vis)**2.)
        subbatch_idx = np.ones((ref_target.shape[0]),dtype=np.int32)*-1
        num_example_list = []

        alpha_list = list(range(101))
        subloss_list = []
        
        for alpha in alpha_list:

            #pre = torch.cuda.memory_allocated(device=cuda)/1024/1024
            #print("current allocated memory:", pre)

            alpha_batch = torch.ones((ref_target.shape[0]),dtype=torch.float32,device=device) * (alpha/100.)
                
            #test_vis_tmp = model.compare(dataset['test_bg'].clone(), dataset['test_fg'].clone(), dataset['test_mask'] * alpha_batch.view(-1,1,1,1))

            # model.set_inputs_tg_ref_alphamap(test_target, test_ref,  dataset['test_mask'] * alpha_batch.view(-1,1,1,1), dataset['test_mask'])
            # model.compute_visibility()
            model.set_alphamap(dataset['test_mask'] * alpha_batch.view(-1,1,1,1))
            # model.set_inputs_tg_ref_alphamap(test_target, test_ref,  dataset['test_mask'] * alpha_batch[:,i].view(-1,1,1,1), dataset['test_mask'])
            model.compute_visibility_wo_weight()
            test_vis_tmp = model.vis_score

            #test_stack.append(test_vis_tmp)
            
            subloss = (test_vis_tmp - ref_vis)**2.
            subloss_list.append(subloss)
        
        subloss_tensor = torch.stack(subloss_list,1)#[batch,alpha]
        exp_tensor = torch.exp(-subloss_tensor)
        sum_exp_tensor = torch.sum(exp_tensor,dim=1,keepdim=True)
        prob_tensor = exp_tensor/sum_exp_tensor

        alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32, device=device)/100.0
        expected_val = (alpha_tensor.view(1,-1)*prob_tensor).sum(dim=1)

        loss = positive_loss + torch.mean(torch.log(sum_exp_tensor))

    if use_mask_loss or use_zero_loss:
        loss += mask_loss

    if show_data:
        ref_vis_np = ref_vis.cpu().detach().numpy()
        test_vis_np = test_vis.cpu().detach().numpy()

        # for i in range(test_vis_np.shape[0]):
        #     if test_vis_np[i] > 9.0 or ref_vis_np[i] > 9.0:
        #         print('ref_vis',ref_vis_np[i])
        #         print('test_vis',test_vis_np[i])
        #         print('condition',dataset["condition"][i])
        #         print('unique_index',dataset["unique_index"][i])
        #         print('ref_fg',dataset["ref_fg_label"][i])
        #         print('ref_bg',dataset["ref_bg_label"][i])
        #         print('test_fg',dataset["test_fg_label"][i])
        #         print('test_bg',dataset["test_bg_label"][i])
        #         print('ref_alpha',dataset['ref_alpha'][i])
        #         print('test_alpha',dataset['test_alpha'][i])

        tmp_df = pd.DataFrame()#columns=['ref_val', 'test_val', 'target_img'])
        tmp_df['unique_index']=dataset["unique_index"]
        tmp_df['ref_val']=ref_vis_np
        tmp_df['test_val']=test_vis_np
        tmp_df['ref foreground']=dataset["ref_fg_label"]
        tmp_df['ref background']=dataset["ref_bg_label"]
        tmp_df['test foreground']=dataset["test_fg_label"]
        tmp_df['test background']=dataset["test_bg_label"]
        tmp_df['condition']=dataset["condition"]
        tmp_df['weight']=dataset["weight"].cpu().detach().numpy()

        if opt_func == 'mle_precise':
            alpha_batch_np = expected_val.cpu().detach().numpy()
            response_np = dataset['test_alpha'].cpu().detach().numpy()
            tmp_df['prediction']=alpha_batch_np
            tmp_df['response']=response_np
            ref_alpha_np = dataset['ref_alpha'].cpu().detach().numpy()
            tmp_df['ref_alpha']= ref_alpha_np
        #tmp_df = tmp_df.sort_values('target_img')
        #print(result_df[0:10])
        #result_df = result_df.append(tmp_df)
        result_df = pd.concat([result_df,tmp_df])
        
        #ax.scatter(ref_vis.data.cpu().numpy(),test_vis.data.cpu().numpy())
        
        #plt.show()
        return loss, result_df#いちいち返して代入してあげないとdfは消失してしまう
    else:
        

        return loss#1.-p_corr

def compute_all(model, dataloader_dict, label_class = 'condition', opt_func = 'mle', precise_mode=False, out_path = None,device=None,datatype='', use_ref_weight = False):

    condition_list = list(dataloader_dict.keys())
    print(condition_list)

    # if out_path is None:
    #     out_path = path_to_plot

    model.eval()

    num_count = 0
    datasetsize = 0
    for dataloader in dataloader_dict.values():
        datasetsize += len(dataloader)

    results_df = pd.DataFrame()
    s_loss = 0.

    loss_dict = {}
    for key, dataloader in dataloader_dict.items():
        s_loss_cond = 0.
        num_count_cond = 0
        for batch_index, dataset in enumerate(dataloader):
            dataload_deivce(dataset, device)
            #print("currently allocated memory (MB):",torch.cuda.memory_allocated(device=cuda)/1024/1024)
            loss, results_df = compute_loss(model, dataset, show_data = True, result_df = results_df, opt_func = opt_func, precise_mode=precise_mode, device=device, use_ref_weight = use_ref_weight)
            #loss = compute_loss(dataset)#.cpu().numpy()
            #s_loss+=loss.item()
            s_loss_cond+=loss.item()

            num_count += 1
            num_count_cond += 1

            print(100*num_count/datasetsize,"percent of data completed.")
        
        loss_dict[key]=s_loss_cond/num_count_cond
        print(key, loss_dict[key])
        s_loss += s_loss_cond
    loss_dict['all'] = s_loss/num_count
    print("loss_all:", loss_dict['all'])

    #print(results_df[0:10])
    corr, pval = spearmanr(results_df['ref_val'].values,results_df['test_val'].values)
    print("vis spearman corr:",corr)
    corr, pval = pearsonr(results_df['ref_val'].values,results_df['test_val'].values)
    print("vis pearson corr:",corr)

    sns.set(style='darkgrid')
    figure = sns.relplot(data=results_df, x='ref_val', y='test_val', style = 'condition', hue=label_class, legend="full", alpha=0.5)#,hue_order=hue_order_list)

    max_val = max(np.max(results_df['ref_val'].values), np.max(results_df['test_val'].values))
    min_val = min(np.min(results_df['ref_val'].values), np.min(results_df['test_val'].values))
    figure.set(ylim=(min_val, max_val),xlim=(min_val, max_val))

    figure.savefig(out_path+'vis_plot.png')
    #plt.show()
    

    if opt_func == 'mle_precise':
        #condition_list = results_df['condition'].unique()
        path_all = '_cond'#'prediction'
        for cond in condition_list:
            path_all += '_'+cond
        
        f = open(out_path+'eval_result'+path_all+'_'+datatype+'.txt', mode='w')

        corr, pval = spearmanr(results_df['response'].values,results_df['prediction'].values)
        print("alpha spearman corr:",corr)
        f.write(path_all+' spearman corr'+': '+str(corr)+'\n')
        corr, pval = pearsonr(results_df['response'].values,results_df['prediction'].values)
        print("alpha pearson corr:",corr)
        f.write(path_all+' pearson corr'+': '+str(corr)+'\n')
        f.write(path_all+' NLL'+': '+str(loss_dict['all'])+'\n')

        sns.set(style='darkgrid')
        figure = sns.relplot(data=results_df, x='prediction', y='response', style = 'condition', hue=label_class, legend="full", alpha=0.5)#,hue_order=hue_order_list)
        figure.set(ylim=(0, 1),xlim=(0, 1))

        #figure.savefig(out_path+'prediction.png')
        figure.savefig(out_path+'prediction'+path_all+'.png')
        

        for cond in condition_list:
            print(cond)
            corr, pval = spearmanr(results_df[results_df['condition']==cond]['response'].values,results_df[results_df['condition']==cond]['prediction'].values)
            print("alpha spearman corr:",corr)
            f.write(cond+' spearman corr'+': '+str(corr)+'\n')
            corr, pval = pearsonr(results_df[results_df['condition']==cond]['response'].values,results_df[results_df['condition']==cond]['prediction'].values)
            print("alpha pearson corr:",corr)
            f.write(cond+' pearson corr'+': '+str(corr)+'\n')
            f.write(cond+' NLL'+': '+str(loss_dict[cond])+'\n')
            
            figure = sns.relplot(data=results_df[results_df['condition']==cond], x='prediction', y='response', style = 'condition', hue=label_class, legend="full", alpha=0.5)#, size=1)
            #max_val = max(np.max(t_df['prediction'].values), np.max(t_df['response'].values))
            figure.set(ylim=(0, 1),xlim=(0, 1))
            figure.savefig(out_path+'prediction'+'_cond_'+cond+'.png')

        f.close()

    # #compute precise rrse
    # weight_np = results_df['weight']#dataset['weight'].cpu().detach().numpy()
    # #mean_ref = np.mean(results_df['ref_val'])
    # mean_ref = np.sum(results_df['ref_val']*weight_np)/np.sum(weight_np)

    # #root relative squared error (RRSE)
    # #numer = np.sum((results_df['test_val'] - results_df['ref_val'])**2.)
    # #denom = np.sum((results_df['ref_val'] - mean_ref)**2.)
    # numer = np.sum(((results_df['test_val'] - results_df['ref_val'])**2.)*weight_np)
    # denom = np.sum(((results_df['ref_val'] - mean_ref)**2.)*weight_np)
    # rrse = np.sqrt(numer / denom)# + 0.1/denom

    # print("correct loss:", rrse)

    #こっちは重みの再計算に使うので前回の重みづけしない
    results_df["loss_per_example"] = ((results_df['test_val'] - results_df['ref_val'])**2.)/np.mean((results_df['ref_val'] - np.mean(results_df['ref_val']))**2.)
    # fig, axs = plt.subplots()
    # stdval = results_df["loss_per_example"].std()
    # mean_val = results_df["loss_per_example"].mean()
    # #target_tensor=np.clip(target_tensor,mean_val-3*stdval,mean_val+3*stdval)

    # axs.hist(results_df["loss_per_example"], bins=50, density=True)
    # #axs.set_xlim(mean_val-3*stdval,mean_val+3*stdval)
    # plt.show()
    return loss_dict, results_df

#%% Gradient based optimization

def gradient_opt(model: VisModel,
                num_epochs: int,
                fold_id: int,
                train_loader_list: dict[str,DataLoader] | list[dict[str,DataLoader]],
                valid_loader_list: list[dict[str,DataLoader]] | None,
                optimizer: Optimizer,
                scheduler: LRScheduler | None=None,
                batch_scheduler=None ,
                get_best: bool = True,
                opt_func: str = 'bruteforce',
                writer: dict[str] | None = None,
                out_path: str | None = None,
                device: torch.device | None = None,
                use_ref_weight: bool = False) -> VisModel:
    # if out_path is None:
    #     out_path = path_to_plot
    
    # simpleGD = True

    run = [0]

    train_loss = []
    valid_loss = []
    train_loss_dict: dict[str,list] = {}
    valid_loss_dict: dict[str,list] = {}
    param_list = []

    epoch_list = []

    if fold_id>=0:
        _train_loader_list: dict[str,DataLoader] = train_loader_list[fold_id]
    else:
        _train_loader_list: dict[str,DataLoader] = train_loader_list
    
    for key in _train_loader_list.keys():
        train_loss_dict[key]=[]
        if fold_id>=0:
            valid_loss_dict[key]=[]


    for epoch in range(num_epochs):
        model.visualize_weights()
        epoch_list.append(epoch)

        model.train()
        t_loss = 0.
        t_loss_dict = {}
        dataset_size_dict = {}
        dataset_size = 0
        data_loader_list = []
        #data_loader_dict = {'iterator':[],'num_batches':[], 'prev_idx':[]}
        #num_batches_per_epoch = 100000000
        max_num_batches = 0
        for key, train_loader in _train_loader_list.items():
            dataset_size += len(train_loader)
            dataset_size_dict[key]=len(train_loader)
            t_loss_dict[key]=0.
            
            offset = 0
            
            data_loader_list.append({'iterator':iter(train_loader),'num_batches':len(train_loader), 'prev_idx':-1, 'condition': key, 'offset':offset, 'offset_count':0})
            
            max_num_batches = max(max_num_batches, len(train_loader))

        #dataset_size = len(data_loader_list) * num_batches_per_epoch
        
        #dataloader_iterator = iter(dataloader)
        for batch_index in range(max_num_batches):
            print("epoch {} batch idx {}".format(epoch, batch_index))

            #for data in data_loader_dict.values():
            for data in data_loader_list:
                #idx = int(batch_index/max_num_batches * data['num_batches'])
                idx = int(data['num_batches'] / max_num_batches * batch_index)
                #idx_base = int(idx / data['num_batches'] * max_num_batches)
                if idx>data['prev_idx']:# and batch_index == idx_base+data['offset']:
                    if data['offset_count'] == data['offset']:
                        data['offset_count']=0
                        #print("local idx {}:".format(idx))
                        print('{} idx {}'.format(data['condition'], idx))
                        data['prev_idx']=idx

                        try:
                            dataset = next(data['iterator'])
                            dataload_deivce(dataset, device)
                        except StopIteration:
                            print("iteration stopped at", idx)
                            # dataloader_iterator = iter(dataloader)
                            # data, target = next(dataloader_iterator)
                        optimizer.zero_grad()

                        if debug_mode:
                            with detect_anomaly():
                                model.visualize_weights()
                                loss = compute_loss(model, dataset, opt_func = opt_func,device=device, use_ref_weight = use_ref_weight)
                                loss.backward()

                        else:
                            loss = compute_loss(model, dataset, opt_func = opt_func,device=device, use_ref_weight = use_ref_weight)
                            loss.backward()
                
                        run[0] += 1
                        #print("run {}:".format(run))
                        print('Loss : {:4f}'.format(loss.item()))
                        #model.showParams()

                        t_loss += loss.item()
                        t_loss_dict[data['condition']]+=loss.item()
                        print()
                        
                        optimizer.step()

                        model.projection()
                    else:
                        data['offset_count']+=1


            # if batch_scheduler is not None:
            #     batch_scheduler.step(epoch+batch_index/num_batches_per_epoch)
            #     print("current_lr:",batch_scheduler.get_lr())

        t_loss/=dataset_size
        train_loss.append(t_loss)

        for key in t_loss_dict.keys():
            t_loss_dict[key]/=dataset_size_dict[key]
            train_loss_dict[key].append(t_loss_dict[key])

        print("training loss:", t_loss)
        if writer is not None:
            writer['train'].add_scalar("loss/all/fold"+str(fold_id), t_loss, epoch)
        
        if fold_id>=0:
            with torch.no_grad():
                training_mode = model.training
                model.eval()
                #loss computation on validation set
                v_loss, _df = compute_all(model, valid_loader_list[fold_id], opt_func = opt_func, out_path = out_path, device=device, use_ref_weight=use_ref_weight)
                model.train(training_mode)
            
            print("validation loss:", v_loss)
            #print("validation RMSE:", v_rmse)
            valid_loss.append(v_loss['all'])

            for key in valid_loss_dict.keys():
                valid_loss_dict[key].append(v_loss[key])

            if writer is not None:
                for key, val in v_loss.items():
                    writer['valid'].add_scalar('loss/'+key + '/fold'+str(fold_id), val, epoch)
    
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #sns.set(style='darkgrid')
        ax.plot(np.array(epoch_list),np.array(train_loss), color='black',  linestyle='solid', linewidth = 1.0, label='train')
        if fold_id>=0:
            ax.plot(np.array(epoch_list),np.array(valid_loss), color='grey',  linestyle='solid', linewidth = 1.0, label='validation')
        
        col_list = ['red','green','blue','purple','orange','gold','brown','greenyellow','yellow']
        for ii, key in enumerate(train_loss_dict.keys()):
            label_train = key+'_train'
            ax.plot(np.array(epoch_list),np.array(train_loss_dict[key]), color=col_list[ii],  linestyle='solid', linewidth = 1.0, label=label_train)
            if fold_id>=0:
                label_valid = key+'_valid'
                ax.plot(np.array(epoch_list),np.array(valid_loss_dict[key]), color=col_list[ii],  linestyle='dashed', linewidth = 1.0, label=label_valid)


        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        #plt.show()
        plt.savefig(out_path+"learning_curve.png")

        param_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                #print(name, param.data)
                param_dict[name]=param.data
        param_list.append(param_dict)

        if scheduler is not None:
            scheduler.step()

        PATH = out_path+'tmp_epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), PATH)

    if get_best:
        #validation scoreが最小のパラメータをとってくる
        min_ind = np.argmin(np.array(valid_loss))

        for name, param in model.named_parameters():
            if param.requires_grad:
                #print(name, param.data)
                #param_dict[name]=param.data
                model.set_param(name,param_list[min_ind][name])


    #print("Final params:")
    #model.showParams()

    return model

#%% Compute Std

def gradient_opt_2024(model: VisModel,
                num_epochs: int,
                fold_id: int,
                train_loader_list: dict[str,DataLoader] | list[dict[str,DataLoader]],
                valid_loader_list: list[dict[str,DataLoader]] | None,
                optimizer: Optimizer,
                scheduler: LRScheduler | None=None,
                batch_scheduler=None ,
                get_best: bool = True,
                opt_func: str = 'bruteforce',
                writer: dict[str] | None = None,
                out_path: str | None = None,
                device: torch.device | None = None,
                use_ref_weight: bool = False,
                use_scale_augment: bool = False,
                use_mask_loss: bool = False,
                use_zero_loss: bool = False,
                restart_threshold = 5.0
                ) -> VisModel:
    
    # 2 epoch終了時に以下のlossを下回らなければ最初からやりなおす
    # restart_threshold = 4.5

    run = [0]

    train_loss = []
    valid_loss = []
    train_loss_dict: dict[str,list] = {}
    valid_loss_dict: dict[str,list] = {}
    param_list = []

    epoch_list = []

    if fold_id>=0:
        _train_loader_list: dict[str,DataLoader] = train_loader_list[fold_id]
    else:
        _train_loader_list: dict[str,DataLoader] = train_loader_list
    
    for key in _train_loader_list.keys():
        train_loss_dict[key]=[]
        if fold_id>=0:
            valid_loss_dict[key]=[]


    for epoch in range(num_epochs):

        if epoch >= 1:
            use_dispersion_loss = False
        else:
            use_dispersion_loss = True

        model.visualize_weights()
        epoch_list.append(epoch)

        model.train()
        t_loss = 0.
        t_loss_dict = {}
        dataset_size_dict = {}
        dataset_size = 0
        data_loader_list = []
        #data_loader_dict = {'iterator':[],'num_batches':[], 'prev_idx':[]}
        #num_batches_per_epoch = 100000000
        max_num_batches = 0
        for key, train_loader in _train_loader_list.items():
            dataset_size += len(train_loader)
            dataset_size_dict[key]=len(train_loader)
            t_loss_dict[key]=0.
            
            offset = 0
            
            data_loader_list.append({'iterator':iter(train_loader),'num_batches':len(train_loader), 'prev_idx':-1, 'condition': key, 'offset':offset, 'offset_count':0})
            
            max_num_batches = max(max_num_batches, len(train_loader))

        #dataset_size = len(data_loader_list) * num_batches_per_epoch
        
        #dataloader_iterator = iter(dataloader)
        for batch_index in range(max_num_batches):
            print("epoch {} batch idx {}".format(epoch, batch_index))

            #for data in data_loader_dict.values():
            for data in data_loader_list:
                #idx = int(batch_index/max_num_batches * data['num_batches'])
                idx = int(data['num_batches'] / max_num_batches * batch_index)
                #idx_base = int(idx / data['num_batches'] * max_num_batches)
                if idx>data['prev_idx']:# and batch_index == idx_base+data['offset']:
                    if data['offset_count'] == data['offset']:
                        data['offset_count']=0
                        #print("local idx {}:".format(idx))
                        print('{} idx {}'.format(data['condition'], idx))
                        data['prev_idx']=idx

                        try:
                            dataset = next(data['iterator'])
                            dataload_deivce(dataset, device)
                        except StopIteration:
                            print("iteration stopped at", idx)
                            # dataloader_iterator = iter(dataloader)
                            # data, target = next(dataloader_iterator)
                        
                        if use_scale_augment:
                            scale_transform(dataset)
                        optimizer.zero_grad()

                        if debug_mode:
                            with detect_anomaly():
                                model.visualize_weights()
                                loss = compute_loss(model,
                                                    dataset,
                                                    opt_func = opt_func,
                                                    device=device,
                                                    use_ref_weight = use_ref_weight,
                                                    use_dispersion_loss=use_dispersion_loss,
                                                    use_mask_loss=use_mask_loss,
                                                    use_zero_loss=use_zero_loss,
                                                    )
                                loss.backward()

                        else:
                            loss = compute_loss(model,
                                                dataset,
                                                opt_func = opt_func,
                                                device=device,
                                                use_ref_weight = use_ref_weight,
                                                use_dispersion_loss=use_dispersion_loss,
                                                use_mask_loss=use_mask_loss,
                                                use_zero_loss=use_zero_loss,
                                                )
                            loss.backward()
                
                        run[0] += 1
                        #print("run {}:".format(run))
                        print('Loss : {:4f}'.format(loss.item()))
                        #model.showParams()

                        t_loss += loss.item()
                        t_loss_dict[data['condition']]+=loss.item()
                        print()
                        
                        optimizer.step()

                        model.projection()
                    else:
                        data['offset_count']+=1


            # if batch_scheduler is not None:
            #     batch_scheduler.step(epoch+batch_index/num_batches_per_epoch)
            #     print("current_lr:",batch_scheduler.get_lr())

        t_loss/=dataset_size
        train_loss.append(t_loss)

        for key in t_loss_dict.keys():
            t_loss_dict[key]/=dataset_size_dict[key]
            train_loss_dict[key].append(t_loss_dict[key])

        print("training loss:", t_loss)
        if writer is not None:
            writer['train'].add_scalar("loss/all/fold"+str(fold_id), t_loss, epoch)
        

        if t_loss > restart_threshold and epoch >= 1:
            print("training did not start to converge... restart model training")
            return model, False
        
        if fold_id>=0:
            with torch.no_grad():
                training_mode = model.training
                model.eval()
                #loss computation on validation set
                v_loss, _df = compute_all(model, valid_loader_list[fold_id], opt_func = opt_func, out_path = out_path, device=device, use_ref_weight=use_ref_weight)
                model.train(training_mode)
            
            print("validation loss:", v_loss)
            #print("validation RMSE:", v_rmse)
            valid_loss.append(v_loss['all'])

            for key in valid_loss_dict.keys():
                valid_loss_dict[key].append(v_loss[key])

            if writer is not None:
                for key, val in v_loss.items():
                    writer['valid'].add_scalar('loss/'+key + '/fold'+str(fold_id), val, epoch)
    
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #sns.set(style='darkgrid')
        ax.plot(np.array(epoch_list),np.array(train_loss), color='black',  linestyle='solid', linewidth = 1.0, label='train')
        if fold_id>=0:
            ax.plot(np.array(epoch_list),np.array(valid_loss), color='grey',  linestyle='solid', linewidth = 1.0, label='validation')
        
        col_list = ['red','green','blue','purple','orange','gold','brown','greenyellow','yellow']
        for ii, key in enumerate(train_loss_dict.keys()):
            label_train = key+'_train'
            ax.plot(np.array(epoch_list),np.array(train_loss_dict[key]), color=col_list[ii],  linestyle='solid', linewidth = 1.0, label=label_train)
            if fold_id>=0:
                label_valid = key+'_valid'
                ax.plot(np.array(epoch_list),np.array(valid_loss_dict[key]), color=col_list[ii],  linestyle='dashed', linewidth = 1.0, label=label_valid)


        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        #plt.show()
        plt.savefig(out_path+"learning_curve.png")

        param_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                #print(name, param.data)
                param_dict[name]=param.data
        param_list.append(param_dict)

        if scheduler is not None:
            scheduler.step()

        PATH = out_path+'tmp_epoch_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), PATH)

    if get_best:
        #validation scoreが最小のパラメータをとってくる
        min_ind = np.argmin(np.array(valid_loss))

        for name, param in model.named_parameters():
            if param.requires_grad:
                #print(name, param.data)
                #param_dict[name]=param.data
                model.set_param(name,param_list[min_ind][name])


    #print("Final params:")
    #model.showParams()

    return model, True

def compute_band_std(model: VisModel,
                    train_loader: dict[str,DataLoader],
                    device: torch.device):

    # 0 tensor([0.0069, 0.0010, 0.0012])
    # 1 tensor([0.0046, 0.0008, 0.0011])
    # 2 tensor([0.0044, 0.0010, 0.0013])
    # 3 tensor([0.0045, 0.0013, 0.0017])
    # 4 tensor([0.0045, 0.0015, 0.0019])
    # 5 tensor([0.3346, 0.0251, 0.0463])

    info_pyr = []
    for i in range(model.level):
        info_pyr.append({'std':torch.zeros((3),device=device),'count':0})

    num_count = 0
    datasetsize = 0
    for dataloader in train_loader.values():
        datasetsize += len(dataloader)

    loss_dict = {}
    for key, dataloader in train_loader.items():
        num_count_cond = 0
        for batch_index, dataset in enumerate(dataloader):
            #print("currently allocated memory (MB):",torch.cuda.memory_allocated(device=cuda)/1024/1024)
            
            info_pyr = model.compute_std(dataset['ref_fg'].to(device),info_pyr,False)        
    
            num_count += 1
            num_count_cond += 1

            print(100*num_count/datasetsize,"percent of data completed.")

    for i in range(model.level):
        info_pyr[i]['std']/=info_pyr[i]['count']
        print(i,info_pyr[i]['std'])
    
    model.set_std_vector(info_pyr)


def running_std_test(model: VisModel,
                    train_loader: dict[str,DataLoader],
                    device: torch.device):
    
    num_count = 0
    datasetsize = 0
    epoch_list=[]
    for dataloader in train_loader.values():
        datasetsize += len(dataloader)

    std_transition:list[list[list]]=[]
    for i in range(model.level-1):
        std_transition.append([[],[],[]])

    for key, dataloader in train_loader.items():
        num_count_cond = 0
        for batch_index, dataset in enumerate(dataloader):
            #model.compute_std_running([dataset['test_bg'].clone(),dataset['test_fg'].clone()])      
            model.compute_std_running([dataset['test_bg'].to(device)])  
    
            num_count += 1
            epoch_list.append(num_count)
            num_count_cond += 1

            print(100*num_count/datasetsize,"percent of data completed.")
            for i in range(model.level-1):
                std_transition[i][0].append(model.std_vector[i,0])
                std_transition[i][1].append(model.std_vector[i,1])
                std_transition[i][2].append(model.std_vector[i,2])
    
    for i in range(model.level-1):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        #sns.set(style='darkgrid')
        ax.plot(np.array(epoch_list),np.array(std_transition[i][0]), color='black',  linestyle='solid', linewidth = 1.0, label='Y')
        ax.plot(np.array(epoch_list),np.array(std_transition[i][1]), color='red',  linestyle='solid', linewidth = 1.0, label='U')
        ax.plot(np.array(epoch_list),np.array(std_transition[i][2]), color='blue',  linestyle='solid', linewidth = 1.0, label='V')

        ax.set_xlabel('epoch')
        ax.set_ylabel('mean squared contrast')
        ax.legend()
        plt.show()

def collate_fn(batch_list: list[dict[str,torch.Tensor]]):
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    # batch_dict['ref'] = torch.stack(batch_dict['ref'])
    #         batch_dict['ref_bg'] = torch.stack(batch_dict['ref_bg'])
    #         batch_dict['ref_DC'] = torch.stack(batch_dict['ref_DC'])
    #         batch_dict['test_fg'] = torch.stack(batch_dict['test_fg'])
    #         batch_dict['test_bg'] = torch.stack(batch_dict['test_bg'])
    #         batch_dict['ref_alpha'] = torch.stack(batch_dict['ref_alpha'])
    #         batch_dict['test_alpha'] = torch.stack(batch_dict['test_alpha'])
    #         batch_dict['response_std'] = torch.stack(batch_dict['response_std'])
    #         batch_dict['dist'] = torch.stack(batch_dict['dist'])
    batch_dict={'ref_fg': [],
                #'ref': [],
                'ref_bg': [],
                #'ref_DC': [],
                'test_fg': [],
                'test_bg': [],
                'ref_mask': [],
                'test_mask': [],
                'ref_fg_label': [],
                'ref_bg_label': [],
                'test_fg_label': [],
                'test_bg_label': [],
                'ref_alpha': [],
                'test_alpha': [],
                'response_std': [],
                'condition': [],
                'unique_index': [],
                'weight': []}
                # 'dist': []}
    for sub_batch in batch_list:
        # if sub_batch['condition'][0]=='small':
        #     #pad images to the same size as large ones
        #     padsize = (256-128)//2
        #     padsizeDC = padsize//(2**(LapLevs-1))
        #     sub_batch['ref'] = F.pad(sub_batch['ref'], (padsize, padsize, padsize, padsize), mode='reflect')
        #     sub_batch['ref_bg'] = F.pad(sub_batch['ref_bg'], (padsize, padsize, padsize, padsize), mode='reflect')
        #     sub_batch['ref_DC'] = F.pad(sub_batch['ref_DC'], (padsizeDC, padsizeDC, padsizeDC, padsizeDC), mode='reflect')
        #     sub_batch['test_fg'] = F.pad(sub_batch['test_fg'], (padsize, padsize, padsize, padsize), mode='reflect')
        #     sub_batch['test_bg'] = F.pad(sub_batch['test_bg'], (padsize, padsize, padsize, padsize), mode='reflect')
        batch_dict['ref_fg'].append(sub_batch['ref_fg'])
        #batch_dict['ref'].append(sub_batch['ref'])
        batch_dict['ref_bg'].append(sub_batch['ref_bg'])
        #batch_dict['ref_DC'].append(sub_batch['ref_DC'])
        batch_dict['test_fg'].append(sub_batch['test_fg'])
        batch_dict['test_bg'].append(sub_batch['test_bg'])
        batch_dict['ref_mask'].append(sub_batch['ref_mask'])
        batch_dict['test_mask'].append(sub_batch['test_mask'])
        batch_dict['ref_fg_label'] += sub_batch['ref_fg_label']
        batch_dict['ref_bg_label'] += sub_batch['ref_bg_label']
        batch_dict['test_fg_label'] += sub_batch['test_fg_label']
        batch_dict['test_bg_label'] += sub_batch['test_bg_label']
        batch_dict['ref_alpha'].append(sub_batch['ref_alpha'])
        batch_dict['test_alpha'].append(sub_batch['test_alpha'])
        batch_dict['response_std'].append(sub_batch['response_std'])
        batch_dict['condition'] += sub_batch['condition']
        batch_dict['unique_index'] += sub_batch['unique_index']
        batch_dict['weight'].append(sub_batch['weight'])
        #batch_dict['dist'].append(sub_batch['dist'])

    batch_dict['ref_fg'] = torch.cat(batch_dict['ref_fg'],dim=0)#.to(device=cuda)
    #batch_dict['ref'] = torch.cat(batch_dict['ref'],dim=0).to(device=cuda)
    batch_dict['ref_bg'] = torch.cat(batch_dict['ref_bg'],dim=0)#.to(device=cuda)
    #batch_dict['ref_DC'] = torch.cat(batch_dict['ref_DC'],dim=0).to(device=cuda)
    batch_dict['test_fg'] = torch.cat(batch_dict['test_fg'],dim=0)#.to(device=cuda)
    batch_dict['test_bg'] = torch.cat(batch_dict['test_bg'],dim=0)#.to(device=cuda)
    batch_dict['ref_mask'] = torch.cat(batch_dict['ref_mask'],dim=0)#.to(device=cuda)
    batch_dict['test_mask'] = torch.cat(batch_dict['test_mask'],dim=0)#.to(device=cuda)
    batch_dict['ref_alpha'] = torch.cat(batch_dict['ref_alpha'],dim=0)#.to(device=cuda)
    batch_dict['test_alpha'] = torch.cat(batch_dict['test_alpha'],dim=0)#.to(device=cuda)
    batch_dict['response_std'] = torch.cat(batch_dict['response_std'],dim=0)#.to(device=cuda)
    batch_dict['weight'] = torch.cat(batch_dict['weight'],dim=0)#.to(device=cuda)
    #batch_dict['dist'] = torch.cat(batch_dict['dist'],dim=0)#.to(device=cuda)

    return batch_dict

def set_loader(vis_dataset: VisDataset,
               condition_list: list[str],
               batch_size: int,
               all_loader_dict: dict[str,DataLoader],
               train_loader_list: list[dict[str,DataLoader]],
               valid_loader_list: list[dict[str,DataLoader]],
               test_loader_dict: dict[str,DataLoader]):

    n_fold = vis_dataset.N_fold #5
        
    for i in range(n_fold):

        train_loader_dict = {}
        valid_loader_dict = {}

        for cond in condition_list:

            if i==0:
                if cond=='standard':
                    test_sampler = SubsetRandomSampler(vis_dataset.get_condition_cv_indices(cond,-1))
                    test_loader_dict[cond] = DataLoader(vis_dataset, batch_size=batch_size, sampler = test_sampler, collate_fn=collate_fn)

                    index_list = []
                    for j in range(n_fold):
                        index_list+=vis_dataset.get_condition_cv_indices(cond,j)
                    index_list = sorted(index_list)
                    all_sampler = SubsetRandomSampler(index_list)
                    all_loader_dict[cond] = DataLoader(vis_dataset, batch_size=batch_size, sampler = all_sampler, collate_fn=collate_fn)
                else:
                    all_sampler = SubsetRandomSampler(vis_dataset.get_condition_indices(cond))
                    all_loader_dict[cond] = DataLoader(vis_dataset, batch_size=batch_size, sampler = all_sampler, collate_fn=collate_fn)


            # if cond not in vis_dataset.condname_strict_cv:
            train_indices = []
            for j in range(n_fold):
                if j==i:
                    valid_indices = vis_dataset.get_condition_cv_indices(cond,j)
                else:
                    train_indices += vis_dataset.get_condition_cv_indices(cond,j)

            train_sampler = SubsetRandomSampler(train_indices)
            train_loader_dict[cond] = DataLoader(vis_dataset, batch_size=batch_size, sampler=train_sampler,collate_fn=collate_fn)
            valid_sampler = SubsetRandomSampler(valid_indices)
            valid_loader_dict[cond] = DataLoader(vis_dataset, batch_size=batch_size, sampler=valid_sampler,collate_fn=collate_fn)

            print(len(train_loader_dict[cond]))

        
        train_loader_list.append(train_loader_dict)
        valid_loader_list.append(valid_loader_dict)
        
def recalc_visibility(model: VisModel,
                      df: pd.DataFrame,
                      stimulus_path: str,
                      device: torch.device,
                      show_img: bool = False) -> tuple(np.ndarray):
    visibility_list = []
    response_list = []
    vis_level_id_list = []
    stimulus_files = os.listdir(stimulus_path)

    show_img = show_img

    for (_, row) in tqdm(df.iterrows()):
        #print(stimulus_path + row['stimulus'])
        
        blend = cv2.imread(stimulus_path + row['stimulus'])
        blend = torch.as_tensor(np.float32(blend.transpose([2,0,1]))/255).to(device).unsqueeze(0)
        target = cv2.imread(stimulus_path + row['test_fg'])
        target = torch.as_tensor(np.float32(target.transpose([2,0,1]))/255).to(device).unsqueeze(0)
        ref = cv2.imread(stimulus_path + row['test_bg'])
        ref = torch.as_tensor(np.float32(ref.transpose([2,0,1]))/255).to(device).unsqueeze(0)
        if show_img:
            ref_img = cv2.imread(stimulus_path + row['ref_img'])
            ref_img = torch.as_tensor(np.float32(ref_img.transpose([2,0,1]))/255).to(device).unsqueeze(0)
            alphamap = cv2.imread(stimulus_path + row['alphamap'],0)
            alphamap = alphamap
            alphamap = torch.as_tensor(np.float32(alphamap)/255).to(device).unsqueeze(0).unsqueeze(0)
            blend_check = target * alphamap + ref * (1 - alphamap)

        mask_name = row['test_fg']
        mask_name = mask_name[:-4] + '_mask.png'
        if mask_name in stimulus_files:
            mask = cv2.imread(stimulus_path + mask_name,0)
            mask = torch.as_tensor(np.float32(mask)/255).to(device).unsqueeze(0).unsqueeze(0)
        else:
            mask = torch.ones_like(target[:,0,:,:]).to(device).unsqueeze(1)
        
        model.set_inputs_tg_ref_blended(target, ref, blend, mask)
        model.compute_visibility()

        if False:#isinstance(model,VisModel_MLP):
            _msk = model.mask_data
            _msk_sum = model.mask_data_sum
            
            model.vis_map = model.vis_map * _msk
            # model.vis_map = model.vis_map
            if model.lp_norm>1:
                visibility_mean = torch.pow( torch.sum(model.vis_map ** 2, dim=(1,2,3))/_msk_sum, 0.5)
            else:
                visibility_mean = torch.sum(model.vis_map, dim=(1,2,3)) / _msk_sum
        else:
            # can be negative
            visibility_mean = model.vis_score

        if False:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(model.vis_map[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=10)
            # vismap_raw = F.avg_pool2d(model.vis_map, 32, stride=1, padding=16)
            # vismap = model.visibility_to_norm(vismap_raw)
            ax[1].imshow(model.norm_vismap[0,0,:,:].detach().cpu().numpy(), vmin=0, vmax=1)
            ax[2].imshow(blend[0].detach().cpu().numpy().transpose([1,2,0])[...,[2,1,0]])
            plt.show()

        # print(visibility_mean)
        # plt.imshow(model.vis_map[0,0,:,:].detach().cpu().numpy())
        # plt.show()

        if show_img:
            def print_img(data, name, color = False):
                if color:
                    out_image = data[0].detach().cpu().numpy()
                    out_image = np.transpose(out_image, [1,2,0])
                    cv2.imwrite(f'{name}.png',np.uint8(255*out_image))
                else:
                    out_image = data[0,0].detach().cpu().numpy()
                    fig, ax = plt.subplots()
                    im = ax.imshow(out_image)
                    fig.colorbar(im, ax=ax)
                    plt.savefig(f'{name}.png')
                    plt.clf()
                    plt.close()
            print_img(target,"calibimg_target",color=True)
            print_img(ref,"calibimg_ref",color=True)
            print_img(blend,"calibimg_blend",color=True)
            print_img(ref_img,"calibimg_ref_img",color=True)
            print_img(blend_check,"calibimg_blend_check",color=True)
            print_img(alphamap,"calibimg_alphamap",color=False)
            print_img(mask,"calibimg_mask",color=False)

            print(f"model:{row['model']}")
            print(f"visibility predict:{visibility_mean}")
            print(f"vis_level_id:{row['vis_level_id']}")
            print(f"vis_level:{row['vis_level']}")
            show_img = False

        visibility_list.append(visibility_mean[0].detach().cpu().numpy().astype(np.float64))
        response_list.append(row['response'])
        vis_level_id_list.append(row['vis_level_id'])
        del blend,target,ref,mask
        torch.cuda.empty_cache()

    return np.array(visibility_list), np.array(response_list), np.array(vis_level_id_list)