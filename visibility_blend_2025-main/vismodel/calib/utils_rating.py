from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import detect_anomaly
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.types import Number
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import sys
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr
import pandas as pd
import cv2
import os
from tqdm import tqdm
import pickle

from .utils import compute_loss, compute_all, dataload_deivce
from vismodel.supermodels.visModel import VisModel
from vismodel.vismodel_mlp import VisModel_MLP

debug_mode = False

#%% Loss function
def dataload_deivce_rating(dataset, device: torch.device)->dict[str]:

    # return unique_index, combination_id, test_bg, test_fg, fg_mod, ref_maskimg, response, blend_mode, alpha
    data_dict = {}
    data_dict['unique_index'] = dataset[0]
    data_dict['combination_id'] = dataset[1]
    data_dict['test_bg'] = dataset[2].to(device)
    data_dict['test_fg'] = dataset[3].to(device)
    data_dict['fg_mod'] = dataset[4].to(device)
    data_dict['ref_maskimg'] = dataset[5].to(device)
    data_dict['response'] = dataset[6].to(device)
    data_dict['blend_mode'] = dataset[7]
    data_dict['alpha'] = dataset[8].to(device)
    
    return data_dict

def compute_loss_rating(model: VisModel, 
                        dataset: dict[str,torch.Tensor], 
                        show_data: bool = False, 
                        result_df: pd.DataFrame | None = None, 
                        device: torch.device | None = None) -> torch.Tensor | tuple[torch.Tensor, pd.DataFrame]:
    
    if not isinstance(model, VisModel_MLP):
        raise NotImplementedError()
    alpha = dataset['alpha']
    test_bg = dataset['test_bg']
    test_fg = dataset['test_fg']
    fg_mod = dataset['fg_mod']
    ref_maskimg = dataset['ref_maskimg']
    response = dataset['response']

    blend_type = []
    for bm in dataset['blend_mode']:
        if bm == 'CLAHE':
            blend_type.append('linear')
        else:
            blend_type.append(bm)

    # if dataset['blend_mode'] == 'CLAHE':
    #     blend_type = 'linear'
    # else:
    #     blend_type = dataset['blend_mode']

    model.set_inputs_tg_ref_alphamap(test_fg, test_bg, ref_maskimg*alpha.view(-1,1,1,1), ref_maskimg, blend_mode=blend_type)
    model.compute_weights()

    model.set_target(fg_mod)
    # 2024/7/11 added これがないとtargetを入れ替えてもblend imageが更新されない
    model.set_alphamap(ref_maskimg*alpha.view(-1,1,1,1), blend_mode=blend_type)
    
    model.compute_visibility_wo_weight()
    # vis_level = model.vis_score

    # vis_rating = model.visibility_to_norm(vis_level) * 4 + 1
    vis_rating = model.norm_score * 4 + 1

    # loss
    loss = F.mse_loss(vis_rating, response)

    if model.mask_loss_weight > 0:

        _mask_data = model.mask_data

        index_mask = _mask_data < 1e-3
        binary_mask = torch.zeros_like(_mask_data)
        binary_mask[index_mask] = 1

        # _mask_dataとbinary_maskを並べて表示
        # plt.subplot(1,2,1)
        # plt.imshow(_mask_data[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(binary_mask[0,0,:,:].cpu().detach().numpy())
        # plt.show()
            
        if False:
            out_vis = torch.sum(model.vis_map * binary_mask, dim=(1,2,3))/torch.sum(binary_mask, dim=(1,2,3))
            mask_loss = model.visibility_to_norm(out_vis, force_compute=True) * 4# + 1

        else:
            # norm_vismap = model.visibility_to_norm(model.vis_map) * 4# + 1
            norm_vismap = model.norm_vismap * 4# + 1
            mask_loss = torch.sum(norm_vismap * binary_mask, dim=(1,2,3))/torch.sum(binary_mask, dim=(1,2,3))

        loss += mask_loss.mean() * model.mask_loss_weight
    

    if show_data:
        ref_vis_np = response.cpu().detach().numpy()
        test_vis_np = vis_rating.cpu().detach().numpy()

        tmp_df = pd.DataFrame()#columns=['ref_val', 'test_val', 'target_img'])
        tmp_df['unique_index']=dataset["unique_index"]
        tmp_df['response']=ref_vis_np
        tmp_df['prediction']=test_vis_np
        tmp_df['blend_mode']=dataset["blend_mode"]
        tmp_df['combination_id']=dataset["combination_id"]
        tmp_df['alpha']=dataset["alpha"].cpu().numpy()
        
        result_df = pd.concat([result_df,tmp_df])
        
        #ax.scatter(ref_vis.data.cpu().numpy(),test_vis.data.cpu().numpy())
        
        #plt.show()
        return loss, result_df#いちいち返して代入してあげないとdfは消失してしまう
    else:
        

        return loss#1.-p_corr

def compute_all_rating(model: VisModel, 
                       dataloader: DataLoader, 
                       out_path: bool = None, 
                       device: torch.device | None = None, 
                       datatype: str = '') -> tuple[Number, pd.DataFrame]:

    results_df = pd.DataFrame()
    model.eval()

    s_loss = 0.
    for dataset in tqdm(dataloader):
        data_dict = dataload_deivce_rating(dataset, device)
        loss, results_df = compute_loss_rating(model, data_dict, show_data = True, result_df = results_df, device=device)

        s_loss += loss.item()
    
    loss_all = s_loss/len(dataloader)
    print("loss_all:", loss_all)

    f = open(out_path+'eval_result_rating_'+datatype+'.txt', mode='w')

    corr, pval = spearmanr(results_df['response'].values,results_df['prediction'].values)
    print("rating spearman corr:",corr)
    f.write('All spearman corr'+': '+str(corr)+'\n')
    corr, pval = pearsonr(results_df['response'].values,results_df['prediction'].values)
    print("rating pearson corr:",corr)
    f.write('All pearson corr'+': '+str(corr)+'\n')
    f.write('All MSE'+': '+str(loss_all)+'\n')

    sns.set(style='darkgrid')
    figure = sns.relplot(data=results_df, x='prediction', y='response', style = 'blend_mode', hue='blend_mode', legend="full", alpha=0.5)#,hue_order=hue_order_list)

    figure.set(ylim=(1, 5),xlim=(1, 5))
    figure.savefig(out_path+'rating_plot.png')

    condition_list = list(results_df['blend_mode'].unique())

    for cond in condition_list:
        print(cond)
        corr, pval = spearmanr(results_df[results_df['blend_mode']==cond]['response'].values,results_df[results_df['blend_mode']==cond]['prediction'].values)
        print("rating spearman corr:",corr)
        f.write(cond+' spearman corr'+': '+str(corr)+'\n')
        corr, pval = pearsonr(results_df[results_df['blend_mode']==cond]['response'].values,results_df[results_df['blend_mode']==cond]['prediction'].values)
        print("rating pearson corr:",corr)
        f.write(cond+' pearson corr'+': '+str(corr)+'\n')

        mse_loss = ((results_df[results_df['blend_mode']==cond]['response'].values - results_df[results_df['blend_mode']==cond]['prediction'].values) ** 2).mean()
        f.write(cond+' MSE'+': '+str(mse_loss)+'\n')
        
        figure = sns.relplot(data=results_df[results_df['blend_mode']==cond], x='prediction', y='response', legend="full", alpha=0.5)#, size=1)
        figure.set(ylim=(1, 5),xlim=(1, 5))
        figure.savefig(out_path+'prediction'+'_mode_'+cond+'.png')

    f.close()

    return loss_all, results_df

#%% Gradient based optimization

def gradient_opt_rating(model: VisModel,
                        num_epochs: int,
                        fold_id: int,
                        train_loader_list: dict[str, DataLoader] | list[dict[str, DataLoader]],
                        valid_loader_list: list[dict[str, DataLoader]] | None,
                        rating_train_loader_list: dict[str, DataLoader] | list[dict[str, DataLoader]],
                        rating_valid_loader_list: list[dict[str, DataLoader]] | None, 
                        blend_mode_list: list[str], 
                        optimizer: Optimizer, 
                        scheduler: LRScheduler | None = None, 
                        batch_scheduler=None ,
                        get_best: bool = True, 
                        opt_func: str = 'bruteforce', 
                        writer: dict[str] = None, 
                        out_path: str = None, 
                        device: torch.Tensor = None, 
                        use_ref_weight: bool = False,
                        weight_rating: float = 1.0, 
                        weight_basic: float = 1.0) -> VisModel:


    run = [0]

    train_loss = []
    valid_loss = []
    train_loss_dict = {}
    valid_loss_dict = {}
    param_list = []

    # rating_train_loss = []
    # rating_valid_loss = []
    # rating_train_loss_dict = {}
    # rating_valid_loss_dict = {}

    epoch_list = []

    if fold_id>=0:
        _train_loader_list: dict[str, DataLoader] = train_loader_list[fold_id]
        _rating_train_loader: dict[str, DataLoader] = rating_train_loader_list[fold_id]
    else:
        _train_loader_list: dict[str, DataLoader] = train_loader_list
        _rating_train_loader: dict[str, DataLoader] = rating_train_loader_list
    
    for key in _train_loader_list.keys():
        train_loss_dict[key]=[]
        if fold_id>=0:
            valid_loss_dict[key]=[]
    
    train_loss_dict['rating']=[]
    if fold_id>=0:
        valid_loss_dict['rating']=[]
    
    # for blend_mode in blend_mode_list:
    #     rating_train_loss_dict[blend_mode]=[]
    #     if fold_id>=0:
    #         rating_valid_loss_dict[blend_mode]=[]


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
        
        dataset_size_rating = len(_rating_train_loader)
        dataset_size_dict['rating']=len(_rating_train_loader)
        t_loss_dict['rating']=0.
        offset = 0
        data_loader_list.append({'iterator':iter(_rating_train_loader),'num_batches':len(_rating_train_loader), 'prev_idx':-1, 'condition': 'rating', 'offset':offset, 'offset_count':0})
        max_num_batches = max(max_num_batches, len(_rating_train_loader))

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
                            
                        except StopIteration:
                            print("iteration stopped at", idx)
                            # dataloader_iterator = iter(dataloader)
                            # data, target = next(dataloader_iterator)
                        
                        if data['condition'] == 'rating' and weight_rating>0:
                            print("train rating")
                            data_dict = dataload_deivce_rating(dataset, device)
                            if debug_mode:
                                with detect_anomaly():
                                    model.visualize_weights()
                                    loss: torch.Tensor = compute_loss_rating(model, data_dict, device=device) * weight_rating
                                    loss.backward()
                            else:
                                loss = compute_loss_rating(model, data_dict, device=device) * weight_rating
                                loss.backward()
                            
                            print('Loss : {:4f}'.format(loss.item()))
                            #model.showParams()

                            t_loss += loss.item() / dataset_size_rating
                            t_loss_dict[data['condition']]+=loss.item()
                            print()
                            
                            optimizer.step()

                            model.projection()

                        elif weight_basic>0:
                            dataload_deivce(dataset, device)
                            optimizer.zero_grad()

                            if debug_mode:
                                with detect_anomaly():
                                    model.visualize_weights()
                                    loss = compute_loss(model, dataset, opt_func = opt_func,device=device, use_ref_weight = use_ref_weight) * weight_basic
                                    loss.backward()

                            else:
                                loss = compute_loss(model, dataset, opt_func = opt_func,device=device, use_ref_weight = use_ref_weight)
                                loss.backward()
                    
                            run[0] += 1
                            #print("run {}:".format(run))
                            print('Loss : {:4f}'.format(loss.item()))
                            #model.showParams()

                            t_loss += loss.item() / dataset_size
                            t_loss_dict[data['condition']]+=loss.item()
                            print()
                            
                            optimizer.step()

                            model.projection()
                    else:
                        data['offset_count']+=1

            # break
            # if batch_scheduler is not None:
            #     batch_scheduler.step(epoch+batch_index/num_batches_per_epoch)
            #     print("current_lr:",batch_scheduler.get_lr())

        # t_loss/=dataset_size
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
                v_loss = {}
                v_loss['all']=0.
                if weight_rating>0:
                    tmp_v_loss, _df = compute_all_rating(model, rating_valid_loader_list[fold_id], out_path = out_path, device=device)
                    v_loss['rating'] = tmp_v_loss
                    v_loss['all'] += tmp_v_loss
                if weight_basic>0:
                    tmp_v_loss, _df = compute_all(model, valid_loader_list[fold_id], opt_func = opt_func, out_path = out_path, device=device, use_ref_weight=use_ref_weight)
                    for key in tmp_v_loss.keys():
                        if key == 'all':
                            v_loss['all'] += tmp_v_loss[key]
                        else:
                            v_loss[key] = tmp_v_loss[key]
                        
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
        ax = fig.add_subplot(3,1,1)
        
        #sns.set(style='darkgrid')
        ax.plot(np.array(epoch_list),np.array(train_loss), color='black',  linestyle='solid', linewidth = 1.0, label='train')
        if fold_id>=0:
            ax.plot(np.array(epoch_list),np.array(valid_loss), color='grey',  linestyle='solid', linewidth = 1.0, label='validation')
        
        ax1 = fig.add_subplot(3,1,2, sharex=ax)
        ax2 = fig.add_subplot(3,1,3, sharex=ax)
        col_list = ['red','green','blue','purple','orange','gold','brown','greenyellow','yellow']
        for ii, key in enumerate(train_loss_dict.keys()):
            label_train = key+'_train'

            if key == 'rating':
                _ax = ax2
            else:
                _ax = ax1
            _ax.plot(np.array(epoch_list),np.array(train_loss_dict[key]), color=col_list[ii],  linestyle='solid', linewidth = 1.0, label=label_train)
            if fold_id>=0:
                label_valid = key+'_valid'
                _ax.plot(np.array(epoch_list),np.array(valid_loss_dict[key]), color=col_list[ii],  linestyle='dashed', linewidth = 1.0, label=label_valid)


        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        ax1.legend()
        ax2.legend()
        #plt.show()
        plt.savefig(out_path+"learning_curve.png")
        with open(out_path + 'train_loss_dict.pkl', 'wb') as f:
            pickle.dump(train_loss_dict, f)

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