from numpy.core.numeric import True_
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import os
import argparse

from vismodel.calib.utils import compute_all, gradient_opt_2024, compute_band_std, set_loader, collate_fn
# from vismodel.calib.utils_rating import gradient_opt_rating, compute_all_rating
from vismodel.calib.data  import VisDataset, filterout_training_stimuli
# from vismodel.calib.data_rating import RatingDataset, filterout_training_stimuli
from vismodel.utils import load_vismodel
from vismodel.supermodels.visModel import VisModel


default_data_path = "/home/Dataset/ExperimentData/"
# default_data_path_rating = "/home/Dataset/Experiment2021_nonlinearblend/"
default_out_path = "./results/calib_result_2024/"
# default_vismodel_path = "./mlp/"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('--device_name', default='cuda:1', help='device name')
parser.add_argument('--data_path', default=default_data_path, help='path to the data directory')
# parser.add_argument('--data_path_rating', default=default_data_path_rating, help='path to the data directory')
parser.add_argument('--output_path', default=default_out_path, help='output directory')
parser.add_argument('--model_type', default="IQA_ciede", help='name of this experiment')#実験名を設定.この文字列に基づいてディレクトリが生成される
# parser.add_argument('--load_initial', action='store_true', help='load initial data')
# parser.add_argument('--vismodel_path', default=default_vismodel_path, help='path to vismodel')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
# parser.add_argument('--weight_rating', default=1.0, type=float, help='weight for rating loss')
parser.add_argument('--condition_list', nargs='*', default=['same','different', 'same2'])# ['same','different', 'same2', 'different2']
parser.add_argument('--eval_condition_list', nargs='*', default=['different2'])
parser.add_argument('--fold_list',nargs='*', default=[-1,0,1,2,3,4])
parser.add_argument('--repeat', default=1, type=int, help='number of repetition')
parser.add_argument('--use_mask_loss', action='store_true', help='use mask_loss')
parser.add_argument('--use_zero_loss', action='store_true', help='use zero_loss')
parser.add_argument('--use_scale_augment', action='store_true', help='if using uniform loss')
# FeatureWeight_Multi
args = parser.parse_args()
# args.use_zero_augment = True
# path_to_vismodel = "/home/visibility_blend_UT/"
# path_to_vismodel = "./"
# path_to_vismodel = args.vismodel_path

if not torch.cuda.is_available():
    args.device_name = "cpu"
    
print("device name:",args.device_name)

dataset_path = args.data_path
# dataset_path_rating = args.data_path_rating
output_path = args.output_path
cuda = args.device_name

model_cond = args.model_type
# load_initial = args.load_initial

filter_out_train_stimuli = True

# initial_path = "./vismodels/new/FeatureWeight_Sig_wd0.005.pth"#visibility_predictor.pth"

# use_ref_weight = False#通常の実験では関係ないので常にFalse

weight_basic = 1.0
# weight_rating = args.weight_rating

# local_runtime = True

# if not local_runtime:
#     from google.colab import drive
#     drive.mount('/content/gdrive')

# cuda = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(cuda)


LapLevs = 6
# NLP_filter = True
eps = 1e-8 #small value to prevent from getting "nan" in grad of p

save_compatible = True


"""# Preparation"""

# condition_list = ['same','different']
condition_list = args.condition_list#['same','different', 'same2', 'different2']
# eval_condition_list = ['same','different']

# blend_mode_list = ['linear','multiply','screen','CLAHE']
# eval_blend_mode_list = ['linear','multiply','screen','CLAHE']

"""#CrossVal"""

opt_method = 'mle'
longer_training = True

use_precomputed_std = True


# if use_ref_weight:
#     model_cond += '_userefweight'


#model_cond = 'ideal_attention_samdif_nlp_z_ave5_sigma_run_asym2'
# test_sigma = 0.1
#model_cond = 'lum_no_attention_mix_nlp_z_run_ave5_scale'

#model_cond = 'skipband_ideal_attention_mix_run_ave5_flip'#skipband=True runningstd=False / control mode=False

# weight decayの値を設定
weight_decay_fc = args.weight_decay#0.005#.001#0.01#0.01#0.005#0.01#.005#0.001#.005#0.0001

model_cond+='_wd'+str(weight_decay_fc)
model_cond+='_lr'+str(args.lr)
# model_cond+='_wr'+str(weight_rating)
# if load_initial:
#     model_cond+='_loadInit'

model_cond_base = model_cond

# randomtransform = None
#randomtransform = randomflip#randomtransform_scale20


optimize_scale_only = False

for rep_ind in range(args.repeat):

    if args.repeat>1:
        model_cond = model_cond_base + '_rep'+str(rep_ind)

    exp_parent_path = output_path + model_cond+'/'
    os.makedirs(exp_parent_path, exist_ok=True)

    fold_list = [int(x) for x in args.fold_list]
    # fold_list = [-1,0,1,2,3,4]#[4,3,-1]#[0,1,2,3,4,-1]#[1]#[-1,0,1,2,3,4]#[-1]#[4,-1]#,2,3,4,-1]#[-1,0,1]#[4]#[-1]#[2,3]#[0,1]#2,3,4]#[-1]#[3,4]#[0,1,2,3,4]#[-1]
    # fold_listの値のうち、-1は全データセットを利用した訓練
    # 0-4は5-fold cross validationの各分割


    vis_dataset = VisDataset(condition_list,
                             dataset_path,
                             exclude_high_alpha=False,
                             transform=None,
                             consistent_transform=True,
                             use_drop_exp_data=True)
    # rating_dataset = RatingDataset(dataset_path_rating, zero_augment=args.use_zero_augment)

    batch_size = 4#4
    # rating_batch_size = 4

    n_fold = vis_dataset.N_fold#5

    all_loader_dict = {}
    test_loader_dict = {}
    train_loader_list = []
    valid_loader_list = []

    set_loader(vis_dataset, condition_list, batch_size, all_loader_dict,train_loader_list,valid_loader_list,test_loader_dict)
    # rating_loader_all, rating_train_loader_list, rating_valid_loader_list = rating_dataset.set_loader(blend_mode_list, rating_batch_size)

    if len(args.eval_condition_list)>0:
        test_dataset = VisDataset(args.eval_condition_list,
                                  dataset_path,
                                  exclude_high_alpha=False,
                                  transform=None,
                                  consistent_transform=True,
                                  use_drop_exp_data=True)

        if filter_out_train_stimuli:
            filtered_idx_dict = filterout_training_stimuli(vis_dataset, None, test_dataset, args.eval_condition_list, condition_list)

        test_loader_dict = {}
        for cond in args.eval_condition_list:
            if filter_out_train_stimuli:
                _sampler = SubsetRandomSampler(filtered_idx_dict[cond])
            else:
                _sampler = SubsetRandomSampler(test_dataset.get_condition_indices(cond))
            test_loader_dict[cond] = DataLoader(test_dataset, batch_size=batch_size, sampler = _sampler, collate_fn=collate_fn)

    else:
        test_loader_dict = None

    param_dict=[]

    for fold_id in fold_list:

        for repeat in range(10):
            #収束するまで最大１０回やり直す
            
            # test_path_train = exp_parent_path+"train"
            # test_path_val = exp_parent_path+"valid"

            model_name = "vismodel_fold"+str(fold_id)

            writer = None


            path_name = exp_parent_path+model_name+'.pth'
    
            model: VisModel = load_vismodel(args.model_type, cuda, load_param = False)
            model._set_target_type("content")
            # if load_initial:

            #     if False:
            #         # "param_fullmodel"以外のパラメータをロード
            #         state_dict = torch.load(initial_path, map_location = cuda)
            #         state_dict['param_fullmodel'] = model.param_fullmodel
            #         model.load_state_dict(state_dict, strict=False)
            #     else:
                    
            #         model.load_state_dict(torch.load(initial_path, map_location = cuda))
            #     for name, param in model.named_parameters():
            #         print(f"{name}_requires_grad: {param.requires_grad}")
                    

            if weight_decay_fc > 0:
                #重み調整のところにだけweight decayをかける

                opt_params = []
                for name, param in model.named_parameters():
                    print(name)
                    name_split = name.split('.')
                    if name_split[0] == 'fc' and name_split[-1] == 'weight':
                    # if "fc.0.weight" in name:
                    # if "fc.0.weight"  in name or "fc.2.weight" in name or "fc.3.weight" in name or "fc.4.weight" in name or "fc.6.weight" in name:
                        opt_params.append({'params':param, 'weight_decay':weight_decay_fc})
                    else:
                        opt_params.append({'params':param})
            else:
                opt_params = model.get_optimize_params()

            # for param in model.parameters():
            #     print(param)

            # if model.running_std:
            #     model.train()
            # else:
            #     model.col_sigma.data=torch.tensor(test_sigma).to(cuda)
                    
            if use_precomputed_std:
                precomputed_std_array = torch.tensor([
                    [0.00685872975736856461, 0.00096498394850641489, 0.00123452930711209774],
                    [0.00456557050347328186, 0.00076206546509638429, 0.00106148154009133577],
                    [0.00442945770919322968, 0.00099897140171378851, 0.00133739155717194080],
                    [0.00454803090542554855, 0.00127012759912759066, 0.00168349011801183224],
                    [0.00447584548965096474, 0.00146052299533039331, 0.00193399202544242144],
                    [0.33458039164543151855, 0.02506467886269092560, 0.04629739001393318176]]).to(cuda)
                model.std_vector = nn.Parameter(precomputed_std_array)
                model.std_vector.requires_grad=False
                print(model.std_vector)
            else:
                compute_band_std(model, all_loader_dict, device=cuda)
    
            # if optimize_scale_only:
            #     for name, param in model.named_parameters():
            #         print(name, "set grad: off")
            #         param.requires_grad = False
            #     model.scaling.requires_grad = True
            

            model.projection()
            print(model.get_optimize_params())


            if longer_training:
            
                num_epochs = 32#32#16# 24
                optimizer = optim.Adam(opt_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
                # optimizer = optim.Adam(opt_params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12, 18, 24], gamma=0.5, last_epoch=-1)
            else:
                num_epochs = 16
                optimizer = optim.Adam(opt_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,12], gamma=0.1, last_epoch=-1)
            batch_scheduler=None

            #torch.autograd.set_detect_anomaly(True)
            if fold_id==-1:
                model, success = gradient_opt_2024(model,
                                          num_epochs,
                                          fold_id,
                                          all_loader_dict,
                                          None,
                                          optimizer=optimizer,
                                          scheduler=scheduler,
                                          batch_scheduler=batch_scheduler,
                                          get_best= False,
                                          opt_func = opt_method,
                                          writer=writer,
                                          out_path = exp_parent_path+model_name,
                                          device=cuda,
                                          use_ref_weight=False,
                                          use_mask_loss=args.use_mask_loss,
                                          use_zero_loss=args.use_zero_loss,
                                          use_scale_augment=args.use_scale_augment
                )
            else:
                model, success = gradient_opt_2024(model,
                                          num_epochs,
                                          fold_id,
                                          train_loader_list,
                                          valid_loader_list,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        batch_scheduler=batch_scheduler,
                                        get_best= False,
                                        opt_func = opt_method,
                                        writer=writer,
                                        out_path = exp_parent_path+model_name,
                                        device=cuda,
                                        use_ref_weight=False,
                                        use_mask_loss=args.use_mask_loss,
                                        use_zero_loss=args.use_zero_loss,
                                        use_scale_augment=args.use_scale_augment
                )
            
            if not success:
                #もう一度やりなおす
                continue

            PATH = path_name
            torch.save(model.state_dict(), PATH)

            

            with torch.no_grad():
                print("Computing final loss...")

                
                if fold_id==-1:
                    _, train_df = compute_all(model, all_loader_dict, opt_func = 'mle_precise',out_path = exp_parent_path+model_name, device=cuda,datatype='all', use_ref_weight=False)
                    train_df.to_csv(exp_parent_path+model_name+'_all.csv', index=False)

                    if test_loader_dict is not None:
                        _, test_df = compute_all(model, test_loader_dict, opt_func = 'mle_precise',out_path = exp_parent_path+model_name, device=cuda,datatype='test', use_ref_weight=False)
                        test_df.to_csv(exp_parent_path+model_name+'_test.csv', index=False)
                else:
                    _, train_df = compute_all(model, train_loader_list[fold_id], opt_func = 'mle_precise',out_path = exp_parent_path+model_name,device=cuda,datatype='train',use_ref_weight=False)
                    train_df.to_csv(exp_parent_path+model_name+'_train.csv', index=False)

                    _, valid_df = compute_all(model, valid_loader_list[fold_id], opt_func = 'mle_precise',out_path = exp_parent_path+model_name,device=cuda,datatype='valid',use_ref_weight=False)
                    valid_df.to_csv(exp_parent_path+model_name+'_valid.csv', index=False)
                
            model.visualize_weights()

            param_dict.append(model.get_params())
            
            # writer['train'].close()
            # writer['valid'].close()

            # if save_compatible:
            #     #save backward compatible model
            #     PATH = exp_parent_path+model_name+'_gpu.pth'
            #     torch.save(model.state_dict(), PATH, _use_new_zipfile_serialization=False)
            #     #save cpu model
            #     PATH = exp_parent_path+model_name+'_cpu.pth'
            #     torch.save(model.to('cpu').state_dict(), PATH, _use_new_zipfile_serialization=False)

            if success:
                #次のfoldへ進む
                break

    pd.to_pickle(param_dict, exp_parent_path+"params.pkl")
