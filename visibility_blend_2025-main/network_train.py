import sys
import argparse
import os
import os.path
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.train_coco_dtd_dataset import CocoDtdDataset
from dataset.train_coco_dataset import CocoDataset
import json
import datetime

#from alpha_blend import compute_loss_multi, compute_loss, calc_band_simga,compute_corr_resp_loss, compute_weighted_resp_loss, get_custom_gaussian_kernel, calc_img_sigma, local_optimization, local_optimization_lab, global_optimization
#from anlp2 import NLP_Z

from networks.train.networkTrainer import NetworkTrainer
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from networks.utils import load_network
from vismodel.utils import load_vismodel


if __name__ == '__main__':

    #DTD datasetへのパスをここに記入
    default_setting_path = os.getcwd()+'/settings_train_half_551_long.json'

    default_out_path = './results/trains/'

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--device', default='cuda:7', help='device name')
    parser.add_argument('--setting', default = default_setting_path, help='file path to input json file')
    parser.add_argument('--output', default=default_out_path, help='output directory')

    parser.add_argument('--epochs', type=int, default=500, help='epoch count')
    parser.add_argument('--visexp', type=float, default=1.0, help='force vis_exp to this value. input -1.0 to use trained value.')
    parser.add_argument('--save_data_interval', type=int, default=100, help='save data interval epochs')
    parser.add_argument('--save_image_interval', type=int, default=20, help='save image interval epochs')
    #parser.add_argument('--log_interval', type=int, default=1, help='log interval epochs')
    parser.add_argument('--batch_size', type=int, default=48, help='epoch count')

    # parser.add_argument('--use_light_model', action='store_true', help='use use AlphaGeneratorLight')
    parser.add_argument('--handle_tv_edge', action='store_true', help='reduce vis weigths on edges in the tv map')
    #parser.add_argument('--use_multistepLR', action='store_true', help='use multistep LR scheduler')

    #parser.add_argument('--model_path', default = default_model_path, help='file path to visibility model')
    #parser.add_argument('--epochs_lr_decay', type=int, default=400, help='epochs to decay lr to zero')
    #parser.add_argument('--epochs_lr_decay_start', type=int, default=100, help='epochs to lr decay start')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    #parser.add_argument('--init_channel', type=int, default=16, help='number of channels at the output of 1st layer')
    #parser.add_argument('--max_channel', type=int, default=128, help='maximum number of channels')

    #parser.add_argument('--abs_loss', action='store_true', help='train by abs loss')
    #parser.add_argument('--normal_train', action='store_true', help='train by normal blend')

    # テストのみ実行するためのフラグを追加
    parser.add_argument('--test_only', action='store_true', help='run test only with trained model')

    args = parser.parse_args()
    print(args)

    # args.non_uniform_tv = True
    # args.use_light_model = True
    #args.handle_tv_edge = True

    if not torch.cuda.is_available():
        args.device = "cpu"
    args.device: torch.device  = torch.device(args.device)
    print("device name:",args.device)

    json_open = open(args.setting, 'r')
    json_load = json.load(json_open)
    print(len(json_load))
    setting_filename = os.path.basename(args.setting)
    setting_filename = os.path.splitext(setting_filename)[0] # split extention

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)

    exp_path = f"{args.output}{setting_filename}_{now.strftime('%y%m%d_%H%M')}/"
    os.makedirs(exp_path, exist_ok=True)
    with open(exp_path+'config.txt', 'a', encoding='utf-8') as f:
        f.write(str(args))
    with open(exp_path+'setting_info.json', 'a') as f:
        json.dump(json_load['setting'], f, indent=4)

    for setting in json_load['setting']:
        # テストのみの場合はtrain=Falseでモデルを読み込む
        if args.test_only:
            network = load_network(setting["network"], args.device, train=False)
        else:
            network = load_network(setting["network"], args.device, train=True)
            
        vismodel = load_vismodel(setting["vismodel"], args.device)
        trainer = NetworkTrainer(network,
                                 vismodel,
                                 setting,
                                 args.lr,
                                 args.visexp,
                                 args.handle_tv_edge,
                                 exp_path,
                                 args.device,
                                setting.get("train_size", 256))
        trainer.train_append_log(args)
        
        dataset = CocoDataset(
            setting["target_type"], 
            args.device, 
            wo_semi = setting["wo_semi"],
            half_semi = setting.get("half_semi", "uniform"),
            all_semi = setting.get("all_semi", False),
            tv_type = setting["tv_type"],
            crop_size = setting.get("data_size", 256),
            categories = setting.get("categories", []),
            num_pairs = setting.get("num_pairs", 8192),
            apply_blur = setting.get("apply_blur", False),
            apply_grayscale = setting.get("apply_grayscale", False),
            apply_colorjitter = setting.get("apply_colorjitter", False),
            )
        assert network.tv_input == dataset.tv_type
        train_loader, valid_loader, test_loader = dataset.make_loader(args.batch_size)
        os.makedirs(f'{exp_path}/{setting["shortname"]}/', exist_ok=True)
        with open(f'{exp_path}/{setting["shortname"]}/dataset.txt', 'a', encoding='utf-8') as f:
            f.write(str(dataset))
        
        if args.test_only:
            # テストのみ実行
            print(f"Running test only for {setting['shortname']}...")
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    trainer.train_process(data, phase="test")
                trainer.train_print_loss(0, phase="test")
                trainer.save_test_df()  # テスト結果を保存
            print(f"Test finished for {setting['shortname']}.")
        else:
            # 通常の訓練ループ
            for epoch in range(1, args.epochs + 1):
                
                for i, data in enumerate(train_loader):
                    trainer.train_process(data)
                    
                if epoch % args.save_data_interval == 0:
                    trainer.train_save(epoch)

                if epoch % args.save_image_interval == 0:
                    trainer.train_save_image(epoch)

                #if epoch % args.log_interval == 0:
                trainer.train_print_loss(epoch)

                trainer.train_update_learning_rate()
                print("epoch:",epoch,"training finished.")

                with torch.no_grad():
                    for i, data in enumerate(valid_loader):
                        trainer.train_process(data,phase="val")
                    trainer.train_print_loss(epoch,phase="val")
                print("epoch:",epoch,"validation finished.")
            trainer.train_save_loss()
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    trainer.train_process(data,phase="test")
                trainer.train_print_loss(0,phase="test")
                print("epoch:",0,"test finished.")
