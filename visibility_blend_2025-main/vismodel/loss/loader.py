from __future__ import annotations
import torch
from .originalLoss import OriginalLoss
from .blurLoss import BlurLoss
#from .iqaLoss import IqaLoss
from .ILossFunction import ILossFunction

def load_lossFunction(loss_type: str, config: dict[str], device: torch.device) -> ILossFunction:
    if loss_type == 'original':
        lossFunction = OriginalLoss(config['lambda_smooth'],
                                    config['lambda_vis'],
                                    config['lambda_fidelity'],
                                    config['calc_max_vis'],
                                    config.get('clip_target_vis',False),
                                    config['use_norm_vis_for_loss'],
                                    config['smooth_loss_grad_weight'],
                                    device,
                                    config.get('l2_loss',False),
                                    config.get('lp_loss',0),
                                    config.get('use_spatial_weight',False),
                                    config.get('asymmetric_loss', False),
                                    config.get('penalty_factor', 1.0),
                                    config.get('aggregate_vismap', False),
                                    config.get('vis_agg_ksize', 15)
                                    )
    elif loss_type =='blur':
        lossFunction = BlurLoss(config['lambda_smooth'],
                                    config['lambda_vis'],
                                    config['lambda_blur'],
                                    config['calc_max_vis'],
                                    config.get('clip_target_vis',False),
                                    config['use_norm_vis_for_loss'],
                                    config['smooth_loss_grad_weight'],
                                    device,
                                    config.get('l2_loss',False),
                                    config.get('lp_loss',0),
                                    config.get('use_spatial_weight',False),
                                    config.get('asymmetric_loss', False),
                                    config.get('penalty_factor', 1.0),
                                    config.get('aggregate_vismap', False),
                                    config.get('vis_agg_ksize', 15),
                                    config.get('base_alpha', 0.5)
                                    )
    # elif loss_type == 'iqa':
    #     lossFunction = IqaLoss(config['iqa_metric'],
    #                             config['lambda_vis'],
    #                             config['lambda_iqa'],
    #                             config['use_norm_vis_for_loss'],
    #                             device)
    else:
        raise Exception()
    
    return lossFunction