import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
import numpy as np
from .supermodels.superModel import SuperModel
from .supermodels.lowerBond import LowerBound

eps = 1e-8
#on_server=True

param_fukiage2014 = [0.81293315, 0.89033534, 0.95519662]

def local_optimization_lab(model, fglab, bglab, target_vis):
    n_rep = 8

    #opt_alpha_map = torch.ones((fglab.shape[0],1,fglab.shape[2],fglab.shape[3])) * 0.5
    opt_alpha_map = torch.ones_like(fglab)[:,0,:,:].unsqueeze(1)*0.5

    step = 0.25

    for i in range(n_rep):

        alpha_map = opt_alpha_map * model.mask_gp[0]

        resp = model.compute_vis_resp_control_lab(bglab, fglab, alpha_map)

        exp_resp = torch.pow(resp + eps, model.channel_exp)
        vis = torch.pow(exp_resp.sum(dim=1)+eps, 1/model.channel_exp).unsqueeze(1) * model.scaling

        vis = F.interpolate(vis, scale_factor=2, mode='nearest')

        vis_sign = vis-target_vis.view(-1,1,1,1)
        opt_alpha_map[vis_sign>0]-=step
        opt_alpha_map[vis_sign<0]+=step

        step /= 2
    
    return opt_alpha_map#*model.mask_gp[0]

class Fukiage2014(SuperModel):
    #stdによる正規化を行う。NLP_Xと同じ

    def __init__(self, levels, device):
        super(Fukiage2014, self).__init__(levels, device)
        self.level = levels
        self.col_conversion = 'lab'

        #lfilt = np.array([0.019849565,-0.043090917,-0.051887936,0.29323120,0.56379618,0.29323120,-0.051887936,-0.043090917,0.019849565]).T
        # weight[0]=0.7973934;
        # weight[1]=0.41472545;
        # weight[2]=-0.073386624;
        # weight[3]=-0.060944743;
        # weight[4]=0.02807382;
        lfilt = np.array([0.02807382,-0.060944743,-0.073386624,0.41472545,0.7973934,0.41472545,-0.073386624,-0.060944743,0.02807382]).T

        # reshape lfilt to column vector
        if len(lfilt.shape) == 1:
            lfilt = lfilt.reshape(len(lfilt), 1)
        elif lfilt.shape[0] == 1:
            lfilt = lfilt.reshape(lfilt.shape[1], 1)

        sz = len(lfilt)
        sz2 = np.ceil(sz/2.0)
        ind = np.array(range(sz-1,-1,-1))
        hfilt = lfilt[ind].T * (-1)**((ind+1)-sz2)

        # matlab version always returns a column vector
        if len(hfilt.shape) == 1:
            hfilt = hfilt.reshape(len(hfilt), 1)
        elif hfilt.shape[0] == 1:
            hfilt = hfilt.reshape(hfilt.shape[1], 1)

        channels = 1
        kernel = torch.tensor(lfilt,dtype=torch.float32,device=device)
        self.lo_filt_v = kernel.repeat(channels, 1, 1, 1)

        kernel = torch.tensor(hfilt,dtype=torch.float32,device=device)
        self.hi_filt_v = kernel.repeat(channels, 1, 1, 1)

        kernel = torch.tensor(lfilt.reshape(1,sz),dtype=torch.float32,device=device)
        self.lo_filt_h = kernel.repeat(channels, 1, 1, 1)

        kernel = torch.tensor(hfilt.reshape(1,sz),dtype=torch.float32,device=device)
        self.hi_filt_h = kernel.repeat(channels, 1, 1, 1)

        pad_num = (sz-1)//2
        self.pad_h = nn.ReflectionPad2d((pad_num,pad_num,0,0))
        self.pad_v = nn.ReflectionPad2d((0,0,pad_num,pad_num))


        ## parameters
        self.channel_exp = nn.Parameter(torch.tensor(4.5))
        self.spat_exp = nn.Parameter(torch.tensor(2.2))

        S_y = []#torch.ones((self.levels)) * 40.0
        S_y_init = [0.0, 0.14518068, 36.638073, 40.0]
        for i in range(self.level):
            if i < len(S_y_init):
                S_y.append(S_y_init[i])
            elif i == self.level - 1:
                S_y.append(0.35)
            else:
                S_y.append(40.0)

        reparam_offset=2**-18
        self.reparam_offset = torch.FloatTensor([reparam_offset])
        self.pedestal = self.reparam_offset**2
            
        weight_min = 1e-8
        self.weight_bound = (weight_min + self.reparam_offset**2)**.5
        self.weight_bound = self.weight_bound.to(device)
        channel_weight = torch.sqrt(torch.tensor(S_y)+self.pedestal).to(device)
        self.channel_weight = nn.Parameter(channel_weight)
        #self.channel_weight = nn.Parameter(torch.ones((self.num_channels_all)))
        
        self.band_sigma = nn.Parameter(torch.tensor(3.0))
        self.freq_sigma = nn.Parameter(torch.tensor(0.25))
        self.interaction = torch.ones(((self.level-1)*3,(self.level-1)*3),dtype=torch.float32).to(device)

    
        self.d_ratio_min = 0.6#(0.25)**.5
        self.d_bound = (self.d_ratio_min + self.reparam_offset**2)**.5
        self.d_bound = self.d_bound.to(device)
        self.d_ratio_max = 1.4
        self.d_upper_bound = (self.d_ratio_max - self.reparam_offset**2)**.5
        self.d_upper_bound = self.d_upper_bound.to(device)
        d_ratio = torch.sqrt(torch.ones((1))*0.8+self.pedestal)
        self.d_ratio = nn.Parameter(d_ratio.to(device))

        self.gamma_min = 0.5#(0.25)**.5
        self.gamma_bound = (self.gamma_min + self.reparam_offset**2)**.5
        self.gamma_bound = self.gamma_bound.to(device)
        self.gamma_max = 3.0
        self.gamma_upper_bound = (self.gamma_max - self.reparam_offset**2)**.5
        self.gamma_upper_bound = self.gamma_upper_bound.to(device)
        gamma = torch.sqrt(torch.ones((1))*1.7+self.pedestal)
        self.gamma = nn.Parameter(gamma.to(device))

        self.beta_min = 1e-6#(0.25)**.5
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.beta_bound = self.beta_bound.to(device)
        beta = torch.sqrt(torch.ones((1))*10.3+self.pedestal)
        self.beta = nn.Parameter(beta.to(device))

        self.pedestal = self.pedestal.to(self.device)

        self.std_vector = nn.Parameter(torch.zeros(((self.level-1)*3)),requires_grad=False)
        self.std_vector.data[0]=self.std_vector.data[2]=0.3/10.3
        self.std_vector.data[1]=0.2/10.3
        self.std_vector.data[3]=self.std_vector.data[5]=0.8/10.3
        self.std_vector.data[4]=0.5/10.3
        self.std_vector.data[6]=self.std_vector.data[8]=1.9/10.3
        self.std_vector.data[7]=1.1/10.3
        self.std_vector.data[9]=self.std_vector.data[11]=4.6/10.3
        self.std_vector.data[10]=2.7/10.3


        self.scaling = nn.Parameter(torch.tensor(2.0))

    def gen_QMFpyr(self, image):

        J = image
        dims = image.shape[1]
        pyr = []
        
        for i in range(0, self.level-1):
            #horizontal
            padded_h = self.pad_h(J)
            lo_h = F.conv2d(padded_h, self.lo_filt_h, stride=(2,1), padding=0,groups=dims)
            hi_h = F.conv2d(padded_h, self.hi_filt_h, stride=(2,1), padding=0,groups=dims)

            padded_lo = self.pad_v(lo_h)
            padded_hi = self.pad_v(hi_h)
            hihi = F.conv2d(padded_hi, self.hi_filt_v, stride=(1,2), padding=0,groups=dims)
            hilo = F.conv2d(padded_hi, self.lo_filt_v, stride=(1,2), padding=0,groups=dims)
            lohi = F.conv2d(padded_lo, self.hi_filt_v, stride=(1,2), padding=0,groups=dims)
            lolo = F.conv2d(padded_lo, self.lo_filt_v, stride=(1,2), padding=0,groups=dims)

            pyr.append(torch.cat([hilo,hihi,lohi],dim=1))

            J=lolo

        pyr.append(J)
        
        return pyr


    
    def set_std_vector(self, info_pyr):
        std_vector = []
        for i in range(self.level-1):
            std_vector.append(info_pyr[i]['std'])
        self.std_vector = nn.Parameter(torch.cat(std_vector,dim=0))
        self.std_vector.requires_grad=False

        # if self.running_std and self.training:
        #     #ここでlowpass dataを埋めておく。lowpass成分はrunning averageで置き換わらないので固定。
        #     for i in range(self.num_running):
        #         self.std_running[i]=self.std_vector.data

    def compute_std(self, img, info_pyr, show_plot=False):
        
        with torch.no_grad():

            labimg = self.bgr2lab(img)

            pyr = self.gen_QMFpyr(labimg[:,0,:,:].unsqueeze(1))
            
            for i in range(self.level):
                contrast = pyr[i]
                info_pyr[i]['std'] += torch.sqrt((contrast**2).mean(dim=(2,3))).sum(dim=0) #[B,C]
                

                info_pyr[i]['count']+=contrast.shape[0]
        return info_pyr
    
    def div_norm(self, pyr):

        channel_weight = LowerBound.apply(self.channel_weight, self.weight_bound)
        channel_weight = channel_weight**2 - self.pedestal 

        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 

        d_ratio = LowerBound.apply(self.d_ratio, self.d_bound)
        d_ratio = LowerBound.apply(-d_ratio, -self.d_upper_bound)
        d_ratio = d_ratio**2 - self.pedestal

        
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = LowerBound.apply(-gamma, -self.gamma_upper_bound)
        gamma = gamma**2 - self.pedestal

        #construct interaction kernel
        interaction = []
        for i in range(self.interaction.shape[0]):
            tmp_kernel = []
            for k in range(self.interaction.shape[1]):
                band_idx_i = i%3
                freq_idx_i = i//3
                band_idx_k = k%3
                freq_idx_k = k//3
                tmp_kernel.append(torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2))
                #self.interaction[i,k]=torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2)
            tmp_kernel = torch.stack(tmp_kernel)
            interaction.append(tmp_kernel)
        interaction = torch.stack(interaction,dim=0)
        interaction = interaction / interaction.sum(dim=1).unsqueeze(1)
        interaction = interaction.view((self.level-1)*3, (self.level-1)*3, 1, 1)
        # self.interaction = self.interaction / self.interaction.sum(dim=1).unsqueeze(1)
        # interaction = self.interaction.view((self.level-1)*3, (self.level-1)*3, 1, 1)

    
        #upsampling 
        for i in range(self.level):
            num_rep = i
            if i==self.level-1:
                num_rep = self.level-2
            for rep in range(num_rep):
                pyr[i] = self.upsample(pyr[i])
        
        cat_pyr = []
        for i in range(self.level-1):
            wc = pyr[i] * channel_weight[i]
            wc[:,1,:,:] *= d_ratio
            # if i<self.level-1:
            #     #diagonal
            #     wc[:,1,:,:] *= d_ratio
            #     #exitation
            #     #ewc = torch.pow(torch.abs(wc)+eps,gamma)
            cat_pyr.append(wc)
        cat_pyr = torch.cat(cat_pyr,dim=1)
        exp_pyr = torch.pow(torch.abs(cat_pyr)+eps,gamma)
        
        i = self.level-1
        lowband = pyr[i] * channel_weight[i]

        beta = torch.pow(self.std_vector * beta, gamma)
        denom = F.conv2d(exp_pyr, interaction, beta)

        resp_cat = torch.sign(cat_pyr)*exp_pyr/denom
        resp = torch.cat([resp_cat,lowband],dim=1)

        return resp


    def compute_vis_resp_control(self, bg, fg, alphamap, blend_mode = 'linear'):
        
        #元の背景画像とブレンド画像の差分を計算する

        #NLPDの場合はYUVをそのままそれぞれNLPDにかけてkernelを学習させる。重みはno attentionに基づいて学習させるのがfairかな
        #alpha-NLPのcontrolでは背景とブレンドそれぞれをband limited contrast -> normalizationにかける

        with torch.no_grad():

            #blendimg = alphamap * fg + (1-alphamap) * bg
            blendimg = self.blending(fg, bg, alphamap, blend_mode)

            blendimg = self.bgr2lab(blendimg)
            bgimg = self.bgr2lab(bg)

            pyr_blend = self.gen_QMFpyr(blendimg[:,0,:,:].unsqueeze(1))
            pyr_bg = self.gen_QMFpyr(bgimg[:,0,:,:].unsqueeze(1))


        # resp_bg = self.GDN_band(pyr_bg, pyr_bginh,self.std_vector)
        # resp_blend = self.GDN_band(pyr_blend, pyr_blendinh,self.std_vector)

        resp_blend = self.div_norm(pyr_blend)
        resp_bg = self.div_norm(pyr_bg)

        # resp=[]
        # for i in range(self.level):
        #     resp.append(resp_blend[i]-resp_bg[i])

        return torch.abs(resp_blend-resp_bg)
    
    def compute_vis_resp_control_lab(self, bg, fg, alphamap):
        
        #元の背景画像とブレンド画像の差分を計算する

        #NLPDの場合はYUVをそのままそれぞれNLPDにかけてkernelを学習させる。重みはno attentionに基づいて学習させるのがfairかな
        #alpha-NLPのcontrolでは背景とブレンドそれぞれをband limited contrast -> normalizationにかける

        with torch.no_grad():

            # blendimg = alphamap * fg + (1-alphamap) * bg

            # blendimg = self.bgr2lab(blendimg)
            # bgimg = self.bgr2lab(bg)

            blendimg = alphamap * fg + (1-alphamap) * bg

            pyr_blend = self.gen_QMFpyr(blendimg[:,0,:,:].unsqueeze(1))
            pyr_bg = self.gen_QMFpyr(bg[:,0,:,:].unsqueeze(1))


        # resp_bg = self.GDN_band(pyr_bg, pyr_bginh,self.std_vector)
        # resp_blend = self.GDN_band(pyr_blend, pyr_blendinh,self.std_vector)

        resp_blend = self.div_norm(pyr_blend)
        resp_bg = self.div_norm(pyr_bg)

        # resp=[]
        # for i in range(self.level):
        #     resp.append(resp_blend[i]-resp_bg[i])

        return torch.abs(resp_blend-resp_bg)

    def projection(self):
        #apply projection (PGD) for constrained optimization
        #with torch.no_grad():
        self.band_sigma.data=self.band_sigma.data.clamp(min=0.01)
        self.freq_sigma.data=self.freq_sigma.data.clamp(min=0.01)

        self.channel_exp.data=self.channel_exp.data.clamp(min=0.5, max = 8.0)
        
        self.spat_exp.data=self.spat_exp.data.clamp(min=0.5, max = 8.0)

        self.scaling.data=self.scaling.data.clamp(min=0.1)
        
    def showParams(self, required_grad_only=True):
        for name, param in self.named_parameters():
            if required_grad_only:
                if param.requires_grad:
                    print(name)#, param.data)
                    #print(param.data.shape)
            else:
                print(name, param.data)
        return
    
    
    def set_param(self, param_name, param):
        return
        
    def get_param_bound(self, param_name):
        return
    
    def get_params(self):
        param={}
        beta = ((self.beta.data)**2-self.pedestal.data).clone().cpu().numpy()
        #beta = ((self.GDN_band.beta.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()
        param['beta']=beta

        channel_weight = (self.channel_weight.data**2 - self.pedestal).clone().cpu().numpy()
        param['channel_weight']=channel_weight

        d_ratio = (self.d_ratio.data**2 - self.pedestal).clone().cpu().numpy()
        param['d_ratio']=d_ratio

        gamma = (self.gamma.data**2 - self.pedestal).clone().cpu().numpy()
        param['gamma']=gamma

        param['band_sigma']=self.band_sigma.data.clone().cpu().numpy()
        param['freq_sigma']=self.freq_sigma.data.clone().cpu().numpy()
        #construct interaction kernel

        interaction = []
        for i in range(self.interaction.shape[0]):
            tmp_kernel = []
            for k in range(self.interaction.shape[1]):
                band_idx_i = i%3
                freq_idx_i = i//3
                band_idx_k = k%3
                freq_idx_k = k//3
                tmp_kernel.append(torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2))
                #self.interaction[i,k]=torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2)
            tmp_kernel = torch.stack(tmp_kernel)
            interaction.append(tmp_kernel)
        interaction = torch.stack(interaction,dim=0)
        interaction = interaction / interaction.sum(dim=1).unsqueeze(1)
        #interaction = interaction.view((self.level-1)*3, (self.level-1)*3, 1, 1)

        interaction = interaction.cpu().detach().numpy()
        param['interaction']=interaction

        param['channel_exp']=self.channel_exp.data.clone().cpu().numpy()
        param['spat_exp']=self.spat_exp.data.clone().cpu().numpy()
        param['scaling']=self.scaling.data.clone().cpu().numpy()

        return param

    def visualize_weights(self, showplot=False):
        #v1filt = self.conv_main.weight.data.clone().cpu().numpy()
        if self.running_std:
            print(self.std_vector)
        
        
        #gamma_weight = ((self.GDN_band.gamma_weight.data)**2-self.GDN_band.pedestal.data).clone().cpu().numpy()#self.GDN_list_high[i].gamma.data.clone().cpu().numpy()
        beta = ((self.beta.data)**2-self.pedestal.data).clone().cpu().numpy()#self.GDN_list_high[i].beta.data.clone().cpu().numpy()
        #gamma_sigma = self.GDN_band.gamma_sigma.data.clone().cpu().numpy()
        #print('bandpass gamma_weight', gamma_weight)
        print('beta band', beta)
        #print('bandpass gamma_sigma', gamma_sigma)
        
        channel_weight = (self.channel_weight.data**2 - self.pedestal).clone().cpu().numpy()
        print('channel weight', channel_weight)

        d_ratio = (self.d_ratio.data**2 - self.pedestal).clone().cpu().numpy()
        print('d_ratio', d_ratio)

        gamma = (self.gamma.data**2 - self.pedestal).clone().cpu().numpy()
        print('gamma', gamma)

        print('band_sigma', self.band_sigma)
        print('freq_sigma', self.freq_sigma)
        #construct interaction kernel

        interaction = []
        for i in range(self.interaction.shape[0]):
            tmp_kernel = []
            for k in range(self.interaction.shape[1]):
                band_idx_i = i%3
                freq_idx_i = i//3
                band_idx_k = k%3
                freq_idx_k = k//3
                tmp_kernel.append(torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2))
                #self.interaction[i,k]=torch.exp(-((band_idx_i-band_idx_k)/self.band_sigma)**2 - ((freq_idx_i-freq_idx_k)/self.freq_sigma)**2)
            tmp_kernel = torch.stack(tmp_kernel)
            interaction.append(tmp_kernel)
        interaction = torch.stack(interaction,dim=0)
        interaction = interaction / interaction.sum(dim=1).unsqueeze(1)
        # interaction = interaction.view((self.level-1)*3, (self.level-1)*3, 1, 1)
        interaction = interaction.cpu().detach().numpy()

        if showplot:
            fig, ax = plt.subplots()
            #ax.imshow(dnfilt[i,0,:,:]/dnfilt[i].max())
            mm=ax.imshow(interaction)
            fig.colorbar(mm, ax=ax)
            ax.grid(False)
            plt.show()
        
        print("channel_exp", self.channel_exp.data)
        print("spat_exp", self.spat_exp.data)
        print("scaling", self.scaling.data)
    
    
    def get_name(self):
        return 'ISMAR2014'#+self.mode
    
   
    def pool(self, resp):

        exp_resp = torch.pow(resp + eps, self.channel_exp)
        
        agg_resp = torch.pow(exp_resp.sum(dim=1)+eps, 1/self.channel_exp)
        
        exp_agg_resp = torch.pow(agg_resp+eps, self.spat_exp)
        out = torch.pow(exp_agg_resp.sum(dim=(1,2))/self.mask_gp_sum[1]+eps, 1/self.spat_exp)
        
        return out

    def compare(self, bg, fg, alphamap, blend_mode='linear'):

        resp = self.compute_vis_resp_control(bg, fg, alphamap, blend_mode=blend_mode)

        vis = self.pool(resp)

        return vis * self.scaling

