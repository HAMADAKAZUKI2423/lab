from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from .lowerBond import LowerBound

eps = 1e-8

class CustomGDN_NLP_Y(nn.Module):

    def __init__(self,
                 ch,
                 ksize,
                 device,
                 level,
                 beta_min=1e-6,
                 #gamma_init=.1,
                 reparam_offset=2**-18):
        super(CustomGDN_NLP_Y, self).__init__()

        self.beta_min = beta_min
        self.reparam_offset = torch.FloatTensor([reparam_offset])        

        self.pad_surround = nn.ReflectionPad2d((ksize-1)//2)
        #self.spat_kernel = nn.Parameter(nn.init.uniform_(torch.empty(ch,1,ksize,ksize).to(device), a=self.gamma_bound.data[0], b=1.0/np.sqrt(ch)))
        
        self.ch=ch
        self.ksize=ksize
        self.level=level
        
        self.build(ch, torch.device(device))
  
    def build(self, ch, device):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2)**.5
        self.beta_bound = self.beta_bound.to(device)
        self.gamma_bound = self.reparam_offset.to(device)
  
        # Create beta param
        if ch==1:
            self.gather_index= torch.tensor([0]).to(device)
        else:
            self.gather_index= torch.tensor([0,0,0]).to(device)
        beta = torch.sqrt(torch.ones((1))+self.pedestal)
    

        self.beta = nn.Parameter(beta.to(device))
        # self.beta_low = nn.Parameter(beta.to(device))
            
        g = torch.ones((1,1,self.ksize,self.ksize))/self.ksize/self.ksize
        g = g + self.pedestal
        dn_filt = torch.sqrt(g)
        self.dn_filt = nn.Parameter(dn_filt.to(device))
        self.dn_filt.requires_grad=False

        self.alpha_min = 0.1#(0.25)**.5
        self.alpha_bound = (self.alpha_min + self.reparam_offset**2)**.5
        self.alpha_bound = self.alpha_bound.to(device)

        alpha = torch.sqrt(torch.ones((1))*0.5+self.pedestal)
        
        self.alpha = nn.Parameter(alpha.to(device))
        # self.alpha_low = nn.Parameter(alpha.to(device))

        self.alpha_max = 0.5
        self.alpha_upper_bound = (self.alpha_max - self.reparam_offset**2)**.5
        self.alpha_upper_bound = self.alpha_upper_bound.to(device)

        self.pedestal = self.pedestal.to(device)

    def forward(self, inputs,inh,beta_base=None):

        _, ch, _, _ = inputs[0].size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal 
        beta = torch.gather(beta,0,self.gather_index)

        # beta_low = LowerBound.apply(self.beta_low, self.beta_bound)
        # beta_low = beta_low**2 - self.pedestal 
        # beta_low = torch.gather(beta_low,0,self.gather_index)
        
        alpha = LowerBound.apply(self.alpha, self.alpha_bound)
        #alpha = alpha**2 - self.pedestal
        alpha = LowerBound.apply(-alpha, -self.alpha_upper_bound)
        alpha = alpha**2 - self.pedestal
        alpha = torch.gather(alpha,0,self.gather_index)

        # alpha_low = LowerBound.apply(self.alpha_low, self.alpha_bound)
        # alpha_low = LowerBound.apply(-alpha_low, -self.alpha_upper_bound)
        # alpha_low = alpha_low**2 - self.pedestal 
        # alpha_low = torch.gather(alpha_low,0,self.gather_index)


        # Gamma bound and reparam
        dn_filt = LowerBound.apply(self.dn_filt, self.gamma_bound)
        dn_filt = dn_filt**2 - self.pedestal
        dn_filt = dn_filt/dn_filt.sum(dim=(2,3),keepdim=True)

        if ch>1:
            dn_filt = torch.stack([dn_filt[0],dn_filt[0],dn_filt[0]],dim=0)
        else:
            dn_filt = dn_filt[0].unsqueeze(0)
    
        resp_list = []
        for i in range(self.level):

            # Norm pool calc
            if i== self.level-1:
                norm_ = 1.
                # if beta_base is not None:
                #     tmp_beta = beta_base[i].view(1,-1,1,1) * beta_low.view(1,-1,1,1)
                # else:
                #     tmp_beta = beta_low.view(1,-1,1,1)
                
                # norm_ = inh[i]**2 + tmp_beta
                # norm_ = torch.pow(norm_, alpha_low.view(1,-1,1,1))
                
            else:
                if beta_base is not None:
                    tmp_beta = beta_base[i].view(1,-1,1,1) * beta.view(1,-1,1,1)
                else:
                    tmp_beta = beta.view(1,-1,1,1)

                norm_ = F.conv2d(self.pad_surround(inh[i]**2), dn_filt, stride=1, padding=0, groups=ch)
                #from IPython.core.debugger import Pdb; Pdb().set_trace()
                norm_ = norm_ + tmp_beta
                #norm_ = norm_ + beta*(beta_base[i].view(1,-1,1,1))
                
                norm_ = torch.pow(norm_, alpha.view(1,-1,1,1))
                
            # Apply norm
            outputs = inputs[i] / (norm_ + eps)

            resp_list.append(outputs)

        return resp_list