## This script calculates the optical factor of human eye based on photometry 
import torch
import pycvvdp
import pycvvdp.display_model as display_model
from pycvvdp.video_source import reshuffle_dims
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import math
deg2rad = math.pi/180
rad2deg = 180/math.pi
mm, um, nm= 1e-3, 1e-6, 1e-9

def pupil_d_unified(L,area,age):
    ## This function calculates the pupil diameter in mm with a respect to 
    ## L: luminance (cd/m^2)
    ## area: area
    ## age: user age
    clamp = lambda x, max_val, min_val: max(min(x,max_val),min_val)

    y0 = 28.58 # reference age from the reference
    y = clamp(age, 20, 83)
   
    La = L * area 
    pd_sd =   7.75 - 5.75 * ((La/846)**(0.41) / ((La/846)**0.41+2))
    pd = pd_sd + (y-y0)*(0.02132 - 0.009562*pd_sd)

    return pd 

## Fourier transform of torch tensor
def FT2(tensor):
    """ Perform 2D fft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ft_shift = torch.fft.fft2(tensor_shift, norm='ortho')
    tensor_ft = torch.fft.fftshift(tensor_ft_shift, dim=(-2,-1))
    return tensor_ft


def iFT2(tensor):
    """ Perform 2D ifft of a tensor for last two dimensions """
    tensor_shift = torch.fft.ifftshift(tensor, dim=(-2,-1))
    tensor_ift_shift = torch.fft.ifft2(tensor_shift, norm='ortho')
    tensor_ift = torch.fft.fftshift(tensor_ift_shift, dim=(-2,-1))
    return tensor_ift

def gaussian(x, mu, sigma):
    """
    Compute the Gaussian function.
    """
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def preview_enlarged_image(f, crop_window=None, center=None, scale=None, num_tick=None, xlabel=None, ylabel=None, title=None):
    '''
    this function previews the cropped image of input tensor with size of (BCFHW: 1C1HW)
    '''
    
    f_np = np.rollaxis(f.cpu().squeeze().numpy(),0,3) 
    if center is None:
        center = (f.shape[-2]//2, f.shape[-1]//2)
    if crop_window is None:
        crop_window = (f.shape[-2],f.shape[-1])
    f_crop = f_np[center[0]-crop_window[0]//2:center[0]+crop_window[0]//2+1, \
                            center[1]-crop_window[1]//2:center[1]+crop_window[1]//2+1,:]
    plt.imshow(f_crop/np.max(f_crop))
    if num_tick is not None and scale is not None:
        ax = plt.gca()
        xx = scale[1]*np.arange(-(crop_window[1]//2), crop_window[1]//2 + 1, crop_window[1]//(num_tick-1))
        yy = scale[0]*np.arange(-(crop_window[0]//2), crop_window[0]//2 + 1, crop_window[0]//(num_tick-1))
        ax.set_xticks(np.arange(0, crop_window[1] + 1, crop_window[1]//(num_tick-1)))
        ax.set_yticks(np.arange(0, crop_window[0] + 1, crop_window[0]//(num_tick-1)))
        ax.set_xticklabels(np.round(xx))
        ax.set_yticklabels(np.round(yy))
    
    if xlabel is not None and ylabel is not None:
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.show()

class optics_model: 
    def __init__(self, age= None, pd=None, test_frame=None, bkg_frame = None, test_dp=None, test_dg=None, bkg_dp=None, bkg_dg=None, device=None,Num_layer=2):
        
        test_frame = test_frame.squeeze() ## C x H x W
        bkg_frame = bkg_frame.squeeze() ## C x H x W
        
        self.test_frame = test_frame
        self.bkg_frame = bkg_frame
        self.test_dp = test_dp
        self.test_dg = test_dg
        self.bkg_dp = bkg_dp
        self.bkg_dg = bkg_dg
        self.device = device
        
        if isinstance(Num_layer, int):
            self.Num_layer = Num_layer
        else:
            RuntimeError('Define integer for number of layers') 

        self.resolution = (test_frame.shape[-2], test_frame.shape[-1]) ## Resolution of test frame (H,W)
        self.size_m = (self.test_dg.display_size_m[-1]/self.test_dg.resolution[-1]*self.resolution[-2], self.test_dg.display_size_m[0]/self.test_dg.resolution[0]*self.resolution[-1]) # (H,W)
        self.display_size_deg = (2*math.atan(self.size_m[0]/self.test_dg.distance_m/2)*rad2deg, 2*math.atan(self.size_m[1]/self.test_dg.distance_m/2)*rad2deg) # (H,W)
        self.area = self.display_size_deg[0] * self.display_size_deg[1] ## Area in deg^2

        self.set_luminance() ## Set luminance based on the display model
        self.set_pd(pd,age) ## Set pupil diameter based on luminance and age

        self.set_dioptric_distance()
        self.set_angle_domain()

    def set_dioptric_distance(self):
        self.D =1/self.test_dg.distance_m - 1/self.bkg_dg.distance_m

    def set_luminance(self):
        '''
        Calculate the luminance based on the display model
        '''
        test_rgb2y = self.test_dp.rgb2xyz_list[1]
        bkg_rgb2y = self.bkg_dp.rgb2xyz_list[1]

        tl = torch.mean(self.test_frame[0,:,:]*test_rgb2y[0] + self.test_frame[1,:,:]* test_rgb2y[1]+self.test_frame[2,:,:]*test_rgb2y[2])
        if len(self.bkg_frame.shape) ==2:
            bl = torch.mean(self.bkg_frame)
        else:
            bl = torch.mean(self.bkg_frame[0,:,:]*bkg_rgb2y[0] + self.bkg_frame[1,:,:]*bkg_rgb2y[1]+self.bkg_frame[2,:,:]*bkg_rgb2y[2])

        self.luminance = (tl+bl).float()

    def set_pd(self,pd=None,age=None):
        '''
        Set pupil diameter based on mean luminance and age
        '''
        if pd is None and age is not None:
            self.pd = pupil_d_unified(self.luminance, self.area, age)
        elif age is None:
            self.pd = pupil_d_unified(self.luminance, self.area, 30)
        if pd is not None:
            self.pd = pd 

    def set_angle_domain(self):
        w = self.test_frame.shape[-1]
        h = self.test_frame.shape[-2] 
        x_deg = torch.linspace(-self.display_size_deg[1]/2,self.display_size_deg[1]/2,w+1).to(self.device) # To include 0 
        y_deg = torch.linspace(-self.display_size_deg[0]/2,self.display_size_deg[0]/2,h+1).to(self.device)
        self.Y_deg, self.X_deg = torch.meshgrid(y_deg, x_deg)

    def calculate_psf_ray(self, D= None,wavelengths=None, is_preview=False):
        '''
        Calculate the psf based on geometric optics (using circle of confusion)
        wavelengths: tuple of wavelength (r,g,b) in a unit of nm
        '''
        w = self.test_frame.shape[-1]
        h = self.test_frame.shape[-2] 
        Y_deg = self.Y_deg
        X_deg = self.X_deg
        psf = torch.zeros_like(self.test_frame)
        if D is None:
            D = self.D

        if wavelengths is None:
            bd_deg = rad2deg*D*self.pd*mm
            if ( bd_deg.device!='cpu'):
                bd_deg = bd_deg.to('cpu')
            sigma = 0.55 * bd_deg / 2 ## Matching size based on Chromablur paper
            tmp_psf = gaussian(torch.sqrt(X_deg**2 + Y_deg**2), 0, sigma)
            tmp_psf = F.interpolate(tmp_psf.unsqueeze(0).unsqueeze(0), size=(h,w), mode='bilinear', align_corners=False).squeeze()
            tmp_psf = tmp_psf / torch.sum(tmp_psf) 
            tmp_psf = reshuffle_dims(tmp_psf, in_dims = "HW", out_dims = "BCFHW") ## Making the domain shape identical
            psf = tmp_psf.repeat(1,3,1,1,1) ## expand in color channel
        else: # Implement chromatic defocus
            raise RuntimeError('Define proper wavelengths')   
        
        psf=psf.to(self.device)
        if is_preview:
            preview_enlarged_image(psf, crop_window=(21,21), \
                        scale=(60*self.display_size_deg[0]/h, 60*self.display_size_deg[1]/w), \
                        num_tick=3, \
                        xlabel ='angle[arcmin]',\
                            ylabel='angle[armin]',\
                                title = f'pd: {self.pd:.2f} mm / D: {D:.2f} D')
        return psf

    def gen_psf(self, PREVIEW = False):
        psf = torch.zeros(size=(self.Num_layer,3,1,*self.resolution)).to(self.device)
        D_vec = torch.linspace(0,self.D, self.Num_layer)

        for d in range(self.Num_layer):
            psf[d,:,:,:,:] =  self.calculate_psf_ray(D = D_vec[d], is_preview=PREVIEW) 

        return psf


    def get_blur_image(self,bkg_frame, psf, is_preview =False):            
        blur_frame = torch.zeros(size=(1,3,1,*self.resolution),device=self.device) 

        if len(psf.size())==4: ## In case of dimension reduction 
            psf = psf.unsqueeze(0)

        for c in range(3):
            if (bkg_frame.shape[1]==1): # gray scale input
                blur_frame[:,c,:,:,:] = torch.abs(iFT2(FT2(bkg_frame)*FT2(psf[:,c,:,:,:])))
                blur_frame[:,c,:,:,:] = blur_frame[:,c,:,:,:] *torch.sum(bkg_frame)/ torch.sum(blur_frame[:,c,:,:,:]) # Energy preservation
            else:
                blur_frame[:,c,:,:,:] = torch.abs(iFT2(FT2(bkg_frame[:,c,:,:,:])*FT2(psf[:,c,:,:,:])))
                blur_frame[:,c,:,:,:] = blur_frame[:,c,:,:,:] *torch.sum(bkg_frame[:,c,:,:,:] )/ torch.sum(blur_frame[:,c,:,:,:]) # Energy preservation
        if is_preview:
            preview_enlarged_image(blur_frame, crop_window=(101,101), center=(270,580),\
                                   title= f'pd: {self.pd:.2f} mm / D: {self.del_D:.2f} D')

        return blur_frame
        
    def get_image(self, frame, bkg_frame, psf, fs_weight=None, mode = 'sum', idx =None):
        image = torch.zeros_like(frame)
        if len(psf.size())==4: ## In case of dimension reduction 
            psf = psf.unsqueeze(0)

        if fs_weight is None: # Put weight on the foreground
            fs_weight = torch.zeros(size=(self.Num_layer,))

        if len(fs_weight) != self.Num_layer:
           raise RuntimeError('Wrong inputs for the focal stack weight')

        if mode == 'sum':
            loop_range = self.Num_layer
        elif mode == 'individual':
            loop_range = 1
        else:
            raise RuntimeError('Please pick the recon mode')
            
        for d in range(loop_range):
            if mode=='individual':
                d = idx
                fs_weight[d] = 1

            if d==0:
                image = image + fs_weight[d] *(frame + self.get_blur_image(bkg_frame,psf[self.Num_layer-1-d,:,:,:,:]))
            elif d==self.Num_layer-1:
                image = image + fs_weight[d] *(self.get_blur_image(frame,psf[d,:,:,:,:]) + bkg_frame)
            else:
                image = image + fs_weight[d] *(self.get_blur_image(frame,psf[d,:,:,:,:]) + self.get_blur_image(bkg_frame,psf[self.Num_layer-1-d,:,:,:,:]))

        return image