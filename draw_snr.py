#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 07:59:37 2022
@author: jlee
"""


# importing necessary modules
import glob
from os import path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


galaxy_name = "NGC1087"


# ----- Loading MUSE spectra data cube ----- #
dir_cube = "./"
filename = glob.glob(dir_cube+galaxy_name+"_PHANGS_DATACUBE_copt_*.fits")[0]
sp = fits.open(filename)
d_sci, h_sci = sp[1].data, sp[1].header
d_var, h_var = sp[2].data, sp[2].header
wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      num=h_sci['NAXIS3'], endpoint=True)


### SPHEREx-rebinned data ###

# ----- Loading the re-binned data ----- #
img_rb = fits.getdata("rebin.fits")
sp2 = fits.open(dir_cube+"DATACUBE_SPHEREx_extcor.fits")
d_sci2 = sp2[1].data
d_var2 = sp2[2].data

# ----- Making the S/N array ----- #
cont = ((wav_obs > 5000.) & (wav_obs < 5500.))
idx_cont = np.flatnonzero(cont)
snr_2D = np.zeros_like(img_rb)
for ix in np.arange(img_rb.shape[1]):
    for iy in np.arange(img_rb.shape[0]):
        snr_2D[iy, ix] = np.nanmedian(d_sci2[idx_cont, iy, ix] / np.sqrt(d_var2[idx_cont, iy, ix]))
fits.writeto("rebin_SNR.fits", snr_2D, overwrite=True)
