#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 07:59:37 2022
@author: jlee
"""


# importing necessary modules
import glob
from os import path

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import pickle


# ----- Loading the Spectra ----- #
# Loading MUSE spectra data cube
dir_cube = "./"
filename = dir_cube + "NGC0628_PHANGS_DATACUBE_copt_0.92asec.fits"
sp = fits.open(filename)

d_sci, h_sci = sp[1].data, sp[1].header
d_var, h_var = sp[2].data, sp[2].header
wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      num=h_sci['NAXIS3'], endpoint=True)

nx, ny = d_sci.shape[2], d_sci.shape[1]
pxs_SPHEREx, pxs_MUSE = 6.2, 0.2    # arcsec/pixel
pixbin = round(pxs_SPHEREx / pxs_MUSE)

xi, xf = (nx % pixbin) // 2, nx - (nx % pixbin - nx % pixbin // 2)
yi, yf = (ny % pixbin) // 2, ny - (ny % pixbin - ny % pixbin // 2)

assert (xf-xi) % pixbin == 0
assert (yf-yi) % pixbin == 0


# ----- Loading the re-binned image ----- #
dat_rb, hdr_rb = fits.getdata("rebin.fits", header=True)
nx_bin, ny_bin = dat_rb.shape[1], dat_rb.shape[0]


# ----- Making the re-binned spectra ----- #
box_spec = {}
box_vari = {}

for ix in tqdm(range(nx_bin)):#tqdm(range(20, 27)):
    for iy in tqdm(range(ny_bin), leave=False):#tqdm(range(22,29), leave=False):#
        key_coord = f"x{ix:03d}_y{iy:03d}"
        x0, x1 = xi+pixbin*ix, xi+pixbin*(1+ix)
        y0, y1 = yi+pixbin*iy, yi+pixbin*(1+iy)
        box_spec[key_coord] = np.nansum(d_sci[:, y0:y1, x0:x1], axis=(1,2))
        box_vari[key_coord] = np.nansum(d_var[:, y0:y1, x0:x1], axis=(1,2))


# Save data
with open("box_spec_total.pickle","wb") as fw:
    pickle.dump(box_spec, fw)
with open("box_vari_total.pickle","wb") as fw:
    pickle.dump(box_vari, fw)


# # load data
# with open('user.pickle', 'rb') as fr:
#     user_loaded = pickle.load(fr)






