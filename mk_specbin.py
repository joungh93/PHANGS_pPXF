#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 07:59:37 2022
@author: jlee
"""


# importing necessary modules
import time
start_time = time.time()

import glob, os, copy
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import wcs
from tqdm import tqdm
import pickle
from reproject import reproject_interp
import extinction
ebv = 0.030
R_V = 3.1
A_V = R_V * ebv


##### For re-binned data #####

def extract_rebin(pxs_target, out, rebin_image, cube_file, pxs_MUSE=0.2, extcor=True):

    sp = fits.open(cube_file)

    hd0 = sp[0].header
    d_sci, h_sci = sp[1].data, sp[1].header
    d_var, h_var = sp[2].data, sp[2].header

    nx, ny = d_sci.shape[2], d_sci.shape[1]
    pixbin = round(pxs_target / pxs_MUSE)

    xi, xf = (nx % pixbin) // 2, nx - (nx % pixbin - nx % pixbin // 2)
    yi, yf = (ny % pixbin) // 2, ny - (ny % pixbin - ny % pixbin // 2)

    assert (xf-xi) % pixbin == 0
    assert (yf-yi) % pixbin == 0

    # ----- Loading the re-binned image ----- #
    dat_rb, hdr_rb = fits.getdata(rebin_image, header=True)
    nx_bin, ny_bin = dat_rb.shape[1], dat_rb.shape[0]

    # ----- Making the re-binned spectra ----- #
    # box_spec = {}
    # box_vari = {}
    sci_cb = np.zeros((d_sci.shape[0], dat_rb.shape[0], dat_rb.shape[1]))
    var_cb = np.zeros((d_var.shape[0], dat_rb.shape[0], dat_rb.shape[1]))
    for ix in tqdm(range(nx_bin)):
        for iy in tqdm(range(ny_bin), leave=False):
            key_coord = f"x{ix:03d}_y{iy:03d}"
            x0, x1 = xi+pixbin*ix, xi+pixbin*(1+ix)
            y0, y1 = yi+pixbin*iy, yi+pixbin*(1+iy)
            # box_spec[key_coord] = np.nansum(d_sci[:, y0:y1, x0:x1], axis=(1,2))
            # box_vari[key_coord] = np.nansum(d_var[:, y0:y1, x0:x1], axis=(1,2))
            if extcor:
                sci_cb[:, iy, ix] = np.nansum(d_sci[:, y0:y1, x0:x1], axis=(1,2)) * 10.**(0.4*Amags_c89)
                var_cb[:, iy, ix] = np.nansum(d_var[:, y0:y1, x0:x1], axis=(1,2)) * 10.**(0.4*Amags_c89)
            else:
                sci_cb[:, iy, ix] = np.nansum(d_sci[:, y0:y1, x0:x1], axis=(1,2))
                var_cb[:, iy, ix] = np.nansum(d_var[:, y0:y1, x0:x1], axis=(1,2))

    # # Save data
    # with open(out_spec,"wb") as fw:
    #     pickle.dump(box_spec, fw)
    # with open(out_vari,"wb") as fw:
    #     pickle.dump(box_vari, fw)

    kwds = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
    # keys_tar = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']#,
                # #'CD1_1' , 'CD2_2',  'CDELT1', 'CDELT2']
    # keys_ref = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']#,
                # #'PC1_1' , 'PC2_2',  'CDELT1', 'CDELT2']
    n_keys = len(kwds)

    fhd0 = fits.PrimaryHDU()
    fhd1 = fits.ImageHDU()
    fhd2 = fits.ImageHDU()
 
    fhd0.header = hd0
 
    for i in range(n_keys):
        h_sci[kwds[i]] = hdr_rb[kwds[i]]
        h_var[kwds[i]] = hdr_rb[kwds[i]]
    h_sci['CD1_1'] = hdr_rb['CDELT1'] * hdr_rb['PC1_1']
    h_sci['CD2_2'] = hdr_rb['CDELT2'] * hdr_rb['PC2_2']
    h_var['CD1_1'] = hdr_rb['CDELT1'] * hdr_rb['PC1_1']
    h_var['CD2_2'] = hdr_rb['CDELT2'] * hdr_rb['PC2_2']
        
    fhd1.data   = sci_cb
    fhd1.header = h_sci
    
    fhd2.data   = var_cb
    fhd2.header = h_var
     
    fcb_hdu = fits.HDUList([fhd0, fhd1, fhd2])
    fcb_hdu.writeto(out, overwrite=True)



# h2d = copy.deepcopy(h_sci)
# del h2d['CRVAL3']
# del h2d['CRPIX3']
# del h2d['CUNIT3']
# del h2d['CTYPE3']
# del h2d['CD3_3']
# del h2d['CD1_3']
# del h2d['CD2_3']
# del h2d['CD3_1']
# del h2d['CD3_2']
# del h2d['NAXIS3']
# h2d['NAXIS']   = 2
# h2d['WCSAXES'] = 2
# w2d = wcs.WCS(h2d)


# def cube_rebin(pxs_target, out, hdr_ref, wcs_ref, shape_ref, pxs_MUSE=0.2):
#     nw = d_sci.shape[0]

#     # sci_cb = np.zeros_like(d_sci)
#     sci_cb = np.zeros((nw, shape_ref[0], shape_ref[1]))
#     # var_cb = np.zeros_like(d_var)
#     var_cb = np.zeros((nw, shape_ref[0], shape_ref[1]))
    
#     nw = d_sci.shape[0]
#     for iw in tqdm(range(nw)):#np.arange(1000, 1010, 1):
#         arr_sci, footprint = reproject_interp((d_sci[iw, :, :], w2d), wcs_ref, shape_out=shape_ref,
#                                               order='nearest-neighbor')
#         # sci_cb[iw, :, :] = (pxs_target/pxs_MUSE)**2. * arr_sci
#         sci_cb[iw, :, :] = (pxs_target/pxs_MUSE)**2. * arr_sci * 10.**(0.4*Amags_c89[iw])
#         arr_var, footprint = reproject_interp((d_var[iw, :, :], w2d), wcs_ref, shape_out=shape_ref,
#                                               order='nearest-neighbor')
#         # var_cb[iw, :, :] = (pxs_target/pxs_MUSE)**2. * arr_var
#         var_cb[iw, :, :] = (pxs_target/pxs_MUSE)**2. * arr_var * 10.**(0.4*Amags_c89[iw])

#     kwds = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
#     # keys_tar = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']#,
#                 # #'CD1_1' , 'CD2_2',  'CDELT1', 'CDELT2']
#     # keys_ref = ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']#,
#                 # #'PC1_1' , 'PC2_2',  'CDELT1', 'CDELT2']
#     n_keys = len(kwds)
    
#     fhd0 = fits.PrimaryHDU()
#     fhd1 = fits.ImageHDU()
#     fhd2 = fits.ImageHDU()
 
#     fhd0.header = hd0
 
#     for i in range(n_keys):
#         h_sci[kwds[i]] = hdr_ref[kwds[i]]
#         h_var[kwds[i]] = hdr_ref[kwds[i]]
#     h_sci['CD1_1'] = hdr_ref['CDELT1'] * hdr_ref['PC1_1']
#     h_sci['CD2_2'] = hdr_ref['CDELT2'] * hdr_ref['PC2_2']
    
#     fhd1.data   = sci_cb
#     fhd1.header = h_sci
    
#     fhd2.data   = var_cb
#     fhd2.header = h_var
     
#     fcb_hdu = fits.HDUList([fhd0, fhd1, fhd2])
#     fcb_hdu.writeto(out, overwrite=True)


### For SPHEREx-binned data

# ----- Loading the Spectra ----- #
# Loading MUSE spectra data cube
dir_cube = "./"
filename = dir_cube + "NGC1087_PHANGS_DATACUBE_copt_0.92asec.fits"
sp = fits.open(filename)

hd0 = sp[0].header
d_sci, h_sci = sp[1].data, sp[1].header
d_var, h_var = sp[2].data, sp[2].header
wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      num=h_sci['NAXIS3'], endpoint=True)

w_sci = wcs.WCS(h_sci)
w_var = wcs.WCS(h_var)

Amags_c89 = extinction.ccm89(wav_obs, A_V, R_V, unit='aa')

pxs_SPHEREx = 6.2  # arcsec/pix
# extract_rebin(pxs_target, out, rebin_image, cube_file, pxs_MUSE=0.2, extcor=True)
extract_rebin(pxs_SPHEREx, dir_cube+"DATACUBE_SPHEREx_extcor.fits", dir_cube+"rebin.fits",
              filename, pxs_MUSE=0.2, extcor=True)

# dd1, hh1 = fits.getdata("rebin.fits", header=True)
# ww1 = wcs.WCS(hh1)

# # cube_rebin(pxs_SPHEREx, "DATACUBE_SPHEREx.fits", hh1, ww1, dd1.shape, pxs_MUSE=0.2)
# cube_rebin(pxs_SPHEREx, "DATACUBE_SPHEREx_extcor.fits", hh1, ww1, dd1.shape, pxs_MUSE=0.2)



# # ### For CIGALE-binned data
# pxs_CIGALE = 3.75  # arcsec/pix
# # extract_rebin(pxs_CIGALE, "box_spec_3.75.pickle", "box_vari_3.75.pickle",
#               # "rebin_3.75.fits", dir_cube + "NGC0628_PHANGS_DATACUBE_copt_0.92asec.fits", pxs_MUSE=0.2)

# # w2d = wcs.WCS(naxis=2)
# # w2d.wcs.ctype = [w_sci.wcs.ctype[0], w_sci.wcs.ctype[1]]
# # w2d.wcs.crval = w_sci.wcs.crval[:2]
# # w2d.wcs.crpix = w_sci.wcs.crpix[:2]
# # w2d.wcs.pc    = w_sci.wcs.cd[:2, :2]

# dd2, hh2 = fits.getdata("rebin_3.75.fits", header=True)
# ww2 = wcs.WCS(hh2)

# cube_rebin(pxs_CIGALE, "DATACUBE_3.75.fits", hh2, ww2, dd2.shape, pxs_MUSE=0.2)

# # extract_rebin(pxs_CIGALE, "box_spec_3.75.pickle", "box_vari_3.75.pickle",
# #               "rebin_3.75.fits", dir_cube + "NGC0628_PHANGS_DATACUBE_copt_0.92asec.fits", pxs_MUSE=0.2)


print(f"--- {time.time()-start_time:.4f} sec ---")

