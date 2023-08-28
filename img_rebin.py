#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 13:23:10 2022
@author: jlee
"""

# Importing necessary modules
import numpy as np
import glob, os, copy
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp


wht_img = "NGC1087_PHANGS_IMAGE_white_copt_0.92asec.fits"

d, h = fits.getdata(wht_img, header=True)
w = wcs.WCS(h)

nx, ny = h['NAXIS1'], h['NAXIS2']


### For the case when pxs_target / pxs_source is integer,
def make_rebin(pxs_target, out, pxs_MUSE=0.2):
    pixbin = round(pxs_target / pxs_MUSE)

    xi, xf = (nx % pixbin) // 2, nx - (nx % pixbin - nx % pixbin // 2) 
    yi, yf = (ny % pixbin) // 2, ny - (ny % pixbin - ny % pixbin // 2)

    assert (xf-xi) % pixbin == 0
    assert (yf-yi) % pixbin == 0

    d2 = d[yi:yf, xi:xf]
    d2_view = d2.reshape(d2.shape[0] // pixbin, pixbin, d2.shape[1] // pixbin, pixbin)
    sd2 = d2_view.sum(axis=3).sum(axis=1)

    w2 = copy.deepcopy(w)
    w2.wcs.crpix[0] = 0.5 + (w.wcs.crpix[0]-xi-0.5) / pixbin
    w2.wcs.crpix[1] = 0.5 + (w.wcs.crpix[1]-yi-0.5) / pixbin
    w2.wcs.cd = w.wcs.cd * pixbin
    h2 = w2.to_header()

    fits.writeto(out, sd2, h2, overwrite=True)


pxs_SPHEREx = 6.2  # arcsec/pixel
make_rebin(pxs_SPHEREx, "rebin.fits")

# pxs_CIGALE = 3.75  # arcsec/pixel
# # make_rebin(pxs_CIGALE, "rebin_3.75.fits")


# ### For the case when pxs_target / pxs_source is not integer,
# def make_rebin2(pxs_target, out, hdr_ref, wcs_ref, shape_ref, pxs_MUSE=0.2):
#     array, footprint = reproject_interp((d, w), wcs_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')
#     # kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#     #            "PC1_1", "PC2_2",
#     #            "CDELT1", "CDELT2"]

#     array[np.isnan(array)] = 0.

#     hdu = fits.PrimaryHDU()
#     # for kwd in kwd_ref:
#     #     hdu.header[kwd] = hdr_ref[kwd]

#     hdu.data = (pxs_target / pxs_MUSE)**2. * array
#     hdu.header = hdr_ref
#     hdu.writeto(out, overwrite=True)

# img_ref = "From_whlee/NGC_628_images/ph_2MASS_J_3.75.fits"
# d_ref, h_ref = fits.getdata(img_ref, header=True)
# w_ref = wcs.WCS(h_ref)
# shape_ref = d_ref.shape

# make_rebin2(pxs_CIGALE, "rebin_3.75.fits", h_ref, w_ref, shape_ref)


