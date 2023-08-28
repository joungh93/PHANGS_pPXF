#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:21:34 2023
@author: jlee
"""

# Importing necessary modules
import numpy as np
import glob, os, copy
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp


c = 2.99792e+5  # km/s


# ----- Galaxy distance ----- #
lum_dist = 15.9   # Mpc
dist_mod = 5.0*np.log10(lum_dist*1.0e+6 / 10.)


# ----- SPHEREx-binned Images ----- #
dat_rb, hdr_rb = fits.getdata("rebin.fits", header=True)
w_ref = wcs.WCS(hdr_rb)
shape_ref = dat_rb.shape
pxs_SPHEREx = 6.2    # arcsec/pixel


# # ----- CIGALE-binned Images ----- #
# dat_cg, hdr_cg = fits.getdata("rebin_3.75.fits", header=True)
# w_cig = wcs.WCS(hdr_cg)
# shape_cig = dat_cg.shape
# pxs_CIGALE = 3.75    # arcsec/pixel


# # ----- Spitzer/IRAC Images ----- #
# pxs_irac = 0.6    # arcsec/pixel
# dir_irac = "/data01/jhlee/DATA/Spitzer/IRAC/"
# img_irac = sorted(glob.glob(dir_irac+"NGC1087/Phot/maic_ch*.fits"))
# unc_irac = sorted(glob.glob(dir_irac+"NGC1087/Phot/munc_ch*.fits"))
# img_irac = img_irac[:2]    # ch1, ch2
# unc_irac = unc_irac[:2]    # ch1, ch2

# for i in range(len(img_irac)):
#   dat, hdr = fits.getdata(img_irac[i], header=True)
#   unc, hd  = fits.getdata(unc_irac[i], header=True)
#   w = wcs.WCS(hdr)
#   array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                       order='nearest-neighbor')
#   error, footprint = reproject_interp((unc, w), w_ref, shape_out=shape_ref,
#                                       order='nearest-neighbor')

#   kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#              "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#   kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#              "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
#   for k in range(len(kwd_ref)):
#       if ((k == 5) | (k == 6)):
#           hdr[kwd_tar[k]] = 0.0
#       else:
#           hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#   fits.writeto(f"IRAC{i+1:1d}_repr.fits", (pxs_SPHEREx/pxs_irac)**2. * array,
#                hdr, overwrite=True)
#   fits.writeto(f"IRAC{i+1:1d}_unc.fits",  (pxs_SPHEREx/pxs_irac)**2. * error,
#                hdr, overwrite=True)


# ----- S4G Images ----- #
pxs_S4G = 0.75    # arcsec/pixel
dir_S4G = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/S4G/"
img_S4G = sorted(glob.glob(dir_S4G+"NGC1087.*.final_sci.fits"))

for i in range(len(img_S4G)):
    dat, hdr = fits.getdata(img_S4G[i], header=True)
    # w = wcs.WCS(hdr)
    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
    w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
    w.wcs.cd    = np.array([[hdr['CD1_1'], hdr['CD1_2']],
                            [hdr['CD2_1'], hdr['CD2_2']]])

    kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
    kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

    ### SPHEREx
    hdr1 = copy.deepcopy(hdr)
    array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
                                        order='nearest-neighbor')

    for k in range(len(kwd_ref)):
        if ((k == 5) | (k == 6)):
            hdr1[kwd_tar[k]] = 0.0
        else:
            hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

    fits.writeto(f"S4G_ch{i+1:1d}_repr.fits", (pxs_SPHEREx/pxs_S4G)**2. * array,
                 hdr1, overwrite=True)

    # ### CIGALE
    # hdr2 = copy.deepcopy(hdr)
    # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
    #                                     order='nearest-neighbor')

    # for k in range(len(kwd_ref)):
    #     if (k < 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
    #     elif (k == 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
    #     elif ((k == 5) | (k == 6)):
    #         hdr2[kwd_tar[k]] = 0.0
    #     else:
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

    # fits.writeto(f"S4G_ch{i+1:1d}_repr_3.75.fits", (pxs_CIGALE/pxs_S4G)**2. * array,
    #              hdr2, overwrite=True)


# # ----- JWST Images ----- #
#
# ---> PSF convolution needed?
#
# pxs_jwst = 0.03    # arcsec/pixel
# dir_jwst = "/data01/jhlee/DATA/PHANGS/JWST/"
# img_jwst = sorted(glob.glob(dir_jwst+"NGC1087/NGC1087_nircam_*.fits"))

# dat, hdr = fits.getdata(img_jwst[0], header=True, ext=1)
# w = wcs.WCS(hdr)
# array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref, order='nearest-neighbor')

# kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#          "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
# kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#          "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
# for k in range(len(kwd_ref)):
#   if ((k == 5) | (k == 6)):
#       hdr[kwd_tar[k]] = 0.0
#   else:
#       hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]
# hdr["CDELT1"] = 1.0
# hdr["CDELT2"] = 1.0

# fits.writeto("NIRCAM_f200w_repr.fits", (pxs_SPHEREx/pxs_jwst)**2. * array, hdr, overwrite=True)


# ----- SDSS Images ----- #
pxs_sdss = 0.2    # arcsec/pixel
dir_sdss = "./"
img_sdss = sorted(glob.glob(dir_sdss+"NGC1087_PHANGS_IMAGE_SDSS_*_copt_0.92asec.fits"))

for i in range(len(img_sdss)):
    band = img_sdss[i].split('.fits')[0].split('_')[4]
    dat, hdr = fits.getdata(img_sdss[i], header=True)
    w = wcs.WCS(hdr)

    kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
    kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

    ### SPHEREx
    hdr1 = copy.deepcopy(hdr)
    array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
                                        order='nearest-neighbor')

    for k in range(len(kwd_ref)):
        if ((k == 5) | (k == 6)):
            hdr1[kwd_tar[k]] = 0.0
        else:
            hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

    fits.writeto("SDSS_"+band+"_repr.fits", (pxs_SPHEREx/pxs_sdss)**2. * array,
                 hdr1, overwrite=True)

    # ### CIGALE
    # hdr2 = copy.deepcopy(hdr)
    # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
    #                                     order='nearest-neighbor')

    # for k in range(len(kwd_ref)):
    #     if (k < 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
    #     elif (k == 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
    #     elif ((k == 5) | (k == 6)):
    #         hdr2[kwd_tar[k]] = 0.0
    #     else:
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

    # fits.writeto("SDSS_"+band+"_repr_3.75.fits", (pxs_CIGALE/pxs_sdss)**2. * array,
    #              hdr2, overwrite=True)


# # ----- GALEX Images ----- #
# pxs_galex = 1.5    # arcsec/pixel
# dir_galex = "/data01/jhlee/DATA/GALEX/NGC1087/"
# img_galex = [dir_galex+"fd-int.fits"  , dir_galex+"nd-int.fits"]
# img_skybg = [dir_galex+"fd-skybg.fits", dir_galex+"nd-skybg.fits"]
# img_rrhr  = [dir_galex+"fd-rrhr.fits" , dir_galex+"nd-rrhr.fits"]

# for i in range(len(img_galex)):
    # band = img_galex[i].split('/')[-1].split('-')[0]

    # dat, hdr = fits.getdata(img_galex[i], header=True)
    # bgr      = fits.getdata(img_skybg[i], header=False)
    # rrhr     = fits.getdata(img_rrhr[i], header=False)
    # w = wcs.WCS(hdr)

    # array, footprint  = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
                                         # order='nearest-neighbor')
    # bkgr,  footprint  = reproject_interp((bgr, w), w_ref, shape_out=shape_ref,
                                         # order='nearest-neighbor')
    # effext, footprint = reproject_interp((rrhr, w), w_ref, shape_out=shape_ref,
                                         # order='nearest-neighbor')

    # kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               # "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
    # kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               # "CD1_1", "CD1_2", "CD2_1", "CD2_2"]
    # for k in range(len(kwd_ref)):
        # if ((k == 5) | (k == 6)):
            # hdr[kwd_tar[k]] = 0.0
        # else:
            # hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

    # fits.writeto("GALEX_"+band+"_int.fits", (pxs_SPHEREx/pxs_galex)**2. * array,
                 # hdr, overwrite=True)
    # fits.writeto("GALEX_"+band+"_skybg.fits", (pxs_SPHEREx/pxs_galex)**2. * bkgr,
                 # hdr, overwrite=True)
    # fits.writeto("GALEX_"+band+"_rrhr.fits", effext,
                 # hdr, overwrite=True)


# # ----- GALEX (DustPedia) Images ----- #
# pxs_galex = 3.2    # arcsec/pixel
# dir_galex = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/DustPedia/"
# img_galex = [dir_galex+"NGC1087_GALEX_FUV.2.fits", dir_galex+"NGC1087_GALEX_NUV.2.fits"]
# galex_bands = ["FUV", "NUV"]

# for i in range(len(img_galex)):
#     dat, hdr = fits.getdata(img_galex[i], header=True)
#     w = wcs.WCS(hdr)
#     # w = wcs.WCS(naxis=2)
#     # w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
#     # w.wcs.crval = np.array([hdr['CRVAL1'], hdr['CRVAL2']])
#     # w.wcs.crpix = [hdr['CRPIX1'], hdr['CRPIX2']]
#     # w.wcs.cd    = np.array([[hdr['CD1_1'], hdr['CD1_2']],
#                             # [hdr['CD2_1'], hdr['CD2_2']]])

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr1[kwd_tar[k]] = 0.0
#         else:
#             hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("GALEX_DP_"+galex_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_galex)**2. * array,
#                  hdr1, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

#     # fits.writeto("GALEX_DP_"+galex_bands[i]+"_repr_3.75.fits", (pxs_CIGALE/pxs_galex)**2. * array,
#     #              hdr2, overwrite=True)


# # ---- 2MASS (DustPedia) Images ----- #
# pxs_tmass = 1.0    # arcsec/pixel
# dir_tmass = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/DustPedia/"
# img_tmass = [dir_tmass+"NGC1087_2MASS_J.2.fits",
#              dir_tmass+"NGC1087_2MASS_H.2.fits",
#              dir_tmass+"NGC1087_2MASS_Ks.2.fits"]
# tmass_bands = ["J", "H", "Ks"]

# for i in range(len(img_tmass)):
#     dat, hdr = fits.getdata(img_tmass[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr1[kwd_tar[k]] = 0.0
#         else:
#             hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("TwoMASS_DP_"+tmass_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_tmass)**2. * array,
#                  hdr1, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

#     # fits.writeto("TwoMASS_DP_"+tmass_bands[i]+"_repr_3.75.fits", (pxs_CIGALE/pxs_tmass)**2. * array,
#     #              hdr2, overwrite=True)


# # ---- Spitzer (IRAC-3/4) Images ----- #
# pxs_irac = 0.6    # arcsec/pixel
# dir_irac = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/DustPedia/"
# img_irac = [dir_irac+"NGC1087_Spitzer_5.8_.2.fits",
#             dir_irac+"NGC1087_Spitzer_8.0_.2.fits"]
# irac_bands = ["IRAC-3", "IRAC-4"]

# for i in range(len(img_irac)):
#     dat, hdr = fits.getdata(img_irac[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr1[kwd_tar[k]] = 0.0
#         else:
#             hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("Spitzer_DP_"+irac_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_irac)**2. * array,
#                  hdr1, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

#     # fits.writeto("Spitzer_DP_"+irac_bands[i]+"_repr_3.75.fits", (pxs_CIGALE/pxs_irac)**2. * array,
#     #              hdr2, overwrite=True)


# # ----- WISE (DP) Images ----- #
# pxs_wise = 1.375  # arcsec/pixel
# dir_wise = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/DustPedia/"
# img_wise = [dir_wise+"NGC1087_WISE_12.2.fits",
#             dir_wise+"NGC1087_WISE_22.2.fits"]
# wise_bands = ["W3", "W4"]

# for i in range(len(img_wise)):
#     dat, hdr = fits.getdata(img_wise[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr[kwd_tar[k]] = 0.0
#         else:
#             hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("WISE_DP_"+wise_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_wise)**2. * array,
#                  hdr, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

#     # fits.writeto("WISE_DP_"+wise_bands[i]+"_repr_3.75.fits", (pxs_CIGALE/pxs_wise)**2. * array,
#     #              hdr2, overwrite=True)


# # ----- SDSS (DP) Images ----- #
# pxs_sdss2 = 0.45  # arcsec/pixel
# dir_sdss2 = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/DustPedia/"
# img_sdss2 = [dir_sdss2+"NGC1087_SDSS_u.2.fits",
#              dir_sdss2+"NGC1087_SDSS_g.2.fits",
#              dir_sdss2+"NGC1087_SDSS_r.2.fits",
#              dir_sdss2+"NGC1087_SDSS_i.2.fits",
#              dir_sdss2+"NGC1087_SDSS_z.2.fits"]
# sdss2_bands = ["u", "g", "r", "i", "z"]

# for i in range(len(img_sdss2)):
#     dat, hdr = fits.getdata(img_sdss2[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr[kwd_tar[k]] = 0.0
#         else:
#             hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("SDSS_DP_"+sdss2_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_sdss2)**2. * array,
#                  hdr, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

#     # fits.writeto("SDSS_DP_"+sdss2_bands[i]+"_repr_3.75.fits", (pxs_CIGALE/pxs_sdss2)**2. * array,
#     #              hdr2, overwrite=True)


# ----- S4G Stellar Mass Images ----- #
def measure_mass(flux, dist=lum_dist, mass_to_light=0.6):
    return 9308.23 * flux * dist**2. * mass_to_light

pxs_S4G = 0.75    # arcsec/pixel
dir_S4G = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/S4G/"
img_mst = [dir_S4G+"NGC1087.stellar.fits", dir_S4G+"NGC1087.nonstellar.fits"]
mst_bands = ["mstar", "fdust"]
ica_msk   = fits.getdata(dir_S4G+"NGC1087.ICAmask.fits")

flx_s1, hdr_s1 = fits.getdata(img_mst[0], header=True)
flx_s1[ica_msk != 0] = 0.
flx_s1_microJy = 1.0e+6 * 1.0e+6 / ((3600.*180./np.pi)/pxs_S4G)**2. * flx_s1  # to microJy

hdr1 = copy.deepcopy(hdr_s1)
ws = wcs.WCS(naxis=2)
ws.wcs.ctype = ['RA---TAN', 'DEC--TAN']
ws.wcs.crval = np.array([hdr_s1['CRVAL1'], hdr_s1['CRVAL2']])
ws.wcs.crpix = [hdr_s1['CRPIX1'], hdr_s1['CRPIX2']]
ws.wcs.cd    = np.array([[hdr_s1['CD1_1'], hdr_s1['CD1_2']],
                         [hdr_s1['CD2_1'], hdr_s1['CD2_2']]])

### SPHEREx flux map ###
array, footprint = reproject_interp((flx_s1_microJy, ws), w_ref, shape_out=shape_ref,
                                    order='nearest-neighbor')
array *= (pxs_SPHEREx/pxs_S4G)**2.

for k in range(len(kwd_ref)):
    if ((k == 5) | (k == 6)):
        hdr1[kwd_tar[k]] = 0.0
    else:
        hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]
hdr1['BUNIT'] = 'microJy/pix2'
fits.writeto("S4G_flux_repr.stellar.fits", array,
             hdr1, overwrite=True)

Mags = 23.90 - 2.5*np.log10(array) - dist_mod
Msts = 10.0 ** (-0.4 * (Mags - 6.02)) * 0.6
fits.writeto("S4G_flux_repr.stmass.fits", Msts,
             hdr1, overwrite=True)


########################

flx_s2 = fits.getdata(img_mst[1], header=False)
flx_s2[ica_msk != 0] = 0.
flx_s2_microJy = 1.0e+6 * 1.0e+6 / ((3600.*180./np.pi)/pxs_S4G)**2. * flx_s2  # to microJy

### SPHEREx flux map ###
array, footprint = reproject_interp((flx_s2_microJy, ws), w_ref, shape_out=shape_ref,
                                    order='nearest-neighbor')
array *= (pxs_SPHEREx/pxs_S4G)**2.

for k in range(len(kwd_ref)):
    if ((k == 5) | (k == 6)):
        hdr1[kwd_tar[k]] = 0.0
    else:
        hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]
hdr1['BUNIT'] = 'microJy/pix2'
fits.writeto("S4G_flux_repr.nonstellar.fits", array,
             hdr1, overwrite=True)
########################

mst = measure_mass(flx_s1, dist=9.8, mass_to_light=0.6)#*1.154)
fst = flx_s2 / (flx_s1 + flx_s2)

mst_array = [mst, fst]
for i in range(len(mst_bands)):

    kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "PC1_1" , "PC1_2" , "PC2_1" , "PC2_2"]
    kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
               "CD1_1" , "CD1_2" , "CD2_1" , "CD2_2"]

    ### SPHEREx
    hdr1 = copy.deepcopy(hdr_s1)
    array, footprint = reproject_interp((mst_array[i], ws), w_ref, shape_out=shape_ref,
                                        order='nearest-neighbor')
    if (i == 0):
        array *= (pxs_SPHEREx/pxs_S4G)**2.
    else:
        array[np.isnan(array)] = 0.

    for k in range(len(kwd_ref)):
        if ((k == 5) | (k == 6)):
            hdr1[kwd_tar[k]] = 0.0
        else:
            hdr1[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

    fits.writeto("S4G_"+mst_bands[i]+"_repr.fits", array,
                 hdr1, overwrite=True)

    # ### CIGALE
    # hdr2 = copy.deepcopy(hdr_s1)
    # array, footprint = reproject_interp((mst_array[i], ws), w_cig, shape_out=shape_cig,
    #                                     order='nearest-neighbor')
    # if (i == 0):
    #     array *= (pxs_CIGALE/pxs_S4G)**2.
    # else:
    #     array[np.isnan(array)] = 0.

    # for k in range(len(kwd_ref)):
    #     if (k < 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
    #     elif (k == 4):
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
    #     elif ((k == 5) | (k == 6)):
    #         hdr2[kwd_tar[k]] = 0.0
    #     else:
    #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']

    # fits.writeto("S4G_"+mst_bands[i]+"_repr_3.75.fits", array,
    #              hdr2, overwrite=True)    


# # ----- WISE (IRSA) Images ----- #
# pxs_wise_irsa = 1.375  # arcsec/pixel
# dir_wise_irsa = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/WISE/"
# img_wise_irsa = [dir_wise_irsa+"NGC1087_WISE_IRSA_W1.fits",
#                  dir_wise_irsa+"NGC1087_WISE_IRSA_W2.fits",
#                  dir_wise_irsa+"NGC1087_WISE_IRSA_W3.fits",
#                  dir_wise_irsa+"NGC1087_WISE_IRSA_W4.fits"]
# sorted(glob.glob(dir_wise_irsa+"*.fits"))
# wise_irsa_bands = ["W1", "W2", "W3", "W4"]

# # from scipy import ndimage
# # dat, hdr = fits.getdata(img_wise_irsa[0], header=True)
# # wc = wcs.WCS(hdr)
# # tar_wcs = wc.deepcopy()
# # tar_wcs.wcs.ctype  # ['RA---SIN', 'DEC--TAN']
# # y, x = np.indices(dat.shape)
# # ra, dec = wc.all_pix2world(x, y, 0)
# # new_x, new_y = tar_wcs.all_world2pix(ra, dec, 0)
# # reprojected_data = ndimage.map_coordinates(dat, [new_y, new_x])
# # fits.writeto('rep.fits', reprojected_data, tar_wcs.to_header())

# for i in range(len(img_wise_irsa)):
#     dat, hdr = fits.getdata(img_wise_irsa[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2",
#                "CTYPE1", "CTYPE2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2",
#                "CTYPE1", "CTYPE2"]

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((dat, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr[kwd_tar[k]] = 0.0
#         else:
#             hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("WISE_IRSA_"+wise_irsa_bands[i]+"_repr.fits",
#                  (pxs_SPHEREx/pxs_wise_irsa)**2. * array, hdr, overwrite=True)

#     # ### CIGALE
#     # hdr2 = copy.deepcopy(hdr)
#     # array, footprint = reproject_interp((dat, w), w_cig, shape_out=shape_cig,
#     #                                     order='nearest-neighbor')

#     # for k in range(len(kwd_ref)):
#     #     if (k < 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]
#     #     elif (k == 4):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT1']
#     #     elif ((k == 5) | (k == 6)):
#     #         hdr2[kwd_tar[k]] = 0.0
#     #     elif (k == 7):
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]] * hdr_cg['CDELT2']
#     #     else:
#     #         hdr2[kwd_tar[k]] = hdr_cg[kwd_ref[k]]

#     # fits.writeto("WISE_IRSA_"+wise_irsa_bands[i]+"_repr_3.75.fits",
#     #              (pxs_CIGALE/pxs_wise_irsa)**2. * array, hdr2, overwrite=True)


# # ----- SDSS (DR12 SAS) Images ----- #
# pxs_sdss3 = 0.396    # arcsec/pixel
# dir_sdss3 = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/SDSS/"
# img_sdss3 = [dir_sdss3+"J013641.00+154701.0-u.fits",
#              dir_sdss3+"J013641.00+154701.0-g.fits",
#              dir_sdss3+"J013641.00+154701.0-r.fits",
#              dir_sdss3+"J013641.00+154701.0-i.fits",
#              dir_sdss3+"J013641.00+154701.0-z.fits"]
# sdss3_bands = ["u", "g", "r", "i", "z"]
# sdss_lam_ef = [0.353, 0.475, 0.622, 0.763, 0.905]  # micrometer

# for i in range(len(img_sdss3)):
#     dat, hdr = fits.getdata(img_sdss3[i], header=True)
#     w = wcs.WCS(hdr)

#     kwd_ref = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "PC1_1", "PC1_2", "PC2_1", "PC2_2"]
#     kwd_tar = ["CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
#                "CD1_1", "CD1_2", "CD2_1", "CD2_2"]

#     mag = 22.5 - 2.5*np.log10(dat)  # AB magnitude
#     if (sdss3_bands[i] == 'u'):
#         mag -= 0.04
#     if (sdss3_bands[i] == 'z'):
#         mag += 0.02
#     fv  = 10.0**(0.4*(23.90-mag))  # microJy
#     fl  = fv * (c / sdss_lam_ef[i]**2.) * 1.0e-4  # 10^-20 erg/s/cm2/A
#     fl[np.isnan(fl)] = 0.

#     ### SPHEREx
#     hdr1 = copy.deepcopy(hdr)
#     array, footprint = reproject_interp((fl, w), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')

#     for k in range(len(kwd_ref)):
#         if ((k == 5) | (k == 6)):
#             hdr[kwd_tar[k]] = 0.0
#         else:
#             hdr[kwd_tar[k]] = hdr_rb[kwd_ref[k]]

#     fits.writeto("SDSS_DR12_"+sdss3_bands[i]+"_repr.fits", (pxs_SPHEREx/pxs_sdss3)**2. * array,
#                  hdr, overwrite=True)

