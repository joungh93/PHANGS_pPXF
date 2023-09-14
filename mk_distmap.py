#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 8 14:08:46 2023
@author: jlee
"""


# importing necessary modules
import numpy as np
from astropy.io import fits


# ----- Reading the rebinned image ----- #
dat_rb, hdr_rb = fits.getdata("rebin.fits", header=True)
ny, nx = dat_rb.shape


# ----- Creating the distance map ----- #
xInd, yInd = 10-1, 15-1    # Central coordiante
lum_dist = 15.9   # Mpc
dist_mod = 5.0*np.log10(lum_dist*1.0e+6 / 10.)
ang_scale = lum_dist * 1.0e+6 * (1./3600) * (np.pi/180.)   # pc/arcsec
print(f"Scale: {ang_scale:.3f} pc/arcsec")

x_arr  = np.arange(nx)
y_arr  = np.arange(ny)
xx, yy = np.meshgrid(x_arr, y_arr)
zdist  = np.sqrt((xx-xInd)**2. + (yy-yInd)**2.)

fits.writeto("distmap.fits", zdist, hdr_rb, overwrite=True)
