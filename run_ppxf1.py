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
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
c = 2.99792458e+5    # light speed in km/s

from ppxf.ppxf import ppxf, robust_sigma, attenuation
import ppxf.ppxf_util as util
import ppxf.miles_util as lib


# # ----- Setting up the Stellar Libraries ----- #
# ppxf_dir1 = "/data01/jlee/Downloads/E-MILES/tmp1/"    # Template directory for stellar kinematics
# ppxf_dir2 = "/data01/jlee/Downloads/E-MILES/tmp2/"    # Template directory for stellar population
# pathname1 = ppxf_dir1 + "Ech1.30*.fits"
# pathname2 = ppxf_dir2 + "Ech1.30*.fits"


# # ----- Loading the data ----- #
# with open("box_spec.pickle", "rb") as fr:
    # box_spec = pickle.load(fr)
# with open("box_vari.pickle", "rb") as fr:
    # box_vari = pickle.load(fr)


# wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      # stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      # num=h_sci['NAXIS3'], endpoint=True)


def draw_spectra(data, observed_wavelength, out, 
                 xlabel=r"Observer-frame Wavelength [$\rm \AA$]",
                 ylabel=r"Flux Density [$10^{-20}~{\rm erg~s^{-1}~cm^{-2}~\AA^{-1}}$]"):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(observed_wavelength, data, '-', linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def clip_outliers(galaxy, bestfit, goodpixels, sigma=3):
    """
    Repeat the fit after clipping bins deviants more than n*sigma
    in relative error until the bad bins don't change any more.
    """
    # print(len(goodpixels), "/", len(galaxy))
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sigma*err)
        # print(len(goodpixels), "/", len(galaxy))
        if np.array_equal(goodpixels, ok_old):
            break
            
    return goodpixels


def mass_to_light(temp, weights, magfile, band="r", quiet=True):

#     vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
    sdss_bands = ["u", "g", "r", "i", "z"]
#     vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
    sdss_sun_mag = [6.45, 5.14, 4.65, 4.54, 4.52]  # values provided by not Elena Ricciardelli, but http://mips.as.arizona.edu/~cnaw/sun_2006.html
    
    i = sdss_bands.index(band)
    sun_mag = sdss_sun_mag[i]
    
    dt = pd.read_csv(magfile, sep=' ')
    
    mass_grid = np.empty_like(weights)
    lum_grid  = np.empty_like(weights)
    for j in range(temp.n_ages):
        for k in range(temp.n_metal):
            p1 = ((np.abs(temp.age_grid[j, k] - dt['Age']) < 0.001) & \
                  (np.abs(temp.metal_grid[j, k] - dt['Z']) < 0.01))
            mass_grid[j, k] = dt['M(*+remn)'][p1]
            lum_grid[j, k] = 10**(-0.4*(dt[band+'_SDSS'][p1] - sun_mag))
                     
    # This is eq.(2) in Cappellari+13
    # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
    mlpop = np.sum(weights*mass_grid)/np.sum(weights*lum_grid)

    if not quiet:
        print(f'(M*/L)_{band}: {mlpop:#.4g}')

    return [mass_grid, lum_grid, mlpop]
    

def run_ppxf1(spectrum, observed_wavelength, mask_region, tempath_kn, tempath_sp,
              wlim_blue=4800., wlim_red=9000., fig_dir="./", fig_suffix="_",
              calc_mass_to_light=True):
    
    # ----- Inital Wavelength Cut ----- #
    eff = ((observed_wavelength >= wlim_blue) & (observed_wavelength <= wlim_red))
    lam = observed_wavelength[eff]
    galaxy = spectrum[eff] / np.median(spectrum[eff])
    noise = np.ones_like(galaxy)
    # noise = np.sqrt(box_variance[eff] / np.median(box_spectrum[eff])**2.)
    # regul_err = np.median(noise)
    # regul_err
    
    # ----- Re-binning of the Spectra with the log scale ----- #    
    # velscale = c * np.diff(np.log(lam[[0, -1]]) / (lam.size-1))    # works for both log- and linear- scales
    velscale = c * np.diff(np.log(lam[-2:]))    # only works for the log- scale  (Smallest velocity step)
    # print(velscale)
    velscale = velscale[0]
    # print(f"Velocity scale per pixel: {velscale:.2f} km/s")

    galaxy2, ln_lam_gal, velscale = util.log_rebin([wlim_blue, wlim_red], galaxy, velscale=velscale)
    noise2 = np.ones_like(galaxy2)
    lam2 = np.exp(ln_lam_gal)

    # ----- Masking the Spectra ----- #
    msk_flag = np.zeros_like(galaxy)
    # msk_region = [[5570., 5590.],
                  # [5882., 5895.],
                  # [6280., 6320.],
                  # [6350., 6380.],
                  # [6850., 6925.],
                  # [7000., 8490.],
                  # [8700., 9000.]]
    for m in mask_region:
        idmsk_l = np.argmin(np.abs(m[0]-lam))
        idmsk_r = np.argmin(np.abs(m[1]-lam))+1
        msk_flag[idmsk_l:idmsk_r] = 1
    goodpixels0 = np.flatnonzero(msk_flag == 0)
    assert msk_flag[goodpixels0].sum() == 0.0

    msk_flag2 = np.zeros_like(galaxy2)
    for m in mask_region:
        idmsk_l = np.argmin(np.abs(m[0]-lam2))
        idmsk_r = np.argmin(np.abs(m[1]-lam2))+1
        msk_flag2[idmsk_l:idmsk_r] = 1
    goodpixels2 = np.flatnonzero(msk_flag2 == 0)
    assert msk_flag2[goodpixels2].sum() == 0.0
    
    
    # ----- Making the Stellar Templates ----- #
    FWHM_gal = 2.62    # Median FWHM resolution of MUSE
    # FWHM_gal = None
    miles = lib.miles(tempath_kn, velscale, FWHM_gal, norm_range=[5070, 5950])
    reg_dim = miles.templates.shape[1:]
    # print(reg_dim)
    # miles.templates.shape
    # miles.age_grid.shape
    # miles.age_grid
    # miles.metal_grid
    # miles.ln_lam_temp
    # miles.lam_temp
        
    kin_templates = miles.templates.reshape(miles.templates.shape[0], -1)
    kin_templates /= np.median(kin_templates) # Normalizes stellar templates by a scalar
    # stars_templates.shape        

    # ----- Initial Running pPXF for Stellar Components ----- #
    if (fig_dir[-1] != "/"):
        fig_dir += "/"

    start = [100., 180., 0.1, 0.1]
    pp = ppxf(kin_templates, galaxy2, noise2,
              velscale, start, goodpixels=goodpixels2,
              moments=4, degree=4, mdegree=0,
              lam=lam2, lam_temp=miles.lam_temp,
              quiet=True)#,
              #reddening=0.1)

    plt.figure(figsize=(9,4))
    pp.plot()
    plt.savefig(fig_dir+"pPXF_results1"+fig_suffix+".png", dpi=300)
    plt.close()

    # ----- Iterative Running pPXF for Stellar Components (for stellar kinematics) ----- #
    goodpixels3 = clip_outliers(galaxy2, pp.bestfit, goodpixels2, sigma=5)
    goodpixels3 = np.intersect1d(goodpixels3, goodpixels2)

    pp = ppxf(kin_templates, galaxy2, noise2,
              velscale, start, goodpixels=goodpixels3,
              moments=4, degree=4, mdegree=0,
              lam=lam2, lam_temp=miles.lam_temp,
              quiet=True)#,
              #reddening=0.1)

    plt.figure(figsize=(9,4))
    pp.plot()
    plt.savefig(fig_dir+"pPXF_results2"+fig_suffix+".png", dpi=300)
    plt.close()
    
    vel, sigma, h3, h4 = pp.sol
    # print(vel, sigma, h3, h4)
    
    # ----- Adding More Templates ----- #
    miles2 = lib.miles(tempath_sp, velscale, FWHM_gal, norm_range=[5070, 5950])
    reg_dim2 = miles2.templates.shape[1:]
    # print(reg_dim2)

    stars_templates = miles2.templates.reshape(miles.templates.shape[0], -1)
    stars_templates /= np.median(stars_templates) # Normalizes stellar templates by a scalar
    # print(stars_templates2.shape)
    
    # ----- Deriving the Stellar Absorption (with fixed stellar kinematics) ----- #
    start = [vel, sigma, h3, h4]
    fixed = [1, 1, 0, 0]
    a_v0 = 0.1
    pp = ppxf(stars_templates, galaxy2, noise2,
              velscale, start, goodpixels=goodpixels3,
              moments=4, degree=-1, mdegree=-1,
              lam=lam2, lam_temp=miles.lam_temp, fixed=fixed,
              reddening=a_v0,
              quiet=True)

    plt.figure(figsize=(9,4))
    pp.plot()
    plt.savefig(fig_dir+"pPXF_results3"+fig_suffix+".png", dpi=300)
    plt.close()
    
    a_v_star = pp.reddening
    # print(a_v_star)
    
    # ----- Deriving the Stellar Population Parameters ----- #
    f_extn = attenuation(lam2, a_v_star)
    
    galaxy3 = galaxy2 / f_extn
    galaxy3 /= np.median(galaxy3)
    
    start = [vel, sigma, h3, h4]
    fixed = [1, 1, 1, 1]
    pp = ppxf(stars_templates, galaxy3, noise2,
              velscale, start, goodpixels=goodpixels3,
              moments=4, degree=-1, mdegree=-1,
              lam=lam2, lam_temp=miles.lam_temp, fixed=fixed,
              quiet=True)

    plt.figure(figsize=(9,4))
    pp.plot()
    plt.savefig(fig_dir+"pPXF_results4"+fig_suffix+".png", dpi=300)
    plt.close()

    # ----- Deriving Age and Metallicity ----- #
    light_weights = pp.weights#[~gas_component]      # Exclude weights of the gas templates
    light_weights = light_weights.reshape(reg_dim2)  # Reshape to (n_ages, n_metal)
    light_weights /= light_weights.sum()            # Normalize to light fractions
    # print(light_weights)
    # Given that the templates are normalized to the V-band, the pPXF weights
    # represent v-band light fractions and the computed ages and metallicities
    # below are also light weighted in the V-band.
    # print("\n--- Luminosity-weighted values ---")
    logAge_lw, Z_lw = miles2.mean_age_metal(light_weights, quiet=True)
    # print(logAge_lw, Z_lw)
    # For the M/L one needs to input fractional masses, not light fractions.
    # For this, I convert light-fractions into mass-fractions using miles.flux
    mass_weights = light_weights/miles2.flux
    mass_weights /= mass_weights.sum()              # Normalize to mass fractions
    # print(mass_weights)
    # print("\n--- Mass-weighted values ---")
    logAge_mw, Z_mw = miles2.mean_age_metal(mass_weights, quiet=True)
    # print(logAge_mw, Z_mw)
    
    if calc_mass_to_light:
        magfile = "/data01/jlee/Downloads/E-MILES/sdss_ch_iTp0.00.MAG"
        mg, lg, ml = mass_to_light(miles2, mass_weights, magfile, band="r")
        
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights, logAge_mw, Z_mw,
                        mg, lg, ml]
    else:
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights, logAge_mw, Z_mw]
        
    return results_list



if (__name__ == '__main__'):

    # ----- Setting up the Stellar Libraries ----- #
    ppxf_dir1 = "/data01/jlee/Downloads/E-MILES/tmp1/"    # Template directory for stellar kinematics
    ppxf_dir2 = "/data01/jlee/Downloads/E-MILES/tmp2/"    # Template directory for stellar population
    pathname1 = ppxf_dir1 + "Ech1.30*.fits"
    pathname2 = ppxf_dir2 + "Ech1.30*.fits"

    # ----- Loading MUSE spectra data cube ----- #
    dir_cube = "./"
    filename = dir_cube + "NGC0628_PHANGS_DATACUBE_copt_0.92asec.fits"
    sp = fits.open(filename)
    d_sci, h_sci = sp[1].data, sp[1].header
    d_var, h_var = sp[2].data, sp[2].header
    wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          num=h_sci['NAXIS3'], endpoint=True)

    # ----- Loading the re-binned data ----- #
    img_rb = fits.getdata("rebin.fits")
        
    # ----- Loading the re-binned data ----- #
    with open("box_spec_total.pickle", "rb") as fr:
        box_spec = pickle.load(fr)
    with open("box_vari_total.pickle", "rb") as fr:
        box_vari = pickle.load(fr)

    # ----- Masking region ----- #
    msk_region = [[5570., 5590.],
                  [5882., 5895.],
                  [6280., 6320.],
                  [6350., 6380.],
                  [6850., 6925.],
                  [7120., 7250.],
                  [7500., 7750.],
                  [8150., 8350.]]
    
    # ----- Running pPXF ----- #
    niters, nparam = 10, 8
    f = open("ppxf0.csv","w")
    f.write("x,y,vel,e_vel,sigma,e_sigma,Av_star,e_Av_star,")
    f.write("logAge_lw,e_logAge_lw,Z_lw,e_Z_lw,logAge_mw,e_logAge_mw,Z_mw,e_Z_mw,")
    f.write("M/L,e_M/L\n")
    for ix in tqdm(range(img_rb.shape[1])):
        for iy in tqdm(range(img_rb.shape[0])):
            keywd = f"x{ix:03d}_y{iy:03d}"
            if (box_spec[keywd].sum() <= 0.0):
                continue
            np.random.seed(0)
            mres = np.zeros((niters, nparam))
            for nn in range(niters):
                spec_data = truncnorm.rvs(-5.0, 5.0, loc=box_spec[keywd], scale=np.sqrt(box_vari[keywd]),
                                          size=box_spec[keywd].size)
                res = run_ppxf1(spec_data, wav_obs, msk_region, pathname1, pathname2,
                                wlim_blue=4800., wlim_red=9000.,
                                fig_dir="Figure_ppxf0", fig_suffix="_"+keywd)
                idxs = [0, 1, 4, 6, 7, 9, 10, 13]    # Result values
                mres[nn] = np.array(list(map(res.__getitem__, idxs)))
            f_res = np.mean(mres, axis=0)
            e_res = np.std(mres, axis=0)
            f.write(f"{ix:03d},{iy:03d},")
            txt = ""
            for nn in range(nparam):
                txt += f"{f_res[nn]:.3f},{e_res[nn]:.3f}"
                if (nn != range(nparam)[-1]):
                    txt += ","
            f.write(txt+"\n")
            # f.write(f"{f_res[0]:.2f},{res[1]:.2f},{res[4]:.2f},")
            # f.write(f"{f_res[6]:.2f},{res[7]:.3f},{res[9]:.2f},{res[10]:.3f},")
            # f.write(f"{f_res[13]:.3f}\n")
    f.close()

