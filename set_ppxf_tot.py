#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 1 07:59:37 2022
@author: jlee
"""


# importing necessary modules
import glob, os, copy
from os import path
import pickle
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import truncnorm
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
c = 2.99792458e+5    # light speed in km/s

from ppxf.ppxf import ppxf, robust_sigma, attenuation
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import extinction
ebv = 0.030
R_V = 3.1
A_V = R_V * ebv

dir_cube = "/data01/jhlee/DATA/PHANGS/MUSE/NGC1087/"
# with open(dir_cube+"box_spec_total.pickle", "rb") as fr:
#     box_spec = pickle.load(fr)
# with open(dir_cube+"box_vari_total.pickle", "rb") as fr:
#     box_vari = pickle.load(fr)
# n_files = len(box_spec)
# run_name, run_array = "part01", np.arange(86*0, np.minimum(86, n_files), 1)



def draw_spectra(data, observed_wavelength, out, 
                 xlabel=r"Observer-frame Wavelength [$\rm \AA$]",
                 ylabel=r"Flux Density [$10^{-20}~{\rm erg~s^{-1}~cm^{-2}~\AA^{-1}}$]"):
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(observed_wavelength, data, '-', linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out, dpi=300)


def clip_outliers(galaxy, bestfit, goodpixels, sigma=3, maxiters=10):
    """
    Repeat the fit after clipping bins deviants more than n*sigma
    in relative error until the bad bins don't change any more.
    """
    # print(len(goodpixels), "/", len(galaxy))
    n_iter = 0
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < sigma*err)
        # print(len(goodpixels), "/", len(galaxy))
        # print(goodpixels)
        # print(len(goodpixels), err)
        # print(ok_old)
        # print(len(ok_old))
        n_iter += 1
        if ((np.array_equal(goodpixels, ok_old)) | (n_iter >= 10)):
            break
            
    return goodpixels


def mass_to_light(temp, weights, magfile, band=["r_SDSS", "IRAC1"], quiet=True):

#     vega_bands = ["U", "B", "V", "R", "I", "J", "H", "K"]
    sdss_bands = ["u_SDSS", "g_SDSS", "r_SDSS", "i_SDSS", "z_SDSS"]
#     vega_sun_mag = [5.600, 5.441, 4.820, 4.459, 4.148, 3.711, 3.392, 3.334]
    sdss_sun_mag_AB = [6.39, 5.11, 4.65, 4.53, 4.50]   # Willmer+18 (http://mips.as.arizona.edu/~cnaw/sun.html)

    spitzer_bands = ["IRAC1", "IRAC2"]
    spitzer_sun_mag_vega = [3.26, 3.28]   # Willmer+18 (http://mips.as.arizona.edu/~cnaw/sun.html)
    
    bands        = sdss_bands + spitzer_bands
    sun_mag_vals = sdss_sun_mag_AB + spitzer_sun_mag_vega
    sun_mag_sel  = []
    for b in band:
        sun_mag_sel.append(sun_mag_vals[bands.index(b)])

    dt = pd.read_csv(magfile, sep=' ')
    
    mass_grid = np.empty_like(weights)
    lum_grid  = np.empty((len(band), mass_grid.shape[0], mass_grid.shape[1]))  #[np.empty_like(weights)]*len(band)

    for j in range(temp.n_ages):
        for k in range(temp.n_metal):
            p1 = ((np.abs(temp.age_grid[j, k] - dt['Age']) < 0.001) & \
                  (np.abs(temp.metal_grid[j, k] - dt['Z']) < 0.01))
            assert np.sum(p1) == 1

            mass_grid[j, k] = dt['M(*+remn)'][p1]
            for n in range(len(band)):
                if (dt[band[n]].values[p1][0] < -50.):
                    lum_grid[n, j, k] = np.nan
                else:
                    lum_grid[n, j, k] = 10**(-0.4*(dt[band[n]].values[p1][0] - sun_mag_sel[n]))
                     
    # This is eq.(2) in Cappellari+13
    # http://adsabs.harvard.edu/abs/2013MNRAS.432.1862C
    mlpop = []
    for n in range(len(band)):
        mlpop.append(np.sum(weights[~np.isnan(lum_grid[n])]*mass_grid[~np.isnan(lum_grid[n])])/ \
                     np.sum(weights[~np.isnan(lum_grid[n])]*lum_grid[n][~np.isnan(lum_grid[n])]))

    if not quiet:
        for n in range(len(band)):
            print(f'(M*/L)_{band[n]}: {mlpop[n]:#.4g}')

    return mlpop
    

def get_noise(spectrum, bestfit, wavelength, goodpixels, lnoi=10.):

    # New noise estimation (depending on wavelength)
    tmp = spectrum - bestfit
    noise_new = np.zeros(len(tmp))
    indn = np.arange(len(tmp))
    for i in indn:
        sel = ((wavelength > wavelength[i]-lnoi) & \
               (wavelength < wavelength[i]+lnoi))
        noise_new[i] = np.nanstd(tmp[sel])
    SN_new = np.nanmedian(spectrum[goodpixels] / noise_new[goodpixels])

    return [noise_new, SN_new]


def init_spec(spectrum, observed_wavelength, mask_region,
              norm_range=[5070, 5950],
              wlim_blue=4800., wlim_red=7000.):

    # ----- Inital Wavelength Cut ----- #
    eff = ((observed_wavelength >= wlim_blue) & (observed_wavelength <= wlim_red))
    lam = observed_wavelength[eff]
    median_galaxy = np.median(spectrum[(observed_wavelength >= norm_range[0]) & \
                                       (observed_wavelength <= norm_range[1]) & \
                                       eff & (spectrum > 0.) & \
                                       (~np.isnan(spectrum) & (~np.isinf(spectrum)))])
    galaxy = spectrum[eff] / median_galaxy
    noise = np.ones_like(galaxy)
    
    # ----- Re-binning of the Spectra with the log scale ----- #    
    galaxy2, ln_lam_gal, velscale = util.log_rebin([wlim_blue, wlim_red], galaxy)#, velscale=velscale)
    #print(f"Velocity scale per pixel: {velscale:.2f} km/s")
    print("\n", velscale)
    # velscale_ratio = 2
    # vscale = velscale / velscale_ratio
    # noise2 = np.ones_like(galaxy2)
    lam2 = np.exp(ln_lam_gal)

    # ----- Masking the Spectra ----- #
    msk_flag2 = np.zeros_like(galaxy2)
    if mask_region is not None:
        for m in mask_region:
            idmsk_l = np.argmin(np.abs(m[0]-lam2))
            idmsk_r = np.argmin(np.abs(m[1]-lam2))+1
            msk_flag2[idmsk_l:idmsk_r] = 1
        msk_flag2[galaxy2 <= 0.] = 1
    goodpixels2 = np.flatnonzero(msk_flag2 == 0)
    assert msk_flag2[goodpixels2].sum() == 0.0

    galaxy2[np.isnan(galaxy2) | np.isinf(galaxy2)] = 0.

    return [galaxy2, lam2, goodpixels2, velscale, median_galaxy]


def run_ppxf1(spectrum, observed_wavelength, tempath_kn, tempath_sp,
              norm_range=[5070, 5950],
              mask_region_kn=None, mask_region_sp=None, 
              wlim_blue_kn=4800., wlim_red_kn=9000.,
              wlim_blue_sp=4800., wlim_red_sp=9000.,
              fig_dir="./", fig_suffix="_", clip_sigma=4.0,
              start0=None, regul=5., lnoi=10.,
              adeg=12, mdeg=12,
              calc_mass_to_light=True, magfile=None,
              dunit=1.0e-20, tunit=3.828e+33, tdist=9.8,
              fix_kn=False, FWHM_gal=2.62, FWHM_tem=2.51,
              fix_Av=False, Av0=None,
              apply_phot=False, bands=None, phot_galaxy=None, phot_noise=None,
              lam_phot=None):

    if (fig_dir[-1] != "/"):
        fig_dir += "/"

    if start0 is None:
        start = [100., 180., 0.1, 0.1]
    else:
        start = start0

    # ----- Initial process (for kinematics) ----- #
    galaxy1, lam1, goodpixels1, velscale, galnorm1 = init_spec(spectrum, observed_wavelength,
                                                               mask_region_kn,
                                                               norm_range=norm_range,
                                                               wlim_blue=wlim_blue_kn,
                                                               wlim_red=wlim_red_kn)
    velscale_ratio = 2
    vscale = velscale / velscale_ratio

    # ----- Making the Stellar Templates ----- #
    #FWHM_gal = 2.62    # Median FWHM resolution of MUSE
    #FWHM_tem = 2.51    # Vazdekis+10 spectra FWHM: 2.51AA

    # ssp, h1 = fits.getdata(sorted(glob.glob(tempath_kn))[0], header=True)
    # lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
    # sspNew, logLam1, velscale_temp = util.log_rebin(lamRange1, ssp, velscale=vscale)
    # print(velscale_temp)



    if not fix_kn:

        miles1 = lib.miles(tempath_kn, vscale, FWHM_gal=FWHM_gal, FWHM_tem=FWHM_tem,
                           norm_range=norm_range)
        reg_dim1 = miles1.templates.shape[1:]
        # print(reg_dim)
        # miles.templates.shape
        # miles.age_grid.shape
        # miles.age_grid
        # miles.metal_grid
        # miles.ln_lam_temp
        # miles.lam_temp
            
        kin_templates = miles1.templates.reshape(miles1.templates.shape[0], -1)
        kin_norm = np.median(kin_templates)
        kin_templates /= kin_norm
        # kin_templates.shape        

        # if apply_phot:
        #     print(phot_galaxy, galnorm1)
        #     phot_lam, phot_templates, ok_temp = util.synthetic_photometry(
        #         kin_templates, miles1.lam_temp, bands, redshift=0.0, quiet=1)
        #     phot = {"templates": phot_templates, "galaxy": phot_galaxy / galnorm1,
        #             "noise": phot_noise / galnorm1, "lam": phot_lam}
        # else:
        #     phot = {}

        # ----- Initial Running pPXF for Stellar Components ----- #
        # ----- Iterative Running pPXF for Stellar Components (for stellar kinematics) ----- #
        # goodpixels3 = clip_outliers(galaxy2, pp.bestfit, goodpixels2, sigma=clip_sigma)
        # goodpixels3 = np.intersect1d(goodpixels3, goodpixels2)
        # goodpixels3 = goodpixels2

        ### 1st - better determine for new noise
        print("\n PPXF Run 1")
        pp1 = ppxf(kin_templates, galaxy1, np.ones_like(galaxy1), velscale, start,
                   moments=4, degree=adeg, mdegree=0, goodpixels=goodpixels1,
                   lam=lam1, lam_temp=miles1.lam_temp,
                   velscale_ratio=velscale_ratio, quiet=True)

        # plt.figure(figsize=(8,4))
        # pp1.plot()
        # plt.tight_layout()
        # plt.savefig(fig_dir+"pPXF_results1"+fig_suffix+".png", dpi=300)
        # plt.close()

        vel, sigma, h3, h4 = pp1.sol
        print(vel, sigma, h3, h4)
        noise1_1, SN1_1 = get_noise(galaxy1, pp1.bestfit, lam1, goodpixels1, lnoi=10.)


        ### 2nd - derive S/N again, better determine the intial guess
        print("\n PPXF Run 2")
        pp1 = ppxf(kin_templates, galaxy1, noise1_1, velscale, start,
                   moments=4, degree=adeg, mdegree=0, goodpixels=goodpixels1,
                   lam=lam1, lam_temp=miles1.lam_temp,
                   velscale_ratio=velscale_ratio, quiet=True,
                   reg_dim=reg_dim1, reg_ord=2, regul=regul,
                   reddening=None, clean=True)

        # plt.figure(figsize=(8,4))
        # pp1.plot()
        # plt.tight_layout()
        # plt.savefig(fig_dir+"pPXF_results2"+fig_suffix+".png", dpi=300)
        # plt.close()

        vel, sigma, h3, h4 = pp1.sol
        print(vel, sigma, h3, h4)
        noise1_2, SN1_2 = get_noise(galaxy1, pp1.bestfit, lam1, goodpixels1, lnoi=10.)
        start = [vel, sigma, h3, h4]


        ### 3rd - for kinematics
        print("\n PPXF Run 3")
        pp1 = ppxf(kin_templates, galaxy1, noise1_2, velscale, start,
                   moments=4, degree=adeg, mdegree=0, goodpixels=goodpixels1,
                   lam=lam1, lam_temp=miles1.lam_temp,
                   velscale_ratio=velscale_ratio, quiet=True,
                   reg_dim=reg_dim1, reg_ord=2, regul=regul,
                   reddening=0.0, clean=False)
        
        # plt.close('all')
        # plt.figure(figsize=(8,4))
        # pp1.plot()
        # plt.tight_layout()
        # plt.show(block=False)
        # plt.savefig(fig_dir+"pPXF_results3"+fig_suffix+".png", dpi=300)
        # plt.close()

        vel, sigma, h3, h4 = pp1.sol
        print(vel, sigma, h3, h4, pp1.reddening)
        e_vel, e_sigma, e_h3, e_h4 = pp1.error * np.sqrt(pp1.chi2)
        # noise_new3, SN_new = get_noise(galaxy1, pp.bestfit, lam1, goodpixels1, lnoi=10.)
        start = [vel, sigma, h3, h4]
        reddening0 = pp1.reddening

    else:
        assert len(start0) == 4
        start = start0
        vel, sigma, h3, h4 = start
        reddening0 = 0.0

    # ----- Initial process (for stellar population) ----- #
    galaxy2, lam2, goodpixels2, velscale, galnorm2 = init_spec(spectrum, observed_wavelength,
                                                               mask_region_sp,
                                                               norm_range=norm_range,
                                                               wlim_blue=wlim_blue_sp,
                                                               wlim_red=wlim_red_sp)
    velscale_ratio = 2
    vscale = velscale / velscale_ratio
    print(galnorm2, vscale)

    # ----- Adding More Templates ----- #
    miles2 = lib.miles(tempath_sp, vscale, FWHM_gal=FWHM_gal, FWHM_tem=FWHM_tem,
                       norm_range=norm_range)
    reg_dim2 = miles2.templates.shape[1:]
    # print(reg_dim2)

    stars_templates = miles2.templates.reshape(miles2.templates.shape[0], -1)
    stars_norm = np.median(stars_templates)
    stars_templates /= stars_norm # Normalizes stellar templates by a scalar
    # print(stars_templates.shape)

    # ----- Photometry ----- #
    # phot_galaxy = np.array([0.505, 0.688, 1.14, 0.967, 0.826, 0.698, 0.499, 0.341, 0.164])   # fluxes
    # phot_noise = phot_galaxy*0.10   # 10% 1sigma uncertainties
    # bands = ['galex1500', 'galex2500',
    #          'SDSS/g', 'SDSS/r', 'SDSS/i',
    #          '2MASS/J', '2MASS/H', '2MASS/K',
    #          'IRAC/irac_tr1', 'IRAC/irac_tr2']
    if apply_phot:
        print(phot_galaxy, galnorm2)
        phot_lam, phot_templates, ok_temp = util.synthetic_photometry(
            stars_templates, miles2.lam_temp, bands, redshift=vel/c, quiet=1)
        phot = {"templates": phot_templates, "galaxy": phot_galaxy/galnorm2,
                "noise": phot_noise/galnorm2, "lam": phot_lam}
    else:
        phot = None

    # ----- Iterative Running pPXF for Stellar Components (for stellar population) ----- #

    ### 4th - better determine for new noise
    print("\n PPXF Run 4")
    pp2 = ppxf(stars_templates, galaxy2, np.ones_like(galaxy2), velscale, start,
               moments=4, degree=-1, mdegree=mdeg, goodpixels=goodpixels2,
               lam=lam2, lam_temp=miles2.lam_temp,
               velscale_ratio=velscale_ratio, quiet=True,
               reg_dim=reg_dim2, reg_ord=2, regul=0.,
               reddening=None, clean=True, fixed=[1,1,1,1],
               phot=phot)

    # plt.figure(figsize=(8,4))
    # pp2.plot()
    # plt.tight_layout()
    # plt.savefig(fig_dir+"pPXF_results4"+fig_suffix+".png", dpi=300)
    # plt.close()

    # vel, sigma, h3, h4 = pp2.sol
    print(vel, sigma, h3, h4)
    noise2_1, SN2 = get_noise(galaxy2, pp2.bestfit, lam2, goodpixels2, lnoi=10.)
    err = 1.0e-5


    if not fix_Av:

        ### 5th - for stellar absorption
        print("\n PPXF Run 5")
        pp2 = ppxf(stars_templates, galaxy2, noise2_1, velscale, start,
                   moments=4, degree=-1, mdegree=0, goodpixels=goodpixels2,
                   lam=lam2, lam_temp=miles2.lam_temp,
                   velscale_ratio=velscale_ratio, quiet=True,
                   reg_dim=reg_dim2, reg_ord=2, regul=0.,
                   reddening=reddening0, clean=False, fixed=[0,0,0,0],
                   bounds=[[vel-err, vel+err], [sigma-err, sigma+err],
                           [h3-err, h3+err], [h4-err, h4+err]],
                   phot=phot)

        # plt.figure(figsize=(8,4))
        # pp2.plot()
        # plt.tight_layout()
        # plt.show(block=False)
        # plt.savefig(fig_dir+"pPXF_results5"+fig_suffix+".png", dpi=300)
        # plt.close()

        # vel, sigma, h3, h4 = pp2.sol
        a_v_star = pp2.reddening
        print(pp2.sol[0], pp2.sol[1], pp2.sol[2], pp2.sol[3], pp2.reddening)
        noise2_2, SN2 = get_noise(galaxy2, pp2.bestfit, lam2, goodpixels2, lnoi=10.)

    else:
        a_v_star = Av0
        noise2_2 = noise2_1

    f_extn = attenuation(lam2, a_v_star)
    galaxy3 = galaxy2 / f_extn
    eff = ((galaxy2 > 0.) & (~np.isnan(galaxy2)) & (~np.isinf(galaxy2)))
    galnorm3 = np.median(galaxy2[eff] * galnorm2 / f_extn[eff])
    print(galnorm3)
    # galaxy3 /= galnorm3

    if apply_phot:
        f_extn_ph = attenuation(lam_phot, a_v_star)
        print(lam_phot)
        print(f_extn_ph, phot_galaxy/galnorm2/f_extn_ph)
        phot = {"templates": phot_templates, "galaxy": phot_galaxy/galnorm2/f_extn_ph,
                "noise": phot_noise/galnorm2/f_extn_ph, "lam": phot_lam}
    else:
        phot = None

    ### 6th - for stellar population
    print("\n PPXF Run 6")
    pp2 = ppxf(stars_templates, galaxy3, noise2_2, velscale, start,
               moments=4, degree=-1, mdegree=mdeg, goodpixels=goodpixels2,
               lam=lam2, lam_temp=miles2.lam_temp,
               velscale_ratio=velscale_ratio, quiet=True,
               reg_dim=reg_dim2, reg_ord=2, regul=0.,
               reddening=None, clean=False, fixed=[1,1,1,1],
               phot=phot)

    plt.figure(figsize=(8,4))
    pp2.plot()
    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig(fig_dir+"pPXF_results6"+fig_suffix+".png", dpi=300)
    plt.close()

    # vel, sigma, h3, h4 = pp2.sol
    print(vel, sigma, h3, h4, pp2.reddening)


    # ----- Deriving Age and Metallicity ----- #
    light_weights = copy.deepcopy(pp2.weights)#[~gas_component]      # Exclude weights of the gas templates
    light_weights = light_weights.reshape(reg_dim2)  # Reshape to (n_ages, n_metal)
    light_weights /= light_weights.sum()            # Normalize to light fractions
    logAge_lw, Z_lw = miles2.mean_age_metal(light_weights, quiet=True)
    # print(logAge_lw, Z_lw)
    mass_weights = light_weights / miles2.flux
    mass_weights /= mass_weights.sum()              # Normalize to mass fractions
    logAge_mw, Z_mw = miles2.mean_age_metal(mass_weights, quiet=True)
    # print(logAge_mw, Z_mw)
    stellar_mass = np.log10(np.sum(pp2.weights) * (galnorm3 / stars_norm) * \
                            (dunit / tunit) * (4*np.pi*(tdist*3.0857e+24)**2.))

    if calc_mass_to_light:
        ml = mass_to_light(miles2, mass_weights, magfile, band=["r_SDSS", "IRAC1"], quiet=False)
        
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights,  logAge_mw, Z_mw,
                        stellar_mass,  ml, pp2, galnorm2]
        print("\n", logAge_lw, Z_lw, logAge_mw, Z_mw, stellar_mass, ml)
    else:
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights,  logAge_mw, Z_mw,
                        stellar_mass,  pp2, galnorm2]

    return results_list


if (__name__ == '__main__'):

    start_time = time.time()

    from photutils.aperture import CircularAperture as CAp
    from photutils.aperture import CircularAnnulus  as CAn
    from scipy.interpolate import interp1d

    splib    = ["EM", "BC"]
    wav_rng  = [7000.]  #[7000., 9000.]
    mdegs    = [12]  #[-1, 12]
    photoapp = [False]#, True]

    n_ppxf = len(splib) * len(wav_rng) * len(mdegs)

    pxs = 6.2    # arcsec/pixel
    R25 = 1.5    # arcmin
    nbin = 20
    rads = 0.05 * (1+np.arange(nbin-1)) * R25 * 60. / pxs
    xInd, yInd = 10-1, 15-1


    # ----- Loading MUSE spectra data cube ----- #
    fitsname = dir_cube + "DATACUBE_SPHEREx_extcor.fits"
    sp = fits.open(fitsname)
    d_sci, h_sci = sp[1].data, sp[1].header
    d_var, h_var = sp[2].data, sp[2].header
    wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          num=h_sci['NAXIS3'], endpoint=True)
    dat_rb   = fits.getdata("rebin.fits")


    # ----- Masking region ----- #
    msk_region1 = [[4865., 4895.], [4970., 4995.], [5020., 5050.], [5215., 5235.],
                   [5570., 5590.], [5861., 5912.], [6310., 6350.], 
                   [6380., 6410.], [6550., 6650.], [6700., 6780.],
                   #[6820., 6991.],
                   [7000., 9000.]]
    msk_region2 = [[4865., 4895.], [4970., 4995.], [5020., 5050.], [5215., 5235.],
                   [5570., 5590.], [5861., 5912.], [6310., 6350.], 
                   [6380., 6410.], [6550., 6650.], [6700., 6780.],
                   #[6820., 6991.],
                   [7120., 7250.], [7500., 7750.], [8150., 8350.]]


    # ----- Photometry ----- #

    # ### GALEX
    # img_galex = [dir_cube+"GALEX_DP_FUV_repr.fits", dir_cube+"GALEX_DP_NUV_repr.fits"]
    # flx_fuv = fits.getdata(img_galex[0])
    # flx_nuv = fits.getdata(img_galex[1])
    # lam_galex = np.array([0.153, 0.227])
    # Amag_galex = extinction.ccm89(lam_galex*1.0e+4, A_V, R_V, unit='aa')
    # flx_fuv *= (c / lam_galex[0]**2) * 1.0e-4 * 10.**(0.4*Amag_galex[0])
    # flx_nuv *= (c / lam_galex[1]**2) * 1.0e-4 * 10.**(0.4*Amag_galex[1])

    ### SDSS (DR12)
    # img_sdss_rb = [dir_cube+"SDSS_DR12_u_repr.fits", dir_cube+"SDSS_DR12_g_repr.fits", dir_cube+"SDSS_DR12_r_repr.fits", dir_cube+"SDSS_DR12_i_repr.fits", dir_cube+"SDSS_DR12_z_repr.fits"]
    # flx_u = fits.getdata(img_sdss_rb[0])
    flx_g = fits.getdata("SDSS_g_repr.fits")
    flx_r = fits.getdata("SDSS_r_repr.fits")
    flx_i = fits.getdata("SDSS_i_repr.fits")
    # flx_z = fits.getdata(img_sdss_rb[4])
    lam_sdss = np.array([0.475, 0.622, 0.763])  #np.array([0.353, 0.475, 0.622, 0.763, 0.905])
    Amag_sdss = extinction.ccm89(lam_sdss*1.0e+4, A_V, R_V, unit='aa')
    # flx_u *= 10.**(0.4*Amag_sdss[0])
    flx_g *= 10.**(0.4*Amag_sdss[0])
    flx_r *= 10.**(0.4*Amag_sdss[1])
    flx_i *= 10.**(0.4*Amag_sdss[2])
    # flx_z *= 10.**(0.4*Amag_sdss[4])

    # ### 2MASS
    # img_tmass = [dir_cube+"TwoMASS_DP_J_repr.fits", dir_cube+"TwoMASS_DP_H_repr.fits", dir_cube+"TwoMASS_DP_Ks_repr.fits"]
    # flx_j = fits.getdata(img_tmass[0])
    # flx_h = fits.getdata(img_tmass[1])
    # flx_k = fits.getdata(img_tmass[2])
    # lam_tmass = np.array([1.24, 1.66, 2.16])
    # Amag_tmass = extinction.ccm89(lam_tmass*1.0e+4, A_V, R_V, unit='aa')
    # flx_j *= (c / lam_tmass[0]**2) * 1.0e-4 * 10.**(0.4*Amag_tmass[0])
    # flx_h *= (c / lam_tmass[1]**2) * 1.0e-4 * 10.**(0.4*Amag_tmass[1])
    # flx_k *= (c / lam_tmass[2]**2) * 1.0e-4 * 10.**(0.4*Amag_tmass[2])
    
    ### S4G
    img_s4g = [dir_cube+"S4G_ch1_repr.fits", dir_cube+"S4G_ch2_repr.fits"]
    flx_ch1 = fits.getdata(img_s4g[0])
    flx_ch2 = fits.getdata(img_s4g[1])
    lam_s4g = np.array([3.6, 4.5])
    Amag_s4g = extinction.ccm89(lam_s4g*1.0e+4, A_V, R_V, unit='aa')
    flx_ch1 *= (c / lam_s4g[0]**2) * 1.0e-4 * 10.**(0.4*Amag_s4g[0])
    flx_ch2 *= (c / lam_s4g[1]**2) * 1.0e-4 * 10.**(0.4*Amag_s4g[1])

    ### S4G mass
    mst_s4g = fits.getdata("S4G_mstar_repr.fits")

    ### Mask array
    phot_mask = np.ones_like(dat_rb).astype('float')
    phot_mask[(dat_rb <= 0.) | (flx_r <= 0.) | (mst_s4g <= 0.)] = 0.

    ### Photometry wavelength
    wav_phot = np.hstack([lam_sdss, lam_s4g])*1.0e+4

    ### Filter response curve

    # SDSS filter response
    sdss_bands = ['u', 'g', 'r', 'i', 'z']
    ds = pd.read_csv("aj330387t4_ascii.txt", sep='\t', na_values='...')
    wave_sdss_cut, resp_sdss_cut, lam_sdss_eff, respfunc_sdss = {}, {}, {}, {}
    for i in range(len(sdss_bands)):
        wave_sdss_cut[sdss_bands[i]] = ds['lambda'].values[~np.isnan(ds[sdss_bands[i]].values)]
        resp_sdss_cut[sdss_bands[i]] = ds[sdss_bands[i]].values[~np.isnan(ds[sdss_bands[i]].values)]
        lam_sdss_eff[sdss_bands[i]]  = np.trapz(wave_sdss_cut[sdss_bands[i]]*resp_sdss_cut[sdss_bands[i]],
                                               x=wave_sdss_cut[sdss_bands[i]]) / \
                                       np.trapz(resp_sdss_cut[sdss_bands[i]], x=wave_sdss_cut[sdss_bands[i]])
        respfunc_sdss[sdss_bands[i]] = interp1d(wave_sdss_cut[sdss_bands[i]],
                                                resp_sdss_cut[sdss_bands[i]])

    # IRAC filter response
    irac_bands = ['ch1', 'ch2']
    di1 = np.genfromtxt("201125ch1trans_full.txt", dtype=None,
                        names=('wave', 'resp'), encoding='ascii')
    di2 = np.genfromtxt("201125ch2trans_full.txt", dtype=None,
                        names=('wave', 'resp'), encoding='ascii')
    dis = [di1, di2]
    wave_irac_cut, resp_irac_cut, lam_irac_eff, respfunc_irac = {}, {}, {}, {}
    for i in range(len(irac_bands)):
        wave_irac_cut[irac_bands[i]] = dis[i]['wave']*1.0e+4
        resp_irac_cut[irac_bands[i]] = dis[i]['resp']
        lam_irac_eff[irac_bands[i]]  = np.trapz(dis[i]['wave']*1.0e+4*dis[i]['resp'], x=dis[i]['wave']*1.0e+4) / \
                                       np.trapz(dis[i]['resp'], x=dis[i]['wave']*1.0e+4)
        respfunc_irac[irac_bands[i]] = interp1d(wave_irac_cut[irac_bands[i]],
                                                resp_irac_cut[irac_bands[i]])


    # ----- pPXF run ----- #
    for spl in splib:
        for wav_fin in wav_rng:
            for md in mdegs:
                for pht in photoapp:

                    if (md == -1.):
                        suffix = spl+f"_{int(wav_fin):d}_mdeg00"
                    if (md == 12.):
                        suffix = spl+f"_{int(wav_fin):d}_mdeg12"
                    if pht:
                        suffix += "_phot"

                    # ----- Setting up the Stellar Libraries ----- #
                    
                    ### E-MILES
                    if (spl == "EM"):       
                        ppxf_dir1 = "/data01/jhlee/DATA/E-MILES/Works/Set1_kin/"    # Template directory for stellar kinematics
                        ppxf_dir2 = "/data01/jhlee/DATA/E-MILES/Works/Set2_stp/"    # Template directory for stellar population
                        pathname1 = ppxf_dir1 + "Ech1.30*.fits"
                        pathname2 = ppxf_dir2 + "Ech1.30*.fits"
                        magfile   = "/data01/jhlee/DATA/E-MILES/Works/Set2_stp/E-MILES_iPp_Ch.mag"
                        fwhm_temp = 2.50

                    ### BC03
                    if (spl == "BC"):
                        ppxf_dir1 = "/data01/jhlee/DATA/bc03/Works/Set1_kin/"    # Template directory for stellar kinematics
                        ppxf_dir2 = "/data01/jhlee/DATA/bc03/Works/Set2_stp/"    # Template directory for stellar population
                        pathname1 = ppxf_dir1 + "Tch1.30*.fits"
                        pathname2 = ppxf_dir2 + "Tch1.30*.fits"
                        magfile   = "/data01/jhlee/DATA/bc03/Works/Set2_stp/Stelib2_Ch.mag"
                        fwhm_temp = 3.00                

                    ### Unbinned data
                    # ----- Running pPXF ----- #
                    csv_dir = "Tests_tot/"
                    if not path.exists(csv_dir):
                        os.system("mkdir "+csv_dir)
                    # run_name = "part01"
                    fcsvname = csv_dir+"test_tot_"+suffix+".csv"

                    fig_dir = "Tests_tot/"
                    if not path.exists(fig_dir):
                        os.system("mkdir "+fig_dir)
                    fig_suffix = "_"+suffix

                    f = open(fcsvname,"w")
                    f.write("test,vel,sigma,h3,h4,Av_star,")
                    f.write("logAge_lw,Z_lw,logAge_mw,Z_mw,")
                    f.write("logMs,M/L_r,M/L_[3.6],")
                    f.write("chi2\n")
                    f.close()

                    circular_spectrum, circular_variance = [], []
                    phot_data = {'bands': np.array([#'galex1500', 'galex2500',
                                                    #'SDSS/u',
                                                    'SDSS/g', 'SDSS/r', 'SDSS/i',
                                                    #'SDSS/z',
                                                    #'2MASS/J', '2MASS/H', '2MASS/K',
                                                    'IRAC/irac_tr1', 'IRAC/irac_tr2'])}
                                          # np.array(['GALEX/FUV', 'GALEX/NUV',
                                          #           'SDSS/u', 'SDSS/g', 'SDSS/r', 'SDSS/i', 'SDSS/z',
                                          #           '2MASS/J', '2MASS/H', '2MASS/Ks',
                                          #           'IRAC1', 'IRAC2'])
                    idx_phot = np.array([0, 1, 2, 3, 4])

                    for i in range(nbin-1):
                    # i = 0
                        if (i == 0):
                            ap = CAp((xInd, yInd), r=rads[i])
                        else:
                            ap = CAn((xInd, yInd), r_in=rads[i-1], r_out=rads[i])
                        ap_msk = ap.to_mask(method='exact')
                        msk = ap_msk.to_image((d_sci.shape[1], d_sci.shape[2]))
                        # print(f"Area: {np.sum(msk):.4f} = {np.pi*rad**2:.4f}? ", np.sum(msk) == np.pi*rad**2)

                        circular_spectrum.append(np.nansum(d_sci*msk*phot_mask, axis=(1, 2)))
                        circular_variance.append(np.nansum(d_var*msk*phot_mask, axis=(1, 2)))
                        # print(circular_spectrum.shape)

                        phot_data[f"radBin{i+1:02d}"] = np.array([#np.nansum(flx_fuv*msk*phot_mask), np.nansum(flx_nuv*msk*phot_mask),
                                                                  #np.nansum(flx_u*msk*phot_mask),
                                                                  np.nansum(flx_g*msk*phot_mask), np.nansum(flx_r*msk*phot_mask), np.nansum(flx_i*msk*phot_mask),
                                                                  #np.nansum(flx_z*msk*phot_mask),
                                                                  #np.nansum(flx_j*msk*phot_mask), np.nansum(flx_h*msk*phot_mask), np.nansum(flx_k*msk*phot_mask),
                                                                  np.nansum(flx_ch1*msk*phot_mask), np.nansum(flx_ch2*msk*phot_mask)])

                    circular_spectrum.append(np.nansum(d_sci*phot_mask, axis=(1, 2)))
                    circular_variance.append(np.nansum(d_var*phot_mask, axis=(1, 2)))
                    phot_data["Total"] = np.array([#np.nansum(flx_fuv*phot_mask), np.nansum(flx_nuv*phot_mask),
                                                   #np.nansum(flx_u*phot_mask),
                                                   np.nansum(flx_g*phot_mask), np.nansum(flx_r*phot_mask), np.nansum(flx_i*phot_mask),
                                                   #np.nansum(flx_z*phot_mask),
                                                   #np.nansum(flx_j*phot_mask), np.nansum(flx_h*phot_mask), np.nansum(flx_k*phot_mask),
                                                   np.nansum(flx_ch1*phot_mask), np.nansum(flx_ch2*phot_mask)])

                    # ----- PPXF routine ----- #
                    idxs = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11]    # Result values

                    lws, mws, tmps = {}, {}, {}
                    keywd = [f"radBin{nn+1:02d}" for nn in range(nbin-1)]
                    keywd += ["Total"]
                    assert len(keywd) == nbin

                    for k in range(len(keywd)):
                        spec_data = circular_spectrum[k]
                        vari_data = circular_variance[k]

                        if pht:
                            bands = phot_data["bands"][idx_phot]
                            lam_phot = wav_phot[idx_phot]
                            phot_galaxy = phot_data[keywd[k]][idx_phot]
                            phot_noise  = 0.1*phot_data[keywd[k]][idx_phot]
                        else:
                            bands = None
                            lam_phot = None
                            phot_galaxy = None
                            phot_noise  = None

                        if (spl == "EM"):
                            vel  = [1523., 100., 0.01, 0.01]
                            start0 = vel
                            fix_kn = False
                        if (spl == "BC"):
                            df_run0 = pd.read_csv(csv_dir+"test_tot_EM_7000_mdeg12.csv")
                            vel  = df_run0[['vel', 'sigma', 'h3', 'h4']]
                            start0 = vel.loc[k, :].values
                            fix_kn = True

                        print(f"--- pPXF routines for test ---")
                        if (spec_data.sum() <= 0.0):
                            pass

                        niters, nparam = 0, len(idxs)+3
                        norm_range = [5070, 5950]
                        np.random.seed(0)
                        mres = np.zeros((1+niters, nparam))
                        for nn in tqdm(range(1+niters)):
                            if (nn == 0):
                                spec_data2 = copy.deepcopy(spec_data)
                            else:
                                spec_data2 = truncnorm.rvs(-3.0, 3.0, loc=spec_data, scale=np.sqrt(vari_data),
                                                           size=spec_data.size)
                            res = run_ppxf1(spec_data2, wav_obs, pathname1, pathname2,
                                            norm_range=norm_range,
                                            mask_region_kn=msk_region1, mask_region_sp=msk_region2, 
                                            wlim_blue_kn=4850., wlim_red_kn=7000.,
                                            wlim_blue_sp=4850., wlim_red_sp=wav_fin,
                                            fig_dir=fig_dir, fig_suffix=fig_suffix+"_"+keywd[k], clip_sigma=4.0,
                                            start0=start0,
                                            regul=5., lnoi=10.,
                                            adeg=12, mdeg=md,
                                            calc_mass_to_light=True, magfile=magfile,
                                            dunit=1.0e-20, tunit=3.828e+33, tdist=9.8,
                                            fix_kn=fix_kn, FWHM_gal=2.62, FWHM_tem=fwhm_temp,
                                            fix_Av=False, Av0=None,
                                            apply_phot=pht, bands=bands, phot_galaxy=phot_galaxy, phot_noise=phot_noise,
                                            lam_phot=lam_phot) 

                            if (nn == 0):
                                lws[keywd[k]], mws[keywd[k]] = res[5], res[8]
                                sigma = res[13].sol[1] / (res[13].velscale / res[13].velscale_ratio)
                                shift = round(res[13].sol[0] / (res[13].velscale / res[13].velscale_ratio))
                                bestemp = res[13].templates_full @ res[13].weights
                                wave_full = res[13].lam_temp_full
                                temp_full = np.roll(ndimage.gaussian_filter1d(bestemp, sigma), shift)
                                if (shift > 0):
                                    wave_full = wave_full[shift:]
                                    temp_full = temp_full[shift:]
                                else:
                                    wave_full = wave_full[:shift]
                                    temp_full = temp_full[:shift]
                                med_bestfit = np.median(res[13].bestfit[(res[13].lam >= norm_range[0]) & \
                                                                        (res[13].lam <= norm_range[1])])
                                med_tmpfull = np.median(temp_full[(wave_full >= norm_range[0]) & \
                                                                  (wave_full <= norm_range[1])])
                                norm_factor = med_bestfit / med_tmpfull

                                tmps["chi2_"+keywd[k]] = res[13].chi2
                                tmps["wave_"+keywd[k]] = res[13].lam
                                tmps["bestfit_"+keywd[k]] = res[13].bestfit
                                tmps["wave_full_"+keywd[k]]  = wave_full
                                tmps["bestfit_full_"+keywd[k]] = temp_full * norm_factor
                                tmps["bestsed_full_"+keywd[k]] = temp_full * norm_factor * res[14] * \
                                                                 attenuation(wave_full, res[4])
                                tmps["norm_"+keywd[k]] = res[14]
                                tmps["mpoly_"+keywd[k]] = res[13].mpoly

                                # SDSS + IRAC expected flux
                                for ib, b in enumerate(sdss_bands+irac_bands):
                                    if (ib < len(sdss_bands)):
                                        wave_dict = wave_sdss_cut
                                        resp_func = respfunc_sdss
                                    if ((ib >= len(sdss_bands)) & (ib < len(sdss_bands+irac_bands))):
                                        wave_dict = wave_irac_cut
                                        resp_func = respfunc_irac
                                    wavcut = ((wave_full >= wave_dict[b][0]) & \
                                              (wave_full <= wave_dict[b][-1]))
                                    wave_full_b = wave_full[wavcut]
                                    sed_full_b  = tmps["bestsed_full_"+keywd[k]][wavcut]
                                    tmps["pPXF_"+b+"_"+keywd[k]] = np.trapz(wave_full_b*resp_func[b](wave_full_b)*sed_full_b) / \
                                                                   np.trapz(wave_full_b*resp_func[b](wave_full_b))

                            mres[nn] = np.hstack([list(map(res.__getitem__, idxs)), res[12], res[13].chi2])
                            # results_list = [vel, sigma, h3, h4, a_v_star,
                            #                 light_weights, logAge_lw, Z_lw,
                            #                 mass_weights,  logAge_mw, Z_mw,
                            #                 stellar_mass,  ml, pp2, galnorm2]

                        # except:
                            # mres[nn] = -99. * np.ones(len(idxs))

                        f_res = np.mean(mres, axis=0)
                        e_res =  np.std(mres, axis=0)
                    
                        f = open(fcsvname,"a")
                        f.write(keywd[k]+",")
                        txt = ""
                        for nn in range(nparam):
                            txt += f"{f_res[nn]:.3f}"#,{e_res[nn]:.3f}"
                            if (nn != range(nparam)[-1]):
                                txt += ","
                        f.write(txt+"\n")
                        f.close()

                        ##### Test for SED plot #####
                        # plt.close('all')
                        fig, ax = plt.subplots(figsize=(9,4))
                        
                        ax.plot(wav_obs[spec_data2 > 0.], spec_data2[spec_data2 > 0.],
                                '-', color='magenta', lw=1.5, alpha=0.9)

                        sed_bestfit = tmps["bestfit_"+keywd[k]]*tmps["norm_"+keywd[k]]* \
                                      attenuation(tmps["wave_"+keywd[k]], f_res[4])
                        ax.plot(tmps["wave_"+keywd[k]], sed_bestfit,
                                '-', color='darkorange', lw=1.5, alpha=0.8)
                        
                        sed_full = tmps["bestsed_full_"+keywd[k]]
                        ax.plot(tmps["wave_full_"+keywd[k]], sed_full,
                                '-', color='dodgerblue', lw=1.5, alpha=0.7)
                        
                        n_band = len(phot_data['bands'])
                        idx_band = np.arange(n_band)
                        syms  = np.array(['s']*2 + ['o']*5 + ['d']*3 + ['*']*2)
                        cols  = np.array(['blueviolet']*2 + ['lime']*5 + ['darkorange']*3 + ['red']*2)
                        mszs  = np.array([9.0]*(2+5+3) + [16.0]*2)
                        
                        if pht:
                            cols[idx_band[~np.in1d(idx_band, idx_phot)]] = 'silver'
                        else:
                            cols  = ['silver']*n_band

                        for nb in range(n_band):
                            ax.plot(wav_phot[nb], phot_data[keywd[k]][nb], syms[nb],
                                    color=cols[nb], ms=mszs[nb], mec='k', mew=1.5, alpha=0.8)

                        ax.set_xscale('log')
                        ax.set_xlim([700., 100000.])
                        ax.set_yscale('log')
                        ax.set_ylim([sed_full.min()/10., sed_full.max()*10.])
                        ax.set_xlabel(r"$\lambda_{\rm obs}~{\rm [\AA]}$", fontsize=12.0, fontweight='bold')
                        ax.set_ylabel(r"$f_{\lambda}~{\rm [10^{-20}~erg~s^{-1}~cm^{-2}~\AA^{-1}]}$", fontsize=12.0, fontweight='bold')
                        ax.text(0.95, 0.95, keywd[k]+fig_suffix,
                                fontsize=13.0, fontweight='bold', color='k',
                                ha='right', va='top', transform=ax.transAxes)       

                        ax.tick_params(axis='both', labelsize=12.0, pad=6.0)
                        ax.tick_params(width=2.0, length=9.0)
                        ax.tick_params(axis='x', width=2.0, length=5.0, which='minor')
                        ax.tick_params(axis='y', width=2.0, length=0.0, which='minor')
                        for axis in ['top','bottom','left','right']:
                            ax.spines[axis].set_linewidth(2.0)
                        
                        plt.tight_layout()
                        # plt.show(block=False)
                        plt.savefig(fig_dir+"SEDs"+fig_suffix+"_"+keywd[k]+".png", dpi=300)
                        plt.close()

                        if (md > 0):
                            fig, ax = plt.subplots(figsize=(9,4))
                            ax.plot(tmps["wave_"+keywd[k]], tmps["mpoly_"+keywd[k]])
                            ax.set_ylim([tmps["mpoly_"+keywd[k]].min()-0.05, tmps["mpoly_"+keywd[k]].max()+0.05])
                            ax.set_xlabel(r"$\lambda_{\rm obs}~{\rm [\AA]}$", fontsize=12.0)
                            ax.set_ylabel("Best-fit multiplicative polynomial", fontsize=12.0)
                            ax.text(0.95, 0.95, keywd[k]+fig_suffix,
                                fontsize=13.0, fontweight='bold', color='k',
                                ha='right', va='top', transform=ax.transAxes)
                            plt.tight_layout()
                            # plt.show(block=False)
                            plt.savefig(fig_dir+"Mpoly"+fig_suffix+"_"+keywd[k]+".png", dpi=300)
                            plt.close()                  
                        #############################

                    with open(csv_dir+"weights_light_tot_"+suffix+".pickle", "wb") as fw:
                        pickle.dump(lws, fw)
                    with open(csv_dir+"weights_mass_tot_"+suffix+".pickle", "wb") as fw:
                        pickle.dump(mws, fw)
                    with open(csv_dir+"templates_tot_"+suffix+".pickle", "wb") as fw:
                        pickle.dump(tmps, fw)
                    with open(csv_dir+"phot_data_"+suffix+".pickle", "wb") as fw:
                        pickle.dump(phot_data, fw)

    print(f"--- {time.time()-start_time:.4f} sec ---")  
