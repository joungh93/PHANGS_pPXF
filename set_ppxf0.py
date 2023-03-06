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


n_files = len(glob.glob("Vorbin/*.pickle"))
# run_name, run_array = "part01", np.arange(1500*0, np.minimum(1500*1, n_files), 1)


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


def init_spec(spectrum, observed_wavelength, mask_region, wlim_blue=4800., wlim_red=7000.):

    # ----- Inital Wavelength Cut ----- #
    eff = ((observed_wavelength >= wlim_blue) & (observed_wavelength <= wlim_red))
    lam = observed_wavelength[eff]
    median_galaxy = np.median(spectrum[eff & (spectrum > 0.) & (~np.isnan(spectrum) & (~np.isinf(spectrum)))])
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
    goodpixels2 = np.flatnonzero(msk_flag2 == 0)
    assert msk_flag2[goodpixels2].sum() == 0.0

    return [galaxy2, lam2, goodpixels2, velscale, median_galaxy]


def run_ppxf1(spectrum, observed_wavelength, tempath_kn, tempath_sp,
              mask_region_kn=None, mask_region_sp=None, 
              wlim_blue_kn=4800., wlim_red_kn=9000.,
              wlim_blue_sp=4800., wlim_red_sp=9000.,
              fig_dir="./", fig_suffix="_", clip_sigma=4.0,
              start0=None, regul=5., lnoi=10.,
              adeg=12, mdeg=12,
              calc_mass_to_light=True, magfile=None,
              dunit=1.0e-20, tunit=3.828e+33, tdist=9.8):

    # ----- Initial process (for kinematics) ----- #
    galaxy1, lam1, goodpixels1, velscale, galnorm1 = init_spec(spectrum, observed_wavelength,
                                                               mask_region_kn,
                                                               wlim_blue=wlim_blue_kn,
                                                               wlim_red=wlim_red_kn)
    velscale_ratio = 2
    vscale = velscale / velscale_ratio

    # ----- Making the Stellar Templates ----- #
    FWHM_gal = 2.62    # Median FWHM resolution of MUSE
    FWHM_tem = 2.51    # Vazdekis+10 spectra FWHM: 2.51AA

    # ssp, h1 = fits.getdata(sorted(glob.glob(tempath_kn))[0], header=True)
    # lamRange1 = h1['CRVAL1'] + np.array([0., h1['CDELT1']*(h1['NAXIS1'] - 1)])
    # sspNew, logLam1, velscale_temp = util.log_rebin(lamRange1, ssp, velscale=vscale)
    # print(velscale_temp)

    miles1 = lib.miles(tempath_kn, vscale, FWHM_gal=FWHM_gal, FWHM_tem=FWHM_tem,
                       norm_range=[5070, 5950])
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

    # ----- Initial Running pPXF for Stellar Components ----- #
    if (fig_dir[-1] != "/"):
        fig_dir += "/"

    if start0 is None:
        start = [100., 180., 0.1, 0.1]
    else:
        start = start0

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

    plt.figure(figsize=(8,4))
    pp1.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results1"+fig_suffix+".png", dpi=300)
    plt.close()

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

    plt.figure(figsize=(8,4))
    pp1.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results2"+fig_suffix+".png", dpi=300)
    plt.close()

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

    plt.figure(figsize=(8,4))
    pp1.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results3"+fig_suffix+".png", dpi=300)
    plt.close()

    vel, sigma, h3, h4 = pp1.sol
    print(vel, sigma, h3, h4, pp1.reddening)
    e_vel, e_sigma, e_h3, e_h4 = pp1.error * np.sqrt(pp1.chi2)
    # noise_new3, SN_new = get_noise(galaxy1, pp.bestfit, lam1, goodpixels1, lnoi=10.)
    start = [vel, sigma, h3, h4]

    # ----- Initial process (for stellar population) ----- #
    galaxy2, lam2, goodpixels2, velscale, galnorm2 = init_spec(spectrum, observed_wavelength,
                                                               mask_region_sp,
                                                               wlim_blue=wlim_blue_sp,
                                                               wlim_red=wlim_red_sp)
    velscale_ratio = 2
    vscale = velscale / velscale_ratio
    print(galnorm2, vscale)
    # ----- Adding More Templates ----- #
    miles2 = lib.miles(tempath_sp, vscale, FWHM_gal=FWHM_gal, FWHM_tem=FWHM_tem,
                       norm_range=[5070, 5950])
    reg_dim2 = miles2.templates.shape[1:]
    # print(reg_dim2)

    stars_templates = miles2.templates.reshape(miles2.templates.shape[0], -1)
    stars_norm = np.median(stars_templates)
    stars_templates /= stars_norm # Normalizes stellar templates by a scalar
    # print(stars_templates.shape)

    # ----- Iterative Running pPXF for Stellar Components (for stellar population) ----- #

    ### 4th - better determine for new noise
    print("\n PPXF Run 4")
    pp2 = ppxf(stars_templates, galaxy2, np.ones_like(galaxy2), velscale, start,
               moments=4, degree=-1, mdegree=mdeg, goodpixels=goodpixels2,
               lam=lam2, lam_temp=miles2.lam_temp,
               velscale_ratio=velscale_ratio, quiet=True,
               reg_dim=reg_dim2, reg_ord=2, regul=0.,
               reddening=None, clean=True, fixed=[1,1,1,1])

    plt.figure(figsize=(8,4))
    pp2.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results4"+fig_suffix+".png", dpi=300)
    plt.close()

    vel, sigma, h3, h4 = pp2.sol
    print(vel, sigma, h3, h4)
    noise2_1, SN2 = get_noise(galaxy2, pp2.bestfit, lam2, goodpixels2, lnoi=10.)
    err = 1.0e-5


    ### 5th - for stellar absorption
    print("\n PPXF Run 5")
    pp2 = ppxf(stars_templates, galaxy2, noise2_1, velscale, start,
               moments=4, degree=-1, mdegree=-1, goodpixels=goodpixels2,
               lam=lam2, lam_temp=miles2.lam_temp,
               velscale_ratio=velscale_ratio, quiet=True,
               reg_dim=reg_dim2, reg_ord=2, regul=0.,
               reddening=pp1.reddening, clean=False, fixed=[0,0,0,0],
               bounds=[[vel-err, vel+err], [sigma-err, sigma+err],
                       [h3-err, h3+err], [h4-err, h4+err]])

    plt.figure(figsize=(8,4))
    pp2.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results5"+fig_suffix+".png", dpi=300)
    plt.close()

    vel, sigma, h3, h4 = pp2.sol
    a_v_star = pp2.reddening
    print(vel, sigma, h3, h4, pp2.reddening)
    noise2_2, SN2 = get_noise(galaxy2, pp2.bestfit, lam2, goodpixels2, lnoi=10.)

    f_extn = attenuation(lam2, a_v_star)
    galaxy3 = galaxy2 / f_extn
    eff = ((galaxy2 > 0.) & (~np.isnan(galaxy2)) & (~np.isinf(galaxy2)))
    galnorm3 = np.median(galaxy2[eff] * galnorm2 / f_extn[eff])
    print(galnorm3)
    # galaxy3 /= galnorm3

    
    ### 6th - for stellar population
    print("\n PPXF Run 6")
    pp2 = ppxf(stars_templates, galaxy3, noise2_2, velscale, start,
               moments=4, degree=-1, mdegree=mdeg, goodpixels=goodpixels2,
               lam=lam2, lam_temp=miles2.lam_temp,
               velscale_ratio=velscale_ratio, quiet=True,
               reg_dim=reg_dim2, reg_ord=2, regul=0.,
               reddening=None, clean=False, fixed=[1,1,1,1])

    plt.figure(figsize=(8,4))
    pp2.plot()
    plt.tight_layout()
    plt.savefig(fig_dir+"pPXF_results6"+fig_suffix+".png", dpi=300)
    plt.close()

    vel, sigma, h3, h4 = pp2.sol
    a_v_star = pp2.reddening
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
        mg, lg, ml = mass_to_light(miles2, mass_weights, magfile, band="r")
        
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights,  logAge_mw, Z_mw,
                        stellar_mass,  ml, pp2]
        print("\n", logAge_lw, Z_lw, logAge_mw, Z_mw, stellar_mass, ml)
    else:
        results_list = [vel, sigma, h3, h4, a_v_star,
                        light_weights, logAge_lw, Z_lw,
                        mass_weights,  logAge_mw, Z_mw,
                        stellar_mass,  pp2]

    return results_list


if (__name__ == '__main__'):

    # ----- Setting up the Stellar Libraries ----- #
    ppxf_dir1 = "/data01/jhlee/Downloads/E-MILES/tmp1/"    # Template directory for stellar kinematics
    ppxf_dir2 = "/data01/jhlee/Downloads/E-MILES/tmp2/"    # Template directory for stellar population
    pathname1 = ppxf_dir1 + "Ech1.30*.fits"
    pathname2 = ppxf_dir2 + "Ech1.30*.fits"
    magfile   = "/data01/jhlee/Downloads/E-MILES/sdss_ch_iTp0.00.MAG"

    # ----- Loading MUSE spectra data cube ----- #
    dir_cube = "./"
    filename = dir_cube + "NGC0628_PHANGS_DATACUBE_copt_0.92asec.fits"
    sp = fits.open(filename)
    d_sci, h_sci = sp[1].data, sp[1].header
    d_var, h_var = sp[2].data, sp[2].header
    wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                          num=h_sci['NAXIS3'], endpoint=True)

    # bin_id = fits.getdata(dir_cube + "NGC0628_MAPS_copt_0.92asec.fits", ext=1)
    #####


    # # ----- Loading the re-binned data ----- #
    # img_rb = fits.getdata("rebin.fits")
        
    # # ----- Loading the re-binned data ----- #
    # with open("box_spec_total.pickle", "rb") as fr:
    #     box_spec = pickle.load(fr)
    # with open("box_vari_total.pickle", "rb") as fr:
    #     box_vari = pickle.load(fr)

    # ----- Masking region ----- #
    msk_region1 = [[5570., 5590.], [5861., 5912.], [6280., 6320.],
                   [6350., 6380.], [6528., 6585.], [6715., 6760.],
                   [6820., 6991.], [7000., 9000.]]

    msk_region2 = [[5570., 5590.], [5861., 5912.], [6280., 6320.],
                   [6350., 6380.], [6528., 6585.], [6715., 6760.],
                   [6820., 6991.], [7120., 7250.], [7500., 7750.],
                   [8150., 8350.]]

    ### Unbinned data
    # ----- Running pPXF ----- #
    start_time = time.time()

    csv_dir = "./"
    # if not path.exists(csv_dir):
    #     os.system("mkdir "+csv_dir)
    run_name = "part01"
    filename = csv_dir+"Results.csv"
    fig_dir = "./"
    # if not path.exists(fig_dir):
    #     os.system("mkdir "+fig_dir)

    niters, nparam = 1, 9
    f = open(filename,"w")
    f.write("BinID,vel,e_vel,sigma,e_sigma,Av_star,e_Av_star,")
    f.write("logAge_lw,e_logAge_lw,Z_lw,e_Z_lw,logAge_mw,e_logAge_mw,Z_mw,e_Z_mw,")
    f.write("logMs,e_logMs,M/L,e_M/L\n")
    f.close()

    lws, mws, tmps = {}, {}, {}
    for i in [45330]:  #run_array:#np.arange(0, 12000, 1):
        keywd = f"bin{i:05d}"
        with open(dir_cube+f"Vorbin/vbn_{i:05d}.pickle", "rb") as fr:
            vbn = pickle.load(fr)
        spec_data = vbn['spec']
        vari_data = vbn['vari']
        print(f"--- pPXF routines for the Voronoi bin {i:05d} ---")
        if (spec_data.sum() <= 0.0):
            continue

        np.random.seed(0)
        mres = np.zeros((niters, nparam))
        for nn in tqdm(range(niters)):
            # spec_data2 = truncnorm.rvs(-3.0, 3.0, loc=spec_data, scale=np.sqrt(vari_data),
            #                            size=spec_data.size)
            idxs = [0, 1, 4, 6, 7, 9, 10, 11, 12]    # Result values
            try:
            # def run_ppxf1(spectrum, observed_wavelength, tempath_kn, tempath_sp,
            #               mask_region_kn=None, mask_region_sp=None, 
            #               wlim_blue_kn=4800., wlim_red_kn=9000.,
            #               wlim_blue_sp=4800., wlim_red_sp=9000.,
            #               fig_dir="./", fig_suffix="_", clip_sigma=4.0,
            #               start0=None, regul=5., lnoi=10.,
            #               adeg=12, mdeg=12,
            #               calc_mass_to_light=True, magfile=None,
            #               dunit=1.0e-20, tunit=3.828e+33, tdist=9.8):
                res = run_ppxf1(spec_data, wav_obs, pathname1, pathname2,
                                mask_region_kn=msk_region1, mask_region_sp=msk_region2, 
                                wlim_blue_kn=4800., wlim_red_kn=7000.,
                                wlim_blue_sp=4800., wlim_red_sp=9000.,
                                fig_dir=fig_dir, fig_suffix="_"+keywd, clip_sigma=4.0,
                                start0=[659., 49., -0.030, 0.097], regul=5., lnoi=10.,
                                adeg=12, mdeg=-1,
                                calc_mass_to_light=True, magfile=magfile,
                                dunit=1.0e-20, tunit=3.828e+33, tdist=9.8)                    
                
                mres[nn] = np.array(list(map(res.__getitem__, idxs)))
                
                lws[keywd], mws[keywd] = res[5], res[8]

                sigma = res[13].sol[1] / (res[13].velscale[0] / res[13].velscale_ratio)
                shift = round(res[13].sol[0] / (res[13].velscale[0] / res[13].velscale_ratio))
                bestemp = res[13].templates_full @ res[13].weights
                wave_full = res[13].lam_temp_full
                temp_full = np.roll(ndimage.gaussian_filter1d(bestemp, sigma), shift)
                if (shift > 0):
                    wave_full = wave_full[shift:]
                    temp_full = temp_full[shift:]
                else:
                    wave_full = wave_full[:shift]
                    temp_full = temp_full[:shift]
                tmps["wave_"+keywd] = res[13].lam
                tmps["bestfit_"+keywd] = res[13].bestfit
                tmps["wave_full_"+keywd]  = wave_full
                tmps["bestfit_full_"+keywd] = temp_full

            except:
                mres[nn] = -99. * np.ones(len(idxs))

        f_res = np.mean(mres, axis=0)
        e_res = np.std(mres, axis=0)
        
        f = open(filename,"a")
        f.write(f"{i:05d},")
        txt = ""
        for nn in range(nparam):
            txt += f"{f_res[nn]:.3f},{e_res[nn]:.3f}"
            if (nn != range(nparam)[-1]):
                txt += ","
        f.write(txt+"\n")
        f.close()

    with open(csv_dir+"weights_light_"+run_name+".pickle", "wb") as fw:
        pickle.dump(lws, fw)
    with open(csv_dir+"weights_mass_"+run_name+".pickle", "wb") as fw:
        pickle.dump(mws, fw)
    with open(csv_dir+"templates_"+run_name+".pickle", "wb") as fw:
        pickle.dump(tmps, fw)

print(f"--- {time.time()-start_time:.4f} sec ---")        

