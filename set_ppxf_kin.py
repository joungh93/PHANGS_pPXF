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
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
c = 2.99792458e+5    # light speed in km/s

from ppxf.ppxf import ppxf, robust_sigma, attenuation
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
import extinction


# ---- Basic properties ----- #
galaxy_name = "NGC1087"
ebv = 0.030
R_V = 3.1
A_V = R_V * ebv
lum_dist = 15.9   # Mpc


# ----- Loading MUSE spectra data cube ----- #
dir_cube = "/data01/jhlee/DATA/PHANGS/MUSE/"+galaxy_name+"/"
fitsname = dir_cube + "DATACUBE_SPHEREx_extcor.fits"
sp = fits.open(fitsname)
d_sci, h_sci = sp[1].data, sp[1].header
d_var, h_var = sp[2].data, sp[2].header
dat_rb   = fits.getdata(dir_cube + "rebin.fits")
wav_obs = np.linspace(start=h_sci['CRVAL3']+(1-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      stop=h_sci['CRVAL3']+(h_sci['NAXIS3']-h_sci['CRPIX3'])*h_sci['CD3_3'],
                      num=h_sci['NAXIS3'], endpoint=True)

# dir_cube = "/data01/jhlee/DATA/PHANGS/MUSE/NGC628/"
# with open(dir_cube+"Data_dict/box_spec_total.pickle", "rb") as fr:
#     box_spec = pickle.load(fr)
# with open(dir_cube+"Data_dict/box_vari_total.pickle", "rb") as fr:
#     box_vari = pickle.load(fr)



# ----- Splitting the data for parallelization ----- #
nl, ny, nx = d_sci.shape
d_sci2d = d_sci.reshape(nl, -1)
d_var2d = d_var.reshape(nl, -1)
spec_sum = np.nansum(d_sci2d, axis=0)
idx_nonzero = np.flatnonzero(spec_sum > 0.)

n_spec, n_core = len(idx_nonzero), 16
n_pixel = n_spec // n_core + 1
part_name, run_array = "total_kin", idx_nonzero


# ----- Functions ----- #
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
        plt.figure(figsize=(8,4))
        pp1.plot()
        plt.tight_layout()
        # plt.show(block=False)
        plt.savefig(fig_dir+"pPXF_results3"+fig_suffix+".png", dpi=300)
        plt.close()

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

    results_list = [vel, sigma, h3, h4, reddening0, pp1]

    return results_list


if (__name__ == '__main__'):

    splib, wav_fin, mdeg, photoapp, fix_kn = "EM", 7000., 12, False, False
    start0 = [1523., 100., 0.01, 0.01]


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


    # ----- Setting up the Stellar Libraries ----- #
    
    ### E-MILES
    if (splib == "EM"):       
        ppxf_dir1 = "/data01/jhlee/DATA/E-MILES/Works/Set1_kin/"    # Template directory for stellar kinematics
        ppxf_dir2 = "/data01/jhlee/DATA/E-MILES/Works/Set2_stp/"    # Template directory for stellar population
        pathname1 = ppxf_dir1 + "Ech1.30*.fits"
        pathname2 = ppxf_dir2 + "Ech1.30*.fits"
        magfile   = "/data01/jhlee/DATA/E-MILES/Works/Set2_stp/E-MILES_iPp_Ch.mag"
        fwhm_temp = 2.50


    # ----- Running pPXF ----- #
    start_time = time.time()

    csv_dir = "Kinematics/"
    if not path.exists(csv_dir):
        os.system("mkdir "+csv_dir)
    fcsvname = csv_dir+part_name+".csv"

    fig_dir = "Kinematics/"
    if not path.exists(fig_dir):
        os.system("mkdir "+fig_dir)

    f = open(fcsvname,"w")
    f.write("x,y,vel,e_vel,sigma,e_sigma,h3,e_h3,h4,e_h4,Av_star,e_Av_star,")
    f.write("chi2,e_chi2\n")
    f.close()

    lws, mws, tmps = {}, {}, {}
    for i in run_array:
        iy, ix = i // nx, i % nx
        keywd = f"x{ix:03d}_y{iy:03d}"

        spec_data = d_sci2d[:, i]#box_spec[keywd]
        # d_sci2d[:, i]
        vari_data = d_var2d[:, i]#box_vari[keywd]
        # d_var2d[:, i]


        # ----- pPXF routine ----- #
        idxs = [0, 1, 2, 3, 4]  # Result values
        print(f"\n--- pPXF routines for ({ix:d},{iy:d}) ---")
        if (spec_data.sum() <= 0.0):
            continue

        niters, nparam = 0, len(idxs)+1
        norm_range = [5070, 5950]
        np.random.seed(0)
        mres = np.zeros((1+niters, nparam))
        for nn in tqdm(range(1+niters)):
            if (nn == 0):
                spec_data2 = copy.deepcopy(spec_data)
            else:
                spec_data2 = truncnorm.rvs(-3.0, 3.0, loc=spec_data, scale=np.sqrt(vari_data),
                                           size=spec_data.size)

            try:
                # def run_ppxf1(spectrum, observed_wavelength, tempath_kn, tempath_sp,
                #               norm_range=[5070, 5950],
                #               mask_region_kn=None, mask_region_sp=None, 
                #               wlim_blue_kn=4800., wlim_red_kn=9000.,
                #               wlim_blue_sp=4800., wlim_red_sp=9000.,
                #               fig_dir="./", fig_suffix="_", clip_sigma=4.0,
                #               start0=None, regul=5., lnoi=10.,
                #               adeg=12, mdeg=12,
                #               calc_mass_to_light=True, magfile=None,
                #               dunit=1.0e-20, tunit=3.828e+33, tdist=9.8,
                #               fix_kn=False, FWHM_gal=2.62, FWHM_tem=2.51,
                #               fix_Av=False, Av0=None,
                #               apply_phot=False, bands=None, phot_galaxy=None, phot_noise=None,
                #               lam_phot=None):
                res = run_ppxf1(spec_data2, wav_obs, pathname1, pathname2,
                                norm_range=norm_range,
                                mask_region_kn=msk_region1, mask_region_sp=msk_region2, 
                                wlim_blue_kn=4850., wlim_red_kn=7000.,
                                wlim_blue_sp=4850., wlim_red_sp=wav_fin,
                                fig_dir=fig_dir, fig_suffix="_"+keywd, clip_sigma=4.0,
                                start0=start0,
                                regul=5., lnoi=10.,
                                adeg=12, mdeg=mdeg,
                                calc_mass_to_light=True, magfile=magfile,
                                dunit=1.0e-20, tunit=3.828e+33, tdist=lum_dist,
                                fix_kn=fix_kn, FWHM_gal=2.62, FWHM_tem=fwhm_temp,
                                fix_Av=False, Av0=None,
                                apply_phot=False, bands=None,
                                phot_galaxy=None, phot_noise=None,
                                lam_phot=None)
                mres[nn] = np.hstack([list(map(res.__getitem__, idxs)), res[5].chi2])                   

            except:
                mres[nn] = -99. * np.ones(nparam)

        f_res = np.mean(mres, axis=0)
        e_res = np.std(mres, axis=0)
        
        f = open(fcsvname,"a")
        f.write(f"{ix:03d},{iy:03d},")
        txt = ""
        for ipar in range(nparam):
            txt += f"{f_res[ipar]:.3f},{e_res[ipar]:.3f}"
            if (ipar != range(nparam)[-1]):
                txt += ","
        f.write(txt+"\n")
        f.close()

    print(f"--- {time.time()-start_time:.4f} sec ---")
