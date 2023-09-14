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
# from scipy.stats import truncnorm
from astropy.io import fits
# from astropy.cosmology import FlatLambdaCDM
c = 2.99792458e+5    # light speed in km/s

# from ppxf.ppxf import ppxf, robust_sigma, attenuation
# import ppxf.ppxf_util as util
# import ppxf.miles_util as lib

from scipy.interpolate import interp1d
# from matplotlib import cm
# import matplotlib as mpl
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import truncate_colormap as tc


# ----- Reading the data ----- #
dir_output = "Tests_tot/"
csv_list = sorted(glob.glob(dir_output+"test_tot_*.csv"))

dfs = {}
for i in range(len(csv_list)):
    keywd = csv_list[i].split("/")[-1].split("test_tot_")[-1].split(".csv")[0]
    df = pd.read_csv(csv_list[i])
    dfs[keywd] = df


# ----- Radial information ----- #
galaxy = "NGC1087"
pxs = 6.2   # arcsec/pixel
R25 = 1.5   # arcmin
nbin = 20
rads = 0.05 * (1+np.arange(nbin-1)) * R25 * 60. / pxs
rR25 = 0.05 * (1+np.arange(nbin-1))
xInd, yInd = 10-1, 15-1

lum_dist = 15.9   # Mpc
dist_mod = 5.0*np.log10(lum_dist*1.0e+6 / 10.)
ang_scale = lum_dist * 1.0e+6 * (1./3600) * (np.pi/180.)   # pc/arcsec
print(f"Scale: {ang_scale:.3f} pc/arcsec")


# ----- Absolute magnitudes ----- #
Mag_r_Sun_AB, Mag_ch1_Sun_AB  = 4.65, 6.02
lam_eff_r, lam_eff_ch1 = 0.622, 3.6    # micrometer


# ----- Reading data from Pessa+23 ----- #
dir_P23 = "/data01/jhlee/DATA/PHANGS/MUSE/Pessa+23/"
dfA = pd.read_csv(dir_P23+galaxy+"_logAge.csv")
dfZ = pd.read_csv(dir_P23+galaxy+"_Z.csv")


# ----- Reading data from S4G ----- #
dat_rb  = fits.getdata("rebin.fits")
flx_r   = fits.getdata("SDSS_r_repr.fits")
mst_s4g = fits.getdata("S4G_mstar_repr.fits")
fst_s4g = fits.getdata("S4G_flux_repr.stellar.fits")  # microJy
nst_s4g = fits.getdata("S4G_flux_repr.nonstellar.fits")  # microJy
flx_s4g = fits.getdata("S4G_ch1_repr.fits")  # microJy

phot_mask = np.ones_like(dat_rb).astype('float')
phot_mask[(dat_rb <= 0.) | (flx_r <= 0.) | (mst_s4g <= 0.)] = 0.
area = np.hstack([np.pi * np.diff(np.hstack([0., rads**2.])), np.sum(phot_mask)])

from photutils.aperture import CircularAperture as CAp
from photutils.aperture import CircularAnnulus  as CAn

Mst_S4G, Fst_S4G, Dst_S4G, Flx_S4G = [], [], [], []
for i in range(nbin-1):
    if (i == 0):
        ap = CAp((xInd, yInd), r=rads[i])
    else:
        ap = CAn((xInd, yInd), r_in=rads[i-1], r_out=rads[i])
    ap_msk = ap.to_mask(method='exact')
    msk = ap_msk.to_image((dat_rb.shape[0], dat_rb.shape[1]))
    Mst_S4G.append(np.nansum(mst_s4g*msk*phot_mask))
    Fst_S4G.append(np.nansum(fst_s4g*msk*phot_mask))
    Dst_S4G.append(np.nansum(nst_s4g*msk*phot_mask))
    Flx_S4G.append(np.nansum(flx_s4g*msk*phot_mask))

Mst_S4G.append(np.nansum(mst_s4g*phot_mask))
Fst_S4G.append(np.nansum(fst_s4g*phot_mask))
Dst_S4G.append(np.nansum(nst_s4g*phot_mask))
Flx_S4G.append(np.nansum(flx_s4g*phot_mask))

Mst_S4G, Fst_S4G = np.array(Mst_S4G), np.array(Fst_S4G)
Dst_S4G, Flx_S4G = np.array(Dst_S4G), np.array(Flx_S4G)
Rat_S4G = Dst_S4G / (Fst_S4G + Dst_S4G)


# ----- Plotting the RDPs ----- #
fig_dir = "Tests_tot/Figures/"
if not os.path.exists(fig_dir):
    os.system("mkdir "+fig_dir)

# plt.close('all')

for i in range(len(dfs)):

    keywd = list(dfs.keys())[i]

    # ----- Columns in the dataframe ----- #
    for col in ['Av_star', 'logAge_lw', 'Z_lw', 'M/L_r', 'M/L_[3.6]', 'chi2']:

        fig, ax = plt.subplots(figsize=(6,5))

        if (keywd[:2] == 'BC'):
            splib, symcol, titcol = "BC03", "orange", "darkorange"
        if (keywd[:2] == 'EM'):
            splib, symcol, titcol = "E-MILES", "dodgerblue", "royalblue"
        
        if (col == 'Av_star'):
            ymin, ymax, ylab = -0.05, 0.90, r"$A_{V}~[{\rm mag}]$"
            ytick = [0.0, 0.2, 0.4, 0.6, 0.8]
            out = col
        if (col == 'logAge_lw'):
            ymin, ymax, ylab =   7.8, 10.2, "log Age(LW) [yr]"
            ytick = [8.0, 8.5, 9.0, 9.5, 10.0]
            xcmp, ycmp = dfA[dfA.columns[0]].values, dfA[dfA.columns[1]].values
            out = col
        if (col == 'Z_lw'):
            ymin, ymax, ylab = -1.45,  0.5, r"$[Z/Z_{\odot}]$"
            ytick = [-1.5, -1.0, -0.5, 0.0, 0.5]
            xcmp, ycmp = dfZ[dfZ.columns[0]].values, dfZ[dfZ.columns[1]].values
            out = col
        if (col == 'M/L_r'):
            ymin, ymax, ylab =  -0.1,  3.0, r"$M_{\ast}/L_{r}$"
            ytick = [0.0, 1.0, 2.0, 3.0]
            out = "ML_r"
        if (col == 'M/L_[3.6]'):
            ymin, ymax, ylab =  -0.1,  1.0, r"$M_{\ast}/L_{\rm 3.6\mu m}$"
            ytick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            out = "ML_[3.6]"
        if (col == 'chi2'):
            ymin, ymax, ylab =  -0.1, 10.0, r"$\chi^{2}_{\nu}$"
            ytick = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
            out = col

        ax.plot(rR25, dfs[keywd][col].values[:-1], '-',
                lw=1.8, color=symcol, alpha=0.8, zorder=2)        
        ax.plot(rR25, dfs[keywd][col].values[:-1], 'o',
                ms=6.0, color=symcol, mec='k', mew=1.0, alpha=0.8, zorder=3)
        if (col in ['logAge_lw', 'Z_lw']):
            ax.plot(xcmp, ycmp, '-', color='gray', lw=1.5, alpha=0.7, zorder=1,
                    label="From PHANGS team (Pessa+23)")
            ax.legend(loc=(0.30, 0.05), labelcolor='dimgray', fontsize=10.5)
            ff = interp1d(xcmp, ycmp)
            resid = dfs[keywd][col].values[:-1] - ff(rR25)
            RMSE = np.sqrt(np.mean(resid**2.))
            ax.text(0.95, 0.15, f"RMSE = {RMSE:.2f} dex",
                    fontsize=10.5, fontweight='bold', color='dimgray',
                    ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.50, 1.03, galaxy, fontsize=15.0, fontweight='bold', color='k',
                ha='center', va='bottom', transform=ax.transAxes)
        if (keywd.split('_')[-1] == 'phot'):
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+", phot)",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)
        else:
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+")",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_yticks(ytick)
        # ax.set_xticklabels(['0.1', '0.5', '1', '5', '10'])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(r"Galactocentric Distance [$R_{25}$]", fontsize=15.0, fontweight='bold')
        ax.set_ylabel(ylab, fontsize=15.0, fontweight='bold')
        ax.tick_params(axis='both', labelsize=15.0, pad=7.0)
        ax.tick_params(width=1.5, length=7.0)
        # ax.tick_params(axis='x', width=1.5, length=5.0, which='minor')
        # ax.tick_params(axis='y', width=1.5, length=0.0, which='minor')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        plt.tight_layout()
        # plt.show(block=False)
        plt.savefig(fig_dir+galaxy+"_"+keywd+"_"+out+".png", dpi=300)
        plt.close()


    # ----- Stellar mass comparison ----- #

    ### PPXF-expected data
    with open(dir_output+"templates_tot_"+keywd+".pickle", "rb") as fr:
        temps = pickle.load(fr)
    
    kwds_ppxf_r = [f"pPXF_r_radBin{rbin+1:02d}" for rbin in range(len(rads))] + \
                  ["pPXF_r_Total"]
    flx_r_pPXF = np.array([temps[kw] for kw in kwds_ppxf_r]) * (lam_eff_r**2. / c) * 1.0e+4  # to microJy
    Mag_r_pPXF_AB = 23.90 - 2.5*np.log10(flx_r_pPXF) - dist_mod
    Mst_r_pPXF = 10.0 ** (-0.4*(Mag_r_pPXF_AB - Mag_r_Sun_AB)) * dfs[keywd]['M/L_r'].values#[-1]
    Msd_r_pPXF = Mst_r_pPXF / (area * ang_scale**2. * pxs**2.)
    dfs[keywd]['flx_r_pPXF'] = flx_r_pPXF
    dfs[keywd]['Mst_r_pPXF'] = Mst_r_pPXF
    dfs[keywd]['Msd_r_pPXF'] = Msd_r_pPXF
   
    kwds_ppxf_ch1 = [f"pPXF_ch1_radBin{rbin+1:02d}" for rbin in range(len(rads))] + \
                    ["pPXF_ch1_Total"]
    flx_ch1_pPXF = np.array([temps[kw] for kw in kwds_ppxf_ch1]) * (lam_eff_ch1**2. / c) * 1.0e+4  # to microJy
    Mag_ch1_pPXF_AB = 23.90 - 2.5*np.log10(flx_ch1_pPXF) - dist_mod
    Mst_ch1_pPXF = 10.0 ** (-0.4*(Mag_ch1_pPXF_AB - Mag_ch1_Sun_AB)) * dfs[keywd]['M/L_[3.6]'].values#[-1]
    Msd_ch1_pPXF = Mst_ch1_pPXF / (area * ang_scale**2. * pxs**2.)
    dfs[keywd]['flx_[3.6]_pPXF'] = flx_ch1_pPXF
    dfs[keywd]['Mst_[3.6]_pPXF'] = Mst_ch1_pPXF
    dfs[keywd]['Msd_[3.6]_pPXF'] = Msd_ch1_pPXF
   
    ### Photometric data
    with open(dir_output+"phot_data_"+keywd+".pickle", "rb") as fr:
        phots = pickle.load(fr)

    kwds_rads = [f"radBin{rbin+1:02d}" for rbin in range(len(rads))] + \
                ["Total"]
    idx_r = phots['bands'].tolist().index('SDSS/r')
    flx_r_obs = np.array([phots[kw][idx_r] for kw in kwds_rads]) * (lam_eff_r**2. / c) * 1.0e+4  # to microJy
    Mag_r_obs_AB = 23.90 - 2.5*np.log10(flx_r_obs) - dist_mod
    Mst_r_obs = 10.0 ** (-0.4*(Mag_r_obs_AB - Mag_r_Sun_AB)) * dfs[keywd]['M/L_r'].values#[-1]
    Msd_r_obs = Mst_r_obs / (area * ang_scale**2. * pxs**2.)
    dfs[keywd]['flx_r_obs'] = flx_r_obs

    idx_ch1 = phots['bands'].tolist().index('IRAC/irac_tr1')
    flx_ch1_obs = np.array([phots[kw][idx_ch1] for kw in kwds_rads]) * (lam_eff_ch1**2. / c) * 1.0e+4  # to microJy
    Mag_ch1_obs_AB = 23.90 - 2.5*np.log10(flx_ch1_obs) - dist_mod
    Mst_ch1_obs = 10.0 ** (-0.4*(Mag_ch1_obs_AB - Mag_ch1_Sun_AB)) * dfs[keywd]['M/L_[3.6]'].values#[-1]
    Msd_ch1_obs = Mst_ch1_obs / (area * ang_scale**2. * pxs**2.)
    dfs[keywd]['flx_[3.6]_obs'] = flx_ch1_obs
    
    # Mst_ch1_S4G = 10.0 ** (-0.4*(Mag_ch1_obs_AB - Mag_ch1_Sun_AB) * 0.6
    # Msd_ch1_S4G = Mst_ch1_S4G / (area * ang_scale**2. * pxs**2.)

    ### Figures
    Mtot_pPXF = [Mst_r_pPXF[-1], Mst_ch1_pPXF[-1]]
    Mtot_obs  = [Mst_r_obs[-1] , Mst_ch1_obs[-1]]
    yData = [Msd_r_pPXF[:-1]   / Msd_r_obs[:-1],
             Msd_ch1_pPXF[:-1] / Msd_ch1_obs[:-1]]
    ylab  = [r"$\Sigma_{\ast}$(pPXF)/$\Sigma_{\ast}$(obs) [$M_{\ast}/L_{r}$]",
             r"$\Sigma_{\ast}$(pPXF)/$\Sigma_{\ast}$(obs) [$M_{\ast}/L_{\rm 3.6\mu m}$]"]
    out_band = ["r", "IRAC1"]

    for j in range(len(yData)):
        fig, ax = plt.subplots(figsize=(6,5))

        ax.axhline(1.0, 0.0, 1.0, ls='--', color='gray', lw=1.5, alpha=0.6, zorder=1)
        ax.plot(rR25, yData[j], 'o',
                ms=6.0, color=symcol, mec='k', mew=1.0, alpha=0.8, zorder=3)
        ax.plot(rR25, yData[j], '-',
                lw=1.8, color=symcol, alpha=0.8, zorder=2)

        ax.text(0.50, 1.03, galaxy, fontsize=15.0, fontweight='bold', color='k',
                ha='center', va='bottom', transform=ax.transAxes)
        if (keywd.split('_')[-1] == 'phot'):
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+", phot)",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)
        else:
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+")",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_yticks([0.5, 1.0, 1.5])
        # ax.set_xticklabels(['0.1', '0.5', '1', '5', '10'])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.1, 1.9])
        ax.set_xlabel(r"Galactocentric Distance [$R_{25}$]", fontsize=15.0, fontweight='bold')
        ax.set_ylabel(ylab[j], fontsize=15.0, fontweight='bold')
        ax.tick_params(axis='both', labelsize=15.0, pad=7.0)
        ax.tick_params(width=1.5, length=7.0)
        # ax.tick_params(axis='x', width=1.5, length=5.0, which='minor')
        # ax.tick_params(axis='y', width=1.5, length=0.0, which='minor')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        ax.text(0.95, 0.15, r"log $M_{\ast}/M_{\odot}$ (pPXF) = "+ \
                f"{np.log10(Mtot_pPXF[j]):.2f}",
                fontsize=13.0, fontweight='bold', color='gray',
                ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.95, 0.05, r"log $M_{\ast}/M_{\odot}$ (obs) = "+ \
                f"{np.log10(Mtot_obs[j]):.2f}",
                fontsize=13.0, fontweight='bold', color='gray',
                ha='right', va='bottom', transform=ax.transAxes)

        plt.tight_layout()
        # plt.show(block=False)
        plt.savefig(fig_dir+galaxy+"_"+keywd+"_"+"dMsden_"+out_band[j]+".png", dpi=300)
        plt.close()

    ### Figures (S4G)
    Msd_S4G = Mst_S4G / (area * ang_scale**2. * pxs**2.)
    dfs[keywd]['Mst_S4G'] = Mst_S4G
    dfs[keywd]['Msd_S4G'] = Msd_S4G

    yData = [Msd_r_pPXF[:-1]   / Msd_S4G[:-1],
             Msd_ch1_pPXF[:-1] / Msd_S4G[:-1]]
    ylab  = [r"$\Sigma_{\ast}$(pPXF)/$\Sigma_{\ast}$(S4G) [$M_{\ast}/L_{r}$]",
             r"$\Sigma_{\ast}$(pPXF)/$\Sigma_{\ast}$(S4G) [$M_{\ast}/L_{\rm 3.6\mu m}$]"]

    for j in range(len(yData)):
        fig, ax = plt.subplots(figsize=(6,5))

        ax.axhline(1.0, 0.0, 1.0, ls='--', color='gray', lw=1.5, alpha=0.6, zorder=1)
        ax.plot(rR25, yData[j], 'o',
                ms=6.0, color=symcol, mec='k', mew=1.0, alpha=0.8, zorder=3)
        ax.plot(rR25, yData[j], '-',
                lw=1.8, color=symcol, alpha=0.8, zorder=2)

        ax.text(0.50, 1.03, galaxy, fontsize=15.0, fontweight='bold', color='k',
                ha='center', va='bottom', transform=ax.transAxes)
        if (keywd.split('_')[-1] == 'phot'):
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+", phot)",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)
        else:
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+")",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_yticks([0.5, 1.0, 1.5])
        # ax.set_xticklabels(['0.1', '0.5', '1', '5', '10'])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.1, 1.9])
        ax.set_xlabel(r"Galactocentric Distance [$R_{25}$]", fontsize=15.0, fontweight='bold')
        ax.set_ylabel(ylab[j], fontsize=15.0, fontweight='bold')
        ax.tick_params(axis='both', labelsize=15.0, pad=7.0)
        ax.tick_params(width=1.5, length=7.0)
        # ax.tick_params(axis='x', width=1.5, length=5.0, which='minor')
        # ax.tick_params(axis='y', width=1.5, length=0.0, which='minor')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        ax.text(0.95, 0.15, r"log $M_{\ast}/M_{\odot}$ (pPXF) = "+ \
                f"{np.log10(Mtot_pPXF[j]):.2f}",
                fontsize=13.0, fontweight='bold', color='gray',
                ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.95, 0.05, r"log $M_{\ast}/M_{\odot}$ (S4G) = "+ \
                f"{np.log10(Mst_S4G[-1]):.2f}",
                fontsize=13.0, fontweight='bold', color='gray',
                ha='right', va='bottom', transform=ax.transAxes)

        plt.tight_layout()
        # plt.show(block=False)
        plt.savefig(fig_dir+galaxy+"_"+keywd+"_"+"dMsden_"+out_band[j]+"_S4G.png", dpi=300)
        plt.close()


    # ----- Flux comparison ----- #
    yData = [flx_r_pPXF[:-1]   / flx_r_obs[:-1],
             flx_ch1_pPXF[:-1] / flx_ch1_obs[:-1],
             flx_ch1_pPXF[:-1] / Fst_S4G[:-1]]
    dfs[keywd]['fSt_S4G'] = Fst_S4G
    dfs[keywd]['nSt_S4G'] = Dst_S4G

    ylab = ["Flux(pPXF) / Flux(obs) [SDSS r]",
            "Flux(pPXF) / Flux(obs) [IRAC1]",
            "Flux(pPXF) / Flux(S4G) [IRAC1, stellar]"]
    ylim = [[0.75, 1.25], [0.1, 1.9], [0.1, 1.9]]
    ytck = [[0.8, 1.0, 1.2], [0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]
    out_band = ["r", "IRAC1", "S4G"]

    for j in range(len(yData)):
        fig, ax = plt.subplots(figsize=(6,5))

        ax.axhline(1.0, 0.0, 1.0, ls='--', color='gray', lw=1.5, alpha=0.6, zorder=1)
        ax.plot(rR25, yData[j], 'o',
                ms=6.0, color=symcol, mec='k', mew=1.0, alpha=0.8, zorder=3)
        ax.plot(rR25, yData[j], '-',
                lw=1.8, color=symcol, alpha=0.8, zorder=2)

        ax.text(0.50, 1.03, galaxy, fontsize=15.0, fontweight='bold', color='k',
                ha='center', va='bottom', transform=ax.transAxes)
        if (keywd.split('_')[-1] == 'phot'):
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+", phot)",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)
        else:
            ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                    r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+")",
                    fontsize=12.5, fontweight='bold', color=titcol,
                    ha='left', va='top', transform=ax.transAxes)

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_yticks(ytck[j])
        # ax.set_xticklabels(['0.1', '0.5', '1', '5', '10'])
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim(ylim[j])
        ax.set_xlabel(r"Galactocentric Distance [$R_{25}$]", fontsize=15.0, fontweight='bold')
        ax.set_ylabel(ylab[j], fontsize=15.0, fontweight='bold')
        ax.tick_params(axis='both', labelsize=15.0, pad=7.0)
        ax.tick_params(width=1.5, length=7.0)
        # ax.tick_params(axis='x', width=1.5, length=5.0, which='minor')
        # ax.tick_params(axis='y', width=1.5, length=0.0, which='minor')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1.5)

        plt.tight_layout()
        # plt.show(block=False)
        plt.savefig(fig_dir+galaxy+"_"+keywd+"_"+"dFlux_"+out_band[j]+".png", dpi=300)
        plt.close()


    # ----- Flux ratio comparison ----- #
    fig, ax = plt.subplots(figsize=(6,5))

    # ax.axhline(1.0, 0.0, 1.0, ls='--', color='gray', lw=1.5, alpha=0.6, zorder=1)
    ax.plot(rR25, Dst_S4G[:-1] / (Fst_S4G[:-1] + Dst_S4G[:-1]), '--',
            color='gray', lw=1.5, alpha=0.6, zorder=1)
    ax.plot(rR25, (Flx_S4G[:-1] - flx_ch1_pPXF[:-1]) / Flx_S4G[:-1], 'o',
            ms=6.0, color=symcol, mec='k', mew=1.0, alpha=0.8, zorder=3)
    ax.plot(rR25, (Flx_S4G[:-1] - flx_ch1_pPXF[:-1]) / Flx_S4G[:-1], '-',
            lw=1.8, color=symcol, alpha=0.8, zorder=2)

    ax.text(0.50, 1.03, galaxy, fontsize=15.0, fontweight='bold', color='k',
            ha='center', va='bottom', transform=ax.transAxes)
    if (keywd.split('_')[-1] == 'phot'):
        ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+", phot)",
                fontsize=12.5, fontweight='bold', color=titcol,
                ha='left', va='top', transform=ax.transAxes)
    else:
        ax.text(0.04, 0.96, splib+" (4850-"+keywd.split('_')[1]+ \
                r"${\rm \AA}$"+", MDEG="+keywd.split('_')[2][4:]+")",
                fontsize=12.5, fontweight='bold', color=titcol,
                ha='left', va='top', transform=ax.transAxes)

    # ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    # ax.set_xticklabels(['0.1', '0.5', '1', '5', '10'])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.05, 0.75])
    ax.set_xlabel(r"Galactocentric Distance [$R_{25}$]", fontsize=15.0, fontweight='bold')
    ax.set_ylabel(r"Non-stellar flux ratio [$3.6{\rm \mu m}$]", fontsize=15.0, fontweight='bold')
    ax.tick_params(axis='both', labelsize=15.0, pad=7.0)
    ax.tick_params(width=1.5, length=7.0)
    # ax.tick_params(axis='x', width=1.5, length=5.0, which='minor')
    # ax.tick_params(axis='y', width=1.5, length=0.0, which='minor')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    ax.text(0.95, 0.15, r"$f_{\rm dust}$ (pPXF) = "+ \
            f"{(Flx_S4G[-1] - flx_ch1_pPXF[-1]) / Flx_S4G[-1]:.2f}",
            fontsize=13.0, fontweight='bold', color='gray',
            ha='right', va='bottom', transform=ax.transAxes)
    ax.text(0.95, 0.05, r"$f_{\rm dust}$ (S4G) = "+ \
            f"{Dst_S4G[-1] / (Fst_S4G[-1] + Dst_S4G[-1]):.2f}",
            fontsize=13.0, fontweight='bold', color='gray',
            ha='right', va='bottom', transform=ax.transAxes)

    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig(fig_dir+galaxy+"_"+keywd+"_"+"fFlux_"+out_band[j]+".png", dpi=300)
    plt.close()


