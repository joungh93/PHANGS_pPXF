#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 6 09:32:28 2023
@author: jlee
"""


# importing necessary modules
import numpy as np
import glob, os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from astropy.io import fits
import pickle


# ----- Basic properties ----- #
galaxy_name = "NGC1087"
dist_lum = 15.9  # Mpc
dist_mod = 5*np.log10(dist_lum*1.0e+6 / 10.)  # Distance modulus
pos_angle0, pos_x0, pos_y0, pxs0 =  359.1, 707-1, 757-1, 0.2  # Voronoi-binned data
pos_angle , pos_x , pos_y , pxs  =  359.1,  23-1,  25-1, 6.2  # SPHEREx-binned data


# ----- Constants ----- #
c = 2.99792e+5  # km/s
Mag_r_Sun_AB, Mag_ch1_Sun_AB  = 4.65, 6.02  # mag
lam_eff_r, lam_eff_ch1 = 0.622, 3.6  # micrometer


# ----- Reading the Data ----- #

### Released (un-binned / vornoi-binned) data
wht_img = glob.glob(galaxy_name+"_PHANGS_IMAGE_white_copt_*.fits")[0]
wht_data = fits.getdata(wht_img, ext=1, header=False)
wht_stat = fits.getdata(wht_img, ext=2, header=False)

wht_stat[wht_stat == 0] = np.nanmedian(wht_stat[wht_stat > 0.])
wht_SNR = wht_data / np.sqrt(wht_stat)

cube_img = glob.glob(galaxy_name+"_MAPS_copt_*.fits")[0]
vbin_data = fits.getdata(cube_img, ext=1)
h0 = fits.getheader(cube_img, ext=0)
v0 = float(h0['REDSHIFT'])

### SPHEREx-rebinned / CIGALE-rebinned data
img_rebin = fits.getdata("rebin.fits")
snr_rebin = fits.getdata("rebin_SNR.fits")
# cig_rebin = fits.getdata("rebin_3.75.fits")
# snr_cigal = fits.getdata("rebin_SNR_3.75.fits")


# ----- Figure scales ----- #
col_name = ["ML_[3.6]", "vel_star", "sigma_star",
            "logAge_lw", "logAge_mw", "Z_lw", "Z_mw", "Av_star"]
vmins = [0.0, v0-50.0,  25.,  8.9,  8.9, -1.25, -1.25, 0.0]
vmaxs = [1.0, v0+50.0, 100., 10.1, 10.1,  0.25,  0.25, 1.0]


# ----- Function for reading the pPXF results ----- #
def read_all_parts(dir_run):
    if (dir_run[-1] != "/"):
        dir_run += "/"

    # Reading 'part*.csv'
    file_run = sorted(glob.glob(dir_run+"Results/part*.csv"))
    n_file = len(file_run)

    df = pd.read_csv(file_run[0])
    for i in range(1, n_file, 1):
        tmp = pd.read_csv(file_run[i])
        df = pd.concat([df, tmp], axis=0, ignore_index=True)

    # Reading 'templates_part*.pickle'
    file_run = sorted(glob.glob(dir_run+"Results/templates*.pickle"))
    n_file = len(file_run)

    with open(file_run[0], "rb") as fr:
        dct = pickle.load(fr)
    for i in range(1, n_file, 1):
        with open(file_run[i], "rb") as fr:
            tmp = pickle.load(fr)
        dct = dct | tmp

    return [df, dct]

df_run1S, tmp_run1S  = read_all_parts("Run1S")
df_run2S, tmp_run2S  = read_all_parts("Run2S")


# ----- Plotting maps ----- #
def draw_map(data_frame, column, out, shape, xyid=True, vorbins=None):
    par_map = np.zeros(shape)
    for i in range(len(data_frame)):
        if xyid:
            ix, iy = data_frame['x'].values[i], data_frame['y'].values[i]
            par_map[iy, ix] = data_frame[column].values[i]
        else:
            vb = (vorbins == data_frame['BinID'].values[i])
            par_map[vb] = data_frame[column].values[i]
    fits.writeto(out, par_map, overwrite=True)


def draw_map_from_ch1(data_dict, data_frame, out_flux, out_mass, shape, dist_mod):
    flux_map = np.zeros(shape)
    mass_map = np.zeros(shape)
    
    dict_key = np.array(list(data_dict.keys()))
    ch1_keys = dict_key[pd.Series(dict_key).str.startswith("pPXF_ch1_").values]
    n_keywd  = len(ch1_keys)

    for i in range(n_keywd):
        ix = int(ch1_keys[i].split('_')[2][1:])
        iy = int(ch1_keys[i].split('_')[3][1:])
        keywd   = f"x{ix:03d}_y{iy:03d}"
        xycnd   = ((data_frame['x'] == ix) & (data_frame['y'] == iy))

        flx_ch1 = data_dict['pPXF_ch1'+'_'+keywd] * (lam_eff_ch1**2. / c) * 1.0e+4  # to microJy
        if (flx_ch1 > 0.):
            Mag_ch1 = 23.90 - 2.5*np.log10(flx_ch1) - dist_mod  # AB magnitude
            ML_ch1  = data_frame.loc[xycnd, 'M/L_[3.6]'].values[0]
            Mst_ch1 = 10.0 ** (-0.4*(Mag_ch1 - Mag_ch1_Sun_AB)) * ML_ch1
            flux_map[iy, ix] = flx_ch1
            mass_map[iy, ix] = Mst_ch1
        else:
            continue

    fits.writeto(out_flux, flux_map, overwrite=True)
    fits.writeto(out_mass, mass_map, overwrite=True)


def plot_2Dmap(plt_Data, title, v_low, v_high, out, cmap='gray_r',
               add_cb=True, add_or=True, add_sc=True, add_pa=True,
               cb_label=None, 
               x0=-2.75, y0=1.25, sign=-1, L=0.6, theta0=0.0*(np.pi/180.0),
               xN=-1.90, yN=1.25, xE=-2.95, yE=2.10,
               ang_scale=0.04751, pixel_scale=6.2,
               pos_angle=20.7, pos_x=707., pos_y=757.):

    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    plt.suptitle(title, x=0.5, ha='center', y=0.96, va='top',
                 fontsize=17.0)
    xmin, xmax = -plt_Data.shape[1]*pixel_scale/2, plt_Data.shape[1]*pixel_scale/2
    ymin, ymax = -plt_Data.shape[0]*pixel_scale/2, plt_Data.shape[0]*pixel_scale/2
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    # ax.set_xticks([-3,-2,-1,0,1,2,3])
    # ax.set_yticks([-2,-1,0,1,2])
    # ax.set_xticklabels([r'$-3$',r'$-2$',r'$-1$',0,1,2,3], fontsize=15.0)
    # ax.set_yticklabels([r'$-2$',r'$-1$',0,1,2], fontsize=15.0)
    ax.set_xlabel(r'$\Delta X$ [arcsec]', fontsize=13.0) 
    ax.set_ylabel(r'$\Delta Y$ [arcsec]', fontsize=13.0)
    ax.tick_params(axis='both', direction='in', width=1.0, length=5.0, labelsize=14.0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.0)

    im = ax.imshow(plt_Data, cmap=cmap, vmin=v_low, vmax=v_high, aspect='equal',
                   extent=[xmin, xmax, ymin, ymax],
                   origin='lower')

    if add_cb:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(im, cax=cax)
        cb.set_label(cb_label, size=12.0, labelpad=10.0)
        cb.ax.tick_params(direction='in', labelsize=12.0)

    if add_or:
        ax.arrow(x0+sign*0.025, y0, L*np.sin(theta0), L*np.cos(theta0), width=0.8,
                 head_width=5.0, head_length=5.0, fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.arrow(x0, y0+sign*0.025, -L*np.cos(theta0), L*np.sin(theta0), width=0.8,
                 head_width=5.0, head_length=5.0, fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.text(xN, yN, 'N', fontsize=11.0, fontweight='bold', color='blueviolet')
        ax.text(xE, yE, 'E', fontsize=11.0, fontweight='bold', color='blueviolet')
    
    if add_sc:
        kpc2 = 2.0 / ang_scale
        ax.arrow(75.0, -120.0, kpc2, 0., width=0.8, head_width=0., head_length=0.,
                  fc='blueviolet', ec='blueviolet', alpha=0.9)
        ax.text(80.0, -112.0, '2 kpc', fontsize=11.0, fontweight='bold', color='blueviolet')

    if add_pa:
        x_arr = np.linspace(xmin, xmax, 1000)
        ax.plot(x_arr, np.tan(np.pi/2. + pos_angle*np.pi/180.)*(x_arr-pixel_scale*(pos_x-plt_Data.shape[1]/2)) + \
                pixel_scale*(pos_y-plt_Data.shape[0]/2),
                color='magenta', ls='--', lw=1.5, alpha=0.8)

    # plt.savefig(out+'.pdf', dpi=300)
    plt.savefig(out, dpi=300)
    plt.close()

dir_output = "Figure_maps"
if not os.path.exists(dir_output):
    os.system("mkdir "+dir_output)

cols = ['vel', 'sigma', 'Av_star', 'logAge_lw', 'Z_lw', 'logAge_mw', 'Z_mw', 'M/L_[3.6]']
out_suffix = ['vel_star', 'sigma_star', 'Av_star', 'logAge_lw', 'Z_lw', 'logAge_mw', 'Z_mw', 'ML_ch1']


# def draw_map(data_frame, column, out, shape, xyid=True, vorbins=None):

### Run 1S
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run1S_"+out_suffix[i]+".fits"
    draw_map(df_run1S, cols[i], outfile, img_rebin.shape,
             xyid=True)#, vorbins=vbin_data)

### Run 2S
for i in range(len(cols)):
    outfile = str(Path(dir_output))+"/Map_run2S_"+out_suffix[i]+".fits"
    draw_map(df_run2S, cols[i], outfile, img_rebin.shape,
             xyid=True)#, vorbins=vbin_data)


# draw_map_from_ch1(data_dict, data_frame, out_flux, out_mass, shape, dist_mod):

### Run 1S
outflux = str(Path(dir_output))+"/Map_run1S_Flx_ch1_pPXF.fits"
outmass = str(Path(dir_output))+"/Map_run1S_Mst_ch1_pPXF.fits"
draw_map_from_ch1(tmp_run1S, df_run1S, outflux, outmass, img_rebin.shape, dist_mod)

### Run 2S
outflux = str(Path(dir_output))+"/Map_run2S_Flx_ch1_pPXF.fits"
outmass = str(Path(dir_output))+"/Map_run2S_Mst_ch1_pPXF.fits"
draw_map_from_ch1(tmp_run2S, df_run2S, outflux, outmass, img_rebin.shape, dist_mod)


# ----- Plotting images ----- #

### Unbinned images
print("----- Unbinned -----")
vmin, vmax = np.percentile(wht_data, [2.0, 98.0])
plot_2Dmap(wht_data, "White map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_white.png",
           cmap='gray_r', add_cb=False, add_or=True, add_sc=True,
           x0=-80., y0=80., sign=-1, L=30., theta0=0.0*(np.pi/180.0),
           xN=-85., yN=120., xE=-130., yE=75., pixel_scale=0.2,
           add_pa=True, pos_angle=pos_angle0, pos_x=pos_x0, pos_y=pos_y0)

im = fits.getdata(cube_img, ext=2)
im += v0
vmin, vmax = vmins[1], vmaxs[1]    #np.percentile(im[~np.isnan(im)], [50.0, 95.0])
im[(wht_SNR <= 100.) | (np.isnan(wht_SNR)) | (im <= 0)] = np.nan
plot_2Dmap(im, "Radial velocity map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_unbinned_vel_star.png",
           cmap='viridis', cb_label=r"$v_{\rm rad}~{\rm [km~s^{-1}]}$",
           add_cb=True, add_or=False, add_sc=False, pixel_scale=0.2,
           add_pa=True, pos_angle=pos_angle0, pos_x=pos_x0, pos_y=pos_y0)
print(f"V_rad (star): {np.average(im[~np.isnan(im)], weights=wht_data[~np.isnan(im)]):.2f} km/s")

im = fits.getdata(cube_img, ext=4)
vmin, vmax = vmins[2], vmaxs[2]   #np.percentile(im[~np.isnan(im)], [50.0, 95.0])
im[(wht_SNR <= 100.) | (np.isnan(wht_SNR)) | (im <= 0)] = np.nan
plot_2Dmap(im, "Velocity dispersion map (NGC 628)", vmin, vmax, str(Path(dir_output))+"/Map_unbinned_vdisp_star.png",
           cmap='viridis', cb_label=r"$\sigma_{v}({\rm star})~{\rm [km~s^{-1}]}$",
           add_cb=True, add_or=False, add_sc=False, pixel_scale=0.2,
           add_pa=True, pos_angle=pos_angle0, pos_x=pos_x0, pos_y=pos_y0)
print(f"V_disp (star): {np.average(im[~np.isnan(im)], weights=wht_data[~np.isnan(im)]):.2f} km/s")


### Rebinned images
run_name = ["run1S", "run2S"]
map_name = [r"$M/L_{\rm 3.6\mu m}$", "Radial velocity", "Velocity dispersion",
            "Age (LW)", "Age (MW)", r"[$Z/Z_{\odot}$] (LW)", r"[$Z/Z_{\odot}$] (MW)",
            r"$A_{V,\ast}$"]
colorbar = [r"$M/L_{\rm 3.6\mu m}$", r"$v_{\rm rad}~{\rm [km~s^{-1}]}$", r"$\sigma_{v}({\rm star})~{\rm [km~s^{-1}]}$",
            "log Age [yr]", "log Age [yr]", r"$[Z/Z_{\odot}]$", r"$[Z/Z_{\odot}]$",
            r"$A_{V,\ast}$ [mag]"]

for ppxf_run in run_name:
    print("\n----- "+ppxf_run+" -----")
    for i in range(len(cols)):
        im = fits.getdata(str(Path(dir_output))+"/Map_"+ppxf_run+"_"+out_suffix[i]+".fits")
        if ((ppxf_run[3] != '0') & ((ppxf_run[-2] != "W"))):
            # pos_angle, pos_x, pos_y = 20.7, 23-1, 25-1
            img_2D, snr_2D = img_rebin, snr_rebin
            snr_cut = 5.0*31
            pixel_scale = 6.2
        elif (ppxf_run[-2] == "W"):
            # pos_angle, pos_x, pos_y = 20.7, 78-1, 78-1
            img_2D, snr_2D = cig_rebin, snr_cigal
            snr_cut = 5.0*3.75/0.2
            pixel_scale = 3.75            
        else:
            # pos_angle, pos_x, pos_y = 20.7, 707-1, 757-1
            img_2D, snr_2D = wht_data, wht_SNR
            snr_cut = 5.0*1
            pixel_scale = 0.2
        im[(snr_2D <= snr_cut) | (np.isnan(snr_2D)) | (im == 0) | (im == -99.)] = np.nan
        plot_2Dmap(im, map_name[i]+" map (NGC 628)", vmins[i], vmaxs[i],
                   str(Path(dir_output))+"/Map_"+ppxf_run+"_"+out_suffix[i]+".png",
                   cmap='viridis', cb_label=colorbar[i],
                   add_cb=True, add_or=False, add_sc=False, pixel_scale=pixel_scale,
                   add_pa=True, pos_angle=pos_angle, pos_x=pos_x, pos_y=pos_y)
        print(cols[i]+f": {np.average(im[~np.isnan(im)], weights=img_2D[~np.isnan(im)]):.3f}")
