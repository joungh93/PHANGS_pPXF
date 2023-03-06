#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 5 14:42:13 2023

@author: jlee
"""

import time
start_time = time.time()

import numpy as np
import glob, os, copy
from pathlib import Path
from astropy.io import fits
from scipy import interpolate


# ----- Define the function ----- #
def ised_to_fits(ised_file, dir_output, age_bin=[1.0e+9, 5.0e+9, 1.0e+10],
	             wav_start=1000., wav_end=50000., wav_intv=1.0,
	             magfile="test.mag"):

	# Output directories
	if not os.path.exists(dir_output):
		os.system("mkdir "+dir_output)

	# Reading the file
	sed = np.genfromtxt(ised_file, dtype=None, encoding='ascii',
		                names=('Age','logZ','lam','F_lam'))

	# Wavelength information
	wav_new = np.linspace(start=wav_start, stop=wav_end,
                          num=1+int((wav_end-wav_start)/wav_intv),
                          endpoint=True)

	# Metallicity information
	logZ = sed['logZ'][0]
	metal = "Z"
	if (logZ < 0.0):
		metal += "m"
	else:
		metal += "p"
	metal += f"{np.abs(logZ):.2f}"

	# IMF information
	imf = ised_file.split('/')[-1].split('.')[1].split('z')[0]
	if (imf == "kr"):
		imf_str = "kr"
	if (imf == "ss"):
		imf_str = "sa"	

	# Spectral library (HB morphology...)
	splib = ised_file.split('/')[-1].split('.')[-1]
	if (splib == "rhb"):
		splib_str = "MR"
	if (splib == "bhb"):
		splib_str = "MB"

	# Reading 'ml_SSP_SDSS.tab' files
	if (imf_str == "sa"):
		sm_file = "/data01/jhlee/DATA/Maraston05/Archive/stellarmass.salpeter"
		if (splib_str == "MB"):
			nl0, nn = 18, 300
		if (splib_str == "MR"):
			nl0, nn = 321, 300
	if (imf_str == "kr"):
		sm_file = "/data01/jhlee/DATA/Maraston05/Archive/stellarmass.kroupa"
		if (splib_str == "MB"):
			nl0, nn = 627, 300
		if (splib_str == "MR"):
			nl0, nn = 930, 300

	ml_file = "/data01/jhlee/DATA/Maraston05/Archive/ml_SSP_SDSS.tab"
	mls = np.genfromtxt(ml_file, dtype=None, encoding='ascii',
		                names=('logZ','Age','ML_u','ML_g','ML_r','ML_i','ML_z'),
		                skip_header=nl0, max_rows=nn)
	mst = np.genfromtxt(sm_file, dtype=None, encoding='ascii', usecols=(0,1,2),
		                names=('logZ','Age','M_tot'))

	# Age information
	age = sed['Age']
	if (len(age_bin) == 0):
		raise ValueError("The input age bin is empty.")
	else:
		for i in range(len(age_bin)):
			# if (i == 0):
			# 	f = open(magfile, "w")
			# 	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
			# 	# f.write("ML(u_SDSS) ML(g_SDSS) ML(r_SDSS) ML(i_SDSS) ML(z_SDSS)\n")
			# else:
			f = open(magfile, "a")

			idx_age = np.argmin(np.abs(age-age_bin[i]*1.0e-9))
			wavelength = sed['lam'][age == age[idx_age]]
			sed2 = sed['F_lam'][age == age[idx_age]]
			func = interpolate.interp1d(wavelength, sed2, kind='linear')
			sed_interp = func(wav_new)

			out = splib_str+imf_str+"1.30"+metal+f"T{age[idx_age]:07.4f}"
			out += "_iPp0.00_baseFe_linear_FWHM_variable.fits"

			hdu = fits.PrimaryHDU()
			hdu.header['CRVAL1'] = wav_new[0]
			hdu.header['CDELT1'] = wav_intv
			hdu.header['CRPIX1'] = 1
			fits.writeto(str(Path(dir_output) / out), sed_interp, hdu.header, overwrite=True)

			mls_Z = mls[np.where(mls['logZ'] == logZ)[0]]
			idx2_age = np.argmin(np.abs(mls_Z['Age'] - age[idx_age]))
			M_tot = mst['M_tot'][(mst['logZ'] == logZ) & (mst['Age'] == mls_Z['Age'][idx2_age])][0]
			sdss_sun_mag = [6.45, 5.14, 4.65, 4.54, 4.52]  # values provided by not Elena Ricciardelli, but http://mips.as.arizona.edu/~cnaw/sun_2006.html
			f.write(out+f" 1.30 {logZ:.5f} {mls_Z['Age'][idx2_age]:.5f} {M_tot:.5f} ")
			f.write(f"{sdss_sun_mag[0]+2.5*np.log10(mls_Z['ML_u'][idx2_age]/M_tot):.5f} ")
			f.write(f"{sdss_sun_mag[1]+2.5*np.log10(mls_Z['ML_g'][idx2_age]/M_tot):.5f} ")
			f.write(f"{sdss_sun_mag[2]+2.5*np.log10(mls_Z['ML_r'][idx2_age]/M_tot):.5f} ")
			f.write(f"{sdss_sun_mag[3]+2.5*np.log10(mls_Z['ML_i'][idx2_age]/M_tot):.5f} ")
			f.write(f"{sdss_sun_mag[4]+2.5*np.log10(mls_Z['ML_z'][idx2_age]/M_tot):.5f}\n")
			f.close()


# ----- Files & Directories ----- #
dir_mar05 = "/data01/jhlee/DATA/Maraston05/"
dir_sa = dir_mar05+"Archive/Sed_Mar05_SSP_Salpeter/"
dir_kr = dir_mar05+"Archive/Sed_Mar05_SSP_Kroupa/"
# /data01/jhlee/DATA/Maraston05/Archive/Sed_Mar05_SSP_Kroupa/sed.krz002.rhb

dir_output = ["Set1_kin", "Set2_stp"]
for i in range(len(dir_output)):
	if not os.path.exists(dir_output[i]):
		os.system("mkdir "+dir_output[i])


# ----- Running ised_to_fits ----- #

### File 1: Kinematics
age_bin1 = [0.10e+9, 0.20e+9, 0.40e+9, 0.70e+9, 1.25e+9,
            2.00e+9, 3.25e+9, 5.00e+9, 8.50e+9, 1.40e+10]

# Salpeter IMF
ised_file_1_sa = [dir_sa+"sed.ssz0001.rhb",
                  dir_sa+"sed.ssz001.rhb",
                  dir_sa+"sed.ssz002.rhb"]

# Kroupa IMF
ised_file_1_kr = [dir_kr+"sed.krz0001.rhb",
                  dir_kr+"sed.krz001.rhb",
                  dir_kr+"sed.krz002.rhb"]

magfile = ["Maraston05_Sa.mag", "Maraston05_Kr.mag"]
ised_files = [ised_file_1_sa, ised_file_1_kr]
for j in range(len(ised_files)):
	f = open(str(Path(dir_output[0]) / magfile[j]), "w")
	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
	f.close()
	for i in range(len(ised_files[j])):
		print(ised_files[j][i], magfile[j])
		ised_to_fits(ised_files[j][i], dir_output[0], age_bin=age_bin1,
		             wav_start=1000., wav_end=50000., wav_intv=1.0,
		             magfile=str(Path(dir_output[0]) / magfile[j])			         )

### File 2: Stellar populations
age_bin2 = [0.03e+9, 0.05e+9, 0.08e+9, 0.10e+9, 0.20e+9,
            0.40e+9, 0.60e+9, 1.00e+9, 1.75e+9, 3.00e+9,
            5.00e+9, 8.50e+9, 1.35e+10]

# Salpeter IMF
ised_file_2_sa = [dir_sa+"sed.ssz0001.rhb",
                  dir_sa+"sed.ssz001.rhb",
                  dir_sa+"sed.ssz002.rhb",
                  dir_sa+"sed.ssz004.rhb"]

# Kroupa IMF
ised_file_2_kr = [dir_kr+"sed.krz0001.rhb",
                  dir_kr+"sed.krz001.rhb",
                  dir_kr+"sed.krz002.rhb",
                  dir_kr+"sed.krz004.rhb"]

magfile = ["Maraston05_Sa.mag", "Maraston05_Kr.mag"]
ised_files = [ised_file_2_sa, ised_file_2_kr]
for j in range(len(ised_files)):
	f = open(str(Path(dir_output[1]) / magfile[j]), "w")
	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
	f.close()
	for i in range(len(ised_files[j])):
		print(ised_files[j][i], magfile[j])
		ised_to_fits(ised_files[j][i], dir_output[1], age_bin=age_bin2,
		             wav_start=1000., wav_end=50000., wav_intv=1.0,
		             magfile=str(Path(dir_output[1]) / magfile[j])			         )
