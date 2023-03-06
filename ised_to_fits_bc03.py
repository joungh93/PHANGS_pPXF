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
	with open(ised_file, "r") as f:
		ll = f.readlines()

	# Reading '*.1ABmag' and '*.4color' files
	ab1 = np.genfromtxt(ised_file.split(".ised_ASCII")[0]+".1ABmag", dtype=None, encoding='ascii',
		                usecols=(0,1,2,3,4,5), names=('log_age_yr','u','g','r','i','z'))
	col4 = np.genfromtxt(ised_file.split(".ised_ASCII")[0]+".4color", dtype=None, encoding='ascii',
		                 usecols=(0,5,6,10), names=('log_age_yr','M_liv','M_remn','M_tot'))

	# Wavelength information
	n_wav = int(ll[6].split()[0])
	wavelength = np.array(ll[6].split()[1:]).astype('float')
	wav_new = np.linspace(start=wav_start, stop=wav_end,
                          num=1+int((wav_end-wav_start)/wav_intv),
                          endpoint=True)

	# Metallicity information
	logZ = np.log10(float(ll[3].split()[-1].split('=')[-1]) / 0.02)
	metal = "Z"
	if (logZ < 0.0):
		metal += "m"
	else:
		metal += "p"
	metal += f"{np.abs(logZ):.2f}"

	# IMF information
	imf = ised_file.split('/')[-1].split('_')[4]
	imf_str = imf[:2]
	# if (imf == "chab"):
	# 	imf_str = "ch"
	# if (imf == "kroup"):
	# 	imf_str = "kr"
	# if (imf == "salp"):
	# 	imf_str = "sa"

	# Spectral library
	splib = ised_file.split('/')[-1].split('_')[2]
	if (splib == "BaSeL"):
		splib_str = "S"
	if (splib == "xmiless"):
		splib_str = "X"
	if (splib == "stelib"):
		splib_str = "T"
	# splib_str = splib[1].upper()

	# Age information

	age = np.array(ll[0].split()[1:]).astype('float')
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

			idx_age = np.argmin(np.abs(age-age_bin[i]))
			sed2 = np.array(ll[7+idx_age].split()[1:n_wav+1]).astype('float')
			func = interpolate.interp1d(wavelength, sed2, kind='linear')
			sed_interp = func(wav_new)

			out = splib_str+imf_str+"1.30"+metal+f"T{age[idx_age]*1.0e-9:07.4f}"
			out += "_iPp0.00_baseFe_linear_FWHM_variable.fits"

			hdu = fits.PrimaryHDU()
			hdu.header['CRVAL1'] = wav_new[0]
			hdu.header['CDELT1'] = wav_intv
			hdu.header['CRPIX1'] = 1
			fits.writeto(str(Path(dir_output) / out), sed_interp, hdu.header, overwrite=True)

			idx2_age = np.argmin(np.abs(ab1['log_age_yr'] - np.log10(age[idx_age])))
			f.write(out+f" 1.30 {logZ:.5f} {10.**(ab1['log_age_yr'][idx2_age]-9):.5f} {col4['M_tot'][idx2_age]:.5f} ")
			f.write(f"{ab1['u'][idx2_age]:.5f} {ab1['g'][idx2_age]:.5f} {ab1['r'][idx2_age]:.5f} {ab1['i'][idx2_age]:.5f} {ab1['z'][idx2_age]:.5f}\n")
			f.close()


	# wav_new = np.linspace(start=wav_start, stop=wav_end,
    #                       num=1+int((wav_end-wav_start)/ic.wav_intv), endpoint=True)

#####

# ----- Files & Directories ----- #
dir_bc03 = "/data01/jhlee/DATA/bc03/"
# "/data/jlee/DATA/bc03/Stelib_Atlas/Chabrier_IMF/"
# ised_file = dir_ssp+"bc2003_hr_stelib_m62_chab_ssp.ised_ASCII"
dir_output = ["Set1_kin", "Set2_stp"]
for i in range(len(dir_output)):
	if not os.path.exists(dir_output[i]):
		os.system("mkdir "+dir_output[i])


# ----- Running ised_to_fits ----- #

# ### File 1: Kinematics
# ised_file_1 = sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m32_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m52_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m62_*.ised_ASCII"))

### File 2: Stellar populations

# Stelib + Chabrier IMF
ised_file_ch = sorted(glob.glob(dir_bc03+"Stelib_Atlas/Chabrier_IMF/bc2003_*_m32_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Chabrier_IMF/bc2003_*_m42_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Chabrier_IMF/bc2003_*_m52_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Chabrier_IMF/bc2003_*_m62_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Chabrier_IMF/bc2003_*_m72_*.ised_ASCII"))

# Stelib + Chabrier IMF
ised_file_kr = sorted(glob.glob(dir_bc03+"Stelib_Atlas/Kroupa_IMF/bc2003_*_m32_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Kroupa_IMF/bc2003_*_m42_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Kroupa_IMF/bc2003_*_m52_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Kroupa_IMF/bc2003_*_m62_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Kroupa_IMF/bc2003_*_m72_*.ised_ASCII"))

# Stelib + Salpeter IMF
ised_file_sa = sorted(glob.glob(dir_bc03+"Stelib_Atlas/Salpeter_IMF/bc2003_*_m32_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Salpeter_IMF/bc2003_*_m42_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Salpeter_IMF/bc2003_*_m52_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Salpeter_IMF/bc2003_*_m62_*.ised_ASCII")) + \
               sorted(glob.glob(dir_bc03+"Stelib_Atlas/Salpeter_IMF/bc2003_*_m72_*.ised_ASCII"))

# ised_file_2 = sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m32_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m42_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m52_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m62_*.ised_ASCII")) + \
#               sorted(glob.glob(dir_bc03+"*_Atlas/*_IMF/bc2003_*_m72_*.ised_ASCII"))

# age_bin1 = [0.15e+9, 0.25e+9, 0.40e+9, 0.70e+9, 1.25e+9,
#             2.00e+9, 3.25e+9, 5.00e+9, 8.50e+9, 1.40e+10]
age_bin2 = [0.03e+9, 0.05e+9, 0.08e+9, 0.15e+9, 0.25e+9,
            0.40e+9, 0.60e+9, 1.00e+9, 1.75e+9, 3.00e+9,
            5.00e+9, 8.50e+9, 1.35e+10]

# magfile_1 = ["BaSeL.mag", "Miles.mag", "Stelib.mag"]
# for i in range(len(magfile_1)):
# 	f = open(str(Path(dir_output[0]) / magfile_1[i]), "w")
# 	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
# 	f.close()
# for i in range(len(ised_file_1)):
# 	magfile = magfile_1[i % 3]
# 	ised_to_fits(ised_file_1[i], dir_output[0], age_bin=age_bin1,
# 		         wav_start=1000., wav_end=50000., wav_intv=1.0,
# 		         magfile=str(Path(dir_output[0]) / magfile))

###
magfile = ["Stelib2_Ch.mag", "Stelib2_Kr.mag", "Stelib2_Sa.mag"]
ised_files = [ised_file_ch, ised_file_kr, ised_file_sa]
for j in range(len(ised_files)):
	f = open(str(Path(dir_output[1]) / magfile[j]), "w")
	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
	f.close()
	for i in range(len(ised_files[j])):
		# print(ised_files[j][i], magfile[j])
		ised_to_fits(ised_files[j][i], dir_output[1], age_bin=age_bin2,
		             wav_start=1000., wav_end=50000., wav_intv=1.0,
		             magfile=str(Path(dir_output[1]) / magfile[j])			         )


# magfile_2 = ["BaSeL2.mag", "Miles2.mag", "Stelib2.mag"]
# for i in range(len(magfile_2)):
# 	f = open(str(Path(dir_output[1]) / magfile_2[i]), "w")
# 	f.write("model mu Z Age M(*+remn) u_SDSS g_SDSS r_SDSS i_SDSS z_SDSS \n")
# 	f.close()
# for i in range(len(ised_file_2)):
# 	magfile = magfile_2[i % 3]
# 	ised_to_fits(ised_file_2[i], dir_output[1], age_bin=age_bin2,
# 		         wav_start=1000., wav_end=50000., wav_intv=1.0,
# 		         magfile=str(Path(dir_output[1]) / magfile))

