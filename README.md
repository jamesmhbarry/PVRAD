# PVRAD
Software to extract irradiance and atmospheric optical properties from PV power plants, developed in the context of the MetPVNet project

## aeronetmystic
- Use AERONET data to create log-linear fit and extract Angstrom parameters
- Plot daily Angstrom fits if required

## configs
- All config files used by the different submodules are saved here

## cosmomystic
- Combine US standard atmosphere with data from COSMO weather model
- Create atmosphere input files for libRadtran radiative transfer simulation
- Find grid boxes closest to relevant PV stations

## cosmopvcal
- Extract surface data from COSMO weather model, relevant to PV modelling
- Ambient temperature and wind speed - calculate wind speed from components
- Find grid boxes closest to relevant PV stations

## cosmopvcod
- Extract cloud information from COSMO data (specifically COD)
- Find grid boxes closest to relevant PV stations

## gti2ghi_lookup_table
- Contains a NetCDF file with a LUT for transforming GTI to GHI, using a MYSTIC-based simulation

## msgseviripvcod
- Extract COD data from MSG SEVIRI data
- Find pixels closest to relevant PV station

## pvcal_invert2rad
- Main part of the software
- Calibrate PV systems
- Invert PV power onto tilted irradiance, optical depth and horizontal irradiance

## pvdataprocess
- Data process functions to pre-process raw data from measurement campaigns
- Remove duplicates, resample

