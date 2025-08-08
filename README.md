# PRAS
Polynomial Reconstruction and Sampling method

This repository outlines the PRAS approach, as well as the AEE, RORR methods 

This is a research code, designed for methodological testing, rather than an optimised practical implementation.

## Testing

The `testing' directory contains the code used to produce individual k- and p- tables for comparing to the pre-mixed values.

Run the make_k_table.py and make_p_table.py routines first, then you can mix together the opacities using the k_table_overlap_\*.py routines, which will use the various methods to mix the opacities.
the comp_k_poly_\*.py codes then read in the data and make the comparison plots.
A reference lbl calculation can be performed using k_table_overlap_lbl_ref.py

The `input.yaml' file controls the input to the testing routines.


## RT_core

This calculates the shortwave and longwave fluxes using a 1D vertical and VMR profile, read in from the T_P_VMR_input directory.
The routine will interpolate the k- or p- table to the atmospheric layers, then use AEE, RORR or PRAS (see yaml file) for each profile in the layer to mix the opacities.
Shortwave and longwave fluxes are then calculated using the RT_flux.py methods.

The plot_\*.py files can then be used to compare each methods output.

The RT_input.yaml file controls the input to the RT routines.


