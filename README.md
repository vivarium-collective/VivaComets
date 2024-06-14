
# Vivarium-COMETS

## Description
Vivarium-COMETS is a multiscale modeling project that leverages the capabilities of the Vivarium package to implement the 
COMETS methodology. This project aims to provide a comprehensive framework for simulating and analyzing complex 
microbial systems in spatial environments, enhancing our understanding of dynamic interactions within various 
environments.

## Features
- **Implementation of the COMETS Methodology Using Vivarium:** Utilizes the robust features of Vivarium to bring the COMETS methodology to life.
- **Analysis and Visualization Capabilities:** Includes advanced capabilities for analyzing system interactions, with a focus on Flux Balance Analysis (FBA). This allows for the detailed examination and visualization of metabolic fluxes within the system, facilitating a deeper understanding of the underlying biological processes.

## Jupyter Notebook Demonstration
- **[Diffusion notebook](https://github.com/vivarium-collective/VivaComets/blob/main/notebooks/diffusion.ipynb):** Demonstrates simulating and visualizing molecular concentrations in a 2D field influenced by diffusion, advection, and sinking.

- **[Spatial dFBA notebook](https://github.com/vivarium-collective/VivaComets/blob/main/notebooks/Spatial_DFBA.ipynb):** This Jupyter notebook models the dynamic metabolic behavior of microorganisms over time using dynamic Flux Balance Analysis (dFBA), it incorporates two-dimensional partial differential equations (PDEs) to simulate the diffusion and advection of substances for added spatial resolution and realism.
- **[VivaComets Notebook](https://github.com/vivarium-collective/VivaComets/blob/main/notebooks/VivaComets.ipynb):** The VivaComets notebook integrates the functionalities of two previously developed notebooks to create a comprehensive simulation environment. This combined notebook accounts for the diffusion and advection of molecules and species biomass within a 3D environment. Additionally, it performs dynamic Flux Balance Analysis (dFBA) for organisms in this spatially structured environment.


Key Features:

**Diffusion and Advection:** Models the movement and distribution of molecules and biomass through diffusion and advection processes, simulating realistic environmental interactions.

**Dynamic Flux Balance Analysis (dFBA):** Computes the growth and metabolic rates of organisms dynamically over time, considering the spatial distribution of resources and environmental conditions.

**3D Environment:** Simulates the interactions and growth of multiple species in a three-dimensional space, providing a more realistic and detailed analysis of ecological and metabolic dynamics.


## MIT License

Copyright (c) 2024 Vivarium Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

