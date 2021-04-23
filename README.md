# Chrono Simulation

## Install
Follow the official [tutorial](http://api.projectchrono.org/tutorial_table_of_content_install.html) to build the Chrono and the optional modules. Please also follow the steps of each module before building them. The develop branch is the most stable one. Following are some notes for some specific modules.

### Visual Studio
Visual Studio 2017 and 2019 should both work for the develop branch.

### FSI & GPU
CUDA Toolkit needs to be installed from [here](https://developer.nvidia.com/cuda-downloads). CUDA Toolkit 11.3 is tested.

### Pardiso MKL
For this module, three components need to be downloaded and installed from [here](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html) before building the module.

- Intel oneAPI DPC++/C++ Compiler Runtime for Windows
- Intel oneAPI DPC++/C++ Compiler for Windows
- Intel oneAPI Math Kernel Library for Windows

Before running the program, some environment variables need to be set using the script located at `INSTALL\LOCATION\Intel\oneAPI\mkl\latest\env\vars.bat`

### Note
- For FSI, the BiCGStab and GMRES linear solvers do not work correctly. Jacobi is more stable especially for problems with a large number of particles.
