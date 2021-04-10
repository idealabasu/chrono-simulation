# Chrono Simulation

## Install
Follow the official [tutorial](http://api.projectchrono.org/tutorial_table_of_content_install.html) to build the Chrono and the optional modules. Please also follow the steps of each module before building them. Following are some notes for some specific modules.  

### FSI & GPU
CUDA Toolkit needs to be installed from [here](https://developer.nvidia.com/cuda-downloads).

### Pardiso MKL
For this module, three components need to be downloaded and installed from [here](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html) before building the module.

- Intel oneAPI DPC++/C++ Compiler Runtime for Windows
- Intel oneAPI DPC++/C++ Compiler for Windows
- Intel oneAPI Math Kernel Library for Windows

Before running the program, some environment variables need to be set using the script located at `INSTALL\LOCATION\Intel\oneAPI\mkl\latest\env\vars.bat`
