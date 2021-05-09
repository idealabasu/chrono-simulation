# Chrono Simulation

## Windows Install
Follow the "Building Chrono" section in the [official tutorial](http://api.projectchrono.org/tutorial_table_of_content_install.html) to build the Chrono and the optional modules. Please also follow the steps of each module before building them. The develop branch is the most stable one.

Follow the "Building a project that uses Chrono" section to build this repository with the built Chrono.

### Visual Studio
Visual Studio 2017 and 2019 should both work for the develop branch.

### Eigen
No installer for Eigen. Just extract it somewhere and point to the directory in CMAKE.

### FSI & GPU
CUDA Toolkit needs to be installed from [here](https://developer.nvidia.com/cuda-downloads). CUDA Toolkit 11.3 is tested.

### Pardiso MKL
For this module, three components need to be downloaded and installed from [here](https://software.intel.com/content/www/us/en/develop/articles/oneapi-standalone-components.html) before building the module.

- Intel oneAPI DPC++/C++ Compiler Runtime for Windows
- Intel oneAPI DPC++/C++ Compiler for Windows
- Intel oneAPI Math Kernel Library for Windows

### MultiCore
Download Blaze source code [here](https://bitbucket.org/blaze-lib/blaze/src/master/) and in cmake set `BLAZE_DIR` to root directory of source code. Thrust should be already included in the CUDA Toolkit.

### OpenGL
Download GLM [here](https://github.com/g-truc/glm/releases). \
In CMake, set `GLM_INCLUDE_DIR` to root directory of the library.

Download pre-compiled binaries of GLFW [here](https://www.glfw.org/download.html). \
In CMake, set `GLFW_INCLUDE_DIR` to `LIBRARY_ROOT/include`. \
In CMake, set `GLFW_LIBRARY` to `LIBRARY_ROOT/lib-vc2019/glfw3.lib`. \
In CMake, set `GLFW_DLL` to `LIBRARY_ROOT/lib-vc2019/glfw3.dll`. \
If using VC 2017, select the corresponding folder.

Download the binaries of GLEW [here](http://glew.sourceforge.net/). \
In CMake, set `GLEW_INCLUDE_DIR` to `LIBRARY_ROOT/include`. \
In CMake, set `GLEW_LIBRARY` to `LIBRARY_ROOT/lib/Release/x64/glew32.lib`. \
In CMake, set `GLEW_DLL` to `LIBRARY_ROOT/bin/Release/x64/glew32.dll`.

## Simulation
- If the program uses MKL, some environment variables need to be set using the script located at `INSTALL\LOCATION\Intel\oneAPI\mkl\latest\env\vars.bat`.
- Adjusting fluid/solid time step and under relaxation should be enough to make the simulation converge.
- CFL number is for adaptive time stepping.   
- For FSI, the BiCGStab and GMRES linear solvers do not work correctly. Jacobi is more stable especially for problems with a large number of particles.
