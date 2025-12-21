# GEMMul8 for MATLAB

## Overview

This repository provides a MATLAB interface for [GEMMul8 v1.0.2](https://github.com/RIKEN-RCCS/GEMMul8), enabling its use on MATLAB running on Windows.

> [!NOTE]
> Legacy cuBLAS INT8 GEMM implementations may not be supported on newer-generation GPUs.  
> Installation of the latest CUDA Toolkit is therefore recommended.

## Building the Library on Windows

1. Open `Gemmul8_MATLAB/build.bat` with a text editor such as Notepad  
   (do not double-click, as this will execute the script).

   Modify the following lines according to your environment:

   ```bat
   set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
   set GPU_ARCH=89
   ```

   Target GPU architecture (`GPU_ARCH`) can be found from [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus).

2. Open the standard Windows Command Prompt.

3. In the Command Prompt, run the following command
   (replace `2022` with the installed Visual Studio version if necessary):

   ```bat
   "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
   ```

4. Change the directory to the `Gemmul8_MATLAB` folder.

   ```bat
   cd PATH_TO_Gemmul8_MATLAB
   ```

5. To build the static library of GEMMul8, execute:

   ```bat
   build.bat
   ```

6. In the MATLAB Command Window, execute:

   ```matlab
   oz2_compile
   ```

## Running Tests

1. In the MATLAB Command Window, execute:

   ```matlab
   oz2_test
   ```

This will run basic functionality tests to verify correct integration of GEMMul8 with MATLAB.

## Notations

The generated MATLAB function introduce non-negligible runtime overhead, including dynamic allocation and deallocation of working memory, as well as additional costs inherent to MATLAB execution and the MEX interface.
Consequently, their performance is inferior to that of the native GEMMul8 library.
For performance-critical workloads, direct utilization of the original GEMMul8 implementation is strongly recommended.
