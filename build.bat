@echo off

set CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set GPU_ARCH=89

set PATH=%CUDA_PATH%\bin;%PATH%

set COMPILER=nvcc
set FLAGS=-std=c++20 -O3 -Xcompiler "/MD"
set ARCH=-gencode arch=compute_%GPU_ARCH%,code=sm_%GPU_ARCH%

:: Compile
%COMPILER% %FLAGS% %ARCH% -c src\gemmul8.cu -o src\gemmul8.obj

:: Create Windows static lib
if not exist lib mkdir lib
lib /OUT:lib\gemmul8.lib src\gemmul8.obj

echo done
