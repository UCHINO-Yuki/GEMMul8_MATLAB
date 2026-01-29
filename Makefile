#===============
# options
#===============
# path to cuda
CUDA_PATH ?= /usr/local/cuda

# path to hip
HIP_PATH ?= /opt/rocm

# GPU Backend selection (cuda, hip, or auto)
BACKEND ?= auto

# Compute Capability (e.g., 90 for NVIDIA H200, gfx942 for AMD CDNA3, or auto)
GPU_ARCH ?= auto

#===============
# Auto-detect backend (CUDA or HIP)
#===============
ifeq ($(BACKEND),auto)
ifneq ($(shell command -v nvidia-smi 2>/dev/null),)
BACKEND := cuda
else ifneq ($(shell command -v rocminfo 2>/dev/null),)
BACKEND := hip
else
$(error Neither NVIDIA (CUDA) nor AMD (ROCm) GPU environment detected!)
endif
endif


#===============
# CUDA setup
#===============
ifeq ($(BACKEND),cuda)

# Auto-detect arch
ifeq ($(GPU_ARCH),auto)
GPU_ARCH = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
endif

# path
export PATH := $(CUDA_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(CUDA_PATH)/lib64:$(LD_LIBRARY_PATH)

# compiler & flags
COMPILER := nvcc
FLAGS := -std=c++20 -O3
LIBS := -lcublas -lcudart -lcuda -lnvidia-ml -ldl
ARCH := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

endif


#===============
# HIP setup
#===============
ifeq ($(BACKEND),hip)

# Auto-detect arch
ifeq ($(GPU_ARCH),auto)
GPU_ARCH := $(shell amd-smi static --asic --csv | grep -o 'gfx[0-9]\+' | head -n 1)
endif

# path
export PATH := $(HIP_PATH)/bin:$(PATH)
export LD_LIBRARY_PATH := $(HIP_PATH)/lib:$(LD_LIBRARY_PATH)

# compiler & flags
COMPILER := hipcc
FLAGS := -std=c++20 -O3
FLAGS += -ffp-contract=off
FLAGS += -Wno-unused-result -Wno-unused-command-line-argument
FLAGS += -DOCML_BASIC_ROUNDED_OPERATIONS
LIBS := -lamd_smi -lamdhip64 -lhipblas -ldl
ARCH := --offload-arch=$(GPU_ARCH)

endif

MAKE_FLAGS := \
CUDA_PATH=$(CUDA_PATH) \
HIP_PATH=$(HIP_PATH) \
BACKEND=$(BACKEND) \
GPU_ARCH=$(GPU_ARCH)


#===============
# static library
#===============
LIB_TARGET := lib/libgemmul8.a
HEADERS := include/gemmul8.hpp
CU_FILES := src/gemmul8.cu
CU_OBJS := $(CU_FILES:.cu=.o)

all: $(LIB_TARGET)

$(LIB_TARGET): $(CU_OBJS)
	mkdir -p lib
	ar rcs $@ $^

%.o: %.cu $(HEADERS)
	$(COMPILER) $(FLAGS) $(ARCH) -c $< -o $@

clean:
	rm -f $(CU_OBJS) $(LIB_TARGET)

