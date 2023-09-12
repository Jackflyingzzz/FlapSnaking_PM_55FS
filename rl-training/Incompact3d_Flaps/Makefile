#=======================================================================
# Makefile for Imcompact3D
#=======================================================================

#FC = mpif90
#OPTFC = -O3 -funroll-loops -ftree-vectorize -fcray-pointer -cpp
#CC = mpicc
#CFLAGS = -O3

FC = gfortran
OPTFC = -cpp -O3 -fdefault-real-8 -fdefault-double-8 -w -mcmodel=large -march=native
#OPTFC = -cpp -Og -g -ffpe-trap=zero,invalid -fdefault-real-8 -fdefault-double-8 -fbacktrace -Wall -Wextra -ffpe-trap=overflow -w -mcmodel=large


# Add this flag to allow memory of more than 2GB if a too fine mesh is not working: -mcmodel=large


# To save a log file
# make 2> make.log instead of make
# less make.log to view

# Or use gdb ./incompact and then type: run
# For debugging



ODIR = obj
MODDIR = mod
SRCDIR = src

SRC = module_param.f90 incompact3d.f90 geom_complex.f90 stats.f90 schemes.f90 derive.f90 waves.f90 tools.f90 poisson.f90 parameters.f90 body.f90 navier.f90 slfft2d_shift.f90 slfft3d_shift.f90 fft.f90 slfft2d.f90 convdiff.f90 lift_drag.f90

_OBJ = $(SRC:.f90=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

all: dirs incompact3d

dirs:
	mkdir -p $(MODDIR)
	mkdir -p $(ODIR)
	
incompact3d : $(OBJ)
	$(FC) $(OPTFC) -o $@ $(OBJ) $(LIBFFT)

$(ODIR)/%.o : $(SRCDIR)/%.f90
	$(FC) $(OPTFC) $(OPTIONS) $(INC) -c $< -o $@ -J $(MODDIR)

.PHONY: clean
clean:
	rm -f $(ODIR)/*.o $(MODDIR)/*.mod $(SRCDIR)/*.mod *.vtr mass.txt incompact3d
