#=======================================================================
# Makefile for Imcompact3D
#=======================================================================
FC = gfortran
OPTFC = -O3 -fdefault-real-8 -fdefault-double-8 -w -mcmodel=large
#OPTFC = -Og -g -ffpe-trap=zero,invalid -fdefault-real-8 -fdefault-double-8 -fbacktrace -Wall -Wextra -ffpe-trap=overflow -w


# Add this flag to allow memory of more than 2GB if a too fine mesh is not working: -mcmodel=large


# To save a log file
# make 2> make.log instead of make
# less make.log to view

# Or use gdb ./incompact and then type: run
# For debugging 




SRC = module_param.f90 incompact3d.f90 geom_complex.f90 stats.f90 schemes.f90 derive.f90 waves.f90 tools.f90 poisson.f90 parameters.f90 body.f90 navier.f90 slfft2d_shift.f90 slfft3d_shift.f90 fft.f90 slfft2d.f90 convdiff.f90 lift_drag.f90

OBJ =   $(SRC:.f90=.o)

all: incompact3d

incompact3d : $(OBJ)
	$(FC) $(OPTFC) -o $@ $(OBJ) $(LIBFFT)

%.o : %.f90
	$(FC) $(OPTFC) $(OPTIONS) $(INC) -c $<

.PHONY: clean
clean:
	rm -f *.o *.mod *.vtr mass.txt incompact3d
