!********************************************************************
!
subroutine error_calc (ux,uy,epsi,ppm,err_u,err_pp,err_ux,err_uy) 
!
!********************************************************************

USE param
USE variables

implicit none 

integer                        :: i,j,k
real(8),dimension(nx,ny,nz)    :: ux,uy,uz,epsi, vel_u, tx, txx, di, di1, sy8
real(8),dimension(nx,ny)       :: err_ux, err_uy, err_u, ana_u, pp_ana_norm, err_pp
real(8),dimension(nxm,nym,nzm) :: ppm, ppm_norm
real(8),dimension(nxm,nym)     :: err_p, err_p_dt, p_ana_norm
real(8)                        :: sum_ux, sum_uy, sum_p, sum_p_dt, area, error_ux, error_uy, error_p, error_p_dt, error_u, sum_u
real(8)                        :: sum_pp, error_pp

! P-P_inf
ppm_norm(:,:,:) =  ppm(:,:,:) - ppm(1,1,1) 
pp_ana_norm(:,:) =  pp_ana(:,:) - pp_ana(1,1)

vel_u(:,:,:) = (ux(:,:,:)**2+uy(:,:,:)**2)**0.5
ana_u(:,:) = (ux_ana(:,:)**2+uy_ana(:,:)**2)**0.5

!PRESSURE ON MAIN MESH
call interi6(sy8,ppm_norm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1)
call interiy6(tx,sy8,di1,di,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nx,nym,ny,nz,1)
tx(:,:,:)=tx(:,:,:)/dt
txx(:,:,:)=(1.-epsi(:,:,:))*tx(:,:,:)


sum_ux = 0.0
sum_uy = 0.0
sum_u  = 0.0
sum_pp = 0.0
do j=1,ny
do i=1,nx
   ! L2 Norm Error
   err_ux(i,j) = (1.0-epsi(i,j,1))* ((ux(i,j,1) - ux_ana(i,j))**2)
   err_uy(i,j) = (1.0-epsi(i,j,1))* ((uy(i,j,1) - uy_ana(i,j))**2) 
   err_u (i,j) = (1.0-epsi(i,j,1))* ((vel_u(i,j,1) - ana_u(i,j))**2)
   err_pp (i,j) = (1.0-epsi(i,j,1))* ((txx(i,j,1) - pp_ana_norm(i,j))**2)
   sum_ux = sum_ux +  err_ux(i,j)
   sum_uy = sum_uy +  err_uy(i,j)
   sum_u  = sum_u  +  err_u(i,j)
   sum_pp = sum_pp +  err_pp(i,j)
enddo
enddo

!print*, err_ux(1,:)**0.5
!print*, err_ux(nx,:)**0.5
!print*, err_uy(1,:)**0.5
!print*, err_uy(nx,:)**0.5
!print*, '------------------------ Here ----------------------------'
!print*, err_ux(:,1)**0.5
!print*, err_ux(:,ny)**0.5
!print*, err_uy(:,1)**0.5
!print*, err_uy(:,ny)**0.5
!stop


area = (dx*dy)/(xlx*yly)
error_ux = sqrt(area*sum_ux)
error_uy = sqrt(area*sum_uy)
error_pp = sqrt(area*sum_pp)
error_u = sqrt(area*sum_u)

if (itime.eq.idebut)then
   open(97,file='error.dat')
   close(97) 
endif 

open(97,file='error.dat',status='OLD')
write(97,*) t, error_u, error_ux, error_uy, error_pp

return
end subroutine error_calc
!********************************************************************
!
subroutine initial(ux,uy,uz,gx,gy,gz,fx,fy,fz,phi,ppm,epsi)
!
!********************************************************************

USE param
USE IBM
USE variables

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,uz,gx,gy,gz,fx,fy,fz,epsi
real(8),dimension(nx,ny,nz,nphi) :: phi
real(8),dimension(nxm,nym,nzm) :: ppm
real(8) :: y,r,um,r1,r2,r3,heigh,yl
integer :: k,j,i,l
integer :: l1,l2,nxyz
character*80 nfichier

nxyz=nx*ny*nz	
ux=0.;uy=0.;uz=0.			
   call random_number(ux)
   call random_number(uy)
   call random_number(uz)

if (ilit==0) then

   call ecoule(ux,uy,uz,epsi)
   do k=1,nz
   do j=1,ny
!heigh=(yly)/2.
!yl=yp(j)-heigh
!um=0.5*(u1+u2)*(1.-((yl*yl)/(heigh*heigh)))
      y=yp(j)-yly/2.
      um=exp(-0.2*y*y)
!
      do i=1,nx
       if (epsi(i,j,k).eq.1.) then
         ux(i,j,k)=0.
         uy(i,j,k)=0.
         uz(i,j,k)=0.
         gx(i,j,k)=ux(i,j,k)
         gy(i,j,k)=uy(i,j,k)
         gz(i,j,k)=uz(i,j,k)
       else         
         ux(i,j,k)=bxx1(j,k)+0.1*um*ux(i,j,k)
         uy(i,j,k)=bxy1(j,k)+0.1*um*uy(i,j,k)
         uz(i,j,k)=bxz1(j,k)
         gx(i,j,k)=ux(i,j,k)
         gy(i,j,k)=uy(i,j,k)
         gz(i,j,k)=uz(i,j,k)
        endif
      enddo
   enddo
   enddo
endif

if (ilit.eq.1) then

open(11,file='restart',form='unformatted',status='unknown')
if (iscalaire==0) then
   if (nz.gt.1) then
      read(11) ux,uy,uz,ppm,gx,gy,gz,dpdyx1,dpdyxn,dpdzx1,dpdzxn,dpdxy1,dpdxyn,dpdzy1,dpdzyn,dpdxz1,dpdxzn,dpdyz1,dpdyzn
   else
      read(11) ux,uy,ppm,gx,gy,dpdyx1,dpdyxn,dpdxy1,dpdxyn
   endif
else
   if (nz.gt.1) then
      read(11) ux,uy,uz,phi,ppm,gx,gy,gz,dpdyx1,dpdyxn,dpdzx1,dpdzxn,dpdxy1,dpdxyn,dpdzy1,dpdzyn,dpdxz1,dpdxzn,dpdyz1,dpdyzn
   else
      read(11) ux,uy,phi,ppm,gx,gy,dpdyx1,dpdyxn,dpdxy1,dpdxyn
   endif
endif
close(11)

endif

return
end subroutine initial

!********************************************************************
!
subroutine  intt (ux,uy,uz,gax,gay,gaz,gox,goy,goz,gx,gy,gz,hx,hy,hz,epsi)
! 
!********************************************************************

USE param
USE variables

implicit none

integer :: ijk,nxyz
real(8),dimension(nx,ny,nz) :: ux,uy,uz,hx,hy,hz,gx,gy,gz,gox,goy,goz,gax,gay,gaz,epsi

nxyz=nx*ny*nz
!
if (nschema.eq.1) then       ! Euler
   if (nz.gt.1) then
      do ijk=1,nxyz
         ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1) 
         uz(ijk,1,1)=gdt(itr)*hz(ijk,1,1)+uz(ijk,1,1)
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
         gz(ijk,1,1)=hz(ijk,1,1)            
      enddo
   else
      do ijk=1,nxyz
         ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1)   
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
      enddo
   endif
endif
!
if ((nschema.eq.2.and.itime.eq.1.and.ilit.eq.0).or.&
     (nschema.eq.5.and.itr.eq.1)) then     ! AB2 or RK3
   if (nz.gt.1) then
      do ijk=1,nxyz
         ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1) 
         uz(ijk,1,1)=gdt(itr)*hz(ijk,1,1)+uz(ijk,1,1)
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
         gz(ijk,1,1)=hz(ijk,1,1)            
      enddo
   else
      do ijk=1,nxyz
         ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1)   
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
      enddo
   endif
else
   if (nz.gt.1) then
      do ijk=1,nxyz
         ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)   
         uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+uz(ijk,1,1)
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
         gz(ijk,1,1)=hz(ijk,1,1)            
      enddo
   else
      do ijk=1,nxyz
!      if (epsi(ijk,1,1).eq.1.) then 
!         gx(ijk,1,1)=ux(ijk,1,1)
!         gy(ijk,1,1)=uy(ijk,1,1)
!      else
         ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
         uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)   
         gx(ijk,1,1)=hx(ijk,1,1)
         gy(ijk,1,1)=hy(ijk,1,1)
!      endif   
      enddo
   endif
endif

if (nschema.eq.6) then ! RK4
   if (nz.gt.1) then
      if (adt(itr)==0.) then
         do ijk=1,nxyz
            gx(ijk,1,1)=dt*hx(ijk,1,1)
            gy(ijk,1,1)=dt*hy(ijk,1,1)
            gz(ijk,1,1)=dt*hz(ijk,1,1)
         enddo
      else
         do ijk=1,nxyz
            gx(ijk,1,1)=adt(itr)*gx(ijk,1,1)+dt*hx(ijk,1,1)
            gy(ijk,1,1)=adt(itr)*gy(ijk,1,1)+dt*hy(ijk,1,1)
            gz(ijk,1,1)=adt(itr)*gz(ijk,1,1)+dt*hz(ijk,1,1)
         enddo
      endif
      do ijk=1,nxyz
         ux(ijk,1,1)=ux(ijk,1,1)+bdt(itr)*gx(ijk,1,1)
         uy(ijk,1,1)=uy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)
         uz(ijk,1,1)=uz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)
      enddo
   else
      if (adt(itr)==0.) then
         do ijk=1,nxyz
            gx(ijk,1,1)=dt*hx(ijk,1,1)
            gy(ijk,1,1)=dt*hy(ijk,1,1)
         enddo
      else
         do ijk=1,nxyz
            gx(ijk,1,1)=adt(itr)*gx(ijk,1,1)+dt*hx(ijk,1,1)
            gy(ijk,1,1)=adt(itr)*gy(ijk,1,1)+dt*hy(ijk,1,1)
         enddo
      endif
      do ijk=1,nxyz
         ux(ijk,1,1)=ux(ijk,1,1)+bdt(itr)*gx(ijk,1,1)
         uy(ijk,1,1)=uy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)
      enddo
   endif
endif
!
if (nschema.eq.3) then            ! AB3
  if (itime.eq.1.and.ilit.eq.0) then
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1) 
		     uz(ijk,1,1)=gdt(itr)*hz(ijk,1,1)+uz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1)   
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  elseif (itime.eq.2.and.ilit.eq.0) then 	   
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)   
		     uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+uz(ijk,1,1)
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     goz(ijk,1,1)=gz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)  
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  else
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+uy(ijk,1,1)   
		     uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+cdt(itr)*goz(ijk,1,1)+uz(ijk,1,1)
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     goz(ijk,1,1)=gz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+uy(ijk,1,1)  
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  endif	   
endif
!
if (nschema.eq.4) then     ! AB4
  if (itime.eq.1.and.ilit.eq.0) then
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1) 
		     uz(ijk,1,1)=gdt(itr)*hz(ijk,1,1)+uz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=gdt(itr)*hx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=gdt(itr)*hy(ijk,1,1)+uy(ijk,1,1)   
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  elseif (itime.eq.2.and.ilit.eq.0) then 	   
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)   
		     uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+uz(ijk,1,1)
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     goz(ijk,1,1)=gz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+uy(ijk,1,1)  
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  elseif (itime.eq.3.and.ilit.eq.0) then 
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+uy(ijk,1,1)   
		     uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+cdt(itr)*goz(ijk,1,1)+uz(ijk,1,1)
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     goz(ijk,1,1)=gz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+uy(ijk,1,1)  
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif
  else 
	   if (nz.gt.1) then
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ddt(itr)*gax(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+ddt(itr)*gay(ijk,1,1)+uy(ijk,1,1)   
		     uz(ijk,1,1)=adt(itr)*hz(ijk,1,1)+bdt(itr)*gz(ijk,1,1)+cdt(itr)*goz(ijk,1,1)+ddt(itr)*gaz(ijk,1,1)+uz(ijk,1,1)
		     gax(ijk,1,1)=gox(ijk,1,1)
		     gay(ijk,1,1)=goy(ijk,1,1)
		     gaz(ijk,1,1)=goz(ijk,1,1)		     
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     goz(ijk,1,1)=gz(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		     gz(ijk,1,1)=hz(ijk,1,1)            
		  enddo
	   else
		  do ijk=1,nxyz
		     ux(ijk,1,1)=adt(itr)*hx(ijk,1,1)+bdt(itr)*gx(ijk,1,1)+cdt(itr)*gox(ijk,1,1)+ddt(itr)*gax(ijk,1,1)+ux(ijk,1,1)
		     uy(ijk,1,1)=adt(itr)*hy(ijk,1,1)+bdt(itr)*gy(ijk,1,1)+cdt(itr)*goy(ijk,1,1)+ddt(itr)*gay(ijk,1,1)+uy(ijk,1,1)  
		     gax(ijk,1,1)=gox(ijk,1,1)
		     gay(ijk,1,1)=goy(ijk,1,1)
		     gox(ijk,1,1)=gx(ijk,1,1)
		     goy(ijk,1,1)=gy(ijk,1,1)
		     gx(ijk,1,1)=hx(ijk,1,1)
		     gy(ijk,1,1)=hy(ijk,1,1)
		  enddo
	   endif	   
  endif	   
endif
!
return
end subroutine intt

!********************************************************************
!
subroutine corgp (ux,uy,uz,px,py,pz,epsi)
! 
!********************************************************************

USE param
USE variables

implicit none

integer :: ijk,nxyz,i,j,k
real(8),dimension(nx,ny,nz) :: ux,uy,uz,px,py,pz,epsi
real(8) :: can,ut3,ut


nxyz=nx*ny*nz

if (nz.gt.1) then
   do ijk=1,nxyz
      uy(ijk,1,1)=-py(ijk,1,1)+uy(ijk,1,1) 
      uz(ijk,1,1)=-pz(ijk,1,1)+uz(ijk,1,1) 
      ux(ijk,1,1)=-px(ijk,1,1)+ux(ijk,1,1)
   enddo
else
   do ijk=1,nxyz
!   if ((epsi(ijk,1,1).eq.0.).and.(epsi(ijk-1,1,1).eq.1.)) then
!!      uy(ijk,1,1)=-py(ijk+8,1,1)*(1.-epsi(ijk,1,1))+uy(ijk,1,1) 
!!      ux(ijk,1,1)=-px(ijk+8,1,1)*(1.-epsi(ijk,1,1))+ux(ijk,1,1)
!      uy(ijk,1,1)=uy(ijk,1,1) 
!      ux(ijk,1,1)=ux(ijk,1,1)
!   else
      uy(ijk,1,1)=-py(ijk,1,1)+uy(ijk,1,1) 
      ux(ijk,1,1)=-px(ijk,1,1)+ux(ijk,1,1)
!   endif   
   enddo
endif

if (iecoule.eq.5) then
   ut3=0.
   do k=1,nz
   do i=1,nx
      ut=0.
      do j=1,ny-1
         ut=ut+(yp(j+1)-yp(j))*(ux(i,j+1,k)-0.5*(ux(i,j+1,k)-ux(i,j,k)))
      enddo
      ut=ut/yly
      ut3=ut3+ut
   enddo
   enddo
   ut3=ut3/nz/nx

   can=-(2./3.-ut3) ! gradient de pression constant

   do k=1,nz
   do i=1,nx
   do j=2,ny-1
      ux(i,j,k)=-can+ux(i,j,k)
   enddo
   enddo
   enddo
endif

return
end subroutine corgp

!*********************************************************
!
subroutine inflow (ux,uy,uz,phi,epsi)
!  
!*********************************************************

USE param
USE IBM
USE variables

implicit none

integer  :: k,j,idum
real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
real(8) :: r1,r2,r3,y,um,rand6
real(8),external :: rand2 
real(8),dimension(nx,ny,nz,nphi) :: phi


idum=-67

call ecoule(ux,uy,uz,epsi)


if (ientree.eq.1) then  
   if (nz.gt.1) then
      do k=1,nz
      do j=1,ny
         r1=rand2(idum)-0.5
         r2=rand2(idum)-0.5
         r3=rand2(idum)-0.5
         bxx1(j,k)=bxx1(j,k)+r1*bruit
         bxy1(j,k)=bxy1(j,k)+r2*bruit
         bxz1(j,k)=bxz1(j,k)+r3*bruit
      enddo
      enddo
   else
      do j=1,ny
      y=yp(j)-yly/2.
      um=exp(-0.2*y*y)
      call random_number(rand6)
      if (epsi(1,j,1).eq.1.) then
         bxx1(j,1)=bxx1(j,1)
         bxy1(j,1)=bxy1(j,1)
         if(iscalaire.eq.1)then
            phi(1,j,1,1)=1.
         endif
      else 
   
         r1=rand2(idum)-0.5
         r2=rand2(idum)-0.5
         bxx1(j,1)=bxx1(j,1)!+r1*bruit+0.01*rand6*um
         bxy1(j,1)=bxy1(j,1)!+r2*bruit+0.01*rand6*um
         if(iscalaire.eq.1)then
            phi(1,j,1,1)=1.
         endif
       endif
      enddo
   endif
endif

return
end subroutine inflow 

!*********************************************************
!
subroutine outflow (ux,uy,uz,phi,epsi)
!
!*********************************************************

USE param
USE variables

implicit none

integer :: j,k,i
real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
real(8) :: udx,udy,udz,uddx,uddy,uddz,uxmax,uxmin,vphase,cx,coef,uxmax1,uxmin1
real(8),dimension(nx,ny,nz,nphi) :: phi


byxn(:,:)=0.;byyn(:,:)=0.;byzn(:,:)=0.;bzxn(:,:)=0.;bzyn(:,:)=0.;bzzn(:,:)=0.;
udx=1./dx
udy=1./dy
udz=1./dz
uddx=0.5/dx
uddy=0.5/dy
uddz=0.5/dz
 cx=0.5*(u1+u2)*gdt(itr)*udx    

uxmax=-1.e10
uxmin=1.e10
do k=1,nz
do j=1,ny
   if (ux(nx-1,j,k).gt.uxmax) uxmax=ux(nx-1,j,k)
   if (ux(nx-1,j,k).lt.uxmin) uxmin=ux(nx-1,j,k)
enddo
enddo
vphase=0.5*(uxmax+uxmin)
!cx=vphase*gdt(itr)*udx
coef=gdt(itr)*dx/(dx+cx*gdt(itr))

if (iecoule.ne.9) then
   if (nz.gt.1) then
      do k=1,nz
      do j=1,ny
         bxxn(j,k)=ux(nx,j,k)-cx*(ux(nx,j,k)-ux(nx-1,j,k))
         bxyn(j,k)=uy(nx,j,k)-cx*(uy(nx,j,k)-uy(nx-1,j,k))
         bxzn(j,k)=uz(nx,j,k)-cx*(uz(nx,j,k)-uz(nx-1,j,k))
      enddo
      enddo
   else
      do j=1,ny
      if (epsi(nx,j,1).eq.0.) then
         bxxn(j,1)=ux(nx,j,1)-cx*(ux(nx,j,1)-ux(nx-1,j,1))
         bxyn(j,1)=uy(nx,j,1)-cx*(uy(nx,j,1)-uy(nx-1,j,1))
!         bxxn(j,1)=uxout(j)
!         bxyn(j,1)=uyout(j)
         	phi(nx,j,1,1)=phi(nx,j,1,1)-cx*(phi(nx,j,1,1)-phi(nx-1,j,1,1))
      else
         bxxn(j,1)=0.
         bxyn(j,1)=0.
         	phi(nx,j,1,1)=phi(nx,j,1,1)-cx*(phi(nx,j,1,1)-phi(nx-1,j,1,1))
      endif     
      enddo
   endif
else
   do k=2,nz-1
   do i=2,nx-1
      byxn(i,k)=ux(i,ny-1,k)-(uy(i+1,ny-1,k)-uy(i-1,ny-1,k))*udx*0.5
      byzn(i,k)=uz(i,ny-1,k)-(uz(i-1,ny-1,k)-uz(i-1,ny-1,k))*udz*0.5
      byyn(i,k)=0.
   enddo
   enddo
endif


return
end subroutine outflow 

!********************************************************************
!
subroutine gradpression(ppm,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,di1,di2)
!
!********************************************************************

USE param 
USE variables

implicit none

real(8),dimension(mx,my,mz) :: di2,di1
real(8),dimension(nxm,ny,nzm) :: sy4
real(8),dimension(nxm,ny,nz) :: sy7
real(8),dimension(nx,nym,nzm) :: sy8,sy9
real(8),dimension(nx,ny,nzm) :: sy6
real(8),dimension(nx,nym,nz) :: sy5
real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3
real(8),dimension(nxm,nym,nzm) :: ppm
integer :: i,j,k,mxyz,ijk
real(8):: x,y,z

 call interiy6(sy4,ppm,di1,di2,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nxm,nym,ny,nz,1)
 call interi6(sy5,ppm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1)
if (nz.gt.1) then
   call interi6(sy9,ppm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1) 
   call interiy6(sy6,sy9,di1,di2,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nx,nym,ny,nz,1)
   call interiz6(sy7,sy4,di1,sz,cifip6z,cisip6z,ciwip6z,cifz6,cisz6,ciwz6,nxm,ny,nzm,nz,1)
   call interiz6(sy8,sy5,di1,sz,cifip6z,cisip6z,ciwip6z,cifz6,cisz6,ciwz6,nx,nym,nzm,nz,1)
   call deciz6(sy3,sy6,di1,sz,cfip6z,csip6z,cwip6z,cfz6,csz6,cwz6,nx,ny,nzm,nz,1)
endif
if (nz.gt.1) then
   call deci6(sy1,sy7,di1,sx,cfip6,csip6,cwip6,cfx6,csx6,cwx6,nxm,nx,ny,nz,1)
   call deciy6(sy2,sy8,di1,di2,sy,cfip6y,csip6y,cwip6y,cfy6,csy6,cwy6,ppy,nx,nym,ny,nz,1)
else
   call deci6(sy1,sy4,di1,sx,cfip6,csip6,cwip6,cfx6,csx6,cwx6,nxm,nx,ny,nz,1)
   call deciy6(sy2,sy5,di1,di2,sy,cfip6y,csip6y,cwip6y,cfy6,csy6,cwy6,ppy,nx,nym,ny,nz,1)
endif

!      byzn(i,k)=0.

do j=1,ny
do i=1,nx
   dpdxz1(i,j)=sy1(i,j,1)/gdt(itr)
   dpdyz1(i,j)=sy2(i,j,1)/gdt(itr)
   dpdxzn(i,j)=sy1(i,j,nz)/gdt(itr)
   dpdyzn(i,j)=sy2(i,j,nz)/gdt(itr)
enddo
enddo

do k=1,nz
do i=1,nx
   dpdxy1(i,k)=sy1(i,1,k)/gdt(itr)
   dpdxyn(i,k)=sy1(i,ny,k)/gdt(itr)
   dpdzy1(i,k)=sy3(i,1,k)/gdt(itr)
   dpdzyn(i,k)=sy3(i,ny,k)/gdt(itr)
enddo
enddo
do k=1,nz
do j=1,ny    
   dpdyxn(j,k)=sy2(nx,j,k)/gdt(itr)
   dpdyx1(j,k)=sy2(1,j,k)/gdt(itr)
   dpdzx1(j,k)=sy3(1,j,k)/gdt(itr)
   dpdzxn(j,k)=sy3(nx,j,k)/gdt(itr)
enddo
enddo

return
end subroutine gradpression

!********************************************************************
!
subroutine divergence (ppm,ux,uy,uz,sy4,sy5,sy6,di1,di2,&
                       sy7,sy8,sy9,sy10,sy11,sy12,&
                       epsi,epsidec,sy1,sy2,sy3,work,table,nlock)    
! 
!********************************************************************

USE param
USE IBM
USE variables
   
implicit none

   integer :: i,j,k,ijk,nxyz,l,imax,kmax
   integer :: nlock
   real(8),dimension(nwork) :: work,table
   real(8),dimension(nx,ny,nz) :: ux,uy,uz,di1,di2,epsi,sy1,sy2,sy3
   real(8),dimension(nx,nym,nzm) :: sy10
   real(8),dimension(nxm,ny,nzm) ::sy11
   real(8),dimension(nx,nym,nz) :: sy7
   real(8),dimension(nxm,ny,nz) :: sy8,sy12
   real(8),dimension(nxm,nym,nz) :: sy9
   real(8),dimension(nxm,nym,nzm) :: ppm,epsidec,sy4,sy5,sy6
   real(8) :: tmax,tmoy,tmax1,tmoy1,xl2,x,y,z
   real(8),dimension(mx,my) :: tb11
   real(8) :: omega1, alfa2, x1, y1, cexx, ceyy, ceyy2, cexx4, radiu
   real(8) :: ceyy3, ceyy4, ahmlength, ahmwidth, alfab, alfat
   
   real(8) :: ahmlength2, radiu2, cex_2, cexx_1, cexx_4, ceyy_1, ceyy_2, ceyy_3, ceyy_4
   real(8) :: omega2, alfat2, alfab2   

   nxyz=nx*ny*nz

! Fundamental Constants for Ahmed
 ahmlength = aspec
 ahmwidth = 1.0

! Constants for Ahmed Body
 if (geoah.eq.1) radiu = ahmlength/10.4396   ! Smooth Corner Radius 
 if (geoah.eq.0) radiu = 0.                  ! Square Cylinder (Sharp Corner) 
 cexx4 = cex + radiu                         ! Right face of Ahmed Body
 ceyy3 = cey + ahmwidth/2.0 - radiu          ! Top Face of Ahmed Body
 ceyy4 = cey - ahmwidth/2.0 + radiu          ! Bottom Face of Ahmed Body

! Constants for Flaps   
omega1 = (amax*pi/180.)*2.0*pi*freq*cos(2.0*pi*freq*t) ! Rotational Velocity  
if (angexc.eq.0) then                        ! Exceed 0 degrees
   alfat = amax*sin(2.0*pi*freq*t)           ! Angle of top flap
else
   alfat = amax*sin(2.0*pi*freq*t)-amax  ! Angle of top flap
endif

if (flap.eq.0) then
   alfab = alfat      ! Flapping
else
   alfab = -alfat     ! Clapping
endif
! Constants for flaps
 cexx = cex + ahmlength                 ! Centre of both Flaps (x-dir)
 ceyy = ceyy3 + radiu  -thick*0.5                 ! Centre of top Flap (y-dir)
 ceyy2 = ceyy4 - radiu +thick*0.5                 ! Centre of Bot Flap (y-dir)



! Second Body Constants
	! Fundamental Constants for Ahmed
	 ahmlength2 = aspec2

	! Constants for Ahmed Body
	 if (geoah.eq.1) radiu2 = ahmlength2/10.4396   ! Smooth Corner Radius 
	 if (geoah.eq.0) radiu2 = 0.                  ! Square Cylinder (Sharp Corner) 
	 
	 ! Displace body by certain distance (horizontal direction)
	 cex_2 = cex + ahmlength + distance
	 
	 cexx_4 = cex_2 + radiu2                         ! Right face of Ahmed Body
	 ceyy_3 = cey + ahmwidth/2.0 - radiu2          ! Top Face of Ahmed Body
	 ceyy_4 = cey - ahmwidth/2.0 + radiu2          ! Bottom Face of Ahmed Body

	! Constants for Flaps   
	omega2 = (amax2*pi/180.)*2.0*pi*freq2*cos(2.0*pi*freq2*t) ! Rotational Velocity
	if (angexc.eq.0) then                        ! Exceed 0 degrees
	   alfat2 = amax2*sin(2.0*pi*freq2*t)           ! Angle of top flap
	else
	   alfat2 = amax2*sin(2.0*pi*freq2*t)-amax2      ! Angle of top flap
	endif

	if (flap.eq.0) then
	   alfab2 = alfat2      ! Flapping
	else
	   alfab2 = -alfat2     ! Clapping
	endif
	! Constants for flaps
	 cexx_1 = cex_2 + ahmlength2                 ! Centre of both Flaps (x-dir)
	 ceyy_1 = ceyy_3 + radiu2 -thick*0.5                  ! Centre of top Flap (y-dir)
	 ceyy_2 = ceyy_4 - radiu2 +thick*0.5                 ! Centre of Bot Flap (y-dir)

 
do k=1,nz
do j=1,ny
y1=yp(j)
do i=1,nx
    if (epsi(i,j,k).eq.0.) then
         sy1(i,j,k)=ux(i,j,k)
         sy2(i,j,k)=uy(i,j,k)        
    else
         x1=(i-1)*dx
         if (x1.lt.cex_2-0.05) then ! Body 1
		 if (x1.lt.cex+ahmlength) then
		    sy1(i,j,k)=0.0
		    sy2(i,j,k)=0.0   
		 else
		   if(y1.gt.cey) then
		      sy1(i,j,k)=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)
		      sy2(i,j,k)=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)
		   else
		      if (flap.eq.0) then
		         sy1(i,j,k)=-sin(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
		         sy2(i,j,k)=cos(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
		      else
		         sy1(i,j,k)=sin(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
		         sy2(i,j,k)=-cos(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
		      endif
		   endif
		 endif   
          else               ! Body 2
		 if (x1.lt.cex_2+ahmlength2) then
		    sy1(i,j,k)=0.0
		    sy2(i,j,k)=0.0   
		 else
		   if(y1.gt.cey) then
		      sy1(i,j,k)=-sin(pi*alfat2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_1)**2)
		      sy2(i,j,k)=cos(pi*alfat2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_1)**2)
		   else
		      if (flap.eq.0) then
		         sy1(i,j,k)=-sin(pi*alfab2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_2)**2)
		         sy2(i,j,k)=cos(pi*alfab2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_2)**2)
		      else
		         sy1(i,j,k)=sin(pi*alfab2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_2)**2)
		         sy2(i,j,k)=-cos(pi*alfab2/180.)*omega2*sqrt((x1- cexx_1)**2 + (y1-ceyy_2)**2)
		      endif
		   endif
		 endif             
          endif		 
    endif
enddo
enddo
enddo

!call paraview_3d(sy1,sy2,epsi)
!   if (nz.gt.1) then
!      do i=1,nxyz
!         sy1(i,1,1)=(1.-epsi(i,1,1))*ux(i,1,1)
!         sy2(i,1,1)=(1.-epsi(i,1,1))*uy(i,1,1)
!         sy3(i,1,1)=(1.-epsi(i,1,1))*uz(i,1,1)
!      enddo
!   else
!      do i=1,nxyz
!         sy1(i,1,1)=(1.-epsi(i,1,1))*ux(i,1,1)
!         sy2(i,1,1)=(1.-epsi(i,1,1))*uy(i,1,1)
!      enddo
!   endif
   call intery6(sy7,sy1,di1,di2,sy,cifyp6,cisyp6,ciwyp6,nx,ny,nym,nz,1)
   call inter6(sy8,sy2,di1,sx,cifxp6,cisxp6,ciwxp6,nx,nxm,ny,nz,1)
   if (nz.gt.1) then
      call inter6(sy12,sy3,di1,sx,cifxp6,cisxp6,ciwxp6,nx,nxm,ny,nz,1)
      call intery6(sy9,sy12,di1,di2,sy,cifyp6,cisyp6,ciwyp6,nxm,ny,nym,nz,1)
      call interz6(sy10,sy7,di1,sz,cifzp6,ciszp6,ciwzp6,nx,nym,nz,nzm,1)
      call interz6(sy11,sy8,di1,sz,cifzp6,ciszp6,ciwzp6,nxm,ny,nz,nzm,1)
      call decz6(sy6,sy9,di1,sz,cfz6,csz6,cwz6,nxm,nym,nz,nzm,0)
   endif
   if (nz.gt.1) then
      call decx6(sy4,sy10,di1,sx,cfx6,csx6,cwx6,nx,nxm,nym,nz,0)
      call decy6(sy5,sy11,di1,di2,sy,cfy6,csy6,cwy6,ppyi,nxm,ny,nym,nz,0)
   else
      call decx6(sy4,sy7,di1,sx,cfx6,csx6,cwx6,nx,nxm,nym,nz,0)
      call decy6(sy5,sy8,di1,di2,sy,cfy6,csy6,cwy6,ppyi,nxm,ny,nym,nz,0) 
   endif

   if (nz.gt.1) then
      nxyz=nxm*nym*nzm
      do ijk=1,nxyz
         ppm(ijk,1,1)=sy4(ijk,1,1)+sy5(ijk,1,1)+sy6(ijk,1,1)
         if(nlock.eq.1)then
            ppm(ijk,1,1)=ppm(ijk,1,1)+epsidec(ijk,1,1)
         endif
      enddo
      if (nlock==2) then
         xl2=ppm(1,1,1)
         do k=1,nzm
         do j=1,nym
         do i=1,nxm
            ppm(i,j,k)=ppm(i,j,k)-xl2
         enddo
         enddo
         enddo
      endif
   else
      do j=1,nym
      do i=1,nxm
         ppm(i,j,1)=sy4(i,j,1)+sy5(i,j,1)
      enddo
      enddo
      if (nlock==2) then
         xl2=ppm(1,1,1)
         do j=1,nym
         do i=1,nxm
               ppm(i,j,1)=ppm(i,j,1)-xl2
         enddo
         enddo
      endif
!      do j=1,nym
!      do i=1,nxm
!         ppm(i,j,1)=ppm(i,j,1)*(1.-epsidec(i,j,1))
!         ppm(i,j,1)=ppm(i,j,1)*epsidec(i,j,1)
!      enddo
!      enddo
   endif

   tmax=-1609.
   tmoy=0.
   do k=1,nzm
   do j=1,nym
   do i=1,nxm
      if (ppm(i,j,k).gt.tmax) tmax=ppm(i,j,k)     
      tmoy=tmoy+abs(ppm(i,j,k))
   enddo
   enddo
   enddo
   tmoy=tmoy/nxm/nym/nzm
   if (nlock==2) then
      write(*,1001) tmax,tmoy
 1001 format('Div(u) final Max, Moy= ',2E10.3)
   else
      write(*,1002) tmax,tmoy
 1002 format('Div(u*) Max, Moy     = ',2E10.3)
   endif

return
end subroutine divergence

!**********************************************************************
!
subroutine ecoule (ux,uy,uz,epsi)
!
!**********************************************************************

USE param
USE IBM
USE variables

implicit none

integer  :: i,j,k,jj1,jj2 
real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
real(8)  :: x,y,ym,heigh,yl
real(8) :: r1,r2,r3,r
real(8) :: uh,ud,um,xxk1,xxk2,xv,bruit1

bxx1=0.;bxy1=0.;bxz1=0.
byx1=0.;byy1=0.;byz1=0.
bzx1=0.;bzy1=0.;bzz1=0. 

!IECOULE=1 --> Constant flow field
!IECOULE=2 --> Mixing layer hyperbolic tangent profile
!IECOULE=3 --> Wake flow
!IECOULE=4 --> Mixing layer with splitter plate
!IECOULE=5 --> Channel flow
!IECOULE=6 --> Taylor Green vortices
!IECOULE=7 --> Cavity flow
!IECOULE=8 --> Flat plate Boundary layer
!IECOULE=9 --> Tank 

if (iecoule.eq.1) then
!   um=0.5*(u1+u2)
   do k=1,nz
   do j=1,ny
!      bxx1(j,k)=uxin(j)        To impose alanytical BC
!      bxy1(j,k)=uyin(j)        To impose alanytical BC
heigh=(yly)/2.
yl=yp(j)-heigh
um=0.5*(u1+u2)*(1.-((yl*yl)/(heigh*heigh)))

!
	if (epsi(1,j,k).eq.0.) then
	um=0.5*(u1+u2)
		  bxx1(j,k)=um
		  bxy1(j,k)=0.
		  bxz1(j,k)=0.
!      bxx1(j,k)=uxin(j)
!      bxy1(j,k)=uyin(j)
!      bxz1(j,k)=0.
	else
		  bxx1(j,k)=0.
		  bxy1(j,k)=0.
		  bxz1(j,k)=0.      
	endif 
   enddo
   enddo
endif


if (iecoule.eq.2) then
   uh=0.5*(u1+u2)
   ud=0.5*(u1-u2)
   do k=1,nz
   do j=1,ny
      y=yp(j)-yly/2.
      bxx1(j,k)=uh+ud*tanh(2.*y)
      bxy1(j,k)=0.
      bxz1(j,k)=0.
   enddo
   enddo
endif

if (iecoule.eq.3) then
   do k=1,nz
   do j=1,ny
      y=yp(j)-yly/2.
      bxx1(j,k)=1-exp(-log(2.)*y**2)
      bxy1(j,k)=0.
      bxz1(j,k)=0.
   enddo
   enddo
endif

if (iecoule.eq.4) then
   if (istret.ne.1) then
       print *,'ECOULEMENT IMPOSSIBLE SI ISTRET <> 1!!'
   endif
   i5=int(cex/dx)
   jj=int(cey/dy)+1
   j1=1
   do j=2,jj
      r=abs(yp(j)-cey)
      if (r.gt.ra) j1=j+1
   enddo
   j2=1
   do j=ny,jj,-1
      r=abs(yp(j)-cey)
      if (r.gt.ra) j2=j-1
   enddo

   um=0.5*(u1+u2)
   do k=1,nz
   do j=1,ny
      bxx1(j,k)=um
      bxy1(j,k)=0.
      bxz1(j,k)=0.
   enddo
   enddo
   do j=ny,jj,-1
      r=abs(yp(j)-cey)
      if (r.gt.(0.5+(11./19.))) jj2=j
   enddo
   do k=1,nz
   do j=j2,jj2
      ym=(yp(j)-yp(j2))/(yp(jj2)-yp(j2))
      ym=ym*0.772588621523 
      bxx1(j,k)=2.*ym*u1-5.*ym*ym*ym*ym*u1+6.*ym*ym*ym*ym*ym*u1-2.*ym*ym*ym*ym*ym*ym*u1
      if ((j>j2+1).and.(bxx1(j,k) < bxx1(j-1,k))) bxx1(j,k)=u1
   enddo
   do j=jj2+1,ny
      bxx1(j,k)=u1
   enddo
   do j=j1,j2
      bxx1(j,k)=0.
   enddo
   enddo
   do j=2,jj
      r=abs(yp(j)-cey)
      if (r.gt.(ra+(9.5/19.))) jj1=j+1
   enddo
   do k=1,nz
   do j=j1,jj1,-1
      ym=(yp(j)-yp(j1))/(yp(jj1)-yp(j1))
      ym=ym*0.772588621523
      bxx1(j,k)=2.*ym*u2-5.*ym*ym*ym*ym*u2+6.*ym*ym*ym*ym*ym*u2-2.*ym*ym*ym*ym*ym*ym*u2
      if ((j<j1-1).and.(bxx1(j,k) < bxx1(j+1,k))) bxx1(j,k)=u2
   enddo
   do j=1,jj1-1
      bxx1(j,k)=u2
   enddo
   enddo
endif

if (iecoule.eq.5) then
   bruit1=0.125
   if (ilit.eq.0) then
      do k=1,nz
      do j=1,ny
      do i=1,nx
         x=(i-1)*dx-xlx/2.
         y=yp(j)-yly/2.
         if (istret.eq.1.or.istret.eq.3) then
            print *,'Not possible with this refinement parameter'
            stop
         endif
         ux(i,j,k)=1.-y*y+bruit1*r1
         uy(i,j,k)=bruit1*r2
         uz(i,j,k)=bruit1*r3
      enddo
      enddo
      enddo
   endif
endif

if (iecoule.eq.6) then
   if (nz.gt.1) then 
      print *,'ECOULEMENT UNIQUEMENT 2D'
      stop
   else
   xv=1./100.
   xxk1=twopi/xlx
   xxk2=twopi/yly
   do k=1,nz
   do j=1,ny
      y=yp(j)
      do i=1,nx
         x=(i-1)*dx
         ux(i,j,k)=sin(xxk1*x)*cos(xxk2*y)*exp(-(xxk1*xxk1+xxk2*xxk2)*xv*t)
         uy(i,j,k)=-xxk1/xxk2*sin(xxk2*y)*cos(xxk1*x)*exp(-(xxk1*xxk1+xxk2*xxk2)*xv*t)
         bxx1(j,k)=0.
         bxy1(j,k)=0.
      enddo
   enddo
   enddo
   endif
endif

if (iecoule.eq.7) then
   if (nz.gt.1) then 
      print *,'ECOULEMENT UNIQUEMENT 2D'
      stop
   else
      do k=1,nz
      do j=1,ny
         y=(j-1)*dy
         do i=1,nx
            x=(i-1)*dx
            ux(i,j,k)= 8.*(x*x*x*x-2.*x*x*x+x*x)*(4.*y*y*y-2.*y)
            uy(i,j,k)=-8.*(4.*x*x*x-6.*x*x+2.*x)*(y*y*y*y-y*y)
            bxx1(j,k)=0.
            bxy1(j,k)=0.
         enddo
      enddo
      enddo
   endif
endif

if (iecoule.eq.8) then
   do k=1,nz
   do j=1,ny
      ym=yp(j)*2.
      bxx1(j,k)=2.*ym*u1-5.*ym*ym*ym*ym*u1+6.*ym*ym*ym*ym*ym*u1-2.*ym*ym*ym*ym*ym*ym*u1
      if (ym.gt.1.) bxx1(j,k)=u1
      bxy1(j,k)=0.
      bxz1(j,k)=0.
      byx1(j,k)=1.
      byy1(j,k)=0.
      byz1(j,k)=0.
   enddo
   enddo
endif

if (iecoule.eq.9) then
   bxx1=0.
   bxy1=0.
   bxz1=0.
   bzx1=0.
   bzy1=0.
   bzz1=0.
endif

return
end subroutine ecoule

!********************************************************************
!
subroutine cavite (uy,nx,ny,nz)
!
!********************************************************************

USE param

implicit none

integer :: nx,ny,nz
real(8),dimension(nx,ny,nz) :: cav,uy
integer :: nxyz,j,i,k
real(8) :: x,y,xy1,xy2,xy3,xy4,xy5,xy6,xy7,xy8

nxyz=nx*ny*nz

do k=1,nz
do j=1,ny
   y=(j-1.)*dy
   do i=1,nx
      x=(i-1.)*dx
      xy1=x*x*x*x-2.*x*x*x+x*x
      xy2=y*y*y*y-y*y
      xy3=0.2*x*x*x*x*x-0.5*x*x*x*x+(1./3.)*x*x*x
      xy4=-4.*x*x*x*x*x*x+12.*x*x*x*x*x-14.*x*x*x*x+8.*x*x*x-2.*x*x
      xy5=0.5*xy1*xy1
      xy6=-24.*y*y*y*y*y+8.*y*y*y-4.*y
      xy7=4.*x*x*x-6.*x*x+2.*x
      xy8=4.*y*y*y-2.*y
      cav(i,j,k)=-8.*xnu*(24.*xy3+2.*xy7*(12.*y*y-2.)+xy2*(24.*x-12.))&
           -64.*(xy5*xy6-xy2*xy8*xy4)
   enddo
enddo
enddo
do i=1,nxyz
   uy(i,1,1)=-cav(i,1,1)+uy(i,1,1)
enddo

return
end subroutine cavite

