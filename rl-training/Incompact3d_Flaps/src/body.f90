!*******************************************************************
!
subroutine solid_body(ux,uy,uz,epsi,ppm,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,di1,di2)
!
!*******************************************************************

USE param
USE IBM 
USE variables

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3
real(8),dimension(nx,ny,nz) :: sy4,sy5,sy6
real(8),dimension(nx,ny,nz) :: sy7,sy8,sy9,di1,di2
real(8),dimension(nxm,nym,nz) :: ppm


if ((ivirtuel.eq.1).or.(ipen.eq.1)) then
   call gradpression(ppm,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,di1,di2)
   call corgp_IBM(ux,uy,uz,sy1,sy2,sy3,1)
   call forcage_original(ux,uy,uz,epsi)
   call corgp_IBM(ux,uy,uz,sy1,sy2,sy3,2)
endif

return
end subroutine solid_body

!********************************************************************
!
subroutine corgp_IBM (ux,uy,uz,px,py,pz,nlock)
! 
!********************************************************************

USE param
USE IBM
USE variables

implicit none

integer :: ijk,nlock,nxyz
real(8),dimension(nx,ny,nz) :: ux,uy,uz,px,py,pz

nxyz=nx*ny*nz

if (nlock.eq.1) then
   if (nz.gt.1) then
      do ijk=1,nxyz
         uy(ijk,1,1)=-py(ijk,1,1)+uy(ijk,1,1) 
         uz(ijk,1,1)=-pz(ijk,1,1)+uz(ijk,1,1) 
         ux(ijk,1,1)=-px(ijk,1,1)+ux(ijk,1,1)
      enddo
   else
      do ijk=1,nxyz
         uy(ijk,1,1)=-py(ijk,1,1)+uy(ijk,1,1) 
         ux(ijk,1,1)=-px(ijk,1,1)+ux(ijk,1,1)
      enddo
   endif
endif
if (nlock.eq.2) then
   if (nz.gt.1) then
      do ijk=1,nxyz
         uy(ijk,1,1)=py(ijk,1,1)+uy(ijk,1,1) 
         uz(ijk,1,1)=pz(ijk,1,1)+uz(ijk,1,1) 
         ux(ijk,1,1)=px(ijk,1,1)+ux(ijk,1,1)
      enddo
   else
      do ijk=1,nxyz
         uy(ijk,1,1)=py(ijk,1,1)+uy(ijk,1,1) 
         ux(ijk,1,1)=px(ijk,1,1)+ux(ijk,1,1)
      enddo
   endif
endif

return
end subroutine corgp_IBM


!*******************************************************************
!
subroutine forcage_original(ux,uy,uz,epsi)
!
!*******************************************************************

USE param
USE IBM 
USE variables

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
integer :: j, i, k, np, i2
real(8) :: xm,ym,r,func,x0,x1,xa,xb,uy1,cexx2
real(8) :: d1,d2,d3,d4,aaa
real(8) :: ceyy1, ceyy2, radiu, angle, fr, theta, y1, y2
real(8) :: cexx, ceyy, omega1, y0
real(8) :: Area, Area1, Area2, Area3, Area4, Area5, TArea
real(8) :: Ax, Ay, Bx, By, Ex, Ey, Fx, Fy
real(8) :: Axr, Ayr, Bxr, Byr, Exr, Eyr, Fxr, Fyr
real(8) :: Ax2, Ay2, Bx2, By2, Ex2, Ey2, Fx2, Fy2
real(8) :: Axr2, Ayr2, Bxr2, Byr2, Exr2, Eyr2, Fxr2, Fyr2
real(8) :: x2, alfat, alfab
real(8) :: ceyy3, ceyy4, cexx4, ahmlength, ahmwidth
 
if (nz==1) then
   np=0 
   epsi=0.

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
omega1 = amax*2.0*pi*freq*cos(2.0*pi*freq*t) ! Rotational Velocity 
if (angexc.eq.0) then                        ! Exceed 0 degrees
   alfat = amax*sin(2.0*pi*freq*t)           ! Angle of top flap
else
   alfat = amax*sin(2.0*pi*freq*t)-amax      ! Angle of top flap
endif

if (flap.eq.0) then
   alfab = alfat      ! Flapping
else
   alfab = -alfat     ! Clapping
endif
! Constants for flaps
 cexx = cex + ahmlength                 ! Centre of both Flaps (x-dir)
 ceyy = ceyy3 + radiu -thick*0.5                  ! Centre of top Flap (y-dir)
 ceyy2 = ceyy4 - radiu +thick*0.5                 ! Centre of Bot Flap (y-dir)

! Rectangle Points (Top Flap)
 Ax = cexx
 Ay = ceyy + 0.5*thick
 Bx = cexx + length
 By = ceyy + 0.5*thick
 Ex = cexx + length
 Ey = ceyy - 0.5*thick
 Fx = cexx
 Fy = ceyy - 0.5*thick
 
! Rectangle Points (Bottom Flap)
 Ax2 = cexx
 Ay2 = ceyy2 + 0.5*thick
 Bx2 = cexx + length
 By2 = ceyy2 + 0.5*thick
 Ex2 = cexx + length
 Ey2 = ceyy2 - 0.5*thick
 Fx2 = cexx
 Fy2 = ceyy2 - 0.5*thick 
 
 ! Calculate Total Area of Rectangle 
 Area = (Bx-Ax)*(Ay-Fy)

! Rotate Rectangle Points (Top Flap)
   Axr=cos(pi*alfat/180.)*(Ax-cexx)-sin(pi*alfat/180.)*(Ay-ceyy) + cexx
   Ayr=sin(pi*alfat/180.)*(Ax-cexx)+cos(pi*alfat/180.)*(Ay-ceyy) + ceyy
   Bxr=cos(pi*alfat/180.)*(Bx-cexx)-sin(pi*alfat/180.)*(By-ceyy) + cexx
   Byr=sin(pi*alfat/180.)*(Bx-cexx)+cos(pi*alfat/180.)*(By-ceyy) + ceyy
   Exr=cos(pi*alfat/180.)*(Ex-cexx)-sin(pi*alfat/180.)*(Ey-ceyy) + cexx
   Eyr=sin(pi*alfat/180.)*(Ex-cexx)+cos(pi*alfat/180.)*(Ey-ceyy) + ceyy
   Fxr=cos(pi*alfat/180.)*(Fx-cexx)-sin(pi*alfat/180.)*(Fy-ceyy) + cexx
   Fyr=sin(pi*alfat/180.)*(Fx-cexx)+cos(pi*alfat/180.)*(Fy-ceyy) + ceyy

! Rotate Rectangle Points (Top Flap)
   Axr2=cos(pi*alfab/180.)*(Ax2-cexx)-sin(pi*alfab/180.)*(Ay2-ceyy2) + cexx
   Ayr2=sin(pi*alfab/180.)*(Ax2-cexx)+cos(pi*alfab/180.)*(Ay2-ceyy2) + ceyy2
   Bxr2=cos(pi*alfab/180.)*(Bx2-cexx)-sin(pi*alfab/180.)*(By2-ceyy2) + cexx
   Byr2=sin(pi*alfab/180.)*(Bx2-cexx)+cos(pi*alfab/180.)*(By2-ceyy2) + ceyy2
   Exr2=cos(pi*alfab/180.)*(Ex2-cexx)-sin(pi*alfab/180.)*(Ey2-ceyy2) + cexx
   Eyr2=sin(pi*alfab/180.)*(Ex2-cexx)+cos(pi*alfab/180.)*(Ey2-ceyy2) + ceyy2
   Fxr2=cos(pi*alfab/180.)*(Fx2-cexx)-sin(pi*alfab/180.)*(Fy2-ceyy2) + cexx
   Fyr2=sin(pi*alfab/180.)*(Fx2-cexx)+cos(pi*alfab/180.)*(Fy2-ceyy2) + ceyy2
   

do j=1,ny
   y1=yp(j)
do i=1,nx
   x1=(i-1)*dx

   if (x1.lt.cex+ahmlength) then         ! For the Ahmed Body or Square
     if (x1.lt.cex) cycle
     if (x1.gt.(cex+ahmlength)) cycle
     if (y1.gt.(cey+ahmwidth/2.0)) cycle
     if (y1.lt.(cey-ahmwidth/2.0)) cycle  
   
     if ((x1.lt.(cex+radiu)).and.(y1.gt.(cey+ahmwidth/2.0-radiu))) then ! Top Corner
        r=sqrt((x1-cexx4)*(x1-cexx4)+(y1-ceyy3)*(y1-ceyy3))
        if (r-radiu.ge.0.) cycle 
     endif
   
     if ((x1.lt.(cex+radiu)).and.(y1.lt.(cey-ahmwidth/2.0+radiu))) then ! Bot Corner
        r=sqrt((x1-cexx4)*(x1-cexx4)+(y1-ceyy4)*(y1-ceyy4))
        if (r-radiu.ge.0.) cycle 
     endif  
     
      epsi(i,j,1)=1.
      ux(i,j,1)=0.
      uy(i,j,1)=0.     
   else                                   ! For the Flaps
   
   ! Area of each triangle (Top Flap)
   Area1 = abs(x1*(Byr-Eyr) + Bxr*(Eyr-y1) + Exr*(y1-Byr))*0.5
   Area2 = abs(Axr*(Byr-y1) + Bxr*(y1-Ayr) + x1*(Ayr-Byr))*0.5
   Area3 = abs(x1*(Fyr-Eyr) + Fxr*(Eyr-y1) + Exr*(y1-Fyr))*0.5
   Area4 = abs(Axr*(Fyr-y1) + Fxr*(y1-Ayr) + x1*(Ayr-Fyr))*0.5
   TArea = Area1 + Area2 + Area3 + Area4  
   if (TArea.le.Area+0.001) then   ! Top Flap
      epsi(i,j,1)=1.
      ux(i,j,1)=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)
      uy(i,j,1)=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)
   endif    

   ! Area of each triangle (Bottom Flap)
   Area1 = abs(x1*(Byr2-Eyr2) + Bxr2*(Eyr2-y1) + Exr2*(y1-Byr2))*0.5
   Area2 = abs(Axr2*(Byr2-y1) + Bxr2*(y1-Ayr2) + x1*(Ayr2-Byr2))*0.5
   Area3 = abs(x1*(Fyr2-Eyr2) + Fxr2*(Eyr2-y1) + Exr2*(y1-Fyr2))*0.5
   Area4 = abs(Axr2*(Fyr2-y1) + Fxr2*(y1-Ayr2) + x1*(Ayr2-Fyr2))*0.5
   TArea = Area1 + Area2 + Area3 + Area4  
   if (TArea.le.Area+0.001) then   ! Bottom Flap
      epsi(i,j,1)=1.
      if (flap.eq.0) then
         ux(i,j,1) = -sin(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
         uy(i,j,1) =  cos(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
      else
         ux(i,j,1) =  sin(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
         uy(i,j,1) = -cos(pi*alfab/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)
      endif
   endif  
        
   endif
enddo
enddo
!
else
   np=0
   epsi=0.
   do k=1,nz
   do j=1,ny
   do i=1,nx
      xm=(i-1)*dx
      ym=yp(j)
      r=sqrt((xm-cex)*(xm-cex)+(ym-cey)*(ym-cey))
      if (r-ra >= 0.) cycle
      ux(i,j,k)=0.
      uy(i,j,k)=0.
      uz(i,j,k)=0.
      np=np+1
      epsi(i,j,k)=1.
   enddo
   enddo
   enddo
endif

!

return  
end subroutine forcage_original













