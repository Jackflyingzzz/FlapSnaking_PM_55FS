#include "debug.h"
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi)

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      USE param
      USE IBM
      USE variables
      USE aeroforce

      implicit none

      integer :: i,j,k,icvlf,icvrt,jcvlw,jcvup
      real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,sy10,sy11,sy12,ux,uy,uz,di,di1,di2,tx,txx,epsi
      real(8),dimension(nx,ny,nz) :: uxm1,uym1,uxm2,uym2
      real(8),dimension(nxm,nym,nzm) :: ppm
      real :: time
      real(8) :: sumx,sumy,tvcx,tvcy,spress
!
      real(8) :: cexx,ceyy,deltax,deltay,omega1,theta
      integer(16) :: x_probe, y_probe, z_probe


!     At the start
!

        if (itime.eq.idebut)then
        do k=1,nz
          do j=1,ny
             do i=1,nx
               uxm2(i,j,k)=ux(i,j,k)
               uym2(i,j,k)=uy(i,j,k)
             enddo
          enddo
        enddo
        open(95,file='aerof6.dat') 
        open(66,file='probe.dat')
        close(66)         
        return       
        else if (itime.eq.(idebut+1))then
        do k=1,nz
          do j=1,ny
            do i=1,nx
               uxm1(i,j,k)=ux(i,j,k)
               uym1(i,j,k)=uy(i,j,k)
            enddo
          enddo
        enddo
        return
        endif

!*****************************************************************
!     Definition of the Control Volume
!*****************************************************************
! To have a moving box:
! omega1=0.5
! theta=(omega1*t)
! deltax=-2.0*sin(theta)
! deltay=2.0*cos(theta)
!
 cexx=cex
 ceyy=cey

! Control Volume Points
      xld=1.0
      xud=4.0
      yld=1.0
      yud=1.0

      icvlf = nint((cexx-xld)/dx)+1
      icvrt = nint((cexx+xud)/dx)+1
      jcvlw = nint((ceyy-yld)/dy)+1
      jcvup = nint((ceyy+yud)/dy)+1  
!************************************************************* 
!        Calculation of the momentum terms
!*************************************************************
! 
      call rmomentum(ux,uy,uz,icvlf,icvrt,jcvlw,jcvup,uxm1,uxm2,&
                     uym1,uym2,sumx,sumy,tvcx,tvcy,epsi)
!
!*************************************************************      
!       Calculation of forces on the external CV surface
!*************************************************************
!
      call sforce(ux,uy,uz,icvlf,icvrt,jcvlw,jcvup,ppm,spress)
!       Calculation of the aerodynamic force components
!
!
!       Adjusting length scales reference. Multiply by the 
!       airfoil maximum thickness and divide by chord. One 
!       should observe that, in fact, calculated values   
!       of lift and drag correspond already to the coefficients,
!       because the program works with nondimensonalized 
!       variables. The point here is that the original reference
!       for lengths is the maximum thickness, and the force 
!       coefficients are obtained considering as reference 
!       the airfoil chord.

         yforce = 2.*(f2y-ymom) !Ja foi dividido por dz em rmomentum e sforce 
         xforce = 2.*(f2xx-xmom)
         sumx=2.*sumx
         sumy=2.*sumy
         tvcx=2.*tvcx
         tvcy=2.*tvcy
         f2xx=2.*f2xx
         f2y=2.*f2y
!                         
#if DEBUG
            print *,'xDrag,yLift=',xforce,yforce
            time=itime*dt
            write(95,*) xforce, yforce, time
#else
            write(95,*) xforce, yforce
#endif
       
      ! Write a probe according to a distribution defined in 'probe_layout.txt' if iteration = ifin
      if (itime.eq.ifin) then

         ! Calculate proper pressures
         call interi6(sy8,ppm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1)
         call interiy6(tx,sy8,di1,di,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nx,nym,ny,nz,1)
         tx(:,:,:)=tx(:,:,:)/dt
         txx(:,:,:)=(1.-epsi(:,:,:))*tx(:,:,:)
         !
         open (10, file='probe_layout.txt')
         open(66,file='probe.dat',status='OLD')
         do 
            read (10,*, END=10) x_probe, y_probe, z_probe
            write(66,*) txx(x_probe, y_probe, z_probe)
         end do
         10 continue

         close(10)
         close(66)
         close(95)
      end if
       
!
 1002 FORMAT(12F15.8)
!     For other instants of time
!
      do k=1,nz
        do j=1,ny
          do i=1,nx
            uxm2(i,j,k)=uxm1(i,j,k)
            uym2(i,j,k)=uym1(i,j,k)
          enddo
        enddo
      enddo    
      
      do k=1,nz
        do j=1,ny
          do i=1,nx
            uxm1(i,j,k)=ux(i,j,k)
            uym1(i,j,k)=uy(i,j,k)
          enddo
        enddo
      enddo    

      return
      end subroutine aerof2d
!
!
!***********************************************************************
      subroutine rmomentum(ux,uy,uz,icvlf,icvrt,jcvlw,jcvup,uxm1,uxm2,&
                           uym1,uym2,sumx,sumy,tvcx,tvcy,epsi)
!***********************************************************************
!
!
      USE param
      USE IBM
      USE variables
      USE aeroforce

      implicit none

      integer :: i,j,icvlf,icvrt,jcvlw,jcvup,ii
      real(8),dimension(nx,ny,nz) :: ux,uy,uz,epsi
      real(8) :: sumx,sumy,tvcx,tvcy
      real(8) :: d,r,xit,yit
      real(8), dimension(nx,ny,nz) :: uxm1,uym1
      real(8), dimension(nx,ny,nz) :: uxm2,uym2
      real(8) :: fac,dudt1,dudt2,dudt3,dudt4,duxdt,duydt
      real(8) :: fxab,fyab,fxbc,fybc,fxcd,fycd,fxda,fyda
      real(8) :: pxab,pyab,pxbc,pybc,pxcd,pycd,pxda,pyda
      real(8) :: uxm,uym,r3,x0,y0,x,y,x1,y1

      sumx=0.
      sumy=0.

      do j=jcvlw,jcvup-1
      do i=icvlf,icvrt-1

         dudt1=0.0
         dudt2=0.0
         dudt3=0.0
         dudt4=0.0

         fac   = ux(i,j,nz)-uxm1(i,j,nz)
         dudt1 = fac/dt
         fac   = ux(i+1,j,nz)-uxm1(i+1,j,nz)
         dudt2 = fac/dt
         fac   = ux(i+1,j+1,nz)-uxm1(i+1,j+1,nz)
         dudt3 = fac/dt
         fac   = ux(i,j+1,nz)-uxm1(i,j+1,nz)
         dudt4 = fac/dt
         
         duxdt = (dudt1+dudt2+dudt3+dudt4)/4.0
         
         
         fac = uy(i,j,nz)-uym1(i,j,nz)
         dudt1 = fac/dt
         fac   = uy(i+1,j,nz)-uym1(i+1,j,nz)
         dudt2 = fac/dt
         fac   = uy(i+1,j+1,nz)-uym1(i+1,j+1,nz)
         dudt3 = fac/dt
         fac   = uy(i,j+1,nz)-uym1(i,j+1,nz)
         dudt4 = fac/dt

         duydt = (dudt1+dudt2+dudt3+dudt4)/4.0

         xit=duxdt*dx*(yp(j+1)-yp(j))!dy
         sumx=sumx+xit*(1.-epsi(i,j,1))*(1.-epsi(i+1,j,1))*(1.-epsi(i+1,j+1,1))*(1.-epsi(i,j+1,1))
         yit=duydt*dx*(yp(j+1)-yp(j))!dy
         sumy=sumy+yit!*(1.-epsi(i,j,1))
!
        enddo
      enddo


!
!
!        Secondly, the surface momentum fluxes
!
         fxab=0.0
         fyab=0.0
         i=icvlf
         do j=jcvlw,jcvup-1
            uxm = (ux(i,j,nz)+ux(i,j+1,nz))/2.0
            uym = (uy(i,j,nz)+uy(i,j+1,nz))/2.0
            pxab=-uxm*uxm*(yp(j+1)-yp(j))!dy
            fxab= fxab+pxab
            pyab=-uxm*uym*(yp(j+1)-yp(j))!dy
            fyab= fyab+pyab
         enddo
         

         fxbc=0.0
         fybc=0.0
         j=jcvup
         do i=icvlf,icvrt-1
            uxm = (ux(i,j,nz)+ux(i+1,j,nz))/2.0
            uym = (uy(i,j,nz)+uy(i+1,j,nz))/2.0
            pxbc=+uxm*uym*dx
            fxbc= fxbc+pxbc
            pybc=+uym*uym*dx
            fybc= fybc+pybc
         enddo

         fxcd=0.0
         fycd=0.0
         i=icvrt
         do j=jcvlw,jcvup-1
            uxm = (ux(i,j,nz)+ux(i,j+1,nz))/2.0
            uym = (uy(i,j,nz)+uy(i,j+1,nz))/2.0
            pxcd=+uxm*uxm*(yp(j+1)-yp(j))!dydy
            fxcd= fxcd+pxcd
            pycd=+uxm*uym*(yp(j+1)-yp(j))!dydy
            fycd= fycd+pycd
         enddo

         fxda=0.0
         fyda=0.0
         j=jcvlw
         do i=icvlf,icvrt-1
            uxm = (ux(i,j,nz)+ux(i+1,j,nz))/2.0
            uym = (uy(i,j,nz)+uy(i+1,j,nz))/2.0
            pxda=-uxm*uym*dx
            fxda= fxda+pxda
            pyda=-uym*uym*dx
            fyda= fyda+pyda
         enddo
!        The components of the total momentum

         tvcx=fxab+fxbc+fxcd+fxda
         tvcy=fyab+fybc+fycd+fyda
         xmom=sumx+fxab+fxbc+fxcd+fxda
         ymom=sumy+fyab+fybc+fycd+fyda

      return
      end subroutine rmomentum

!
!***********************************************************************
      subroutine sforce(ux,uy,uz,icvlf,icvrt,jcvlw,jcvup,ppm,spress)
!***********************************************************************
!
        USE param
        USE IBM
        USE variables
        USE aeroforce

        implicit none

        real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,sy10,sy11,sy12,ux,uy,uz,di1,di2
        integer :: i,j,icvlf,icvrt,jcvlw,jcvup
        real(8) :: foxab,foyab,foxbc,foybc,foxcd,foycd,foxda,foyda
        real(8) :: dudxm,dudym,dvdxm,dvdym,rho
        real(8) :: pxab,pyab,pxbc,pybc,pxcd,pycd,pxda,pyda,pab,pcd,difp,pxp,padx
        real(8),dimension(nxm,nym,nzm) :: ppm
        real(8) :: pbc,pyp,foypr,foxpr,spress

        call derx (sy1,ux,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
        call dery (sy2,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
        call derx (sy3,uy,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
        call dery (sy4,uy,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
!       if (nrang==0) print *,'icvlf,icvrt,jcvlw,jcvup=',icvlf,icvrt,jcvlw,jcvup
!       stop
!
!     Force calculation
!
!      Along CV's entrance face AB
!
!       if (nrang==2) print *,'ppm_sforce=',ppm(180,108,1)
         i=icvlf
         foxab=0.0
         foyab=0.0
         do j=jcvlw,jcvup-1
            dudxm = (sy1(i,j,nz)+sy1(i,j+1,nz))/2.0
            dudym = (sy2(i,j,nz)+sy2(i,j+1,nz))/2.0
            dvdxm = (sy3(i,j,nz)+sy3(i,j+1,nz))/2.0
            pxab  =-2.0*xnu*dudxm*(yp(j+1)-yp(j))!dydy
            foxab = foxab+pxab
            pyab  =-xnu*(dvdxm+dudym)*(yp(j+1)-yp(j))!dydy
            foyab = foyab+pyab
         enddo

!
!        Along CV's upper face BC
!
         j=jcvup
         foxbc=0.0
         foybc=0.0
         do i=icvlf,icvrt-1
            dudym = (sy2(i,j,nz)+sy2(i+1,j,nz))/2.0
            dvdxm = (sy3(i,j,nz)+sy3(i+1,j,nz))/2.0
            dvdym = (sy4(i,j,nz)+sy4(i+1,j,nz))/2.0
            pxbc  =+xnu*(dudym+dvdxm)*dx
            foxbc = foxbc+pxbc
            pybc  =+2.0*xnu*dvdym*dx
            foybc = foybc+pybc
         enddo
!
!        Along CV's exit face CD
!
         i=icvrt
         foxcd=0.0
         foycd=0.0
         do j=jcvlw,jcvup-1
            dudxm = (sy1(i,j,nz)+sy1(i,j+1,nz))/2.0
            dudym = (sy2(i,j,nz)+sy2(i,j+1,nz))/2.0
            dvdxm = (sy3(i,j,nz)+sy3(i,j+1,nz))/2.0
            pxcd  =+2.0*xnu*dudxm*(yp(j+1)-yp(j))!dydy
            foxcd = foxcd+pxcd
            pycd  =+xnu*(dvdxm+dudym)*(yp(j+1)-yp(j))!dydy
            foycd = foycd+pycd
          enddo
!        Along CV's lower face DA
!
         j=jcvlw
         foxda=0.0
         foyda=0.0
         do i=icvlf,icvrt-1
            dudym = (sy2(i,j,nz)+sy2(i+1,j,nz))/2.0
            dvdxm = (sy3(i,j,nz)+sy3(i+1,j,nz))/2.0
            dvdym = (sy4(i,j,nz)+sy4(i+1,j,nz))/2.0
            pxda  =-xnu*(dudym+dvdxm)*dx
            foxda = foxda+pxda
            pyda  =-2.0*xnu*dvdym*dx
            foyda = foyda+pyda
         enddo

      call interi6(sy8,ppm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1)
      call interiy6(sy9,sy8,di1,di2,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nx,nym,ny,nz,1)

      !        The pressure contribution. 
!
!        Note: We shall pretend that pressure is given by 
!        a vector pp(i,j,1). In this instance, we may obtain
!        the pressure difference between two given points in
!        the field. In reality this pressure difference has
!        to be drawn from somewhere in the code.
!
!
!        The pressure force between planes AB and CD
!
!         if (nrang==1) print *,'dt=',dt
!        if (nrang==2) print *,'pp=',ppm(180,108,1)
!
!
         rho=1.0
         foxpr=0.0
         do j=jcvlw,jcvup-1
            i     = icvlf
            pab   = (sy9(i,j,nz)/dt+sy9(i,j+1,nz)/dt)/2.0
            i     = icvrt
            pcd   = (sy9(i,j,nz)/dt+sy9(i,j+1,nz)/dt)/2.0
            difp  = (pab-pcd)/rho
            pxp   = difp*(yp(j+1)-yp(j))!dydy
            foxpr = foxpr+pxp
         enddo
!        The pressure force between planes AD and BC
!
         foypr=0.0
         do i=icvlf,icvrt-1
            j     = jcvlw
            padx   = (sy9(i,j,nz)/dt+sy9(i+1,j,nz)/dt)/2.0
            j     = jcvup
            pbc   = (sy9(i,j,nz)/dt+sy9(i+1,j,nz)/dt)/2.0
            difp  = (padx-pbc)/rho
            pyp   = difp*dx
            foypr = foypr+pyp
         enddo

         f2xx=foxab+foxbc+foxcd+foxda+foxpr
         f2y=foyab+foybc+foycd+foyda+foypr
         spress=sy9(icvlf,jcvlw,nz)


      return
      end subroutine sforce

