#include "debug.h"
!********************************************************************
!
PROGRAM incompact3d
!
!********************************************************************
USE param
USE parfiX;USE parfiY;USE parfiZ
USE derivX;USE derivY;USE derivZ
USE aeroforce
USE variables

implicit none

real(8) :: count1,count2
real(8),dimension(220,2) :: up_p,low_p,sum_up,sum_low
real(8) :: itime_start

real(8),dimension(nx,ny,nz) :: ux,uy,uz,gx,gy,gz,fx,fy,fz,gox,goy,goz,gax,gay,gaz,uxxol,uyyol
real(8),dimension(nx,ny,nz,nphi) :: phi,phis,phiss
real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3,sy4,sy5,sy6
real(8),dimension(mx,my,mz) :: sy7,sy8,sy9,sy10,sy11,sy12
real(8),dimension(nx,ny,nz) :: di1,di2,epsi,epsik
real(8),dimension(nxm,nym,nzm) :: ppm,ppm1
real(8),dimension(nx,ny,nz) :: uxm1,uym1,uzm1,uxm2,uym2,uzm2
real(8),dimension(nx,ny,nz) ::uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz
!
real(8),dimension(mx,mz,ny/2,5) :: a,a2
real(8),dimension(mx,mz,ny,5) :: a3 
real(8),dimension(mx,mz,ny) :: d,d1,e,c
real(8),dimension(mx,mz) :: sr
real(8),dimension(mx,mz) :: a1,b1
!
real(8),dimension(nwork) :: work
real(8),dimension(100+2*(nxm+nym+nzm)) :: table
!
real(8),dimension(nxm,nym,nzm) :: epsidec
real(8),dimension(nx,ny,nz) :: px,py,pz,p_x,p_y,p_z
integer                     :: i,j,numvis,k
integer,dimension(nx,ny,nz) :: epsin
character(len=4) suffixe
!
real(8) :: theta,x,y,xmin,xmax,xmoyn,xmoyp
integer :: inumn,inump,imax,imin,jmin,jmax,kk
!
real(8),dimension(144,192) :: mat_A
real(8),dimension(192,192) :: lhs
integer,dimension(192) :: ipiv
real(8),dimension(2,64) :: coord_sources
!
real(8),dimension(nxm,ny,nzm) :: sy34
real(8),dimension(nxm,ny,nz)  :: sy37
real(8),dimension(nx,nym,nzm) :: sy38,sy39
real(8),dimension(nx,ny,nzm)  :: sy36
real(8),dimension(nx,nym,nz)  :: sy35
real(8),dimension(nx,ny,nz)   :: sy31,sy32,sy33

real(8),dimension(nx,ny)       :: err_ux, err_uy, err_u, ana_u, err_pp
!
   call parametre()
   call schemas()
   call waves ()  
   
   !new_IBM
   if(ilag.eq.1)then
   t=0.
      if(nz.eq.1)call gene_epsi_2D(epsi,t,0)
      call verif_epsi(epsi)
   else
      epsi=0.
   endif
!
   t=0.
   if ((ivirtuel.eq.1).or.(ipen.eq.1)) then
      call forcage_original(ux,uy,uz,epsi)
   endif
   call initial (ux,uy,uz,gx,gy,gz,fx,fy,fz,phi,ppm,epsi)
   do itime=idebut,ifin
      t=(itime-1)*dt
#if DEBUG
      write(*,1001) itime,t
      1001 format('Time step =',i7,', Time unit =',F9.3)
#endif
!
      do itr=1,iavance_temps
         call inflow    (ux,uy,uz,phi,epsi)
         call outflow   (ux,uy,uz,phi,epsi)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
         if (itr.eq.1) then
         ! Re-set the position
           if(ilag.eq.1)then
              if(nz.eq.1)call gene_epsi_2D(epsi,t,0)
              call verif_epsi(epsi)
           endif    
         endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         call gradpression(ppm,sy31,sy32,sy33,sy34,sy35,sy36,sy37,sy38,sy39,di1,di2)
         call corgp_IBM(ux,uy,uz,sy31,sy32,sy33,1)
         call convdiff  (ux,uy,uz,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,sy10,sy11,sy12,di1,di2)
         call corgp_IBM(ux,uy,uz,sy31,sy32,sy33,2)
         
         call intt      (ux,uy,uz,gax,gay,gaz,gox,goy,goz,gx,gy,gz,sy7,sy8,sy9,epsi)

         call solid_body(ux,uy,uz,epsi,ppm,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,di1,di2)
         call pre_correc(ux,uy,uz,epsi)
         call divergence(ppm,ux,uy,uz,sy4,sy5,sy6,di1,di2,&
                         sy7,sy8,sy9,sy10,sy11,sy12,&
                         epsi,epsidec,sy1,sy2,sy3,work,table,1)
         call poisson   (ppm,sy7,sy8,sy9,sy10,sy11,sy12,&
                         work,table,a,a3,e,c,a2,sr,a1,b1,d1,d)
         call gradpression(ppm,p_x,p_y,p_z,sy4,sy5,sy6,sy7,sy8,sy9,di1,di2)
         call corgp       (ux,uy,uz,p_x,p_y,p_z,epsi)
         call divergence  (ppm1,ux,uy,uz,sy4,sy5,sy6,di1,di2,&
                           sy7,sy8,sy9,sy10,sy11,sy12,&
                           epsi,epsidec,sy1,sy2,sy3,work,table,2)
!
      enddo
      call stats (ux,uy,uz,gx,gy,gz,ppm,phi,phiss,epsi,&
                  sy1,sy2,sy3,sy4,sy5,sy6,di1,di2,sy7,sy8,sy9,&
                  uxm1,uym1,uzm1,uxm2,uym2,uzm2,p_x,p_y,p_z,&
                  err_u,err_pp,err_ux,err_uy)
      call moyt(uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz,ux,uy,uz)
!
   enddo
!
end PROGRAM incompact3d

