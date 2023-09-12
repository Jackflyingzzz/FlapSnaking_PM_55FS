#include "debug.h"
!***************************************************************************
!
subroutine stats(ux,uy,uz,gx,gy,gz,ppm,phi,phiss,epsi,&
                 tx,ty,tz,fx,fy,fz,di1,di2,px,py,pz,&
                 uxm1,uym1,uzm1,uxm2,uym2,uzm2,p_x,p_y,p_z,&
                 err_u,err_pp,err_ux,err_uy)
!
!***************************************************************************
!
USE param
USE variables
USE aeroforce
!
implicit none
!
integer  :: i,j,k
real(8),dimension(nx,ny,nz) :: ux,uy,uz,gx,gy,gz,epsi,p_x,p_y,p_z,walls,angle,indi
real(8),dimension(nx,ny,nz) :: tx,ty,tz,fx,fy,fz,di1,di2,px,py,pz
real(8),dimension(nx,ny,nz) :: uxm1,uym1,uzm1,uxm2,uym2,uzm2
real(8),dimension(nx,ny,nz,nphi) :: phi,phiss
real(8),dimension(nxm,nym,nzm) :: ppm
integer :: nxyz

real(8),dimension(nx,ny)       :: err_ux, err_uy, err_u, ana_u, err_pp
!
nxyz=nx*ny*nz
call minmax(ux,nxyz,'Ux')
call minmax(uy,nxyz,'Uy')
if (nz.gt.1) call minmax(uz,nxyz,'Uz')
!
if (itime.eq.ifin) then ! Save restart on the last iteration
   call save_restart(ux,uy,uz,gx,gy,gz,ppm,phi,phiss) 
endif
!
call aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi)
if (mod(itime - isave,imodulo).eq.0.and.(itime.ge.isave)) then
!
!   call paraview_3d_scalar(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,walls,angle,indi,epsi)
   call paraview_3d_scalar(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,epsi,err_u,err_pp,err_ux,err_uy)
endif
!
!call aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi)
!
return
end subroutine stats
!
!********************************************************************
!
subroutine snapshots(ux,uy,uz)
!
!********************************************************************

USE param
USE variables

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,uz
integer :: longueur,num,i,j 
real(8) :: wzmin, wzmax 
character(len=3) suffix
character(len=20) nfichier

num=isave
call numcar (num,suffix)
longueur=index(nchamp,' ')-1
nfichier=nchamp(1:longueur)//suffix
longueur=index(nfichier,' ')-1
#if DEBUG
print *,nfichier(1:longueur)
#endif
open(12,file=nfichier(1:longueur),form='unformatted',status='unknown')
if (nz.gt.1) then
   write(12) ux,uy,uz
else
   write(12) ux,uy
endif
close(12)


!
return
end subroutine snapshots

!********************************************************************
!
subroutine paraview_3d_scalar(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,err_u,err_pp,err_ux,err_uy)
!
!********************************************************************
!
USE param
USE variables
USE aeroforce

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,tx,ty,tz,di,di1,px,py,pz,sy8,walls,angle,indi,epsi,tzz,uxx,uyy,txx
integer::i,j,k,nfil,num,longueur,o
real(8),dimension(nx) :: xx,xxnew,yynew
real(8),dimension(ny) :: yy
real(8),dimension(nz) :: zz
real(8),dimension(nxm,nym,nzm) :: ppm
real(8) :: heit,ymax,ymin
character(len=3) suffix
character(len=20) nfichier
character(len=20) :: filename

real(8),dimension(nx,ny)       :: err_ux, err_uy, err_u, err_pp
real(8),dimension(nx,ny)       :: erra_ux, erra_uy, erra_u, erra_pp
real(8)                        :: max_ux, max_uy, max_u, max_pp

801 format('snapshot',I4.4)
write(filename, 801) itime/imodulo

do i=1,nx
   xx(i)=(i-1)*dx
enddo
do j=1,ny
   yy(j)=yp(j)
enddo
do k=1,nz
   zz(k)=(k-1)*dz
enddo
!CALCULATION OF THE VORTICITY

if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,2)!IBM
call derx (tx,uy,di,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
call dery (ty,ux,di,di1,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
do k=1,nz
do j=1,ny
do i=1,nx
   tz(i,j,k)=(tx(i,j,k)-ty(i,j,k))
   walls(i,j,k)=ty(i,j,k)  ! Calculate Wall Shear
enddo
enddo
enddo

tzz(:,:,:)=(1.-epsi(:,:,:))*tz(:,:,:)
 
uxx(:,:,:)=(1.-epsi(:,:,:))*ux(:,:,:)
uyy(:,:,:)=(1.-epsi(:,:,:))*uy(:,:,:)

!PRESSURE ON MAIN MESH
call interi6(sy8,ppm,di1,sx,cifip6,cisip6,ciwip6,cifx6,cisx6,ciwx6,nxm,nx,nym,nz,1)
call interiy6(tx,sy8,di1,di,sy,cifip6y,cisip6y,ciwip6y,cify6,cisy6,ciwy6,nx,nym,ny,nz,1)
tx(:,:,:)=tx(:,:,:)/dt
txx(:,:,:)=(1.-epsi(:,:,:))*tx(:,:,:)
!
nfil=41
open(nfil,file='snapshots/'//trim(filename(1:12))//'.vtr')
write(nfil,*)'<VTKFile type="RectilinearGrid" version="0.1"',&
     ' byte_order="LittleEndian">'
write(nfil,*)'  <RectilinearGrid WholeExtent=',&
     '"1 ',nx,' 1 ',ny,' 1 ',nz,'">'
write(nfil,*)'    <Piece Extent=',&
     '"1 ',nx,' 1 ',ny,' 1 ',nz,'">'
write(nfil,*)'      <Coordinates>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="X_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (xx(i),i=1,nx)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Y_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (yy(j),j=1,ny)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Z_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (zz(k),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </Coordinates>'
write(nfil,*)'      <PointData Scalars="scalar">'
write(nfil,*)'        <DataArray Name="vorticity"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((tzz(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="velocity_ux"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((ux(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="velocity_uy"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uy(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="pressure"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((txx(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="epsi"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((epsi(i,j,k),i=1,nx),j=1,ny),k=1,nz)

!write(nfil,*)'        </DataArray>'
!write(nfil,*)'        <DataArray Name="angle"',&
!     ' type="Float32"',&
!     ' NumberOfComponents="1"',&
!     ' format="ascii">'
!write(nfil,*) (((angle(i,j,k),i=1,nx),j=1,ny),k=1,nz)
!write(nfil,*)'        </DataArray>'
!write(nfil,*)'        <DataArray Name="Height"',&
!     ' type="Float32"',&
!     ' NumberOfComponents="1"',&
!     ' format="ascii">'
!write(nfil,*) (heit)
!write(nfil,*)'        </DataArray>'
!write(nfil,*)'        <DataArray Name="xn"',&
!     ' type="Float32"',&
!     ' NumberOfComponents="1"',&
!     ' format="ascii">'
!write(nfil,*) (xxnew(j),j=1,o)
!write(nfil,*)'        </DataArray>'
!write(nfil,*)'        <DataArray Name="yn"',&
!     ' type="Float32"',&
!     ' NumberOfComponents="1"',&
!     ' format="ascii">'
!write(nfil,*) (yynew(j),j=1,o)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </PointData>'
!write(nfil,*)'      <CellData Scalars="scalar">'
!write(nfil,*)'      </CellData>'
write(nfil,*)'    </Piece>'
write(nfil,*)'  </RectilinearGrid>'
write(nfil,*)'</VTKFile>'
close(nfil)
!
return
end subroutine paraview_3d_scalar



!********************************************************************
!
subroutine save_restart(ux,uy,uz,gx,gy,gz,ppm,phi,phiss)
!
!*******************************************************************

USE param
USE variables

implicit none

integer :: num,longueur
real(8),dimension(nx,ny,nz) :: ux,uy,uz,gx,gy,gz
real(8),dimension(nx,ny,nz,nphi) :: phi,phiss
real(8),dimension(nxm,nym,nzm) :: ppm
character(len=4) suffix
character(len=20) nfichier


open(11,file='restart',form='unformatted',status='unknown')
if (iscalaire==0) then
   if (nz.gt.1) then
      write(11) ux,uy,uz,ppm,gx,gy,gz,dpdyx1,dpdyxn,dpdzx1,dpdzxn,dpdxy1,dpdxyn,dpdzy1,dpdzyn,dpdxz1,dpdxzn,dpdyz1,dpdyzn
   else
      write(11) ux,uy,ppm,gx,gy,dpdyx1,dpdyxn,dpdxy1,dpdxyn
   endif
else
   if (nz.gt.1) then
      write(11) ux,uy,uz,phi,ppm,gx,gy,gz,dpdyx1,dpdyxn,dpdzx1,dpdzxn,dpdxy1,dpdxyn,dpdzy1,dpdzyn,dpdxz1,dpdxzn,dpdyz1,dpdyzn
   else
      write(11) ux,uy,phi,ppm,gx,gy,dpdyx1,dpdyxn,dpdxy1,dpdxyn
   endif
endif
close(11)


return
end subroutine save_restart

!*************************************************************
!
subroutine moyt(uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz,ux,uy,uz)
!
!*************************************************************

USE param
USE variables

implicit none

integer :: i,nxyz
real(8),dimension(nx,ny,nz) ::uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz,ux,uy,uz

nxyz=nx*ny*nz

do i=1,nxyz
   uxmt(i,1,1)=uxmt(i,1,1)+ux(i,1,1)
   uymt(i,1,1)=uymt(i,1,1)+uy(i,1,1)
   uzmt(i,1,1)=uzmt(i,1,1)+uz(i,1,1)
   uxux(i,1,1)=uxux(i,1,1)+ux(i,1,1)*ux(i,1,1)
   uyuy(i,1,1)=uyuy(i,1,1)+uy(i,1,1)*uy(i,1,1)
   uzuz(i,1,1)=uzuz(i,1,1)+uz(i,1,1)*uz(i,1,1)
   uxuy(i,1,1)=uxuy(i,1,1)+ux(i,1,1)*uy(i,1,1)
   uxuz(i,1,1)=uxuz(i,1,1)+ux(i,1,1)*uz(i,1,1)
   uyuz(i,1,1)=uyuz(i,1,1)+uy(i,1,1)*uz(i,1,1)
enddo

if (mod(itime,isave).eq.0) then
     open(75,file='moyt.dat',form='unformatted')
     write(75) uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz
     close(75)
endif


return
end subroutine moyt
!********************************************************************
!
subroutine paraview_3d(ux,uy,epsi)
!
!********************************************************************
!
USE param
USE variables
USE aeroforce

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,tx,ty,tz,di,di1,px,py,pz,sy8,walls,angle,indi,epsi
integer::i,j,k,nfil,num,longueur,o
real(8),dimension(nx) :: xx,xxnew,yynew
real(8),dimension(ny) :: yy
real(8),dimension(nz) :: zz
real(8),dimension(nxm,nym,nzm) :: ppm
real(8) :: heit,ymax,ymin
character(len=3) suffix
character(len=20) nfichier
character(len=20) :: filename
!
!ux(:,:,:)=(1.-epsi(:,:,:))*ux(:,:,:)
!uy(:,:,:)=(1.-epsi(:,:,:))*uy(:,:,:)
!
801 format('snapshot',I4.4)
write(filename, 801) itime/imodulo

do i=1,nx
   xx(i)=(i-1)*dx
enddo
do j=1,ny
   yy(j)=yp(j)
enddo
do k=1,nz
   zz(k)=(k-1)*dz
enddo

nfil=41
open(nfil,file=filename(1:12)//'.vtr')
write(nfil,*)'<VTKFile type="RectilinearGrid" version="0.1"',&
     ' byte_order="LittleEndian">'
write(nfil,*)'  <RectilinearGrid WholeExtent=',&
     '"1 ',nx,' 1 ',ny,' 1 ',nz,'">'
write(nfil,*)'    <Piece Extent=',&
     '"1 ',nx,' 1 ',ny,' 1 ',nz,'">'
write(nfil,*)'      <Coordinates>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="X_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (xx(i),i=1,nx)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Y_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (yy(j),j=1,ny)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Z_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (zz(k),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </Coordinates>'
write(nfil,*)'      <PointData Scalars="scalar">'
write(nfil,*)'        <DataArray Name="velocity_ux"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((ux(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="velocity_uy"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uy(i,j,k),i=1,nx),j=1,ny),k=1,nz)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </PointData>'
!write(nfil,*)'      <CellData Scalars="scalar">'
!write(nfil,*)'      </CellData>'
write(nfil,*)'    </Piece>'
write(nfil,*)'  </RectilinearGrid>'
write(nfil,*)'</VTKFile>'
close(nfil)
!
return
end subroutine paraview_3d