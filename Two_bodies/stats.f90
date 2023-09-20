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
if (mod(itime,isave).eq.0) then
   call save_restart(ux,uy,uz,gx,gy,gz,ppm,phi,phiss) 
endif
!
call aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi,1)
call aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi,2)
!call aerof2d(ux,uy,uz,uxm1,uxm2,uym1,uym2,ppm,epsi,3)

if (mod(itime,imodulo).eq.0 .and. itime.ge.idebmod) then 
!
!   call paraview_3d_scalar(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,walls,angle,indi,epsi)
   call paraview_3d_scalar(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,epsi,err_u,err_pp,err_ux,err_uy)
endif

! Phase average
!call phase_ave(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,epsi)
!!! Time average
!call time_ave(ux,uy,tx,ty,tz,ppm,p_x,p_y,p_z,epsi)

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

print *,nfichier(1:longueur)

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
subroutine paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,err_u,err_pp,err_ux,err_uy)
!
!********************************************************************
!
USE param
USE variables
USE aeroforce

implicit none

real(8),dimension(nx,ny,nz) :: ux,uy,tx,ty,tz,di,di1,px,py,pz,sy8,walls,angle,indi,epsi,tzz,uxx,uyy,txx
real(8),dimension(nx,ny,nz) :: uxx2, uyy2, uxy2, txx2
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

counter = counter + 1

601 format('posi',I3.3)
write(filename, 601) counter

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

uxx2(:,:,:) = ux(:,:,:)*ux(:,:,:)
uyy2(:,:,:) = uy(:,:,:)*uy(:,:,:)
uxy2(:,:,:) = ux(:,:,:)*uy(:,:,:)
txx2(:,:,:) = txx(:,:,:)*txx(:,:,:)
!
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

write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="uxx"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uxx2(i,j,k),i=1,nx),j=1,ny),k=1,nz)

write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="uyy"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uyy2(i,j,k),i=1,nx),j=1,ny),k=1,nz)

write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="uxy"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uxy2(i,j,k),i=1,nx),j=1,ny),k=1,nz)

write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="PP"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((txx2(i,j,k),i=1,nx),j=1,ny),k=1,nz)

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
end subroutine paraview_3d_pos

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
!********************************************************************
!
subroutine phase_ave(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi)
!
!********************************************************************
!
USE param
USE ibm
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
character(len=30) :: filename



integer :: tamax, bamax, tamax2up, tamax2do, bamax2up, bamax2do, Zamax2up, Zamax2do
real(8) :: alfat, alfat2, toler, alfat0
real(8) :: amaxT2, amaxTm2, amaxB2, amaxBm2, amaxZ


! Constants for Flaps   
if (angexc.eq.0) then                          ! Exceed 0 degrees
   alfat = amax*sin(2.0*pi*freq*t)             ! Angle of top flap
   alfat2 = amax*sin(2.0*pi*freq*(t+dt))       ! Angle of top flap (t+Dt)
   alfat0 = amax*sin(2.0*pi*freq*(t-dt))       ! Angle of top flap (t-Dt)
   amaxT2 = amax
   amaxTm2 = amax/2.0
   amaxB2 = -amax
   amaxBm2 = -amax/2.0
   amaxZ = 0.0
else
   alfat = amax*sin(2.0*pi*freq*t)-amax        ! Angle of top flap
   alfat2 = amax*sin(2.0*pi*freq*(t+dt))-amax  ! Angle of top flap (t+Dt)
   alfat0 = amax*sin(2.0*pi*freq*(t-dt))-amax  ! Angle of top flap (t-Dt)
   amaxT2 = amax-amax
   amaxTm2 = amax/2.0 - amax  
   amaxB2 = -amax-amax   
   amaxBm2 = -amax/2.0 - amax  
   amaxZ = 0.0 - amax
endif

! Define Tolerance
toler = 0.05

! Find time for Top Angle
tamax = 0
if (alfat.eq.amaxT2) then
   tamax = 1
elseif (abs(alfat-amaxT2).le.toler) then
   if ((alfat2.le.alfat).and.(alfat0.le.alfat)) tamax = 1
endif

! Find time for Bot Angle
bamax = 0
if (alfat.eq.amaxB2) then
   bamax = 1
elseif (abs(alfat-amaxB2).le.toler) then
   if ((alfat2.ge.alfat).and.(alfat0.ge.alfat)) bamax = 1
endif

 
! Find time for Half Top Angle
tamax2up = 0
tamax2do = 0
if (alfat.eq.amaxTm2) then
   if ((alfat2-alfat).gt.0.) then
      tamax2up = 1
   else
      tamax2do = 1
   endif
elseif ((abs(alfat-amaxTm2).le.toler)) then
   if ((alfat2-alfat).gt.0.) then
      if ((alfat2.ge.amaxTm2).and.(alfat.le.amaxTm2)) tamax2up = 1
   else
      if ((alfat2.le.amaxTm2).and.(alfat.ge.amaxTm2)) tamax2do = 1
   endif
endif

! Find time for Half Bot Angle
bamax2up = 0
bamax2do = 0
if (alfat.eq.amaxBm2) then
   if ((alfat2-alfat).gt.0.) then
      bamax2up = 1
   else
      bamax2do = 1
   endif
elseif ((abs(alfat-amaxBm2).le.toler)) then
   if ((alfat2-alfat).gt.0.) then
      if ((alfat2.ge.amaxBm2).and.(alfat.le.amaxBm2)) bamax2up = 1
   else
      if ((alfat2.le.amaxBm2).and.(alfat.ge.amaxBm2)) bamax2do = 1
   endif
endif

! Find time for Zero Angle
Zamax2up = 0
Zamax2do = 0
if (alfat.eq.amaxZ) then
   if ((alfat2-alfat).gt.amaxZ) then
      Zamax2up = 1
   else
      Zamax2do = 1
   endif
elseif ((abs(alfat-amaxZ).le.toler)) then
   if ((alfat2-alfat).gt.0.) then
      if ((alfat2.ge.amaxZ).and.(alfat.le.amaxZ)) Zamax2up = 1
   else
      if ((alfat2.le.amaxZ).and.(alfat.ge.amaxZ)) Zamax2do = 1
   endif
endif





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



! Write Phase Average Files

! Top Max Angle
if (tamax.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxtm = uxtm + ux *(1.0-epsi)
   uytm = uytm + uy *(1.0-epsi) 
   uxxtm = uxxtm + ux*ux *(1.0-epsi)
   uyytm = uyytm + uy*uy *(1.0-epsi)
   uxytm = uxytm + ux*uy *(1.0-epsi)
   tzztm = tzztm + tzz *(1.0-epsi)   ! Vorticity
   txxtm = txxtm + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*tm1*')
          
       write(filename,"('uxtm1.dat',I7.7)") itime   
       open(950,file=filename,status='NEW')
       write(950,*) uxtm        
       close(950)   
       
       write(filename,"('uytm1.dat',I7.7)") itime   
       open(951,file=filename,status='NEW')
       write(951,*) uytm        
       close(951) 
       
       write(filename,"('uxxtm1.dat',I7.7)") itime   
       open(952,file=filename,status='NEW')
       write(952,*) uxxtm        
       close(952)      
       
       write(filename,"('uyytm1.dat',I7.7)") itime   
       open(953,file=filename,status='NEW')
       write(953,*) uyytm        
       close(953)      
       
       write(filename,"('uxytm1.dat',I7.7)") itime   
       open(954,file=filename,status='NEW')
       write(954,*) uxytm        
       close(954)  
       
       write(filename,"('tzztm1.dat',I7.7)") itime   
       open(955,file=filename,status='NEW')
       write(955,*) tzztm        
       close(955)                    
                
       write(filename,"('txxtm1.dat',I7.7)") itime   
       open(956,file=filename,status='NEW')
       write(956,*) txxtm        
       close(956)  
       
       write(filename,"('epsitm1.dat',I7.7)") itime   
       open(957,file=filename,status='NEW')
       write(957,*) epsi        
       close(957)                         
endif

! Bot Max Angle
if (bamax.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxbm = uxbm + ux *(1.0-epsi)
   uybm = uybm + uy *(1.0-epsi) 
   uxxbm = uxxbm + ux*ux *(1.0-epsi)
   uyybm = uyybm + uy*uy *(1.0-epsi)
   uxybm = uxybm + ux*uy *(1.0-epsi)
   tzzbm = tzzbm + tzz *(1.0-epsi)   ! Vorticity
   txxbm = txxbm + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*bm1*')
         
       write(filename,"('uxbm1.dat',I7.7)") itime   
       open(850,file=filename,status='NEW')
       write(850,*) uxbm        
       close(850)   
       
       write(filename,"('uybm1.dat',I7.7)") itime   
       open(851,file=filename,status='NEW')
       write(851,*) uybm        
       close(851) 
       
       write(filename,"('uxxbm1.dat',I7.7)") itime   
       open(852,file=filename,status='NEW')
       write(852,*) uxxbm        
       close(852)      
       
       write(filename,"('uyybm1.dat',I7.7)") itime   
       open(853,file=filename,status='NEW')
       write(853,*) uyybm        
       close(853)      
       
       write(filename,"('uxybm1.dat',I7.7)") itime   
       open(854,file=filename,status='NEW')
       write(854,*) uxybm        
       close(854)  
       
       write(filename,"('tzzbm1.dat',I7.7)") itime   
       open(855,file=filename,status='NEW')
       write(855,*) tzzbm        
       close(855)                    
                
       write(filename,"('txxbm1.dat',I7.7)") itime   
       open(856,file=filename,status='NEW')
       write(856,*) txxbm        
       close(856) 
       
       write(filename,"('epsibm1.dat',I7.7)") itime   
       open(857,file=filename,status='NEW')
       write(857,*) epsi        
       close(857)                        
endif

! Top Half Max Angle (Going Up)
if (tamax2up.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxtm2u = uxtm2u + ux *(1.0-epsi)
   uytm2u = uytm2u + uy *(1.0-epsi) 
   uxxtm2u = uxxtm2u + ux*ux *(1.0-epsi)
   uyytm2u = uyytm2u + uy*uy *(1.0-epsi)
   uxytm2u = uxytm2u + ux*uy *(1.0-epsi)
   tzztm2u = tzztm2u + tzz *(1.0-epsi)   ! Vorticity
   txxtm2u = txxtm2u + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*tm2u*')
         
       write(filename,"('uxtm2u.dat',I7.7)") itime   
       open(750,file=filename,status='NEW')
       write(750,*) uxtm2u        
       close(750)   
       
       write(filename,"('uytm2u.dat',I7.7)") itime   
       open(751,file=filename,status='NEW')
       write(751,*) uytm2u        
       close(751) 
       
       write(filename,"('uxxtm2u.dat',I7.7)") itime   
       open(752,file=filename,status='NEW')
       write(752,*) uxxtm2u        
       close(752)      
       
       write(filename,"('uyytm2u.dat',I7.7)") itime   
       open(753,file=filename,status='NEW')
       write(753,*) uyytm2u        
       close(753)      
       
       write(filename,"('uxytm2u.dat',I7.7)") itime   
       open(754,file=filename,status='NEW')
       write(754,*) uxytm2u        
       close(754)  
       
       write(filename,"('tzztm2u.dat',I7.7)") itime   
       open(755,file=filename,status='NEW')
       write(755,*) tzztm2u        
       close(755)                    
                
       write(filename,"('txxtm2u.dat',I7.7)") itime   
       open(756,file=filename,status='NEW')
       write(756,*) txxtm2u        
       close(756)           
              
       write(filename,"('epsitm2u.dat',I7.7)") itime   
       open(757,file=filename,status='NEW')
       write(757,*) epsi        
       close(757)       
endif

! Top Half Max Angle (Going Down)
if (tamax2do.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxtm2d = uxtm2d + ux *(1.0-epsi)
   uytm2d = uytm2d + uy *(1.0-epsi) 
   uxxtm2d = uxxtm2d + ux*ux *(1.0-epsi)
   uyytm2d = uyytm2d + uy*uy *(1.0-epsi)
   uxytm2d = uxytm2d + ux*uy *(1.0-epsi)
   tzztm2d = tzztm2d + tzz *(1.0-epsi)   ! Vorticity
   txxtm2d = txxtm2d + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*tm2d*')
         
       write(filename,"('uxtm2d.dat',I7.7)") itime   
       open(650,file=filename,status='NEW')
       write(650,*) uxtm2d        
       close(650)   
       
       write(filename,"('uytm2d.dat',I7.7)") itime   
       open(651,file=filename,status='NEW')
       write(651,*) uytm2d        
       close(651) 
       
       write(filename,"('uxxtm2d.dat',I7.7)") itime   
       open(652,file=filename,status='NEW')
       write(652,*) uxxtm2d        
       close(652)      
       
       write(filename,"('uyytm2d.dat',I7.7)") itime   
       open(653,file=filename,status='NEW')
       write(653,*) uyytm2d        
       close(653)      
       
       write(filename,"('uxytm2d.dat',I7.7)") itime   
       open(654,file=filename,status='NEW')
       write(654,*) uxytm2d        
       close(654)  
       
       write(filename,"('tzztm2d.dat',I7.7)") itime   
       open(655,file=filename,status='NEW')
       write(655,*) tzztm2d        
       close(655)                    
                
       write(filename,"('txxtm2d.dat',I7.7)") itime   
       open(656,file=filename,status='NEW')
       write(656,*) txxtm2d        
       close(656)   
              
       write(filename,"('epsitm2d.dat',I7.7)") itime   
       open(657,file=filename,status='NEW')
       write(657,*) epsi        
       close(657)               
endif

! Bot Half Max Angle (Going Up)
if (bamax2up.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxbm2u = uxbm2u + ux *(1.0-epsi)
   uybm2u = uybm2u + uy *(1.0-epsi) 
   uxxbm2u = uxxbm2u + ux*ux *(1.0-epsi)
   uyybm2u = uyybm2u + uy*uy *(1.0-epsi)
   uxybm2u = uxybm2u + ux*uy *(1.0-epsi)
   tzzbm2u = tzzbm2u + tzz *(1.0-epsi)   ! Vorticity
   txxbm2u = txxbm2u + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*bm2u*')
         
       write(filename,"('uxbm2u.dat',I7.7)") itime   
       open(550,file=filename,status='NEW')
       write(550,*) uxbm2u        
       close(550)   
       
       write(filename,"('uybm2u.dat',I7.7)") itime   
       open(551,file=filename,status='NEW')
       write(551,*) uybm2u        
       close(551) 
       
       write(filename,"('uxxbm2u.dat',I7.7)") itime   
       open(552,file=filename,status='NEW')
       write(552,*) uxxbm2u        
       close(552)      
       
       write(filename,"('uyybm2u.dat',I7.7)") itime   
       open(553,file=filename,status='NEW')
       write(553,*) uyybm2u        
       close(553)      
       
       write(filename,"('uxybm2u.dat',I7.7)") itime   
       open(554,file=filename,status='NEW')
       write(554,*) uxybm2u        
       close(554)  
       
       write(filename,"('tzzbm2u.dat',I7.7)") itime   
       open(555,file=filename,status='NEW')
       write(555,*) tzzbm2u        
       close(555)                    
                
       write(filename,"('txxbm2u.dat',I7.7)") itime   
       open(556,file=filename,status='NEW')
       write(556,*) txxbm2u        
       close(556)    
       
       write(filename,"('epsibm2u.dat',I7.7)") itime   
       open(557,file=filename,status='NEW')
       write(557,*) epsi        
       close(557)                     
endif

! Bot Half Max Angle (Going Down)
if (bamax2do.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   uxbm2d = uxbm2d + ux *(1.0-epsi)
   uybm2d = uybm2d + uy  *(1.0-epsi)
   uxxbm2d = uxxbm2d + ux*ux *(1.0-epsi)
   uyybm2d = uyybm2d + uy*uy *(1.0-epsi)
   uxybm2d = uxybm2d + ux*uy *(1.0-epsi)
   tzzbm2d = tzzbm2d + tzz *(1.0-epsi)   ! Vorticity
   txxbm2d = txxbm2d + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*bm2d*')
      
       write(filename,"('uxbm2d.dat',I7.7)") itime   
       open(450,file=filename,status='NEW')
       write(450,*) uxbm2d        
       close(450)   
       
       write(filename,"('uybm2d.dat',I7.7)") itime   
       open(451,file=filename,status='NEW')
       write(451,*) uybm2d        
       close(451) 
       
       write(filename,"('uxxbm2d.dat',I7.7)") itime   
       open(452,file=filename,status='NEW')
       write(452,*) uxxbm2d        
       close(452)      
       
       write(filename,"('uyybm2d.dat',I7.7)") itime   
       open(453,file=filename,status='NEW')
       write(453,*) uyybm2d        
       close(453)      
       
       write(filename,"('uxybm2d.dat',I7.7)") itime   
       open(454,file=filename,status='NEW')
       write(454,*) uxybm2d        
       close(454)  
       
       write(filename,"('tzzbm2d.dat',I7.7)") itime   
       open(455,file=filename,status='NEW')
       write(455,*) tzzbm2d        
       close(455)                    
                
       write(filename,"('txxbm2d.dat',I7.7)") itime   
       open(456,file=filename,status='NEW')
       write(456,*) txxbm2d        
       close(456)   
              
       write(filename,"('epsibm2d.dat',I7.7)") itime   
       open(457,file=filename,status='NEW')
       write(457,*) epsi        
       close(457)               
endif

! BZero Angle (Going Up)
if (Zamax2up.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   ux02u = ux02u + ux *(1.0-epsi)
   uy02u = uy02u + uy *(1.0-epsi) 
   uxx02u = uxx02u + ux*ux *(1.0-epsi)
   uyy02u = uyy02u + uy*uy *(1.0-epsi)
   uxy02u = uxy02u + ux*uy *(1.0-epsi)
   tzz02u = tzz02u + tzz *(1.0-epsi)   ! Vorticity
   txx02u = txx02u + txx *(1.0-epsi)   ! Pressure
       
       ! Remove previous files
       call system ("rm " //'*02u*')
   
       write(filename,"('ux02u.dat',I7.7)") itime   
       open(350,file=filename,status='NEW')
       write(350,*) ux02u        
       close(350)   

       write(filename,"('uy02u.dat',I7.7)") itime   
       open(351,file=filename,status='NEW')
       write(351,*) uy02u        
       close(351) 
       
       write(filename,"('uxx02u.dat',I7.7)") itime   
       open(352,file=filename,status='NEW')
       write(352,*) uxx02u        
       close(352)      
       
       write(filename,"('uyy02u.dat',I7.7)") itime   
       open(353,file=filename,status='NEW')
       write(353,*) uyy02u        
       close(353)      
       
       write(filename,"('uxy02u.dat',I7.7)") itime   
       open(354,file=filename,status='NEW')
       write(354,*) uxy02u        
       close(354)  
       
       write(filename,"('tzz02u.dat',I7.7)") itime   
       open(355,file=filename,status='NEW')
       write(355,*) tzz02u        
       close(355)                    
                
       write(filename,"('txx02u.dat',I7.7)") itime   
       open(356,file=filename,status='NEW')
       write(356,*) txx02u        
       close(356)   
       
       write(filename,"('epsi02u.dat',I7.7)") itime   
       open(357,file=filename,status='NEW')
       write(357,*) epsi        
       close(357)                      
endif

! Zero Angle (Going Down)
if (Zamax2do.eq.1) then     

call paraview_3d_pos(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi,ux,ux,ux,ux)

   ux02d = ux02d + ux *(1.0-epsi)
   uy02d = uy02d + uy *(1.0-epsi) 
   uxx02d = uxx02d + ux*ux *(1.0-epsi)
   uyy02d = uyy02d + uy*uy *(1.0-epsi)
   uxy02d = uxy02d + ux*uy *(1.0-epsi)
   tzz02d = tzz02d + tzz *(1.0-epsi)   ! Vorticity
   txx02d = txx02d + txx *(1.0-epsi)   ! Pressure

       ! Remove previous files
       call system ("rm " //'*02d*')
      
       write(filename,"('ux02d.dat',I7.7)") itime   
       open(250,file=filename,status='NEW')
       write(250,*) ux02d        
       close(250)   
       
       write(filename,"('uy02d.dat',I7.7)") itime   
       open(251,file=filename,status='NEW')
       write(251,*) uy02d        
       close(251) 
       
       write(filename,"('uxx02d.dat',I7.7)") itime   
       open(252,file=filename,status='NEW')
       write(252,*) uxx02d        
       close(252)      
       
       write(filename,"('uyy02d.dat',I7.7)") itime   
       open(253,file=filename,status='NEW')
       write(253,*) uyy02d        
       close(253)      
       
       write(filename,"('uxy02d.dat',I7.7)") itime   
       open(254,file=filename,status='NEW')
       write(254,*) uxy02d        
       close(254)  
       
       write(filename,"('tzz02d.dat',I7.7)") itime   
       open(255,file=filename,status='NEW')
       write(255,*) tzz02d        
       close(255)                    
                
       write(filename,"('txx02d.dat',I7.7)") itime   
       open(256,file=filename,status='NEW')
       write(256,*) txx02d        
       close(256)  
       
       write(filename,"('epsi02d.dat',I7.7)") itime   
       open(257,file=filename,status='NEW')
       write(257,*) epsi        
       close(257)                       
endif
!
return
end subroutine phase_ave


!********************************************************************
!
subroutine time_ave(ux,uy,tx,ty,tz,ppm,px,py,pz,epsi)
!
!********************************************************************
!
USE param
USE ibm
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
character(len=30) :: filename


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



! Write Time Average Files
   ux_ave = ux_ave + ux *(1.0-epsi)
   uy_ave = uy_ave + uy *(1.0-epsi)
   uxx_ave = uxx_ave + ux*ux *(1.0-epsi)
   uyy_ave = uyy_ave + uy*uy *(1.0-epsi)
   uxy_ave = uxy_ave + ux*uy *(1.0-epsi)
   tzz_ave = tzz_ave + tzz *(1.0-epsi)   ! Vorticity
   txx_ave = txx_ave + txx *(1.0-epsi)   ! Pressure
   epsi_ave = epsi_ave + epsi

if (mod(itime,imodulo).eq.0 .and. itime.ge.idebmod) then
       ! Remove previous files
       call system ("rm " //'*_ave1*')
          
       write(filename,"('ux_ave1.dat',I7.7)") itime   
       open(950,file=filename,status='NEW')
       write(950,*) ux_ave        
       close(950)   
       
       write(filename,"('uy_ave1.dat',I7.7)") itime   
       open(951,file=filename,status='NEW')
       write(951,*) uy_ave        
       close(951) 
       
       write(filename,"('uxx_ave1.dat',I7.7)") itime   
       open(952,file=filename,status='NEW')
       write(952,*) uxx_ave        
       close(952)      
       
       write(filename,"('uyy_ave1.dat',I7.7)") itime   
       open(953,file=filename,status='NEW')
       write(953,*) uyy_ave        
       close(953)      
       
       write(filename,"('uxy_ave1.dat',I7.7)") itime   
       open(954,file=filename,status='NEW')
       write(954,*) uxy_ave        
       close(954)  
       
       write(filename,"('tzz_ave1.dat',I7.7)") itime   
       open(955,file=filename,status='NEW')
       write(955,*) tzz_ave        
       close(955)                    
                
       write(filename,"('txx_ave1.dat',I7.7)") itime   
       open(956,file=filename,status='NEW')
       write(956,*) txx_ave        
       close(956)  
       
       write(filename,"('epsi_ave1.dat',I7.7)") itime   
       open(957,file=filename,status='NEW')
       write(957,*) epsi_ave        
       close(957)                         
endif
!
return
end subroutine time_ave
