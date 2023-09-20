program moy_stat

  implicit none

  integer, parameter :: nx1=512 , ny1=257, nz1=1
  real(8),dimension(nx1,ny1,nz1) :: uxmt,uymt,uzmt,uxux,uyuy
  real(8),dimension(nx1,ny1,nz1) :: uzuz,uxuy,uxuz,uyuz
  integer :: i,j,k,count,nfil,ntime
  real(8) :: pi,x,y,dx,dy,dz,lx,ly,lz
  real(8),dimension(ny1) :: yp,ypi,qstat,qstat2
  real(8),dimension(nx1) :: xx
  real(8),dimension(ny1) :: yy
  real(8),dimension(nz1) :: zz

  ntime=10000
  lx=20.
  ly=12.
  lz=1.
  dx=lx/real(nx1)
  dy=ly/real(ny1-1)
  dz=lz

do i=1,nx1
   xx(i)=(i-1)*dx
enddo

open(10,file='yp.dat', form='formatted')
do j=1,ny1
read(10,*)yy(j)
enddo

do k=1,nz1
   zz(k)=(k-1)*dz
enddo

open(75,file='moyt.dat',form='unformatted')
read(75) uxmt,uymt,uzmt,uxux,uyuy,uzuz,uxuy,uxuz,uyuz
close(75)

uxmt(:,:,:)=uxmt(:,:,:)/real(ntime)
uxux(:,:,:)=uxux(:,:,:)/real(ntime)
uxux(:,:,:)=uxux(:,:,:)-uxmt(:,:,:)*uxmt(:,:,:)


nfil=41
open(nfil,file='moy.vtr')
write(nfil,*)'<VTKFile type="RectilinearGrid" version="0.1"',&
     ' byte_order="LittleEndian">'
write(nfil,*)'  <RectilinearGrid WholeExtent=',&
     '"1 ',nx1,' 1 ',ny1,' 1 ',nz1,'">'
write(nfil,*)'    <Piece Extent=',&
     '"1 ',nx1,' 1 ',ny1,' 1 ',nz1,'">'
write(nfil,*)'      <Coordinates>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="X_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (xx(i),i=1,nx1)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Y_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (yy(j),j=1,ny1)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray type="Float32"',&
     ' Name="Z_COORDINATES"',&
     ' NumberOfComponents="1">'
write(nfil,*) (zz(k),k=1,nz1)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </Coordinates>'
write(nfil,*)'      <PointData Scalars="scalar">'
write(nfil,*)'        <DataArray Name="umean"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uxmt(i,j,k),i=1,nx1),j=1,ny1),k=1,nz1)
write(nfil,*)'        </DataArray>'
write(nfil,*)'        <DataArray Name="uumean"',&
     ' type="Float32"',&
     ' NumberOfComponents="1"',&
     ' format="ascii">'
write(nfil,*) (((uxux(i,j,k),i=1,nx1),j=1,ny1),k=1,nz1)
write(nfil,*)'        </DataArray>'
write(nfil,*)'      </PointData>'
write(nfil,*)'    </Piece>'
write(nfil,*)'  </RectilinearGrid>'
write(nfil,*)'</VTKFile>'


end program moy_stat
