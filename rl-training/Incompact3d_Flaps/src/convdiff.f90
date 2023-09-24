!********************************************************************
!
subroutine convdiff(ux,uy,uz,sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,sy10,sy11,sy12,di1,di2)
! 
!********************************************************************
USE param
USE variables
!
implicit none
!
integer :: nxyz1,ijk,i,j,k,ns
real(8),dimension(nx,ny,nz) :: sy1,sy2,sy3,sy4,sy5,sy6,sy7,sy8,sy9,sy10,sy11,sy12,ux,uy,uz,di1,di2,sy3x,sy3y,uxhg
!
nxyz1=nx*ny*nz 
   if (iskew==0) then !UROTU!
      if (nz.gt.1) then!3D
         if(ilag.eq.1)call forcage_flugrange_y(uz,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy1,uz,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(uy,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy2,uy,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(ux,zi,zf,nobjz,nx,ny,1)!IBM
         call derz (sy3,ux,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(uz,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy7,uz,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy8,uy,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1) 
         if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM         
         call dery (sy9,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         do ijk=1,nxyz1
            sy1(ijk,1,1)=sy1(ijk,1,1)-sy2(ijk,1,1)
            sy3(ijk,1,1)=sy3(ijk,1,1)-sy7(ijk,1,1)
            sy8(ijk,1,1)=sy8(ijk,1,1)-sy9(ijk,1,1)
         enddo
         do ijk=1,nxyz1
            sy2(ijk,1,1)=sy3(ijk,1,1)*uz(ijk,1,1)-sy8(ijk,1,1)*uy(ijk,1,1)
            sy7(ijk,1,1)=sy8(ijk,1,1)*ux(ijk,1,1)-sy1(ijk,1,1)*uz(ijk,1,1)
            sy9(ijk,1,1)=sy1(ijk,1,1)*uy(ijk,1,1)-sy3(ijk,1,1)*ux(ijk,1,1)
         enddo
      else!2D
         if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,2)!IBM
         call derx (sy8,uy,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
         call dery (sy9,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1) 
         do ijk=1,nxyz1
            sy8(ijk,1,1)= sy8(ijk,1,1)-sy9(ijk,1,1)
            sy2(ijk,1,1)=-sy8(ijk,1,1)*uy(ijk,1,1)
            sy7(ijk,1,1)= sy8(ijk,1,1)*ux(ijk,1,1)
         enddo
      endif
   else!skew
      if (nz.gt.1) then!3D
         sy1(:,:,:)=ux(:,:,:)*ux(:,:,:)
         sy4(:,:,:)=uy(:,:,:)*uy(:,:,:)
         sy3(:,:,:)=uz(:,:,:)*uz(:,:,:)
         sy5(:,:,:)=uy(:,:,:)*uz(:,:,:)
         sy8(:,:,:)=ux(:,:,:)*uz(:,:,:)
         sy6(:,:,:)=ux(:,:,:)*uy(:,:,:)
         if(ilag.eq.1)call forcage_flugrange_x(sy1,xi,xf,nobjx,ny,nz,1)!IBM
         call derx (sy2,sy1,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(sy4,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy7,sy4,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(sy3,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy9,sy3,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(sy6,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy1,sy6,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(sy6,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy4,sy6,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(sy8,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy3,sy8,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         sy2(:,:,:)=sy2(:,:,:)+sy4(:,:,:)
         sy7(:,:,:)=sy7(:,:,:)+sy1(:,:,:)
         sy9(:,:,:)=sy9(:,:,:)+sy3(:,:,:)
         if(ilag.eq.1)call forcage_flugrange_z(sy8,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy3,sy8,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(sy5,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy1,sy5,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(sy5,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy4,sy5,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         sy2(:,:,:)=sy2(:,:,:)+sy3(:,:,:)
         sy7(:,:,:)=sy7(:,:,:)+sy4(:,:,:)
         sy9(:,:,:)=sy9(:,:,:)+sy1(:,:,:)
         if(ilag.eq.1)call forcage_flugrange_x(ux,xi,xf,nobjx,ny,nz,1)!IBM
         call derx (sy1,ux,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy4,uy,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(uz,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy3,uz,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy6,uy,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(uz,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy5,uz,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(ux,zi,zf,nobjz,nx,ny,1)!IBM
         call derz (sy8,ux,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         sy2(:,:,:)=0.5*(sy2(:,:,:)+ux(:,:,:)*sy1(:,:,:)+uz(:,:,:)*sy8(:,:,:))
         sy7(:,:,:)=0.5*(sy7(:,:,:)+uy(:,:,:)*sy4(:,:,:)+ux(:,:,:)*sy6(:,:,:))
         sy9(:,:,:)=0.5*(sy9(:,:,:)+uz(:,:,:)*sy3(:,:,:)+uy(:,:,:)*sy5(:,:,:))
         if(ilag.eq.1)call forcage_flugrange_x(uz,xi,xf,nobjx,ny,nz,0)!IBM
         call derx (sy3,uz,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
         call dery (sy1,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_z(uy,zi,zf,nobjz,nx,ny,0)!IBM
         call derz (sy4,uy,di1,sz,ffzp,fszp,fwzp,nx,ny,nz,1)
         sy2(:,:,:)=sy2(:,:,:)+0.5*(uy(:,:,:)*sy1(:,:,:))
         sy7(:,:,:)=sy7(:,:,:)+0.5*(uz(:,:,:)*sy4(:,:,:))
         sy9(:,:,:)=sy9(:,:,:)+0.5*(ux(:,:,:)*sy3(:,:,:))
      else!2D
         sy1(:,:,:)=ux(:,:,:)*ux(:,:,:)
         sy4(:,:,:)=uy(:,:,:)*uy(:,:,:)
         
         if(ilag.eq.1)call forcage_flugrange_x(ux,xi,xf,nobjx,ny,nz,1)!IBM
         if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,2)!IBM
         sy3x(:,:,:)=ux(:,:,:)*uy(:,:,:)
         
         if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
         if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,2)!IBM
         sy3y(:,:,:)=ux(:,:,:)*uy(:,:,:)
         
         if(ilag.eq.1)call forcage_flugrange_x(sy1,xi,xf,nobjx,ny,nz,3)!IBM

         call derx (sy2,sy1,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(sy4,yi,yf,nobjy,nx,nz,4)!IBM
         call dery (sy7,sy4,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_x(sy3x,xi,xf,nobjx,ny,nz,5)!IBM
         call derx (sy1,sy3x,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(sy3y,yi,yf,nobjy,nx,nz,5)!IBM
         call dery (sy4,sy3y,di1,di2,sy,ffy,fsy,fwy,ppy,nx,ny,nz,0)
         sy2(:,:,:)=sy2(:,:,:)+sy4(:,:,:)
         sy7(:,:,:)=sy7(:,:,:)+sy1(:,:,:)
         if(ilag.eq.1)call forcage_flugrange_x(ux,xi,xf,nobjx,ny,nz,1)!IBM
         call derx (sy1,ux,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,2)!IBM
         call dery (sy4,uy,di1,di2,sy,ffy,fsy,fwy,ppy,nx,ny,nz,0)
         if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,2)!IBM
         call derx (sy3,uy,di1,sx,ffxp,fsxp,fwxp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
         call dery (sy5,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         sy2(:,:,:)=0.5*(sy2(:,:,:)+ux(:,:,:)*sy1(:,:,:)+uy(:,:,:)*sy5(:,:,:))
         sy7(:,:,:)=0.5*(sy7(:,:,:)+uy(:,:,:)*sy4(:,:,:)+ux(:,:,:)*sy3(:,:,:))
      endif
   endif
   sy8(:,:,:)=sy7(:,:,:)
   if(ilag.eq.1)call forcage_flugrange_x(ux,xi,xf,nobjx,ny,nz,1)!IBM
   call derxx (sy1,ux,di1,sx,sfx ,ssx ,swx ,nx,ny,nz,0)
   if (istret.ne.0) then !raffinement
      if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
      call deryy (sy4,ux,di1,di2,sy,sfyp,ssyp,swyp,nx,ny,nz,1)
      if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
      call dery (sy5,ux,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
      do k=1,nz
      do j=1,ny
      do i=1,nx
         sy4(i,j,k)=sy4(i,j,k)*pp2y(j)-pp4y(j)*sy5(i,j,k)
      enddo
      enddo
      enddo
   else! pas de raffinement
      if(ilag.eq.1)call forcage_flugrange_y(ux,yi,yf,nobjy,nx,nz,1)!IBM
      call deryy (sy4,ux,di1,di2,sy,sfyp,ssyp,swyp,nx,ny,nz,1) 
   endif
   if (nz.gt.1) then!3D
      if(ilag.eq.1)call forcage_flugrange_z(ux,zi,zf,nobjz,nx,ny,1)!IBM
      call derzz (sy6,ux,di1,sz,sfzp,sszp,swzp,nx,ny,nz,1)
      sy7(:,:,:)=xnu*(sy1(:,:,:)+sy4(:,:,:)+sy6(:,:,:))-sy2(:,:,:)
   else!2D
      do j=1,ny
      do i=1,nx
         sy7(i,j,1)=xnu*(sy1(i,j,1)+sy4(i,j,1))-sy2(i,j,1)
      enddo
      enddo
   endif
!
   if(ilag.eq.1)call forcage_flugrange_x(uy,xi,xf,nobjx,ny,nz,2)!IBM
   call derxx (sy1,uy,di1,sx,sfxp,ssxp,swxp,nx,ny,nz,1)
   if (istret.ne.0) then !raffinement
      if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,2)!IBM
      call deryy (sy4,uy,di1,di2,sy,sfy,ssy,swy,nx,ny,nz,0)
      if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,2)!IBM
      call dery (sy5,uy,di1,di2,sy,ffy,fsy,fwy,ppy,nx,ny,nz,0)
      do k=1,nz
      do j=1,ny
      do i=1,nx
         sy4(i,j,k)=sy4(i,j,k)*pp2y(j)-pp4y(j)*sy5(i,j,k)
      enddo
      enddo
      enddo
   else!pas de raffinement
      if(ilag.eq.1)call forcage_flugrange_y(uy,yi,yf,nobjy,nx,nz,2)!IBM
      call deryy (sy4,uy,di1,di2,sy,sfy,ssy,swy,nx,ny,nz,0)
   endif
   if (nz.gt.1) then!3D
      if(ilag.eq.1)call forcage_flugrange_z(uy,zi,zf,nobjz,nx,ny,2)!IBM
      call derzz (sy6,uy,di1,sz,sfzp,sszp,swzp,nx,ny,nz,1)
      sy8(:,:,:)=xnu*(sy1(:,:,:)+sy4(:,:,:)+sy6(:,:,:))-sy8(:,:,:)
   else!2D
      do j=1,ny
      do i=1,nx
         sy8(i,j,1)=xnu*(sy1(i,j,1)+sy4(i,j,1))-sy8(i,j,1)
      enddo
      enddo
   endif
!
   if (nz.gt.1) then!3D
      if(ilag.eq.1)call forcage_flugrange_x(uz,xi,xf,nobjx,ny,nz,0)!IBM
      call derxx (sy1,uz,di1,sx,sfxp,ssxp,swxp,nx,ny,nz,1)
      if (istret.ne.0) then!raffinement
         if(ilag.eq.1)call forcage_flugrange_y(uz,yi,yf,nobjy,nx,nz,0)!IBM
         call deryy (sy4,uz,di1,di2,sy,sfyp,ssyp,swyp,nx,ny,nz,1)
         if(ilag.eq.1)call forcage_flugrange_y(uz,yi,yf,nobjy,nx,nz,0)!IBM
         call dery (sy5,uz,di1,di2,sy,ffyp,fsyp,fwyp,ppy,nx,ny,nz,1)
         do k=1,nz
         do j=1,ny
         do i=1,nx
            sy4(i,j,k)=sy4(i,j,k)*pp2y(j)-pp4y(j)*sy5(i,j,k)
         enddo
         enddo
         enddo
      else!pas de raffinement
         if(ilag.eq.1)call forcage_flugrange_y(uz,yi,yf,nobjy,nx,nz,0)!IBM
         call deryy (sy4,uz,di1,di2,sy,sfyp,ssyp,swyp,nx,ny,nz,1)
      endif
      if(ilag.eq.1)call forcage_flugrange_z(uz,zi,zf,nobjz,nx,ny,0)!IBM
      call derzz (sy6,uz,di1,sz,sfz ,ssz ,swz ,nx,ny,nz,0)
      sy9(:,:,:)=xnu*(sy1(:,:,:)+sy4(:,:,:)+sy6(:,:,:))-sy9(:,:,:)
   endif
return
end subroutine convdiff
