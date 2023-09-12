!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine gene_epsi_2D(epsi,time,endix)
   USE param
   USE variables
   implicit none
!
   real(8),dimension(nx,ny,nz)      :: epsi
   integer,dimension(ny,nz)         :: nobjxraf
   integer,dimension(nx,nz)         :: nobjyraf
   real(8),dimension(  nxraf,ny,nz) :: xepsi
   real(8),dimension(  nx,nyraf,nz) :: yepsi
   real(8),dimension(        nyraf) :: ypraf
   real(8)                          :: dxraf,time
   integer                          :: i,j,k,endix
   integer                          :: ii,jj
   integer                          :: inum,jnum
   integer                          :: ibug,jbug
   integer                          :: iobj,jobj
   integer                          :: iflu,jflu
   integer                          :: isol,jsol
   integer                          :: iraf,jraf
   integer                          :: nobjxmax ,nobjymax
   integer                          :: nobjxmaxraf,nobjymaxraf
   integer                          :: idebraf,jdebraf
   integer                          :: ifinraf,jfinraf
   character(len=4) suffixe
   integer                          :: numvis
!
!  générer 'epsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
   epsi(:,:,:)=0.
   if (endix.eq.1) then
     call forcage22(epsi,nx,ny,yp,dx,1.,time,1)
   else
     call forcage22(epsi,nx,ny,yp,dx,1.,time,0)
   endif
!
!  générer 'xepsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
   if(nclx.eq.0)then
      dxraf =xlx/nxraf
   elseif(nclx.eq.1.or.nclx.eq.2)then
      dxraf =xlx/(nxraf-1)
   endif
   xepsi(:,:,:)=0.
   call forcage22(xepsi,nxraf,ny,yp,dxraf,1.,time,0)
!
!  générer 'yepsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
   do j=1,ny-1
      do jraf=1,nraf
         ypraf(jraf+nraf*(j-1))=yp(j)+(jraf-1)*(yp(j+1)-yp(j))/nraf
      enddo
   enddo
   if(ncly.ne.0)ypraf(nyraf)=yp(ny)
   yepsi(:,:,:)=0.
   call forcage22(yepsi,nx,nyraf,ypraf,dx,1.,time,0)
!
!  Search fro max Number of obstacles in 1D:
!  x-direction:
   nobjx(:,:)=0
   nobjxmax=0
   do k=1,nz
   do j=1,ny
      inum=0
      if(epsi(1,j,k).eq.1.)then
         inum=1
         nobjx(j,k)=1
      endif

      do i=1,nx-1
         if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.1.)then
            inum=inum+1
            nobjx(j,k)=nobjx(j,k)+1
         endif
      enddo
      if(inum.gt.nobjxmax)then
         nobjxmax=inum
      endif
   enddo
   enddo
!         print*,nobjxmax
!
   nobjxraf(:,:)=0
   ibug=0
   nobjxmaxraf=0
   inum=0
   do k=1,nz
   do j=1,ny
      inum=0
      if(xepsi(1,j,k).eq.1.)then
         inum=1
         nobjxraf(j,k)=1
      endif
      do i=1,nxraf-1
         if(xepsi(i,j,k).eq.0..and.xepsi(i+1,j,k).eq.1.)then
            inum=inum+1
            nobjxraf(j,k)=nobjxraf(j,k)+1
         endif
      enddo
      if(inum.gt.nobjxmaxraf)then
         nobjxmaxraf=inum
      endif
      if(nobjx(j,k).ne.nobjxraf(j,k))then
         ibug=ibug+1
      endif
   enddo
   enddo
!
!  y-direction:
   nobjy(:,:)=0
   nobjymax=0
   do k=1,nz
   do i=1,nx
      jnum=0
      if(epsi(i,1,k).eq.1.)then
         jnum=1
         nobjy(i,k)=1
      endif
      do j=1,ny-1
         if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.1.)then
            jnum=jnum+1
            nobjy(i,k)=nobjy(i,k)+1
         endif
      enddo
      if(jnum.gt.nobjymax)then
         nobjymax=jnum
      endif
   enddo
   enddo

!
   nobjyraf(:,:)=0
   jbug=0
   nobjymaxraf=0
   jnum=0
   do k=1,nz
   do i=1,nx
      jnum=0
      if(yepsi(i,1,k).eq.1.)then
         jnum=1
         nobjyraf(i,k)=1
      endif
      do j=1,nyraf-1
         if(yepsi(i,j,k).eq.0..and.yepsi(i,j+1,k).eq.1.)then
            jnum=jnum+1
            nobjyraf(i,k)=nobjyraf(i,k)+1
         endif
      enddo
      if(jnum.gt.nobjymaxraf)then
         nobjymaxraf=jnum
      endif
      if(nobjy(i,k).ne.nobjyraf(i,k))then
         jbug=jbug+1
      endif
   enddo
   enddo
!
!  Find approximate position of Boundaries:
!  x-direction:
   do k=1,nz
   do j=1,ny
      inum=0
      if(xepsi(1,j,k).eq.1.)then
         inum=inum+1
         xi(inum,j,k)=0.
      endif
      do i=1,nxraf-1
         if(xepsi(i,j,k).eq.0..and.xepsi(i+1,j,k).eq.1.)then
            inum=inum+1
            xi(inum,j,k)=dxraf*(i-1)+dxraf/2.
         elseif(xepsi(i,j,k).eq.1..and.xepsi(i+1,j,k).eq.0.)then
            xf(inum,j,k)=dxraf*(i-1)+dxraf/2.
         endif
      enddo
      if(xepsi(nxraf,j,k).eq.1.)then
         xf(inum,j,k)=xlx
      endif
   enddo
   enddo
!
!  Mesh cpnflict between refined/unrefined mesh:
   if(ibug.ne.0)then
      do k=1,nz
      do j=1,ny
         if(nobjx(j,k).ne.nobjxraf(j,k))then
            iobj=0
            if(epsi(1,j,k).eq.1.)iobj=iobj+1
            do i=1,nx-1
               if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.1.)iobj=iobj+1
               if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.0.)iflu=1
               if(epsi(i,j,k).eq.1..and.epsi(i+1,j,k).eq.1.)isol=1
               do iraf=1,nraf
                  if(xepsi(iraf+nraf*(i-1)  ,j,k).eq.0..and.&
                     xepsi(iraf+nraf*(i-1)+1,j,k).eq.1.)idebraf=iraf+nraf*(i-1)+1
                  if(xepsi(iraf+nraf*(i-1)  ,j,k).eq.1..and.&
                     xepsi(iraf+nraf*(i-1)+1,j,k).eq.0.)ifinraf=iraf+nraf*(i-1)+1
               enddo
               if(idebraf.ne.0.and.ifinraf.ne.0.and.&
                  idebraf.lt.ifinraf.and.iflu.eq.1)then
                  iobj=iobj+1
                  do ii=iobj,nobjmax-1
                     xi(ii,j,k)=xi(ii+1,j,k)
                     xf(ii,j,k)=xf(ii+1,j,k)
                  enddo
                  iobj=iobj-1
               endif
               if(idebraf.ne.0.and.ifinraf.ne.0.and.&
                  idebraf.gt.ifinraf.and.isol.eq.1)then
                  iobj=iobj+1
                  do ii=iobj,nobjmax-1
                     xi(ii,j,k)=xi(ii+1,j,k)
                  enddo
                  iobj=iobj-1
                  do ii=iobj,nobjmax-1
                     xf(ii,j,k)=xf(ii+1,j,k)
                  enddo
               endif
               idebraf=0
               ifinraf=0
               iflu=0
            enddo
         endif
      enddo
      enddo
   endif
!
!  y-direction:
   do k=1,nz
   do i=1,nx
      jnum=0
      if(yepsi(i,1,k).eq.1.)then
         jnum=jnum+1
         yi(jnum,i,k)=0.
      endif
      do j=1,nyraf-1
         if(yepsi(i,j,k).eq.0..and.yepsi(i,j+1,k).eq.1.)then
!         print*,'1111111111111111'
            jnum=jnum+1
            yi(jnum,i,k)=ypraf(j)+(ypraf(j+1)-ypraf(j))/2.
         elseif(yepsi(i,j,k).eq.1..and.yepsi(i,j+1,k).eq.0.)then
            yf(jnum,i,k)=ypraf(j)+(ypraf(j+1)-ypraf(j))/2.
         endif
      enddo
!      print*,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
      if(yepsi(i,nyraf,k).eq.1.)then
         yf(jnum,i,k)=yly
      endif
!      print*,yf(1:jnum,i,k),'yf'
!      print*,yi(1:jnum,i,k),'yi'
   enddo
   enddo
!
!  conflit maillage raffiné/non-raffiné en y :
   if(jbug.ne.0)then
      do k=1,nz
      do i=1,nx
         if(nobjy(i,k).ne.nobjyraf(i,k))then
            jobj=0
            if(epsi(i,1,k).eq.1.)jobj=jobj+1
            do j=1,ny-1
               if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.1.)jobj=jobj+1
               if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.0.)jflu=1
               if(epsi(i,j,k).eq.1..and.epsi(i,j+1,k).eq.1.)jsol=1
               do jraf=1,nraf
                  if(yepsi(i,jraf+nraf*(j-1)  ,k).eq.0..and.&
                     yepsi(i,jraf+nraf*(j-1)+1,k).eq.1.)jdebraf=jraf+nraf*(j-1)+1
                  if(yepsi(i,jraf+nraf*(j-1)  ,k).eq.1..and.&
                     yepsi(i,jraf+nraf*(j-1)+1,k).eq.0.)jfinraf=jraf+nraf*(j-1)+1
               enddo
               if(jdebraf.ne.0.and.jfinraf.ne.0.and.&
                  jdebraf.lt.jfinraf.and.jflu.eq.1)then
                  jobj=jobj+1
                  do jj=jobj,nobjmax-1
                     yi(jj,i,k)=yi(jj+1,i,k)
                     yf(jj,i,k)=yf(jj+1,i,k)
                  enddo
                  jobj=jobj-1
               endif
               if(jdebraf.ne.0.and.jfinraf.ne.0.and.&
                  jdebraf.gt.jfinraf.and.jsol.eq.1)then
                  jobj=jobj+1
                  do jj=jobj,nobjmax-1
                     yi(jj,i,k)=yi(jj+1,i,k)
                  enddo
                  jobj=jobj-1
                  do jj=jobj,nobjmax-1
                     yf(jj,i,k)=yf(jj+1,i,k)
                  enddo
               endif
               jdebraf=0
               jfinraf=0
               jflu=0
            enddo
         endif
      enddo
      enddo
   endif
!
   return
end subroutine gene_epsi_2D
!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine verif_epsi(epsi)
!
USE param
USE variables
   implicit none
!
   real(8),dimension(nx,ny,nz) :: epsi
   integer                     :: i,j,k
   integer                     :: inum ,jnum ,knum
   integer                     :: iflu ,jflu ,kflu
   integer                     :: ising,jsing,ksing,itest
!
!  Repérer les singularités sur epsilon :
!  balayage sur x :
   nxipif(:,:,:)=npif
   nxfpif(:,:,:)=npif
   ising=0
   do k=1,nz
   do j=1,ny
      inum=0
      iflu=0
      if(epsi(1,j,k).eq.1.)inum=inum+1
      if(epsi(1,j,k).eq.0.)iflu=iflu+1
      do i=2,nx
         if(epsi(i  ,j,k).eq.0.)iflu=iflu+1
         if(epsi(i-1,j,k).eq.0..and.&
            epsi(i  ,j,k).eq.1.)then
            inum=inum+1
            if(inum.eq.1)then
               if(izap.eq.1)nxipif(inum  ,j,k)=iflu-1
               if(izap.eq.0)nxipif(inum  ,j,k)=iflu
               if(izap.eq.1.and.iflu-1.lt.npif)ising=ising+1
               if(izap.eq.0.and.iflu  .lt.npif)ising=ising+1
               iflu=0
            else
               if(izap.eq.1)nxipif(inum  ,j,k)=iflu-1
               if(izap.eq.0)nxipif(inum  ,j,k)=iflu
               if(izap.eq.1)nxfpif(inum-1,j,k)=iflu-1
               if(izap.eq.0)nxfpif(inum-1,j,k)=iflu
               if(izap.eq.1.and.iflu-1.lt.npif)ising=ising+1
               if(izap.eq.0.and.iflu  .lt.npif)ising=ising+1
               iflu=0
            endif
         endif
         if(epsi(i,j,k).eq.1.)iflu=0
      enddo
      if(epsi(nx,j,k).eq.0..and.izap.eq.1)then
         nxfpif(inum,j,k)=iflu-1
         if(iflu-1.lt.npif)ising=ising+1
      endif
      if(epsi(nx,j,k).eq.0..and.izap.eq.0)then
         nxfpif(inum,j,k)=iflu
         if(iflu  .lt.npif)ising=ising+1
      endif
   enddo
   enddo
!
!  balayage sur y :
   nyipif(:,:,:)=npif
   nyfpif(:,:,:)=npif
   jsing=0
   do k=1,nz
   do i=1,nx
      jnum=0
      jflu=0
      if(epsi(i,1,k).eq.1.)jnum=jnum+1
      if(epsi(i,1,k).eq.0.)jflu=jflu+1
      do j=2,ny
         if(epsi(i,j  ,k).eq.0.)jflu=jflu+1
         if(epsi(i,j-1,k).eq.0..and.&
            epsi(i,j  ,k).eq.1.)then
            jnum=jnum+1
            if(jnum.eq.1)then
               if(izap.eq.1)nyipif(jnum  ,i,k)=jflu-1
               if(izap.eq.0)nyipif(jnum  ,i,k)=jflu
               if(izap.eq.1.and.jflu-1.lt.npif)jsing=jsing+1
               if(izap.eq.0.and.jflu  .lt.npif)jsing=jsing+1
               jflu=0
            else
               if(izap.eq.1)nyipif(jnum  ,i,k)=jflu-1
               if(izap.eq.0)nyipif(jnum  ,i,k)=jflu
               if(izap.eq.1)nyfpif(jnum-1,i,k)=jflu-1
               if(izap.eq.0)nyfpif(jnum-1,i,k)=jflu
               if(izap.eq.1.and.jflu-1.lt.npif)jsing=jsing+1
               if(izap.eq.0.and.jflu  .lt.npif)jsing=jsing+1
               jflu=0
            endif
         endif
         if(epsi(i,j,k).eq.1.)jflu=0
      enddo
      if(epsi(i,ny,k).eq.0..and.izap.eq.1)then
         nyfpif(jnum,i,k)=jflu-1
         if(jflu-1.lt.npif)jsing=jsing+1
      endif
      if(epsi(i,ny,k).eq.0..and.izap.eq.0)then
         nyfpif(jnum,i,k)=jflu
         if(jflu  .lt.npif)jsing=jsing+1
      endif
   enddo
   enddo
!stop
!
!  balayage sur z :
   if(nz.gt.1)then
      nzipif(:,:,:)=npif
      nzfpif(:,:,:)=npif
      ksing=0
      do j=1,ny
      do i=1,nx
         knum=0
         kflu=0
         if(epsi(i,j,1).eq.1.)knum=knum+1
         if(epsi(i,j,1).eq.0.)kflu=kflu+1
         do k=2,nz
            if(epsi(i,j,k  ).eq.0.)kflu=kflu+1
            if(epsi(i,j,k-1).eq.0..and.&
               epsi(i,j,k  ).eq.1.)then
               knum=knum+1
               if(knum.eq.1)then
                  if(izap.eq.1)nzipif(knum  ,i,j)=kflu-1
                  if(izap.eq.0)nzipif(knum  ,i,j)=kflu
                  if(izap.eq.1.and.kflu-1.lt.npif)ksing=ksing+1
                  if(izap.eq.0.and.kflu  .lt.npif)ksing=ksing+1
                  kflu=0
               else
                  if(izap.eq.1)nzipif(knum  ,i,j)=kflu-1
                  if(izap.eq.0)nzipif(knum  ,i,j)=kflu
                  if(izap.eq.1)nzfpif(knum-1,i,j)=kflu-1
                  if(izap.eq.0)nzfpif(knum-1,i,j)=kflu
                  if(izap.eq.1.and.kflu-1.lt.npif)ksing=ksing+1
                  if(izap.eq.0.and.kflu  .lt.npif)ksing=ksing+1
                  kflu=0
               endif
            endif
            if(epsi(i,j,k).eq.1.)kflu=0
         enddo
         if(epsi(i,j,nz).eq.0..and.izap.eq.1)then
            nzfpif(knum,i,j)=kflu-1
            if(kflu-1.lt.npif)ksing=ksing+1
         endif
         if(epsi(i,j,nz).eq.0..and.izap.eq.0)then
            nzfpif(knum,i,j)=kflu
            if(kflu  .lt.npif)ksing=ksing+1
         endif
      enddo
      enddo
   endif  
!
   return
end subroutine verif_epsi
!
!
!---------------------------------------------------------------------------
!*      *      *      *      *      *      *      *      *      *      *    
!*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *  
!* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
!*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
!************* subroutines de forçage / polynôme de Lagrange ***************
!*** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** ***
!* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
!*   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *   *  
!*      *      *      *      *      *      *      *      *      *      *    
!---------------------------------------------------------------------------
!
!
!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine forcage_flugrange_x(u,xini,xfin,nobj,nyj,nzk,lind)
!
USE param
USE variables
USE IBM
!
implicit none
!
   real(8),dimension(nx,nyj,nzk) :: u
   real(8),dimension(20,nyj,nzk) :: xini,xfin
   integer,dimension   (nyj,nzk) :: nobj
   integer                       :: nyj,nzk
   integer                       :: i,j,k,itest
   integer                       :: ix              != position du point "zappé"
   integer                       :: ipif,ipol,nxpif
   integer                       :: ipoli,ipolf     != positions Initiales et Finales du POLynôme considéré
   real(8)                       :: xpol,ypol,dypol !|variables concernant les polynômes
   real(8),dimension(10)         :: xa,ya           !|de Lagrange. A mettre impérativement en 
   integer                       :: ia,na,lind,inxi,inxf           !|double précision
   
   real(8) :: ceyy1, ceyy2, radiu, angle, fr, theta, y1, y2
   real(8) :: cexx, ceyy, omega1, omega2, y0
   real(8) :: Area, Area1, Area2, Area3, Area4, Area5, TArea
   real(8) :: Ax, Ay, Bx, By, Ex, Ey, Fx, Fy
   real(8) :: Axr, Ayr, Bxr, Byr, Exr, Eyr, Fxr, Fyr
   real(8) :: Ax2, Ay2, Bx2, By2, Ex2, Ey2, Fx2, Fy2
   real(8) :: Axr2, Ayr2, Bxr2, Byr2, Exr2, Eyr2, Fxr2, Fyr2
   real(8) :: x1, x2, alfat, alfab
   real(8) :: ceyy3, ceyy4, cexx4, ahmlength, ahmwidth   
   
!
! Initialise Arrays
xa(:)=0.
ya(:)=0.

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
 omega1 = (alfat_transition(2) - alfat_transition(1)) / ((ifin - idebut + 1)*dt) ! Top Flap Rotational Velocity in deg/s
 omega2 = (alfab_transition(2) - alfab_transition(1)) / ((ifin - idebut + 1)*dt) ! Bottom Flap Rotational Velocity in deg/s
 
 alfat = omega1 * (t - (idebut - 2)*dt) + alfat_transition(1)   ! Angle for top flap in deg
 alfab = omega2 * (t - (idebut - 2)*dt) + alfab_transition(1)   ! Angle for bottom flap in deg

! Constants for flaps
 cexx = cex + ahmlength                 ! Centre of both Flaps (x-dir)
 ceyy = ceyy3 + radiu -thick*0.5                  ! Centre of top Flap (y-dir)
 ceyy2 = ceyy4 - radiu +thick*0.5                 ! Centre of Bot Flap (y-dir)

! Rectangle Points (Top)
 Ax = cexx
 Ay = ceyy + 0.5*thick
 Bx = cexx + length
 By = ceyy + 0.5*thick
 Ex = cexx + length
 Ey = ceyy - 0.5*thick
 Fx = cexx
 Fy = ceyy - 0.5*thick
 
! Rectangle Points (Bottom)
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

! Rotate Rectangle Points (Top)
   Axr=cos(pi*alfat/180.)*(Ax-cexx)-sin(pi*alfat/180.)*(Ay-ceyy) + cexx
   Ayr=sin(pi*alfat/180.)*(Ax-cexx)+cos(pi*alfat/180.)*(Ay-ceyy) + ceyy
   Bxr=cos(pi*alfat/180.)*(Bx-cexx)-sin(pi*alfat/180.)*(By-ceyy) + cexx
   Byr=sin(pi*alfat/180.)*(Bx-cexx)+cos(pi*alfat/180.)*(By-ceyy) + ceyy
   Exr=cos(pi*alfat/180.)*(Ex-cexx)-sin(pi*alfat/180.)*(Ey-ceyy) + cexx
   Eyr=sin(pi*alfat/180.)*(Ex-cexx)+cos(pi*alfat/180.)*(Ey-ceyy) + ceyy
   Fxr=cos(pi*alfat/180.)*(Fx-cexx)-sin(pi*alfat/180.)*(Fy-ceyy) + cexx
   Fyr=sin(pi*alfat/180.)*(Fx-cexx)+cos(pi*alfat/180.)*(Fy-ceyy) + ceyy

! Rotate Rectangle Points (Top)
   Axr2=cos(pi*alfab/180.)*(Ax2-cexx)-sin(pi*alfab/180.)*(Ay2-ceyy2) + cexx
   Ayr2=sin(pi*alfab/180.)*(Ax2-cexx)+cos(pi*alfab/180.)*(Ay2-ceyy2) + ceyy2
   Bxr2=cos(pi*alfab/180.)*(Bx2-cexx)-sin(pi*alfab/180.)*(By2-ceyy2) + cexx
   Byr2=sin(pi*alfab/180.)*(Bx2-cexx)+cos(pi*alfab/180.)*(By2-ceyy2) + ceyy2
   Exr2=cos(pi*alfab/180.)*(Ex2-cexx)-sin(pi*alfab/180.)*(Ey2-ceyy2) + cexx
   Eyr2=sin(pi*alfab/180.)*(Ex2-cexx)+cos(pi*alfab/180.)*(Ey2-ceyy2) + ceyy2
   Fxr2=cos(pi*alfab/180.)*(Fx2-cexx)-sin(pi*alfab/180.)*(Fy2-ceyy2) + cexx
   Fyr2=sin(pi*alfab/180.)*(Fx2-cexx)+cos(pi*alfab/180.)*(Fy2-ceyy2) + ceyy2


!  double-boucle (y,z) :
   do k=1,nzk
   do j=1,nyj
      y1=yp(j)
      if(nobj(j,k).ne.0)then
         ia=0
         do i=1,nobj(j,k)          !boucle sur le nombre d'objets par (j,k)
         !  1st Boundary
            nxpif=npif
            ia=ia+1
            xa(ia)=xini(i,j,k)              ! Approximate BC Position
            x1 = xa(ia)
            ! Square/Ahmed Body + Flaps
            if (x1.lt.cex+ahmlength) then   ! Ahmed/Square
               ubcx = 0.0
               ubcy = 0.0
            else                                ! Flaps
               if(y1.gt.cey) then
                  ubcx=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
                  ubcy=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
               else
                  ubcx=-sin(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
                  ubcy=cos(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
               endif   
            endif
            
            if (lind.eq.1) then
                ya(ia)=ubcx
            elseif (lind.eq.2) then
                ya(ia)=ubcy
            elseif (lind.eq.3) then
                ya(ia)=ubcx*ubcx
            elseif (lind.eq.4) then
                ya(ia)=ubcy*ubcy
            elseif (lind.eq.5) then
                ya(ia)=ubcx*ubcy
            endif
            if(xini(i,j,k).gt.0.)then   ! Immersed Object
               inxi=0
               ix=xini(i,j,k)/dx+1
               ipoli=ix+1
               if(nxipif(i,j,k).lt.npif)nxpif=nxipif(i,j,k)
               do ipif=1,nxpif
                  ia=ia+1
                  if(izap.eq.1)then!zapping
                     xa(ia)=(ix-1)*dx-ipif*dx
                     ya(ia)=u(ix-ipif,j,k)
                  else             !no zapping
                     xa(ia)=(ix-1)*dx-(ipif-1)*dx
                     ya(ia)=u(ix-ipif+1,j,k)
                  endif
               enddo
            else                       ! Immersed Object at the Inflow
               inxi=1
               ipoli=1
               ix=xini(i,j,k)/dx
               ipoli=ix+1
               if(nxipif(i,j,k).lt.npif)nxpif=nxipif(i,j,k)
               do ipif=1,nxpif
                  ia=ia+1
                  if(izap.eq.1)then  !zapping
                     xa(ia)=(ix-1)*dx-ipif*dx
                     ya(ia)=0.
                  else               !no zapping
                     xa(ia)=(ix-1)*dx-(ipif-1)*dx
                     ya(ia)=0.
                  endif              
                  
               enddo
            endif
!   
         !  2nd Boundary
            nxpif=npif
            ia=ia+1
            xa(ia)=xfin(i,j,k)              ! Approximate BC Position
            x1 = xa(ia)
            ! Square/Ahmed Body + Flaps
            if (x1.lt.cex+ahmlength) then   ! Ahmed/Square
               ubcx = 0.0
               ubcy = 0.0
            else                                ! Flaps
               if(y1.gt.cey) then
                  ubcx=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
                  ubcy=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
               else
                  ubcx=-sin(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
                  ubcy=cos(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
               endif
            endif
            
            if (lind.eq.1) then
                ya(ia)=ubcx
            elseif (lind.eq.2) then
                ya(ia)=ubcy
            elseif (lind.eq.3) then
                ya(ia)=ubcx*ubcx
            elseif (lind.eq.4) then
                ya(ia)=ubcy*ubcy
            elseif (lind.eq.5) then
                ya(ia)=ubcx*ubcy
            endif
            if(xfin(i,j,k).lt.xlx)then    ! Immersed Object
               inxf=0
               ix=(xfin(i,j,k)+dx)/dx+1
               ipolf=ix-1
               if(nxfpif(i,j,k).lt.npif)nxpif=nxfpif(i,j,k)
               do ipif=1,nxpif
                  ia=ia+1
                  if(izap.eq.1)then      ! zapping
                     xa(ia)=(ix-1)*dx+ipif*dx
                     ya(ia)=u(ix+ipif,j,k)
                  else                   ! no zapping
                     xa(ia)=(ix-1)*dx+(ipif-1)*dx
                     ya(ia)=u(ix+ipif-1,j,k)
                  endif
               enddo
            else
               inxf=1
               ipolf=nx
               ix=(xfin(i,j,k)+dx)/dx+1
               ipolf=ix-1
               if(nxfpif(i,j,k).lt.npif)nxpif=nxfpif(i,j,k)
               do ipif=1,nxpif
                  ia=ia+1
                  if(izap.eq.1)then      ! zapping
                     xa(ia)=(ix-1)*dx+ipif*dx
                     ya(ia)=0.
                  else                   ! no zapping
                     xa(ia)=(ix-1)*dx+(ipif-1)*dx
                     ya(ia)=0.
                  endif
               enddo
            endif
         !  cas (très) particulier (ne marche que pour les frontières exactes)
            if(xini(i,j,k).eq.xfin(i,j,k))then
               u(ipoli-1,j,k)=0.     !ou u(ipolf+1,j,k)=0.
            endif
         !  calcul du polynôme
            na=ia
            do ipol=ipoli,ipolf
            if ((inxf.eq.1).and.(inxi.eq.1)) then
               u(ipol,j,k)=0.
            else
               xpol=dx*(ipol-1)
               call polint(xa,ya,na,xpol,ypol)
                u(ipol,j,k)=ypol
            endif 
            enddo
            ia=0
         enddo
      endif
   enddo
   enddo
!
   return
end subroutine forcage_flugrange_x
!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine forcage_flugrange_y(u,yini,yfin,nobj,nxi,nzk,lind)
!
USE param
USE variables
USE IBM
!
implicit none
!
   real(8),dimension(nxi,ny,nzk) :: u
   real(8),dimension(20,nxi,nzk) :: yini,yfin
   integer,dimension   (nxi,nzk) :: nobj
   integer                       :: nxi,nzk
   integer                       :: i,j,k
   integer                       :: jy              != position du point "zappé"
   integer                       :: jpif,jpol,nypif
   integer                       :: jpoli,jpolf     != positions Initiales et Finales du POLynôme considéré
   real(8)                       :: xpol,ypol,dypol !|variables concernant les polynômes
   real(8),dimension(10)         :: xa,ya           !|de Lagrange. A mettre impérativement en 
   integer                       :: ia,na,lind           !|double précision

   real(8) :: ceyy1, ceyy2, radiu, angle, fr, theta, y1, y2
   real(8) :: cexx, ceyy, omega1, omega2, y0
   real(8) :: Area, Area1, Area2, Area3, Area4, Area5, TArea
   real(8) :: Ax, Ay, Bx, By, Ex, Ey, Fx, Fy
   real(8) :: Axr, Ayr, Bxr, Byr, Exr, Eyr, Fxr, Fyr
   real(8) :: Ax2, Ay2, Bx2, By2, Ex2, Ey2, Fx2, Fy2
   real(8) :: Axr2, Ayr2, Bxr2, Byr2, Exr2, Eyr2, Fxr2, Fyr2
   real(8) :: x1, x2, alfat, alfab
   real(8) :: ceyy3, ceyy4, cexx4, ahmlength, ahmwidth   
!
! Initialise Arrays
xa(:)=0.
ya(:)=0.
!
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
 omega1 = (alfat_transition(2) - alfat_transition(1)) / ((ifin - idebut + 1)*dt) ! Top Flap Rotational Velocity in deg
 omega2 = (alfab_transition(2) - alfab_transition(1)) / ((ifin - idebut + 1)*dt)  ! Bottom Flap Rotational Velocity in deg
 
 alfat = omega1 * (t - (idebut - 2)*dt) + alfat_transition(1)   ! Angle for top flap in deg
 alfab = omega2 * (t - (idebut - 2)*dt) + alfab_transition(1)   ! Angle for bottom flap in deg

! Constants for flaps
 cexx = cex + ahmlength                 ! Centre of both Flaps (x-dir)
 ceyy = ceyy3 + radiu -thick*0.5                  ! Centre of top Flap (y-dir)
 ceyy2 = ceyy4 - radiu +thick*0.5                 ! Centre of Bot Flap (y-dir)

! Rectangle Points (Top)
 Ax = cexx
 Ay = ceyy + 0.5*thick
 Bx = cexx + length
 By = ceyy + 0.5*thick
 Ex = cexx + length
 Ey = ceyy - 0.5*thick
 Fx = cexx
 Fy = ceyy - 0.5*thick
 
! Rectangle Points (Bottom)
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

! Rotate Rectangle Points (Top)
   Axr=cos(pi*alfat/180.)*(Ax-cexx)-sin(pi*alfat/180.)*(Ay-ceyy) + cexx
   Ayr=sin(pi*alfat/180.)*(Ax-cexx)+cos(pi*alfat/180.)*(Ay-ceyy) + ceyy
   Bxr=cos(pi*alfat/180.)*(Bx-cexx)-sin(pi*alfat/180.)*(By-ceyy) + cexx
   Byr=sin(pi*alfat/180.)*(Bx-cexx)+cos(pi*alfat/180.)*(By-ceyy) + ceyy
   Exr=cos(pi*alfat/180.)*(Ex-cexx)-sin(pi*alfat/180.)*(Ey-ceyy) + cexx
   Eyr=sin(pi*alfat/180.)*(Ex-cexx)+cos(pi*alfat/180.)*(Ey-ceyy) + ceyy
   Fxr=cos(pi*alfat/180.)*(Fx-cexx)-sin(pi*alfat/180.)*(Fy-ceyy) + cexx
   Fyr=sin(pi*alfat/180.)*(Fx-cexx)+cos(pi*alfat/180.)*(Fy-ceyy) + ceyy

! Rotate Rectangle Points (Top)
   Axr2=cos(pi*alfab/180.)*(Ax2-cexx)-sin(pi*alfab/180.)*(Ay2-ceyy2) + cexx
   Ayr2=sin(pi*alfab/180.)*(Ax2-cexx)+cos(pi*alfab/180.)*(Ay2-ceyy2) + ceyy2
   Bxr2=cos(pi*alfab/180.)*(Bx2-cexx)-sin(pi*alfab/180.)*(By2-ceyy2) + cexx
   Byr2=sin(pi*alfab/180.)*(Bx2-cexx)+cos(pi*alfab/180.)*(By2-ceyy2) + ceyy2
   Exr2=cos(pi*alfab/180.)*(Ex2-cexx)-sin(pi*alfab/180.)*(Ey2-ceyy2) + cexx
   Eyr2=sin(pi*alfab/180.)*(Ex2-cexx)+cos(pi*alfab/180.)*(Ey2-ceyy2) + ceyy2
   Fxr2=cos(pi*alfab/180.)*(Fx2-cexx)-sin(pi*alfab/180.)*(Fy2-ceyy2) + cexx
   Fyr2=sin(pi*alfab/180.)*(Fx2-cexx)+cos(pi*alfab/180.)*(Fy2-ceyy2) + ceyy2


   do k=1,nzk
   do i=1,nxi
      x1=(i-1)*dx
      if(nobj(i,k).ne.0)then
         ia=0
         do j=1,nobj(i,k)          !boucle sur le nombre d'objets par (j,k)
         !  1st Boundary
            nypif=npif
            ia=ia+1
            xa(ia)=yini(j,i,k)              ! Approximate BC Position
            y1 = xa(ia)
            ! Square/Ahmed Body + Flaps
            if (x1.lt.cex+ahmlength) then   ! Ahmed/Square
               ubcx = 0.0
               ubcy = 0.0
            else                                ! Flaps
               if(y1.gt.cey) then
                  ubcx=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
                  ubcy=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
               else
                  ubcx=-sin(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
                  ubcy=cos(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
               endif 
            endif
            
            if (lind.eq.1) then
                ya(ia)=ubcx
            elseif (lind.eq.2) then
                ya(ia)=ubcy
            elseif (lind.eq.3) then
                ya(ia)=ubcx*ubcx
            elseif (lind.eq.4) then
                ya(ia)=ubcy*ubcy
            elseif (lind.eq.5) then
                ya(ia)=ubcx*ubcy
            endif
            if(yini(j,i,k).gt.0.)then   ! Immersed Object
               jy=1!jy=yi(j,i,k)/dy+1
               do while(yp(jy).lt.yini(j,i,k))
                  jy=jy+1
               enddo
               jy=jy-1
               jpoli=jy+1
               if(nyipif(j,i,k).lt.npif)nypif=nyipif(j,i,k)
               do jpif=1,nypif
                  ia=ia+1
                  if(izap.eq.1)then  ! zapping
                     xa(ia)=yp(jy-jpif)
                     ya(ia)=u(i,jy-jpif,k)
                  else               ! no zapping
                     xa(ia)=yp(jy-jpif+1)
                     ya(ia)=u(i,jy-jpif+1,k)
                  endif
               enddo
            else              ! Immersed Object at the Bottom Boundary
              jpoli=1                           
              jy=1
               do while(yp(jy).lt.yini(j,i,k))
                  jy=jy+1
               enddo
               jy=jy-1
               jpoli=jy+1
               if(nyipif(j,i,k).lt.npif) nypif=nyipif(j,i,k)
               do jpif=1,nypif
                  ia=ia+1
                  if(izap.eq.1)then         !zapping
                     xa(ia)=yp(1)-(jpif+1)*dy
                     ya(ia)=0.
                  else                      !no zapping
                     xa(ia)=yp(1)-(jpif*dy)
                     ya(ia)=0.
                  endif        
               enddo          
            endif
            
         !  2nd Boundary
            nypif=npif
            ia=ia+1
            xa(ia)=yfin(j,i,k)              ! Approximate BC Position
            y1 = xa(ia)
            ! Square/Ahmed Body + Flaps
            if (x1.lt.cex+ahmlength) then   ! Ahmed/Square
               ubcx = 0.0
               ubcy = 0.0
            else                                ! Flaps
               if(y1.gt.cey) then
                  ubcx=-sin(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
                  ubcy=cos(pi*alfat/180.)*omega1*sqrt((x1- cexx)**2 + (y1-ceyy)**2)*pi/180
               else
                  ubcx=-sin(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
                  ubcy=cos(pi*alfab/180.)*omega2*sqrt((x1- cexx)**2 + (y1-ceyy2)**2)*pi/180
               endif 
            endif
            
            if (lind.eq.1) then
                ya(ia)=ubcx
            elseif (lind.eq.2) then
                ya(ia)=ubcy
            elseif (lind.eq.3) then
                ya(ia)=ubcx*ubcx
            elseif (lind.eq.4) then
                ya(ia)=ubcy*ubcy
            elseif (lind.eq.5) then
                ya(ia)=ubcx*ubcy
            endif
            if(yfin(j,i,k).lt.yly)then   ! Immersed Object
               jy=1
               do while(yp(jy).lt.yfin(j,i,k))
                  jy=jy+1
               enddo
               jpolf=jy-1
               if(nyfpif(j,i,k).lt.npif)nypif=nyfpif(j,i,k)
               do jpif=1,nypif
                  ia=ia+1
                  if(izap.eq.1)then   ! zapping
                     xa(ia)=yp(jy+jpif)
                     ya(ia)=u(i,jy+jpif,k)
                  else                ! no zapping
                     xa(ia)=yp(jy+jpif-1)
                     ya(ia)=u(i,jy+jpif-1,k)
                  endif
               enddo
            else                       ! Immersed Object at the Top Boundary
               jpolf=ny
               jy=1
               do while(yp(jy).lt.yfin(j,i,k))
                  jy=jy+1
               enddo
               jpolf=jy-1
               if(nyfpif(j,i,k).lt.npif)nypif=nyfpif(j,i,k)
               do jpif=1,nypif
                  ia=ia+1
                  if(izap.eq.1)then   ! zapping
                     xa(ia)=yp(ny)+(jpif+1)*dy
                     ya(ia)=0.
                  else                ! no zapping
                     xa(ia)=yp(ny)+(jpif)*dy
                     ya(ia)=0.
                  endif
               enddo
            endif
!         !  cas (très) particulier (ne marche que pour les frontières exactes)
!            if(yini(i,j,k).eq.yfin(i,j,k))then
!               u(i,jpoli-1,k)=0.     !ou u(ipolf+1,j,k)=0.
!            endif
         !  calcul du polynôme
            na=ia
            do jpol=jpoli,jpolf
               xpol=yp(jpol)
               call polint(xa,ya,na,xpol,ypol)
               u(i,jpol,k)=ypol
            enddo
            ia=0
         enddo
      endif
   enddo
   enddo
!
return
end subroutine forcage_flugrange_y
!
!***************************************************************************
!***************************************************************************
!***************************************************************************

!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine forcage_flugrange_z(u,zini,zfin,nobj,nxi,nyj,lind)
!
USE param
USE variables
!
implicit none
!
   real(8),dimension(nxi,nyj,nz) :: u
   real(8),dimension(20,nxi,nyj) :: zini,zfin
   integer,dimension   (nxi,nyj) :: nobj
   integer                       :: nxi,nyj
   integer                       :: i,j,k
   integer                       :: kz              != position du point "zappé"
   integer                       :: kpif,kpol,nzpif
   integer                       :: kpoli,kpolf     != positions Initiales et Finales du POLynôme considéré
   real(8)                       :: xpol,ypol,dypol !|variables concernant les polynômes
   real(8),dimension(10)         :: xa,ya           !|de Lagrange. A mettre impérativement en 
   integer                       :: ia,na,lind           !|double précision
!
! Initialise Arrays
xa(:)=0.
ya(:)=0.
!  double-boucle (y,z) :
   do j=1,nyj
   do i=1,nxi
      if(nobj(i,j).ne.0)then
         ia=0
         do k=1,nobj(i,j)          !boucle sur le nombre d'objets par couple (i,j)
         !  1ère frontière
            nzpif=npif
            ia=ia+1
            xa(ia)=zini(k,i,j)
            ya(ia)=0.
            if(zini(k,i,j).gt.0.)then !pt.d'inf.flu. (objet immergé)
               kz=zini(k,i,j)/dz+1
               kpoli=kz+1
               if(nzipif(k,i,j).lt.npif)nzpif=nzipif(k,i,j)
               do kpif=1,nzpif
                  ia=ia+1
                  if(izap.eq.1)then!zapping
                     xa(ia)=(kz-1)*dz-kpif*dz
                     ya(ia)=u(i,j,kz-kpif)
                  else             !no zapping
                     xa(ia)=(kz-1)*dz-(kpif-1)*dz
                     ya(ia)=u(i,j,kz-kpif+1)
                  endif

               enddo
            else
               kpoli=1
            endif
         !  2ème frontière
            nzpif=npif
            ia=ia+1
            xa(ia)=zfin(k,i,j)
            ya(ia)=0.
            if(zfin(k,i,j).lt.zlz)then !objet immergé
               kz=(zfin(k,i,j)+dz)/dz+1
               kpolf=kz-1
               if(nzfpif(k,i,j).lt.npif)nzpif=nzfpif(k,i,j)
               do kpif=1,nzpif
                  ia=ia+1
                  if(izap.eq.1)then   ! zapping
                     xa(ia)=(kz-1)*dz+kpif*dz
                     ya(ia)=u(i,j,kz+kpif)
                  else                ! no zapping
                     xa(ia)=(kz-1)*dz+(kpif-1)*dz
                     ya(ia)=u(i,j,kz+kpif-1)
                  endif
               enddo
            else
               kpolf=nz
            endif
         !  cas (très) particulier (ne marche que pour les frontières exactes)
            if(zini(k,i,j).eq.zfin(k,i,j))then
               u(i,j,kpoli-1)=0.     
            endif
         !  calcul du polynôme
            na=ia
            do kpol=kpoli,kpolf
               xpol=dz*(kpol-1)
               call polint(xa,ya,na,xpol,ypol)
               u(i,j,kpol)=ypol
               if (lind.eq.1) then
                u(i,j,kpol)=ypol
               else
                u(i,j,kpol)=ypol
               endif 
            enddo
            ia=0
         enddo
      endif
   enddo
   enddo
!
   return
end subroutine forcage_flugrange_z
!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine polint(xa,ya,n,x,y)
   implicit none
!
   integer                 :: n,i,j,nc,nk
   real(8)                 :: x,y,xcc
   real(8),dimension(n)    :: xa,ya
   real(8),dimension(10)   :: xaa,yaa
   real(8)                 :: ypri,yprf
   real(8),dimension(n-2)  :: xx,alpha,cc,zz,ll,aa,yy
   real(8),dimension(n-3)  :: hh,dd,bb,mm
!
! Initialise Arrays
xaa(:)=0.
yaa(:)=0.
! Arrange Points in Correct Order (based on x-coor)
j=n/2
do i=1,n
        if (i.le.n/2) then
                xaa(i)=xa(j)
                yaa(i)=ya(j)
                j=j-1
        else
                xaa(i)=xa(i)
                yaa(i)=ya(i)
        endif
enddo   
!print*, xaa
!print*, yaa, 'Original Values'

!
ypri=(yaa(3)-yaa(1))/(xaa(3)-xaa(1))
yprf=(yaa(n)-yaa(n-2))/(xaa(n)-xaa(n-2))
!
nk=n-1
!
do i=2,nk
    yy(i-1)=yaa(i)
enddo
!
do i=2,nk
    xx(i-1)=xaa(i)
enddo
!
nc=nk-1
!
do i=1,nc
    aa(i)=yy(i)
enddo
!
do i=1,nc-1
    hh(i)=xx(i+1)-xx(i)
enddo
!
alpha(1)=(3.*(aa(2)-aa(1)))/hh(1) - 3.*ypri
alpha(nc)= 3.*yprf - 3.*(aa(nc)-aa(nc-1))/hh(nc-1)
!
do i=2,nc-1
    alpha(i)=(3./hh(i))*(aa(i+1)-aa(i))-(3./hh(i-1))*(aa(i)-aa(i-1))
enddo
ll(1)=2.*hh(1)
mm(1)=0.5
zz(1)=alpha(1)/ll(1)
!
do i=2,nc-1
    ll(i)=2.*(xx(i+1)-xx(i-1))-hh(i-1)*mm(i-1);
    mm(i)=hh(i)/ll(i);
    zz(i)=(alpha(i)-hh(i-1)*zz(i-1))/ll(i);
enddo
!
 ll(nc)=hh(nc-1)*(2.-mm(nc-1));
 zz(nc)=(alpha(nc)-hh(nc-1)*zz(nc-1))/ll(nc);
 cc(nc)=zz(nc);
!
do j=nc-1,1,-1
    cc(j)=zz(j)-mm(j)*cc(j+1);
    bb(j)=(aa(j+1)-aa(j))/hh(j)-(hh(j)/3.)*(cc(j+1)+2.*cc(j));
    dd(j)=(cc(j+1)-cc(j))/(3.*hh(j));
   
enddo
!
do j=1,nc
    xcc=x;
      if (j.eq.1) then
         if (xcc<=xx(j)) then
 print*,'Warning Error In This Thing Hahahaha'
          y= aa(j-1) + bb(j-1)*(xcc-xx(j-1)) + cc(j-1)*(xcc-xx(j-1))**2 + dd(j-1)*(xcc-xx(j-1))**3;
         endif
      else  
         if (xcc<=xx(j) .and. xcc>=xx(j-1)) then
          y= aa(j-1) + bb(j-1)*(xcc-xx(j-1)) + cc(j-1)*(xcc-xx(j-1))**2 + dd(j-1)*(xcc-xx(j-1))**3;
         endif
      endif
enddo 
!
return
end subroutine polint
!
!*******************************************************************
!
subroutine forcage22(epsi,nx,ny,yp,dmx,remp,idts,stox)
! 
!********************************************************************
USE param
USE IBM
!
implicit none
!
real(8),dimension(nx,ny,1) :: epsi
real(8) :: r,x,y,x0,z,x1,remp,dmx,xa,xb,func,idts,cexx2
real(8),dimension(ny)      :: yp
integer :: i,j,k,stox
real(8),dimension(nx) :: xx1
real(8),dimension(ny) :: yy1
integer :: nx,ny
real(8) :: ceyy1, ceyy2, radiu, angle, fr, theta, y1, y2
real(8) :: cexx, ceyy, omega1, omega2, y0
real(8) :: Area, Area1, Area2, Area3, Area4, Area5, TArea
real(8) :: Ax, Ay, Bx, By, Ex, Ey, Fx, Fy
real(8) :: Axr, Ayr, Bxr, Byr, Exr, Eyr, Fxr, Fyr
real(8) :: Ax2, Ay2, Bx2, By2, Ex2, Ey2, Fx2, Fy2
real(8) :: Axr2, Ayr2, Bxr2, Byr2, Exr2, Eyr2, Fxr2, Fyr2
real(8) :: x2, alfat, alfab
real(8) :: ceyy3, ceyy4, cexx4, ahmlength, ahmwidth
!
epsi(:,:,:)=0.
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
omega1 = (alfat_transition(2) - alfat_transition(1)) / ((ifin - idebut + 1)*dt) ! Top Flap Rotational Velocity in deg/s
omega2 = (alfab_transition(2) - alfab_transition(1)) / ((ifin - idebut + 1)*dt)  ! Bottom Flap Rotational Velocity in deg/s

alfat = omega1 * (t - (idebut - 2)*dt) + alfat_transition(1)   ! Angle for top flap in deg
alfab = omega2 * (t - (idebut - 2)*dt) + alfab_transition(1)   ! Angle for bottom flap in deg

! Constants for flaps
 cexx = cex + ahmlength                 ! Centre of both Flaps (x-dir)
 ceyy = ceyy3 + radiu -thick*0.5                  ! Centre of top Flap (y-dir)
 ceyy2 = ceyy4 - radiu +thick*0.5                 ! Centre of Bot Flap (y-dir)

! Rectangle Points (Top)
 Ax = cexx
 Ay = ceyy + 0.5*thick
 Bx = cexx + length
 By = ceyy + 0.5*thick
 Ex = cexx + length
 Ey = ceyy - 0.5*thick
 Fx = cexx
 Fy = ceyy - 0.5*thick
 
! Rectangle Points (Bottom)
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

! Rotate Rectangle Points (Top)
   Axr=cos(pi*alfat/180.)*(Ax-cexx)-sin(pi*alfat/180.)*(Ay-ceyy) + cexx
   Ayr=sin(pi*alfat/180.)*(Ax-cexx)+cos(pi*alfat/180.)*(Ay-ceyy) + ceyy
   Bxr=cos(pi*alfat/180.)*(Bx-cexx)-sin(pi*alfat/180.)*(By-ceyy) + cexx
   Byr=sin(pi*alfat/180.)*(Bx-cexx)+cos(pi*alfat/180.)*(By-ceyy) + ceyy
   Exr=cos(pi*alfat/180.)*(Ex-cexx)-sin(pi*alfat/180.)*(Ey-ceyy) + cexx
   Eyr=sin(pi*alfat/180.)*(Ex-cexx)+cos(pi*alfat/180.)*(Ey-ceyy) + ceyy
   Fxr=cos(pi*alfat/180.)*(Fx-cexx)-sin(pi*alfat/180.)*(Fy-ceyy) + cexx
   Fyr=sin(pi*alfat/180.)*(Fx-cexx)+cos(pi*alfat/180.)*(Fy-ceyy) + ceyy

! Rotate Rectangle Points (Top)
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
   x1=(i-1)*dmx

   if (x1.lt.cex+ahmlength) then         ! For the Ahmed Body
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
   else                                   ! For the Flaps
   
   ! Area of each triangle (Top)
   Area1 = abs(x1*(Byr-Eyr) + Bxr*(Eyr-y1) + Exr*(y1-Byr))*0.5
   Area2 = abs(Axr*(Byr-y1) + Bxr*(y1-Ayr) + x1*(Ayr-Byr))*0.5
   Area3 = abs(x1*(Fyr-Eyr) + Fxr*(Eyr-y1) + Exr*(y1-Fyr))*0.5
   Area4 = abs(Axr*(Fyr-y1) + Fxr*(y1-Ayr) + x1*(Ayr-Fyr))*0.5
   TArea = Area1 + Area2 + Area3 + Area4  
   if (TArea.le.Area+0.000001) then   ! Top Flap
      epsi(i,j,1)=1.
   endif    

   ! Area of each triangle (Bottom)
   Area1 = abs(x1*(Byr2-Eyr2) + Bxr2*(Eyr2-y1) + Exr2*(y1-Byr2))*0.5
   Area2 = abs(Axr2*(Byr2-y1) + Bxr2*(y1-Ayr2) + x1*(Ayr2-Byr2))*0.5
   Area3 = abs(x1*(Fyr2-Eyr2) + Fxr2*(Eyr2-y1) + Exr2*(y1-Fyr2))*0.5
   Area4 = abs(Axr2*(Fyr2-y1) + Fxr2*(y1-Ayr2) + x1*(Ayr2-Fyr2))*0.5
   TArea = Area1 + Area2 + Area3 + Area4  
   if (TArea.le.Area+0.000001) then   ! Bottom Flap
      epsi(i,j,1)=1.
   endif  

   endif
enddo
enddo
!
return
end subroutine forcage22
!
!***************************************************************************
!***************************************************************************
!***************************************************************************
!
subroutine forcing_fresh_x(u,xini,xfin,nobj,nyj,nzk,lind)
   !
   USE param
   USE variables
   !
   implicit none
   !
      real(8),dimension(nx,nyj,nzk) :: u
      real(8),dimension(20,nyj,nzk) :: xini,xfin
      integer,dimension   (nyj,nzk) :: nobj
      integer                       :: nyj,nzk
      integer                       :: i,j,k,itest
      integer                       :: ix              != position du point "zappé"
      integer                       :: ipif,ipol,nxpif
      integer                       :: ipoli,ipolf     != positions Initiales et Finales du POLynôme considéré
      real(8)                       :: xpol,ypol,dypol !|variables concernant les polynômes
      real(8),dimension(10)         :: xa,ya           !|de Lagrange. A mettre impérativement en 
      integer                       :: ia,na,lind,skip          !|double précision
   !
   ! Initialise Arrays
   xa(:)=0.
   ya(:)=0.
   !  double-boucle (y,z) :
      do k=1,nzk
      do j=1,nyj
         if(nobj(j,k).ne.0)then
            ia=0
            do i=1,nobj(j,k)          !boucle sur le nombre d'objets par (j,k)
            !  1st Boundary
               nxpif=npif
               ia=ia+1
               xa(ia)=xfin(i,j,k)
               if (lind.eq.1) then
                 ya(ia)=0.0
               else
                 ya(ia)=0.0
               endif
                  ix=xfin(i,j,k)/dx+1
                  ipoli=ix+1
                  if(nxipif(i,j,k).lt.npif)nxpif=nxipif(i,j,k)
                  do ipif=1,nxpif
                     ia=ia+1
                        xa(ia)=(ix-1)*dx-(ipif-1)*dx
                        ya(ia)=u(ix-ipif+1,j,k)
                  enddo
            !  2nd Boundary
               nxpif=npif
                  ix=(xfin(i,j,k)+dx)/dx+1
                  ipolf=ipoli+1       ! =ipoli -or- =ipoli+1    to interpolate (1) or (2) points accordingly
                  if(nxfpif(i,j,k).lt.npif)nxpif=nxfpif(i,j,k)
                  if (ipolf.eq.ipoli) then ! How many point to skip depending on how many points are interpolated
                     skip=1
                  else
                     skip=2
                  endif
                  do ipif=1,nxpif+1
                     ia=ia+1
                        xa(ia)=(ix-1)*dx+(ipif-1+skip)*dx
                        ya(ia)=u(ix+ipif-1+skip,j,k)
                  enddo
            !  calcul du polynôme
               na=ia
               do ipol=ipoli,ipolf
                  xpol=dx*(ipol-1)
                  call polintx(xa,ya,na,xpol,ypol)
   !               print*,u(ipol,j,k), 'Original'
                  u(ipol,j,k)=ypol
   !               print*,u(ipol,j,k), 'Updated'
               enddo
               ia=0
            enddo
         endif
      enddo
      enddo
   !
      return
   end subroutine forcing_fresh_x
   !
   !***************************************************************************
   !***************************************************************************
   !***************************************************************************
   !
   subroutine polintx(xa,ya,n,x,y)
      implicit none
   !
      integer                 :: n,i,j,nc,nk
      real(8)                 :: x,y,xcc
      real(8),dimension(n)    :: xa,ya
      real(8),dimension(10)   :: xaa,yaa
      real(8)                 :: ypri,yprf
      real(8),dimension(n-2)  :: xx,alpha,cc,zz,ll,aa,yy
      real(8),dimension(n-3)  :: hh,dd,bb,mm
   !
   ! Initialise Arrays
   xaa(:)=0.
   yaa(:)=0.
   ! Arrange Points in Correct Order (based on x-coor)
   j=n/2
   do i=1,n
           if (i.le.n/2) then
                   xaa(i)=xa(j)
                   yaa(i)=ya(j)
                   j=j-1
           else
                   xaa(i)=xa(i)
                   yaa(i)=ya(i)
           endif
   enddo   
   !
   ypri=(yaa(3)-yaa(1))/(xaa(3)-xaa(1))
   yprf=(yaa(n)-yaa(n-2))/(xaa(n)-xaa(n-2))
   !
   nk=n-1
   !
   do i=2,nk
       yy(i-1)=yaa(i)
   enddo
   !
   do i=2,nk
       xx(i-1)=xaa(i)
   enddo
   !
   nc=nk-1
   !
   do i=1,nc
       aa(i)=yy(i)
   enddo
   !
   do i=1,nc-1
       hh(i)=xx(i+1)-xx(i)
   enddo
   !
   alpha(1)=(3.*(aa(2)-aa(1)))/hh(1) - 3.*ypri
   alpha(nc)= 3.*yprf - 3.*(aa(nc)-aa(nc-1))/hh(nc-1)
   !
   do i=2,nc-1
       alpha(i)=(3./hh(i))*(aa(i+1)-aa(i))-(3./hh(i-1))*(aa(i)-aa(i-1))
   enddo
   ll(1)=2.*hh(1)
   mm(1)=0.5
   zz(1)=alpha(1)/ll(1)
   !
   do i=2,nc-1
       ll(i)=2.*(xx(i+1)-xx(i-1))-hh(i-1)*mm(i-1);
       mm(i)=hh(i)/ll(i);
       zz(i)=(alpha(i)-hh(i-1)*zz(i-1))/ll(i);
   enddo
   !
    ll(nc)=hh(nc-1)*(2.-mm(nc-1));
    zz(nc)=(alpha(nc)-hh(nc-1)*zz(nc-1))/ll(nc);
    cc(nc)=zz(nc);
   !
   do j=nc-1,1,-1
       cc(j)=zz(j)-mm(j)*cc(j+1);
       bb(j)=(aa(j+1)-aa(j))/hh(j)-(hh(j)/3.)*(cc(j+1)+2.*cc(j));
       dd(j)=(cc(j+1)-cc(j))/(3.*hh(j));
      
   enddo
   !
   do j=1,nc
       xcc=x;
         if (j.eq.1) then
            if (xcc<=xx(j)) then
    print*,'Warning Error In This Thing Hahahaha'
    stop
             y= aa(j-1) + bb(j-1)*(xcc-xx(j-1)) + cc(j-1)*(xcc-xx(j-1))**2 + dd(j-1)*(xcc-xx(j-1))**3;
            endif
         else  
            if (xcc<=xx(j) .and. xcc>=xx(j-1)) then
             y= aa(j-1) + bb(j-1)*(xcc-xx(j-1)) + cc(j-1)*(xcc-xx(j-1))**2 + dd(j-1)*(xcc-xx(j-1))**3;
            endif
         endif
   enddo 
   !print*,y, 'Final Result'
   !pause
   !print*,y
   !print*,x,'location'
   !
   return
   end subroutine polintx
   !***************************************************************************
   !***************************************************************************
   !***************************************************************************
   !
   subroutine gene_epsi_2Dn(epsi,time)
      USE param
      USE variables
      implicit none
   !
      real(8),dimension(nx,ny,nz)      :: epsi
      integer,dimension(ny,nz)         :: nobjxraf
      integer,dimension(nx,nz)         :: nobjyraf
      real(8),dimension(  nxraf,ny,nz) :: xepsi
      real(8),dimension(  nx,nyraf,nz) :: yepsi
      real(8),dimension(        nyraf) :: ypraf
      real(8)                          :: dxraf,time
      integer                          :: i,j,k
      integer                          :: ii,jj
      integer                          :: inum,jnum
      integer                          :: ibug,jbug
      integer                          :: iobj,jobj
      integer                          :: iflu,jflu
      integer                          :: isol,jsol
      integer                          :: iraf,jraf
      integer                          :: nobjxmax ,nobjymax
      integer                          :: nobjxmaxraf,nobjymaxraf
      integer                          :: idebraf,jdebraf
      integer                          :: ifinraf,jfinraf
      character(len=4) suffixe
      integer                          :: numvis
   !
   !  générer 'epsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
      epsi(:,:,:)=0.
      call forcage22n(epsi,nx,ny,yp,dx,1.,time,1)
   !
   !  générer 'xepsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
      if(nclx.eq.0)then
         dxraf =xlx/nxraf
      elseif(nclx.eq.1.or.nclx.eq.2)then
         dxraf =xlx/(nxraf-1)
      endif
      xepsi(:,:,:)=0.
      call forcage22n(xepsi,nxraf,ny,yp,dxraf,1.,time,0)
   !
   !  générer 'yepsi' : !new_IBM (créer sa/ses propre(s) subroutine(s))
      do j=1,ny-1
         do jraf=1,nraf
            ypraf(jraf+nraf*(j-1))=yp(j)+(jraf-1)*(yp(j+1)-yp(j))/nraf
         enddo
      enddo
      if(ncly.ne.0)ypraf(nyraf)=yp(ny)
      yepsi(:,:,:)=0.
      call forcage22n(yepsi,nx,nyraf,ypraf,dx,1.,time,0)
   !
   !  Search fro max Number of obstacles in 1D:
   !  x-direction:
      nobjx(:,:)=0
      nobjxmax=0
      do k=1,nz
      do j=1,ny
         inum=0
         if(epsi(1,j,k).eq.1.)then
            inum=1
            nobjx(j,k)=1
         endif
   
         do i=1,nx-1
            if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.1.)then
               inum=inum+1
               nobjx(j,k)=nobjx(j,k)+1
            endif
   !         if(epsi(i,j,k).eq.1..and.epsi(i+1,j,k).eq.0.)then         ! For mor than 2 boundaries
   !            inum=inum+1
   !            nobjx(j,k)=nobjx(j,k)+1
   !         endif        
         enddo
   !         if(epsi(nx,j,k).eq.1.)then
   !          inum=inum+1
   !          nobjx(j,k)=nobjx(j,k)+1
   !         endif
   !      print*,inum,'aaaaaaa'
   !      print*,nobjx(j,k)
         if(inum.gt.nobjxmax)then
            nobjxmax=inum
         endif
      enddo
      enddo
   !         print*,nobjxmax
   !
      nobjxraf(:,:)=0
      ibug=0
      nobjxmaxraf=0
      inum=0
      do k=1,nz
      do j=1,ny
         inum=0
         if(xepsi(1,j,k).eq.1.)then
            inum=1
            nobjxraf(j,k)=1
         endif
         do i=1,nxraf-1
            if(xepsi(i,j,k).eq.0..and.xepsi(i+1,j,k).eq.1.)then
               inum=inum+1
               nobjxraf(j,k)=nobjxraf(j,k)+1
            endif
   !         if(xepsi(i,j,k).eq.1..and.xepsi(i+1,j,k).eq.0.)then
   !            inum=inum+1
   !            nobjxraf(j,k)=nobjxraf(j,k)+1
   !         endif         
         enddo
   !       if(xepsi(nx,j,k).eq.1.)then
   !          inum=inum+1
   !          nobjxraf(j,k)=nobjxraf(j,k)+1
   !       endif      
   !      print*,inum,'aaaaaaa'
   !      print*,nobjxraf(j,k)
         if(inum.gt.nobjxmaxraf)then
            nobjxmaxraf=inum
         endif
         if(nobjx(j,k).ne.nobjxraf(j,k))then
            ibug=ibug+1
         endif
      enddo
      enddo
   !
   !  y-direction:
      nobjy(:,:)=0
      nobjymax=0
      do k=1,nz
      do i=1,nx
         jnum=0
         if(epsi(i,1,k).eq.1.)then
            jnum=1
            nobjy(i,k)=1
         endif
         do j=1,ny-1
            if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.1.)then
               jnum=jnum+1
               nobjy(i,k)=nobjy(i,k)+1
            endif
         enddo
         if(jnum.gt.nobjymax)then
            nobjymax=jnum
         endif
      enddo
      enddo
   
   !
      nobjyraf(:,:)=0
      jbug=0
      nobjymaxraf=0
      jnum=0
      do k=1,nz
      do i=1,nx
         jnum=0
         if(yepsi(i,1,k).eq.1.)then
            jnum=1
            nobjyraf(i,k)=1
         endif
         do j=1,nyraf-1
            if(yepsi(i,j,k).eq.0..and.yepsi(i,j+1,k).eq.1.)then
               jnum=jnum+1
               nobjyraf(i,k)=nobjyraf(i,k)+1
            endif
         enddo
         if(jnum.gt.nobjymaxraf)then
            nobjymaxraf=jnum
         endif
         if(nobjy(i,k).ne.nobjyraf(i,k))then
            jbug=jbug+1
         endif
      enddo
      enddo
   !
   !  Find approximate position of Boundaries:
   !  x-direction:
      do k=1,nz
      do j=1,ny
         inum=0
         if(xepsi(1,j,k).eq.1.)then
            inum=inum+1
            xi(inum,j,k)=0.
         endif
         do i=1,nxraf-1
            if(xepsi(i,j,k).eq.0..and.xepsi(i+1,j,k).eq.1.)then
               inum=inum+1
               xi(inum,j,k)=dxraf*(i-1)+dxraf/2.
            elseif(xepsi(i,j,k).eq.1..and.xepsi(i+1,j,k).eq.0.)then
               xf(inum,j,k)=dxraf*(i-1)+dxraf/2.
            endif
         enddo
         if(xepsi(nxraf,j,k).eq.1.)then
            xf(inum,j,k)=xlx
         endif
      enddo
      enddo
   !
   !  Mesh cpnflict between refined/unrefined mesh:
      if(ibug.ne.0)then
         do k=1,nz
         do j=1,ny
            if(nobjx(j,k).ne.nobjxraf(j,k))then
               iobj=0
               if(epsi(1,j,k).eq.1.)iobj=iobj+1
               do i=1,nx-1
                  if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.1.)iobj=iobj+1
                  if(epsi(i,j,k).eq.0..and.epsi(i+1,j,k).eq.0.)iflu=1
                  if(epsi(i,j,k).eq.1..and.epsi(i+1,j,k).eq.1.)isol=1
                  do iraf=1,nraf
                     if(xepsi(iraf+nraf*(i-1)  ,j,k).eq.0..and.&
                        xepsi(iraf+nraf*(i-1)+1,j,k).eq.1.)idebraf=iraf+nraf*(i-1)+1
                     if(xepsi(iraf+nraf*(i-1)  ,j,k).eq.1..and.&
                        xepsi(iraf+nraf*(i-1)+1,j,k).eq.0.)ifinraf=iraf+nraf*(i-1)+1
                  enddo
                  if(idebraf.ne.0.and.ifinraf.ne.0.and.&
                     idebraf.lt.ifinraf.and.iflu.eq.1)then
                     iobj=iobj+1
                     do ii=iobj,nobjmax-1
                        xi(ii,j,k)=xi(ii+1,j,k)
                        xf(ii,j,k)=xf(ii+1,j,k)
                     enddo
                     iobj=iobj-1
                  endif
                  if(idebraf.ne.0.and.ifinraf.ne.0.and.&
                     idebraf.gt.ifinraf.and.isol.eq.1)then
                     iobj=iobj+1
                     do ii=iobj,nobjmax-1
                        xi(ii,j,k)=xi(ii+1,j,k)
                     enddo
                     iobj=iobj-1
                     do ii=iobj,nobjmax-1
                        xf(ii,j,k)=xf(ii+1,j,k)
                     enddo
                  endif
                  idebraf=0
                  ifinraf=0
                  iflu=0
               enddo
            endif
         enddo
         enddo
      endif
   !
   !  y-direction:
      do k=1,nz
      do i=1,nx
         jnum=0
         if(yepsi(i,1,k).eq.1.)then
            jnum=jnum+1
            yi(jnum,i,k)=0.
         endif
         do j=1,nyraf-1
            if(yepsi(i,j,k).eq.0..and.yepsi(i,j+1,k).eq.1.)then
   !         print*,'1111111111111111'
               jnum=jnum+1
               yi(jnum,i,k)=ypraf(j)+(ypraf(j+1)-ypraf(j))/2.
            elseif(yepsi(i,j,k).eq.1..and.yepsi(i,j+1,k).eq.0.)then
               yf(jnum,i,k)=ypraf(j)+(ypraf(j+1)-ypraf(j))/2.
            endif
         enddo
   !      print*,'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
         if(yepsi(i,nyraf,k).eq.1.)then
            yf(jnum,i,k)=yly
         endif
   !      print*,yf(1:jnum,i,k),'yf'
   !      print*,yi(1:jnum,i,k),'yi'
      enddo
      enddo
   !
   !  conflit maillage raffiné/non-raffiné en y :
      if(jbug.ne.0)then
         do k=1,nz
         do i=1,nx
            if(nobjy(i,k).ne.nobjyraf(i,k))then
               jobj=0
               if(epsi(i,1,k).eq.1.)jobj=jobj+1
               do j=1,ny-1
                  if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.1.)jobj=jobj+1
                  if(epsi(i,j,k).eq.0..and.epsi(i,j+1,k).eq.0.)jflu=1
                  if(epsi(i,j,k).eq.1..and.epsi(i,j+1,k).eq.1.)jsol=1
                  do jraf=1,nraf
                     if(yepsi(i,jraf+nraf*(j-1)  ,k).eq.0..and.&
                        yepsi(i,jraf+nraf*(j-1)+1,k).eq.1.)jdebraf=jraf+nraf*(j-1)+1
                     if(yepsi(i,jraf+nraf*(j-1)  ,k).eq.1..and.&
                        yepsi(i,jraf+nraf*(j-1)+1,k).eq.0.)jfinraf=jraf+nraf*(j-1)+1
                  enddo
                  if(jdebraf.ne.0.and.jfinraf.ne.0.and.&
                     jdebraf.lt.jfinraf.and.jflu.eq.1)then
                     jobj=jobj+1
                     do jj=jobj,nobjmax-1
                        yi(jj,i,k)=yi(jj+1,i,k)
                        yf(jj,i,k)=yf(jj+1,i,k)
                     enddo
                     jobj=jobj-1
                  endif
                  if(jdebraf.ne.0.and.jfinraf.ne.0.and.&
                     jdebraf.gt.jfinraf.and.jsol.eq.1)then
                     jobj=jobj+1
                     do jj=jobj,nobjmax-1
                        yi(jj,i,k)=yi(jj+1,i,k)
                     enddo
                     jobj=jobj-1
                     do jj=jobj,nobjmax-1
                        yf(jj,i,k)=yf(jj+1,i,k)
                     enddo
                  endif
                  jdebraf=0
                  jfinraf=0
                  jflu=0
               enddo
            endif
         enddo
         enddo
      endif
   !
      return
   end subroutine gene_epsi_2Dn
   !
   !*******************************************************************
   !
   subroutine forcage22n(epsi,nx,ny,yp,dmx,remp,idts,stox)
   ! 
   !********************************************************************
   USE param
   USE IBM
   !
   implicit none
   !
   real(8),dimension(nx,ny,1) :: epsi
   real(8) :: r,x,y,x0,y0,z,x1,y1,remp,cexx,dmx,xa,xb,func,idts,cexx2
   real(8),dimension(ny)      :: yp
   integer :: i,j,k,stox
   real(8),dimension(nx) :: xx1
   real(8),dimension(ny) :: yy1
   integer :: nx,ny
   !
   epsi(:,:,:)=0.
   !
   do j=1,ny
   do i=1,nx
      x=((i-1)*dmx)
      y=(yp(j))
   !
    cexx2=15.0!-0.5*(t-dt)
   ! cexx2=30.
      r=sqrt((x-cexx2)*(x-cexx2)+(y-cey)*(y-cey)) 
      if (r-rads.ge.0.) cycle
   !
      epsi(i,j,1)=remp
   !
   enddo
   enddo
    cex = cexx2
   ! print*,cex, 'This is the new Time step'
   ! print*,cex, 'lllllllllalalalal', cexx2
   !
   !call paraview_3d(epsi,epsi,epsi)
   !stop
   !
   return
   end subroutine forcage22n
   !
   !***************************************************************************
   !***************************************************************************
   !***************************************************************************
   !
   subroutine fresh_idw_x(ux,uy,xini,xfin,nobj,nyj,nzk,epsik)
   !
   USE param
   USE variables
   !
   implicit none
   !
      real(8),dimension(nx,nyj,nzk) :: ux,uy,epsik
      real(8),dimension(20,nyj,nzk) :: xini,xfin
      integer,dimension   (nyj,nzk) :: nobj
      integer                       :: nyj,nzk
      integer                       :: i,j,k,itest
      integer                       :: ix              != position du point "zappé"
      integer                       :: ipif,ipol
      integer                       :: ipoli,ipolf     != positions Initiales et Finales du POLynôme considéré
      real(8)                       :: xpol,ypol,dypol !|variables concernant les polynômes
      real(8),dimension(10)         :: xa,ya           !|de Lagrange. A mettre impérativement en 
      integer                       :: ia,na,lind,skip          !|double précision
   !
   ! Initialise Arrays
   xa(:)=0.
   ya(:)=0.
   !  double-boucle (y,z) :
      do k=1,nzk
      do j=1,nyj
         if(nobj(j,k).ne.0)then
            do i=1,nobj(j,k)          !boucle sur le nombre d'objets par (j,k)
            !  1st Boundary
                  ix=xfin(i,j,k)/dx+1
                  ipoli=ix+1
            !  2nd Boundary
                  ix=(xfin(i,j,k)+dx)/dx+1
                  ipolf=ipoli+1       ! =ipoli -or- =ipoli+1    to interpolate (1) or (2) points accordingly
            !  calcul du polynôme
               na=ia
               do ipol=ipoli,ipolf
                  call idwfresh(ux,uy,ipol,j,xfin(i,j,k),epsik)
               enddo
            enddo
         endif
      enddo
      enddo
   !
      return
   end subroutine fresh_idw_x
   !
   !*******************************************************************
   !
   subroutine idwfresh(ux,uy,i,j,cexx2,epsik)
   ! 
   !********************************************************************
   USE param
   USE IBM
   USE variables
   !
   implicit none
   !
   real(8),dimension(nx,ny,1) :: ux,uy,epsik,fel
   real(8) :: x,y,cexx2,d1,d2,d3,d4,d5,w1,w2,w3,w4,w5,nom1,nom2,dnom1,powerc
   real(8) :: d6,d7,d8,w6,w7,w8
   integer :: i,j
   !
   powerc=3.
   !
   fel(:,:,:)=(1.-epsik(:,:,:))
   !
      x=((i-1)*dx)
      y=(yp(j))
   !
      d1=x-cexx2
      d2=(dx**2.+dy**2.)**0.5
      d3=dy
      d4=d2
      d5=dx
      d6=d2
      d7=d3
      d8=d2
   !
      if (d1.le.0.000001) then  ! Check for very small values
        print*,'Error',d1
        pause
      endif     
   !
      w1=1./(d1**powerc)
      w2=1./(d2**powerc)
      w3=1./(d3**powerc)
      w4=1./(d4**powerc)
      w5=1./(d5**powerc)
      w6=1./(d6**powerc)
      w7=1./(d7**powerc)
      w8=1./(d8**powerc)
   !
      nom1=w1*(ubcx)+w2*ux(i-1,j+1,1)*fel(i-1,j+1,1)+w3*ux(i,j+1,1)*fel(i,j+1,1)+w4*ux(i+1,j+1,1)*fel(i+1,j+1,1) &
      +w5*ux(i+1,j,1)*fel(i+1,j,1)+w6*ux(i+1,j-1,1)*fel(i+1,j-1,1)+w7*ux(i,j-1,1)*fel(i,j-1,1)+w8*ux(i-1,j-1,1)*fel(i-1,j-1,1) ! For ux
      dnom1=w1+w2+w3+w4+w5+w6+w7+w8
   !
      nom2=w1*(ubcy)+w2*uy(i-1,j+1,1)*fel(i-1,j+1,1)+w3*uy(i,j+1,1)*fel(i,j+1,1)+w4*uy(i+1,j+1,1)*fel(i+1,j+1,1) &
      +w5*uy(i+1,j,1)*fel(i+1,j,1)+w6*uy(i+1,j-1,1)*fel(i+1,j-1,1)+w7*uy(i,j-1,1)*fel(i,j-1,1)+w8*uy(i-1,j-1,1)*fel(i-1,j-1,1) ! For uy
   !
      ux(i,j,1)=nom1/dnom1
   !   print*,ux(i,j,1)
   !   pause
      uy(i,j,1)=nom2/dnom1
   !
   return
   end subroutine idwfresh
   !
   
