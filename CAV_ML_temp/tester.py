!8-8-19: ql10 will be 1 bet-size, but a different approach to the one bet

!ql9 was an attempt at 2 bet-sizes and one bet, as opposed to a half-street

!8-3-19: ql8 the 2 bet size version

!8-2-19 ql7 this is streamlined ql3 with a xpay function

!7-14: ql3 ql2 didn't work; I misunderstood the Q update formula.
!This version does not rely on our opponent making mistakes.

!7-11-19 ql2 changing  so that it is more like playing the game over and over
!6-30-19 ql1.f90 is a first stab at Q learning the half-street game
!pot=2, bet=pot


program ql
  integer, parameter ::  nhands=1000000000,ncards=100
  real, dimension(ncards,3) :: Qx
  real, dimension(ncards,1:2,1:2) ::  Qy
  open(39)
  p=2.0
  b=2.0


  !Qx options are check-fold, check-call and bet
  !Qy options are check or bet if checked to & fold or call if bet into
  Qx=0.0
  Qy=0.0
  alpha=.0001

  do itrip=1,1
     do icount=1,nhands

        call random_number(r)
        ix=int(ncards*r)+1
        call random_number(r)
        iy=int(ncards*r)+1

        call random_number(r)

        if (r.lt..5) then !X will explore
           call random_number(r)
           mx=int(3*r)+1 !X makes a choice
           if (mx.eq.1) then !check-fold
              if (Qy(iy,1,1).gt.Qy(iy,1,2)) then !check back
                 rewardx=xpay(ix,iy,0.0)
              else ! Y bets
                 rewardx=xpay(0,iy,0.0)
              endif
           elseif (mx.eq.2) then !check-call
              if (Qy(iy,1,1).gt.Qy(iy,1,2)) then !check back
                 rewardx=xpay(ix,iy,0.0)
              else ! Y bets
                 rewardx=xpay(ix,iy,b)
              endif
           else !bet
              if (Qy(iy,2,1).gt.Qy(iy,2,2)) then !fold
                 rewardx=xpay(ix,0,0.0) !y folded
              else !call
                 rewardx=xpay(ix,iy,b)
              endif
           endif
           Qx(ix,mx)=(1.0-alpha)*Qx(ix,mx)+alpha*rewardx
        else !Y will explore
           call random_number(r)           
           my=int(2*r)+1 !Y makes a choice           
           if ((Qx(ix,1).gt.Qx(ix,2)).and.(Qx(ix,1).gt.Qx(ix,3))) then 
              if (my.eq.1) then !Y checks back
                 rewardy=-xpay(ix,iy,0.0) 
              else !Y bets
                 rewardy=-xpay(0,iy,0.0) !X folds
              endif
              Qy(iy,1,my)=(1.0-alpha)*Qy(iy,1,my)+alpha*rewardy              
           elseif ((Qx(ix,2).ge.Qx(ix,1)).and.(Qx(ix,2).gt.Qx(ix,3))) then
              if (my.eq.1) then !Y checks back
                 rewardy=-xpay(ix,iy,0.0)
              else !Y bets
                 rewardy=-xpay(ix,iy,b) !X folds
              endif
              Qy(iy,1,my)=(1.0-alpha)*Qy(iy,1,my)+alpha*rewardy
           else !X has bet
              if (my.eq.1) then !Y folds
                 rewardy=-xpay(ix,0,0.0)
              else !Y calls
                 rewardy=-xpay(ix,iy,b)
              endif
              Qy(iy,2,my)=(1.0-alpha)*Qy(iy,2,my)+alpha*rewardy
           endif
        endif
     enddo
     print *,itrip
  enddo
  xav=0.0

  open(37)
  do i=1,ncards
     if (Qx(i,1).gt.max(Qx(i,2),Qx(i,3))) then !ch-fold
        write (37,*) i, Qx(i,1)
        ibet=0
     elseif (Qx(i,2).gt.Qx(i,3)) then !ch-call
        ibet=1
        write (38,*) i, Qx(i,2)
     else
        write (36,*) i, Qx(i,3)
     endif
!!$     if (Qy(i,1).gt.QY(i,2)) then !no call
!!$        icall=0
!!$        write (35,*) i, Qy(i,1)        
!!$     else
!!$        icall=1
!!$        write (36,*) i, Qy(i,2)        
!!$     endif
     xav=xav+max(max(Qx(i,1),Qx(i,2)),Qx(i,3))

     write(39,*) Qx(i,1),Qx(i,2),Qx(i,3)
  enddo
  print *, xav/ncards
  stop

end program ql

real function xpay(ix,iy,bet) !set iy to 0 if y folds
use main

  if (ix.gt.iy) then !X won
     xpay=p/2+bet
  elseif (ix.eq.iy) then !they tied 
     xpay=0.0
  else!if (ix.lt.iy) then !Y won 
     xpay=-p/2-bet
  endif

end function xpay