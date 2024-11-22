import numpy as np
from sistemi_DLTI import *
from upsample import *

def trasformata_wavelet(x,type,direction=1):
  if type=='Coif06':
    scalnum = np.array([1.-np.sqrt(7.), 5.+np.sqrt(7.), 14.+2*np.sqrt(7.), 14.-2*np.sqrt(7.), 1.-np.sqrt(7.), -3.+np.sqrt(7.)]) / (16.*np.sqrt(2.))
    wavenum = np.array([-3.+np.sqrt(7.), -(1.-np.sqrt(7.)), 14.-2*np.sqrt(7.), -(14.+2*np.sqrt(7.)), 5.+np.sqrt(7.), -(1.-np.sqrt(7.))]) / (16.*np.sqrt(2.))
  elif type=='Haar':
    scalnum = np.array([1./np.sqrt(2.), 1./np.sqrt(2.)])
    wavenum = np.array([1./np.sqrt(2.), -1./np.sqrt(2.)])
  elif type=='Daub4':
    scalnum = np.array([1.+np.sqrt(3), 3.+np.sqrt(3), 3.-np.sqrt(3), 1.-np.sqrt(3)]) / (4.*np.sqrt(2))
    wavenum = np.array([1.-np.sqrt(3), -3.+np.sqrt(3), 3.+np.sqrt(3), -1.-np.sqrt(3)]) / (4.*np.sqrt(2))
  elif type=='Daub6':    
    scalnum = np.array([0.332670552950083, 0.806891509311092, 0.459877502118491, -0.135011020010255, -0.0854412738820267, 0.0352262918857095])    
    wavenum = np.array([0.0352262918857095, 0.0854412738820267, -0.135011020010255, -0.459877502118491, 0.806891509311092, -0.332670552950083])
  elif type=='Daub8':
    scalnum = np.array([0.23037781, 0.71484657, 0.63088166, -0.02798377, -0.18703481, 0.03084138, 0.03288301, -0.0105974])
    wavenum = np.array([-0.0105974, -0.03288301, 0.03084138, 0.18703481, -0.02798377, -0.63088166, 0.71484657, -0.23037781])
  elif type=='Coif30':
    scalnum = np.array([-.000149963800,  .000253561200,  .001540245700,  -.002941110800,  -.007163781900,  \
            .016552066400,  .019917804300, -.064997262800,  -.036800073600,   .298092323500, \
            .547505429400,  .309706849000, -.043866050800,  -.074652238900,   .029195879500, \
            .023110777000, -.013973687900, -.006480090000,   .004783001400,   .001720654700, \
           -.001175822200, -.000451227000,  .000213729800,   .000099377600,  -.000029232100, \
           -.000015072000,  .000002640800,  .000001459300,  -.000000118400,  -.000000067300]) * np.sqrt(2.)
    wavenum = np.array([-.000000067300,  .000000118400,  .000001459300,  -.000002640800,  -.000015072000,  \
            .000029232100,  .000099377600, -.000213729800,  -.000451227000,   .001175822200, \
            .001720654700, -.004783001400, -.006480090000,   .013973687900,   .023110777000, \
           -.029195879500, -.074652238900,  .043866050800,   .309706849000,  -.547505429400, \
            .298092323500,  .036800073600, -.064997262800,  -.019917804300,   .016552066400, \
            .007163781900, -.002941110800, -.001540245700,   .000253561200,   .000149963800]) * np.sqrt(2.)
  #endif
  Nc = len(scalnum)
  if x.ndim == 2:
    Nx,Ny = x.shape
    if direction==1:   # trasformata diretta
      for ix in range(Nx):
        u = np.squeeze( x[ix,:] )
        trend = simula_DLTI(scalnum,np.array([1.]),u)
        trend = trend[1:Ny:2]
        flutt = simula_DLTI(wavenum,np.array([1.]),u)
        flutt = flutt[1:Ny:2]
        x[ix,:] = np.concatenate((trend, flutt))
      #endfor
      for iy in range(Ny):
        u = np.squeeze( x[:,iy] )
        trend = simula_DLTI(scalnum,np.array([1.]),u)
        trend = trend[1:Nx:2]
        flutt = simula_DLTI(wavenum,np.array([1.]),u)
        flutt = flutt[1:Nx:2]
        v = np.concatenate((trend, flutt)) 
        #print "v.shape = ", v.shape
        #vv = np.atleast_2d( np.concatenate((trend, flutt)) ).T
        #print "vv.shape = ", vv.shape
        #print "iy = ", iy
        #print "x.shape = ", x.shape
        #x[:,iy] = np.atleast_2d( np.concatenate((trend, flutt)) ).T
        for j in range(Nx):
          x[j,iy] = v[j]
        #endfor
      #endfor
    elif direction==-1:  # trasformata inversa
      for ix in range(Nx):
        ttemp = simula_DLTI(scalnum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[ix,0:int(Ny/2)],2) )
        ttemp = ttemp + simula_DLTI(wavenum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[ix,int(Ny/2):Ny],2) )
        x[ix,:] = ttemp
      #endfor
      for iy in range(Ny):
        ttemp = simula_DLTI(scalnum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[0:int(Nx/2),iy],2) )
        ttemp = ttemp + simula_DLTI(wavenum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[int(Nx/2):Nx,iy],2) )
        x[:,iy] = ttemp.T
      #endfor
    #endif
  else:
    N = len(x)
    if direction==1:   # trasformata diretta
      trend = simula_DLTI(scalnum,np.array([1.]),x)
      trend = trend[1:N:2]
      flutt = simula_DLTI(wavenum,np.array([1.]),x)
      flutt = flutt[1:N:2]
      x = np.concatenate((trend, flutt))
    elif direction==-1:  # trasformata inversa
      ttemp = simula_DLTI(scalnum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[0:int(N/2)],2) )
      ttemp = ttemp + simula_DLTI(wavenum[range(Nc-1,-1,-1)],np.array([1.]), upsample(x[int(N/2):N],2) )
      x = ttemp
    #endif
  #endif
  return x