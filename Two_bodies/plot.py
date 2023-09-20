import matplotlib.pyplot as plt
import numpy as np

from numpy import *
from scipy.signal import lfilter
from scipy.signal import filtfilt
from scipy.signal import *


# --------------------------- Read .dat File --------------------------- # 


itime1, drag1, lift1 = loadtxt('aerof6.dat',unpack=True, usecols=[0,1,2])
itime2, drag2, lift2 = loadtxt('aerof7.dat',unpack=True, usecols=[0,1,2])

fig= plt.figure()
ax2= fig.add_subplot(111)

ax2.plot(itime1,drag1,ls='-',color='black',linewidth=2.0, label='Body1')
ax2.plot(itime2,drag2,ls='-',color='red',linewidth=2.0, label='Bosy2')


ax2.set(ylim=[0.0,3.0])
#ax2.set(xlim=[-5,10])
plt.legend(loc='upper right')

plt.savefig('drag2.png', dpi=600)
