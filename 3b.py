import matplotlib.pyplot as plot
import numpy as nup
K=nup.linspace(2,4,8)
R=nup.linspace(5,7,9)
Q=nup.linspace(0,1,3)
plot.plot(K,K,label='K')
plot.plot(R,R,label='R')
plot.plot(Q,Q,label='Q')
plot.legend()
plot.show()