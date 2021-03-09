from cvxopt import matrix,solvers,log
from math import exp
import numpy as np 

K=[4]
F=matrix([[-1.,1.,0.,1.],[-1.,1.,1.,0.],[-1.,0.,1.,1.]])
g=log(matrix([40.,2.,2.,2.]))
solvers.options['show_progress']=False
sol=solvers.gp(K,F,g)

print(np.exp(np.array(sol['x'])))