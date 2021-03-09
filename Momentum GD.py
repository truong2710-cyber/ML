#find global minimum of f(x)=x^2+10sin(x) using Momentum GD

from math import cos,sin
def f(x):
    return x**2+10*sin(x)

def grad(x):
    return 2*x+10*cos(x)

def Momentum_GD(x_init,grad,gamma,eta):
    x=[x_init]
    v=[0]
    for it in range(200):
        v_new=gamma*v[-1]+eta*grad(x[-1])
        x_new=x[-1]-v_new
        x.append(x_new)
        if abs(grad(x_new))<1e-6:
            break
        v.append(v_new)
    return (x[-1],it)

def NAG(x_init,grad,gamma,eta): #Nesterov accelerated gradient
    x=[x_init]
    v=[0]
    for it in range(200):
        v_new=gamma*v[-1]+eta*grad(x[-1]-gamma*v[-1])
        x_new=x[-1]-v_new
        x.append(x_new)
        if abs(grad(x_new))<1e-6:
            break
        v.append(v_new)
    return (x[-1],it)

x_init=6
(x_min, it)=Momentum_GD(x_init,grad,0.9,0.1)
(x_min1, it1)=NAG(x_init,grad,0.9,0.1)
print("Solutionn found by Momentum GD: x* = %f after %i iterations"%(x_min,it+1))
print("Solutionn found by NAG:         x* = %f after %i iterations"%(x_min1,it1))


