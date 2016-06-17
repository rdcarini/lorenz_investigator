'''
Created on May 31, 2016

@author: rdcarini
'''

import numpy as np
import pylab as p
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def lorenz_system(x_0,y_0,z_0,sigma,roe,beta,end,n):
    def x_prime(t,x,y,z):
        return sigma*(y-x)
    def y_prime(t,x,y,z):
        return (roe*x)-y-(x*z)
    def z_prime(t,x,y,z):
        return (x*y)-(beta*z)
    f = [x_prime,y_prime,z_prime]
    h = float(end)/float(n)
    t = np.linspace(0.0,end,num=n+1)
    approx_1 = np.zeros((3,n+1))
    approx_1[0][0] = x_0
    approx_1[1][0] = y_0
    approx_1[2][0] = z_0
    approx_2 = np.zeros((3,n+1))
    approx_2[0][0] = x_0
    approx_2[1][0] = y_0
    approx_2[2][0] = z_0
    #The following codes were inspired by and built upon from Numerical Analysis 9th Ed. by Burden and Faires
    #First compute an Euler's Method approximation of the solution and save in array approx_1
    for i in range(1,n+1):
        approx_1[0][i] = approx_1[0][i-1]+h*f[0](t[i-1],approx_1[0][i-1],approx_1[1][i-1],approx_1[2][i-1])
        approx_1[1][i] = approx_1[1][i-1]+h*f[1](t[i-1],approx_1[0][i-1],approx_1[1][i-1],approx_1[2][i-1])
        approx_1[2][i] = approx_1[2][i-1]+h*f[2](t[i-1],approx_1[0][i-1],approx_1[1][i-1],approx_1[2][i-1])
        if approx_1[0][i] == np.Inf:
            print 'x-value diverges using Euler Method'
            break
        elif approx_1[1][i] == np.Inf:
            print 'y-value diverges using Euler Method'
            break
        elif approx_1[2][i] == np.Inf:
            print 'z-value diverges using Euler Method'
            break
    #Next compute a Runge Kutta approximation of order 4 to approximate same solution and save in array approx_2
    for i in range(1,n+1):
        k_1_0 = h*f[0](t[i-1],approx_2[0][i-1],approx_2[1][i-1],approx_2[2][i-1])
        k_1_1 = h*f[1](t[i-1],approx_2[0][i-1],approx_2[1][i-1],approx_2[2][i-1])
        k_1_2 = h*f[2](t[i-1],approx_2[0][i-1],approx_2[1][i-1],approx_2[2][i-1])
        k_2_0 = h*f[0](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_1_0,approx_2[1][i-1]+0.5*k_1_1,approx_2[2][i-1]+0.5*k_1_2)
        k_2_1 = h*f[1](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_1_0,approx_2[1][i-1]+0.5*k_1_1,approx_2[2][i-1]+0.5*k_1_2)
        k_2_2 = h*f[2](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_1_0,approx_2[1][i-1]+0.5*k_1_1,approx_2[2][i-1]+0.5*k_1_2)
        k_3_0 = h*f[0](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_2_0,approx_2[1][i-1]+0.5*k_2_1,approx_2[2][i-1]+0.5*k_2_2)
        k_3_1 = h*f[1](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_2_0,approx_2[1][i-1]+0.5*k_2_1,approx_2[2][i-1]+0.5*k_2_2)
        k_3_2 = h*f[2](t[i-1]+h/2.0,approx_2[0][i-1] + 0.5*k_2_0,approx_2[1][i-1]+0.5*k_2_1,approx_2[2][i-1]+0.5*k_2_2)
        k_4_0 = h*f[0](t[i-1]+h,approx_2[0][i-1] + k_3_0,approx_2[1][i-1]+k_3_1,approx_2[2][i-1]+k_3_2)
        k_4_1 = h*f[1](t[i-1]+h,approx_2[0][i-1] + k_3_0,approx_2[1][i-1]+k_3_1,approx_2[2][i-1]+k_3_2)
        k_4_2 = h*f[2](t[i-1]+h,approx_2[0][i-1] + k_3_0,approx_2[1][i-1]+k_3_1,approx_2[2][i-1]+k_3_2)
        approx_2[0][i] = approx_2[0][i-1] + (k_1_0 + 2.0*k_2_0 + 2.0*k_3_0 + k_4_0)/6.0
        approx_2[1][i] = approx_2[1][i-1] + (k_1_1 + 2.0*k_2_1 + 2.0*k_3_1 + k_4_1)/6.0
        approx_2[2][i] = approx_2[2][i-1] + (k_1_2 + 2.0*k_2_2 + 2.0*k_3_2 + k_4_2)/6.0
        if approx_2[0][i] == np.Inf:
            print 'x-value diverges using Runga-Kutta'
            break
        elif approx_2[1][i] == np.Inf:
            print 'y-value diverges using Runga-Kutta'
            break
        elif approx_2[2][i] == np.Inf:
            print 'z-value diverges using Runga-Kutta'
            break
    return (t, approx_1, approx_2)

#Uncomment one of the following to plot or add your own initial conditions

#(t,approx_1,approx_2) = lorenz_system(10.0,1.0,10.0,5.2,15.0,1.0,100,10000)
(t,approx_1,approx_2) = lorenz_system(10.0,12.0,10.0,5.2,15.0,1.0,100,10000)
#(t,approx_1,approx_2) = lorenz_system(0.0,1.0,1.05,10,28,2.667,100,10000)
#(t,approx_1,approx_2) = lorenz_system(0.0,-2.1,0.0,10,28,2.667,100,10000)

t_b = np.array(t)
approx_1b = np.array(approx_1)
approx_2b = np.array(approx_2)
print t_b
print approx_1b
print approx_2b

def distance(array_1, array_2):
    (h,w) = np.shape(array_1)
    dist = np.zeros(w)
    for i in range(0,w):
        dist[i] = np.sqrt(np.abs(array_2[0,i]-array_1[0,i])**2.0 + np.abs(array_2[1,i]-array_1[1,i])**2.0 + np.abs(array_2[2,i]-array_1[2,i])**2.0)
    return dist

dist = distance(approx_1b,approx_2b)
print dist

max = np.amax(dist)
print max
min = np.amin(dist)
print min

#Plot each value (x, y, and z) with versus time
plt.figure(1)
plt.subplot(311)
plt.plot(t_b, approx_1b[0,:], 'b-', t_b, approx_2b[0,:], 'r-')
plt.xscale('log')
plt.subplot(312)
plt.plot(t_b, approx_1b[1,:], 'b-', t_b, approx_2b[1,:], 'r-')
plt.xscale('log')
plt.subplot(313)
plt.plot(t_b, approx_1b[2,:], 'b-', t_b, approx_2b[2,:], 'r-')
plt.xscale('log')


#Plot 3D figure of Lorenz system on same grid for comparison 
#(idea from Tony Allen at http://computationalcyril.blogspot.com/2012/07/fun-with-lorenz-attractor-and-python.html)
fig=p.figure()
ax = p3.Axes3D(fig)

plt.subplot(211)
ax.plot3D(approx_1b[0,:],approx_1b[1,:],approx_1b[2,:])
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
fig.add_axes(ax)

plt.subplot(221)
ax.plot3D(approx_2b[0,:],approx_2b[1,:],approx_2b[2,:])
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
fig.add_axes(ax)
p.show()
