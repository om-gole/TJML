import numpy as np
import math as m

#f0 SHARP POINT WITH x**1/3
#Regular f0 with x**4 - 10

def f0(x):
    # return m.sin(x)
    # return x**(1/3)
    # return x**4 - 10
    return 10/(1+m.e**(-1*x)) - 1

def f1(x):
    # return m.cos(x)
    # return (1/3)*x**(-2/3)
    # return 4*x**3
    return (10*m.e**(-1*x))/((10*m.e**(-1*x))**2)

def f2(x):
    # return -1 * m.sin(x)
    # return 12*x**2
    return (-10 - 4)/(m.e**(0.02*x))

def nrA(f0, f1, f2, x0, tolerance):

    x1 = x0 - f0(x0) / f1(x0) #from sheet

    while abs(x1 - x0) > tolerance:
        x0 = x1
        x1 = x0 - f0(x0) / f1(x0)
    return x1

def nrA2(f0, f1, f2, x0, tolerance):

    x1 = x0 - f0(x0) / f1(x0) #from sheet

    while abs(x1 - x0) > tolerance:
        if abs(f1(x0)) < 1e-6: #Check if derivative is too small. 
            return x1, "ERROR~~~ Deriv is < .000001 (too small)"
        x0 = x1
        x1 = x0 - f0(x0) / f1(x0)
    return x1

def nrB(f0, f1, f2, x0, tolerance):
    x1 = x0 - f1(x0)/f2(x0) #same but switched

    while abs(x1 - x0) > tolerance:
        if abs(f1(x0)) < 1e-6: #Check if derivative is too small. 
            return "ERROR~~~ Deriv is < .000001 (too small)"
        x0 = x1
        x1 = x0 - f1(x0) / f2(x0)
    return x1


# print(nrA2(f0, f1, f2, 1, 0.01))


# #Input should be a numpy array
# def F(x):
#     return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

# def f(x):
#     return x[0]**2 + x[1]**2

# def grad_f(x):
#     return np.array([2*x[0], 2*x[1]])

# def hess_f(x):
#     return np.array([[2, 0], [0, 2]])   

# def jaco_f(x):
#     return np.array([[2*x[0], 2*x[1]], [1, -1]])


# def nrC(f, grad_f, hess_f, x0, tol):
#     """
#     Find the minimum of a real-valued function f of two variables using the basic Newton's method.
    
#     Parameters:
#     f (function): The real-valued function f(x, y) whose minimum is to be estimated.
#     grad_f (function): The gradient of the function f.
#     hess_f (function): The Hessian matrix of the function f.
#     x0 (ndarray): The initial guess for the minimum.
#     tol (float): The tolerance value for the estimate.
    
#     Returns:
#     ndarray: An estimate for the minimum of f.
#     """
#     x1 = x0 - np.linalg.inv(hess_f(x0)).dot(grad_f(x0))
#     while np.linalg.norm(x1 - x0) > tol:
#         x0 = x1
#         x1 = x0 - np.linalg.inv(hess_f(x0)).dot(grad_f(x0))
#     return x1

# def nrD(f, jaco_f, x0, tol):
#     """
#     Find a zero of a vector-valued function F of two variables using the basic Newton's method.
    
#     Parameters:
#     F (function): The vector-valued function F(x, y) whose zero is to be estimated.
#     J (function): The Jacobian matrix of the function F.
#     x0 (ndarray): The initial guess for the zero.
#     tol (float): The tolerance value for the estimate.
    
#     Returns:
#     ndarray: An estimate for the zero of F.
#     """
#     x1 = x0 - np.linalg.inv(jaco_f(x0)).dot(F(x0))
#     while np.linalg.norm(x1 - x0) > tol:
#         x0 = x1
#         x1 = x0 - np.linalg.inv(jaco_f(x0)).dot(F(x0))
#     return x1


# print(nrD(F, jaco_f, np.array([3.0, 100.0]), .01))

import numpy as np

def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

def hess_f(x):
    return np.array([[2, 0], [0, 2]])

def nrC(f, grad_f, hess_f, x0, tolerance):

    x1 = x0 - np.linalg.inv(hess_f(x0)).dot(grad_f(x0))
 #Basically same as A &B
    while np.linalg.norm(x1 - x0) > tolerance:
        x0 = x1
        x1 = x0 - np.linalg.inv(hess_f(x0)).dot(grad_f(x0))
        print(x1)
    return x1

print(nrC(f, grad_f, hess_f, np.array([123,12]), 0.01))

# //------------------------------------------------------
def F(x):
    return np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])

def Jac(x):

    return np.array([[2*x[0], 2*x[1]], [1, -1]])

def nrD(F, Jac, x0, tolerance):
    x1 = x0 - np.linalg.inv(Jac(x0)).dot(F(x0))
 #Basically same as A &B
    while np.linalg.norm(x1 - x0) > tolerance:
        x0 = x1
        x1 = x0 - np.linalg.inv(Jac(x0)).dot(F(x0))
        print(x1)
    return x1

print(nrD(F, Jac, [2,10], 0.01))