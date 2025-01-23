#Import the toolbox
from scipy.optimize import minimize,rosen, rosen_der

#Consider the minimization problem with several constraints

##Objective Function
fun = lambda x: (x[0] - 1)**2 + (x[1]-2.5)**2

#Constraints
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2}, {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6}, {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

#bounds
bnds = ((0, None), (0, None))

#The optimization problem is solved using the SLSQP method as:
#SLSQP = Sequential Least Squares Programming

res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,constraints=cons)

print('Solution = ',res.x)

#Result should be 1.4,1.7
def func_simp(x, a, b):
    return (x[0] -1)**3 + (x-a)**b

def con_simp(x, a, b):
    return x+3 *a + b

# fun_simp = lambda x: (x[0] -1)**3 + (x-6)**2
pbd = 20
thread = -1

# cons_simp = ({'type': 'ineq', 'fun': lambda x:  x[0]+3}, {'type': 'ineq', 'fun': lambda x:  -(x[0]-6)})
cons_simp = ({'type': 'ineq', 'fun': con_simp, 'args': (pbd, thread)}, {'type': 'ineq', 'fun': lambda x:  -(x[0]-6)})

bnds_simp = ((-100, 100),)

res_simp = minimize(func_simp, (-1.19), method='SLSQP', args=(-1, 1.), bounds=bnds_simp, constraints=cons_simp)

print('solution = ', res_simp.x)


