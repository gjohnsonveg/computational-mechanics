---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Initial Value Problems - Project

![Initial condition of firework with FBD and sum of momentum](../images/firework.png)

+++

You are going to end this module with a __bang__ by looking at the
flight path of a firework. Shown above is the initial condition of a
firework, the _Freedom Flyer_ in (a), its final height where it
detonates in (b), the applied forces in the __Free Body Diagram (FBD)__
in (c), and the __momentum__ of the firework $m\mathbf{v}$ and the
propellent $dm \mathbf{u}$ in (d). 

The resulting equation of motion is that the acceleration is
proportional to the speed of the propellent and the mass rate change
$\frac{dm}{dt}$ as such

$$\begin{equation}
m\frac{dv}{dt} = u\frac{dm}{dt} -mg - cv^2.~~~~~~~~(1)
\end{equation}$$

If you assume that the acceleration and the propellent momentum are much
greater than the forces of gravity and drag, then the equation is
simplified to the conservation of momentum. A further simplification is
that the speed of the propellant is constant, $u=constant$, then the
equation can be integrated to obtain an analytical rocket equation
solution of [Tsiolkovsky](https://www.math24.net/rocket-motion/) [1,2], 

$$\begin{equation}
m\frac{dv}{dt} = u\frac{dm}{dt}~~~~~(2.a)
\end{equation}$$

$$\begin{equation}
\frac{m_{f}}{m_{0}}=e^{-\Delta v / u},~~~~~(2.b) 
\end{equation}$$

where $m_f$ and $m_0$ are the mass at beginning and end of flight, $u$
is the speed of the propellent, and $\Delta v=v_{final}-v_{initial}$ is
the change in speed of the rocket from beginning to end of flight.
Equation 2.b only relates the final velocity to the change in mass and
propellent speed. When you integrate Eqn 2.a, you will have to compare
the velocity as a function of mass loss. 

Your first objective is to integrate a numerical model that converges to
equation (2.b), the Tsiolkovsky equation. Next, you will add drag and
gravity and compare the results _between equations (1) and (2)_.
Finally, you will vary the mass change rate to achieve the desired
detonation height.

+++

__1.__ Create a `simplerocket` function that returns the velocity, $v$,
the acceleration, $a$, and the mass rate change $\frac{dm}{dt}$, as a
function of the $state = [position,~velocity,~mass] = [y,~v,~m]$ using
eqn (2.a). Where the mass rate change $\frac{dm}{dt}$ and the propellent
speed $u$ are constants. The average velocity of gun powder propellent
used in firework rockets is $u=250$ m/s [3,4]. 

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = \left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt} \\ \frac{dm}{dt} \end{array}\right]$

Use [an integration method](../module_03/03_Get_Oscillations) to
integrate the `simplerocket` function. Demonstrate that your solution
converges to equation (2.b) the Tsiolkovsky equation. Use an initial
state of y=0 m, v=0 m/s, and m=0.25 kg. 

Integrate the function until mass, $m_{f}=0.05~kg$, using a mass rate change of $\frac{dm}{dt}=0.05$ kg/s. 

> __Hint__: your integrated solution will have a current mass that you can
> use to create $\frac{m_{f}}{m_{0}}$ by dividing state[2]/(initial mass),
> then your plot of velocity(t) vs mass(t)/mass(0) should match
> Tsiolkovsky's
> 
> $\log\left(\frac{m_{f}}{m_{0}}\right) =
> \log\left(\frac{state[2]}{0.25~kg}\right) 
> = \frac{state[1]}{250~m/s} = \frac{-\Delta v+error}{u}$ 
> where $error$ is the difference between your integrated state variable
> and the Tsiolkovsky analytical value.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

```{code-cell} ipython3
def simplerocket(state,dmdt=0.05, u=250):
    '''Computes the right-hand side of the differential equation
    for the acceleration of a rocket, without drag or gravity, in SI units.
    
    Arguments
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt : mass rate change of rocket in kilograms/s default set to 0.05 kg/s
    u    : speed of propellent expelled (default is 250 m/s)
    
    Returns
    -------
    derivs: array of three derivatives [v (u/m*dmdt-g-c/mv^2) -dmdt]^T
    '''
    dstate = np.zeros(np.shape(state))
    dstate[0] = state[1] #velocity
    dstate[1] = (u*dmdt/state[2]) #acceleration
    dstate[2] = -dmdt #dmdt
    return dstate
```

```{code-cell} ipython3
from scipy.integrate import solve_ivp
```

```{code-cell} ipython3
m0=0.25
mf=0.05
dm=0.05
t = np.linspace(0, (m0-mf)/dm, 500)
#implicit solver
sol = solve_ivp(lambda t, y: simplerocket(y), [0,t.max()], [0,0,m0], t_eval=t)

plt.plot(sol.t, sol.y[0], label = 'dm = {}'.format(dm))
```

```{code-cell} ipython3
def simplerocketmass(state,dmdt=0.05, u=250):
    dmass = np.zeros(2)
    dmass[0] = state[2]/m0
    dmass[1] = -u*np.ln(state[2]/m0)
    return dmass
```

```{code-cell} ipython3
#def simplerocketmass(state,dmdt=0.05, u=250):
   # dmass = np.array(state[2]/m0)
    #dmass[0] = state[2]/m0
  #  return dmass
```

```{code-cell} ipython3
y0=0 #initial position
v0=0 #initial velocity
m0=0.25
mf=0.05
dm=0.05
N=500
t = np.linspace(0,(m0-mf)/dm,N)
dt=t[1]-t[0]

#masses = np.zeros(len(t)) #initialize mass array
#for i in masses:
#    masses[i] = state[i][2]/m0

#initialize solution array
num_sol = np.zeros([N,3]) 
masses = np.zeros([N])

#set initial conditions
num_sol[0,0] = y0
num_sol[0,1] = v0
num_sol[0,2] = m0

masses[0] = m0
```

```{code-cell} ipython3
#modified eulers method
def rk2_step(state, rhs, dt):
    '''Update a state to the next time increment using modified Euler's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    
    mid_state = state + rhs(state) * dt*0.5    
    next_state = state + rhs(mid_state)*dt
 
    return next_state
```

```{code-cell} ipython3
for i in range(N-1):
    num_sol[i+1] = rk2_step(num_sol[i], simplerocket, dt)
    masses[i+1] = rk2_step(masses[i], simplerocketmass, dt)
```

```{code-cell} ipython3
#masses = np.zeros(len(t)) #initialize masses array
#for i in masses:
#    masses[i] = state[2]/m0
```

```{code-cell} ipython3
u=250
v_an = -u*np.log(masses)
```

```{code-cell} ipython3
#plot solution with Euler's method
fig = plt.figure(figsize=(6,4))

plt.plot(t, num_sol[:, 1], linewidth=2, linestyle='--', label='Numerical solution')
plt.plot(t, masses, linewidth=2, linestyle='--', label='Analytical solution')
plt.plot(t, v_an, linewidth=1, linestyle='-', label='Analytical solution')
plt.xlabel('Time [s]')
plt.ylabel('$v$ [m/s]')
plt.title('Rocket system with Euler\'s method (dashed line).\n')
```

```{code-cell} ipython3
N=np.array([100, 250, 500, 1000, 5000])
dt_values = np.array(len(N_vals))

t0 = np.linspace(0,(m0-mf)/dm,N[0])
t1 = np.linspace(0,(m0-mf)/dm,N[1])
t2 = np.linspace(0,(m0-mf)/dm,N[2])
t3 = np.linspace(0,(m0-mf)/dm,N[3])
t4 = np.linspace(0,(m0-mf)/dm,N[4])

dt0=t0[1]-t0[0]
dt1=t1[1]-t1[0]
dt2=t2[1]-t2[0]
dt3=t3[1]-t3[0]
dt4=t4[1]-t4[0]

dt_values = np.array([dt0, dt1, dt2, dt3, dt4])

num_sol_time = np.empty_like(dt_values, dtype=np.ndarray)

for j, dt in enumerate(dt_values):

    N = int(T/dt)
    t = np.linspace(0, T, N)
    
    #initialize solution array
    num_sol = np.zeros([N,2])
    
    
    #Set intial conditions
    num_sol[0,0] = x0
    num_sol[0,1] = v0
    
    for i in range(N-1):
        num_sol[i+1] = rk2_step(num_sol[i], springmass, dt)

    num_sol_time[j] = num_sol.copy()
```

```{code-cell} ipython3
def get_error(num_sol):
    
    v_an = u*(np.log(mf)-np.log(m0))+v0 # analytical solution at final time
    
    error =  np.abs(num_sol[-1,0] - v_an)
    
    return error
```

```{code-cell} ipython3

```

```{code-cell} ipython3
error_values = np.empty_like(dt_values)

for j, dt in enumerate(dt_values):
    
    error_values[j] = get_error(num_sol_time[j], T)
```

```{code-cell} ipython3
# plot of convergence for modified Euler's
fig = plt.figure(figsize=(6,6))

plt.loglog(dt_values, error_values, 'ko-')
plt.loglog(dt_values, 5*dt_values**2, 'k:')
plt.grid(True)
plt.axis('equal')
plt.xlabel('$\Delta t$')
plt.ylabel('Error')
plt.title('Convergence of modified Euler\'s method (dotted line: slope 2)\n');
```

__2.__ You should have a converged solution for integrating `simplerocket`. Now, create a more relastic function, `rocket` that incorporates gravity and drag and returns the velocity, $v$, the acceleration, $a$, and the mass rate change $\frac{dm}{dt}$, as a function of the $state = [position,~velocity,~mass] = [y,~v,~m]$ using eqn (1). Where the mass rate change $\frac{dm}{dt}$ and the propellent speed $u$ are constants. The average velocity of gun powder propellent used in firework rockets is $u=250$ m/s [3,4]. 

$\frac{d~state}{dt} = f(state)$

$\left[\begin{array}{c} v\\a\\ \frac{dm}{dt} \end{array}\right] = 
\left[\begin{array}{c} v\\ \frac{u}{m}\frac{dm}{dt}-g-\frac{c}{m}v^2 \\ \frac{dm}{dt} \end{array}\right]$

Use [two integration methods](../notebooks/03_Get_Oscillations.ipynb) to integrate the `rocket` function, one explicit method and one implicit method. Demonstrate that the solutions converge to equation (2.b) the Tsiolkovsky equation. Use an initial state of y=0 m, v=0 m/s, and m=0.25 kg. 

Integrate the function until mass, $m_{f}=0.05~kg$, using a mass rate change of $\frac{dm}{dt}=0.05$ kg/s, . 

Compare solutions between the `simplerocket` and `rocket` integration, what is the height reached when the mass reaches $m_{f} = 0.05~kg?$

```{code-cell} ipython3
def rocket(state,dmdt=0.05, u=250,c=0.18e-3):
    '''Computes the right-hand side of the differential equation
    for the acceleration of a rocket, with drag, in SI units.
    
    Arguments
    ----------    
    state : array of three dependent variables [y v m]^T
    dmdt : mass rate change of rocket in kilograms/s default set to 0.05 kg/s
    u    : speed of propellent expelled (default is 250 m/s)
    c : drag constant for a rocket set to 0.18e-3 kg/m
    Returns
    -------
    derivs: array of three derivatives [v (u/m*dmdt-g-c/mv^2) -dmdt]^T
    '''
    g=9.81
    dstate = np.zeros(np.shape(state))
    dstate[0] = state[1] #velocity
    dstate[1] = u/state[2]*dmdt-g-c/state[2]*state[1]**2 #acceleration
    dstate[2] = -dmdt #dmdt
    return dstate
```

```{code-cell} ipython3
#initialize solution array
num_sol = np.zeros([N,3]) 
#set initial conditions
num_sol[0,0] = y0
num_sol[0,1] = v0
num_sol[0,2] = m0
```

```{code-cell} ipython3
for i in range(N-1):
    num_sol[i+1] = rk2_step(num_sol[i], rocket, dt)
```

```{code-cell} ipython3
def heun_step(state,rhs,dt,etol=0.000001,maxiters = 100):
    '''Update a state to the next time increment using the implicit Heun's method.
    
    Arguments
    ---------
    state : array of dependent variables
    rhs   : function that computes the RHS of the DiffEq
    dt    : float, time increment
    etol  : tolerance in error for each time step corrector
    maxiters: maximum number of iterations each time step can take
    
    Returns
    -------
    next_state : array, updated after one time increment'''
    e=1
    eps=np.finfo('float64').eps
    next_state = state + rhs(state)*dt
    ################### New iterative correction #########################
    for n in range(0,maxiters):
        next_state_old = next_state
        next_state = state + (rhs(state)+rhs(next_state))/2*dt
        e=np.sum(np.abs(next_state-next_state_old)/np.abs(next_state+eps))
        if e<etol:
            break
    ############### end of iterative correction #########################
    return next_state
```

```{code-cell} ipython3
N=500
t = np.linspace(0,(m0-mf)/dm,N)
dt=t[1]-t[0]
#initialize solution array
num_heun = np.zeros([N,3]) 
num_rk2 = np.zeros([N,3]) 

#set initial conditions
num_heun[0,0] = y0
num_heun[0,1] = v0
num_heun[0,2] = m0

num_rk2[0,0] = y0
num_rk2[0,1] = v0
num_rk2[0,2] = m0

for i in range(N-1):
    num_heun[i+1] = heun_step(num_heun[i], rocket, dt)
    num_rk2[i+1] = rk2_step(num_rk2[i], rocket, dt)
```

```{code-cell} ipython3
plt.plot(t,num_heun[:,0],'o-', label='implicit Heun', alpha=0.2)
plt.plot(t,num_rk2[:,0],'s-',label='explicit RK2', alpha=0.2)
plt.axhline(y=v_an, linewidth=1, linestyle='-', label='Analytical solution')#plt.ylim(-8,8)
plt.legend();
plt.xlabel('Time [s]')
plt.ylabel('$v$ [m/s]')
plt.title('Rocket system with implicit and explicit methods.\n')
#plt.xlim(np.max(t)-5,np.max(t))
#plt.xlim(np.max(t)-period,np.max(t))
```

__3.__ Solve for the mass change rate that results in detonation at a height of 300 meters. Create a function `f_dm` that returns the final height of the firework when it reaches $m_{f}=0.05~kg$. The inputs should be 

$f_{m}= f_{m}(\frac{dm}{dt},~parameters)$

where $\frac{dm}{dt}$ is the variable you are using to find a root and $parameters$ are the known values, `m0=0.25, c=0.18e-3, u=250`. When $f_{m}(\frac{dm}{dt}) = 0$, you have found the correct root. 

Plot the height as a function of time and use a star to denote detonation at the correct height with a `'*'`-marker

Approach the solution in two steps, use the incremental search
[`incsearch`](../module_03/04_Getting_to_the_root) with 5-10
sub-intervals _limit the number of times you call the
function_. Then, use the modified secant method to find the true root of
the function.

a. Use the incremental search to find the two closest mass change rates within the interval $\frac{dm}{dt}=0.05-0.4~kg/s.$

b. Use the modified secant method to find the root of the function $f_{m}$.

c. Plot your solution for the height as a function of time and indicate the detonation with a `*`-marker.

```{code-cell} ipython3
def f_dm(dmdt, m0 = 0.25, c = 0.18e-3, u = 250):
    ''' define a function f_dm(dmdt) that returns 
    height_desired-height_predicted[-1]
    here, the time span is based upon the value of dmdt
    
    arguments:
    ---------
    dmdt: the unknown mass change rate
    m0: the known initial mass
    c: the known drag in kg/m
    u: the known speed of the propellent
    
    returns:
    --------
    error: the difference between height_desired and height_predicted[-1]
        when f_dm(dmdt) = 0, the correct mass change rate was chosen
    '''
    
    tmin = 0
    tmax = (mf-m0)/dmdt
    sol = solve_ivp(lambda t, y:simplerocket(y, dmdt=dmdt),
                    [tmin, tmax], [0, 0, m])
    maxheight = sol.y[0, -1] #first state variable (h), last value in time (h[-1])
    error = maxheight - 300
    return error
```

```{code-cell} ipython3
dm = np.linspace(0.05, 0.1)
f_dm_array = np.array([f_dm(dmi) for dmi in dm])

plt.plot(dm, f_dm_array+300)
plt.xlabel('mass rate Kg/s')
plt.ylabel('final height m')
plt.title('ignoring gravity and drag')
```

```{code-cell} ipython3
np.sign(10)
```

```{code-cell} ipython3
def incsearch(func,xmin,xmax,ns=50):
    '''incsearch: incremental search root locator
    xb = incsearch(func,xmin,xmax,ns):
      finds brackets of x that contain sign changes
      of a function on an interval
    arguments:
    ---------
    func = name of function
    xmin, xmax = endpoints of interval
    ns = number of subintervals (default = 50)
    returns:
    ---------
    xb(k,1) is the lower bound of the kth sign change
    xb(k,2) is the upper bound of the kth sign change
    If no brackets found, xb = [].'''
    x = np.linspace(xmin,xmax,ns)
    f = func(x)
    sign_f = np.sign(f)
    delta_sign_f = sign_f[1:]-sign_f[0:-1]
    i_zeros = np.nonzero(delta_sign_f!=0)
    nb = len(i_zeros[0])
    xb = np.block([[ x[i_zeros[0]+1]],[x[i_zeros[0]] ]] )

    
    if nb==0:
      print('no brackets found\n')
      print('check interval or increase ns\n')
    else:
      print('number of brackets:  {}\n'.format(nb))
    return xb
```

```{code-cell} ipython3
incsearch(f_dm, 0.01, 0.5)
```

```{code-cell} ipython3
from scipy.integrate import solve_ivp
dm = 0.1
rocket_dm = lambda t, y: simplerocket(y, dmdt=dm)

sol = solve_ivp(rocket_dm, [0, t.max()], [0, 0, 0.25], )

plt.plot(sol.t, sol.y[0])
```

```{code-cell} ipython3
def mod_secant(func,dx,x0,es=0.0001,maxit=50):
    '''mod_secant: Modified secant root location zeroes
    root,[fx,ea,iter]=mod_secant(func,dfunc,xr,es,maxit,p1,p2,...):
    uses modified secant method to find the root of func
    arguments:
    ----------
    func = name of function
    dx = perturbation fraction
    xr = initial guess
    es = desired relative error (default = 0.0001 )
    maxit = maximum allowable iterations (default = 50)
    p1,p2,... = additional parameters used by function
    returns:
    --------
    root = real root
    fx = func evaluated at root
    ea = approximate relative error ( )
    iter = number of iterations'''

    iter = 0;
    xr=x0
    for iter in range(0,maxit):
        xrold = xr;
        dfunc=(func(xr+dx)-func(xr))/dx;
        xr = xr - func(xr)/dfunc;
        if xr != 0:
            ea = abs((xr - xrold)/xr) * 100;
        else:
            ea = abs((xr - xrold)/1) * 100;
        if ea <= es:
            break
    return xr,[func(xr),ea,iter]
```

## References

1. Math 24 _Rocket Motion_. <https://www.math24.net/rocket-motion/\>

2. Kasdin and Paley. _Engineering Dynamics_. [ch 6-Linear Momentum of a Multiparticle System pp234-235](https://www.jstor.org/stable/j.ctvcm4ggj.9) Princeton University Press 

3. <https://en.wikipedia.org/wiki/Specific_impulse>

4. <https://www.apogeerockets.com/Rocket_Motors/Estes_Motors/13mm_Motors/Estes_13mm_1_4A3-3T>

```{code-cell} ipython3

```
