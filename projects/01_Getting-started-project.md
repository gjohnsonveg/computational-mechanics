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

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
dt=2
T_a=65
T_1=85
T_2=74
dT=T_2-T_1

K=-dT/((T_2-T_a)*(dt))
print(K)
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def K_analytical(T_1, T_2, T_a, t):
    '''
    help file for K_analytical(T_1, T_2, T_a, t)
    computes the empirical constant as a function of two different 
    temperatures, the ambient temperature, and the time elapsed
    Arguments:
    ----------
    T_1 : temperature at time=0
    T_2 : temperature at time=t
    T_a : ambient temperature
    t : time
    Returns:
    --------
    K : empirical constant
    '''
    dt=t
    dT=T_2-T_1
    K=-dT/((T_2-T_a)*dt)
    return K
```

```{code-cell} ipython3
K_analytical(85, 74, 65, 2)
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
T_a = 65
N = 100
t=np.linspace(0, 2, N)
T_num = np.zeros(len(t))
dt = t[1]-t[0]
T_num[0] = 98.6
for i in range(0, len(t)-1):
    T_num[i+1] = T_num[i] - K*(T_num[i]-T_a)*dt
    
    
def T_analytical(T_a,T_0,K,t):
    for i in range(0, len(t)):
        T[i] = T_a + (T_0-T_a)*np.exp(-K*t[i])
    return T

T_ana = T_analytical(T_a, T_num[0], K, t)
    
plt.plot(t, T_num, label='Euler approximation for 100 steps')
plt.plot(t, T_ana, label='analytical')
plt.plot(0.85, T_1, 'o')
plt.title('Newtons Law of Cooling')
plt.xlabel('time (hrs)')
plt.ylabel('Temperature (degrees F)')
plt.legend()

#T_analytical = T_a + (T_0-T_a)*np.exp(-K*t)

#dt = np.diff(t)
#for i in range(0,len(t)-1):
    #T_numerical[i+1]=T_numerical[i]+(T_a+(T_0-T_a)*np.exp(-K*t[i]))*dt[i];
    
```

b. The final temperature as $t \rightarrow \infty$ is $65^o$ F.

c. The time of death was approximately 0.85 hours before 11 AM, or 10:09 AM.

+++

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
time_vals = np.array([6, 7, 8, 9, 10, 11, 12, 13])
Temp_vals = np.array([50, 51, 55, 60, 65, 70, 75, 80])

def ambient_temp(t):
    if t>=6 and t <=7:
        return Temp_vals[0]
    elif t>7 and t<=8:
        return Temp_vals[1]
    elif t>8 and t<=9:
        return Temp_vals[2]
    elif t>9 and t<=10:
        return Temp_vals[3]
    elif t>10 and t<=11:
        return Temp_vals[4]
    elif t>11 and t<=12:
        return Temp_vals[5]
    elif t>12 and t<=13:
        return Temp_vals[6]
    else:
        return 65
    
#option 1
day_Temps = np.array([ambient_temp(t) for t in day_times])

plt.plot(day_times, day_Temps)
plt.title('Ambient Temperatures')
plt.xlabel('hour')
plt.ylabel('Temperature (degrees F)')
```

```{code-cell} ipython3
N=100
t=np.linspace(time_vals[-3], time_vals[-1], N)
t_ana=np.linspace(time_vals[-3], time_vals[-1], N)
T_num = np.zeros(len(t))
dt = t[1]-t[0]
T_num[0] = 98.6
    
for i in range(0, len(t)-1):
    T_a = ambient_temp(t[i])
    T_num[i+1] = T_num[i] - K*(T_num[i]-T_a)*dt
    

#def T_analytical(T_0,K,t):
T_ana = np.zeros(len(t_ana))
for i in range(0, len(t_ana)):
    T_a = ambient_temp(t_ana[i])
    T_ana[i] = T_a + (T_0-T_a)*np.exp(-K*i)
print(T_ana)
    
plt.plot(t, T_num, label='Euler approximation for 100 steps')
plt.plot(t_ana, T_ana, label='analytical')
plt.plot(11.005, 98.6, 'o')
plt.title('Newtons Law of Cooling')
plt.xlabel('time (hrs)')
plt.ylabel('Temperature (degrees F)')
plt.legend()
```

It seems like the time of death was right at 11 AM. 

```{code-cell} ipython3

```
