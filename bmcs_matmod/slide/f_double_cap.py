#!/usr/bin/env python
# coding: utf-8

# # Derivation of the level set function for SLIDE interface model 3.2

# In[32]:


import sympy as sp
import numpy as np

# ![image.png](attachment:image.png)

# # Variables
# 
# **Stress variables**
# 
# $x$ corresponds to $\sigma_N$ and $y$ corresponds to $\sigma_T$

# In[33]:


x, y = sp.symbols('x, y')


# **Unknown parameters of the ellipse cap function**

# In[34]:

x_c = sp.symbols('x_c') # center point of an elips
a = sp.symbols('a', nonnegative=True)
b = sp.symbols('b', nonnegative=True)
c = sp.symbols('c', positive=True)

# **Material parameters**

# In[35]:

x_0 = sp.symbols('x_0')
x_bar, y_bar = sp.symbols(r'x_bar, y_bar', nonnegative=True )
m = sp.symbols('m')

# # Mid part of the threshold function

# In[36]:

f_mid_ = sp.sqrt( y**2 ) - (y_bar - m * (x-x_0))

# # Cap function
# 
# It is introduced in form of an ellipse centered at the position ($x_c, 0$)

f_cap_ = sp.sqrt( (x-x_0-x_c)**2/a**2 + y**2/b**2 ) - c

# Construct the derivatives

df_cap_dx_ = f_cap_.diff(x)
df_cap_dy_ = f_cap_.diff(y)

# # Value and continuity conditions specified in the figure above

eq1 = sp.Eq( f_cap_.subs({x:x_bar, y:0}), 0 )
eq2 = sp.Eq( f_cap_.subs({x:x_0, y:y_bar}), 0)
eq3 = sp.Eq( ( -df_cap_dx_ / df_cap_dy_).subs({x:x_0, y:y_bar}), -m)

# Solve for $a, b$ and $x_c$

abx_subs = sp.solve({eq1,eq2,eq3},{a,b,x_c})[0]
abx_subs

# # Continuity between $f_\mathrm{mid}$ and $f_\mathrm{cap}$
# 

# **Require an identical value for**
# \begin{align}
# f_\mathrm{mid}(x_c, 0) = f_\mathrm{cap}(x_c,0)
# \end{align}

f_mid_c_ = f_mid_.subs({x:abx_subs[x_c]+x_0,y:0})
f_cap_abx_ = f_cap_.subs(abx_subs)
f_cap_abxc_ = f_cap_abx_.subs({x:abx_subs[x_c]+x_0,y:0})
eq4 = sp.Eq( f_cap_abxc_, f_mid_c_ )

# Which can be used to express $c$

c_solved = sp.solve(eq4, c)[0]

# Substitute back the $f_\mathrm{cap}$ to obtain its resolved form

f_cap_solved_ = f_cap_abx_.subs(c, c_solved)

# Test the obtained position of $x_c$
x_c_ = abx_subs[x_c]

# # Domain separation for $f_\mathrm{cap}$ and $f_\mathrm{mid}$
# 
# Define the transition between cap and mid domains by defining a connection 
# line between [$x_c$,0]-[0,$\bar{\tau}$]. 

y_trans_ = (-y_bar / (x_c) * (x - x_0 -x_c)).subs(x_c, x_c_)
f_cap_domain_ = sp.sign(x_bar-x_0) * sp.sign(-m) * (sp.Abs(y) - y_trans_)  > 0

# # Visualization

f_t, f_c, f_c0, tau_bar = sp.symbols('f_t, f_c, f_c0, tau_bar')
subs_tension = {x_0:0, x_bar:f_t, y_bar:tau_bar}
subs_shear = {y_bar:tau_bar, x_0:0}
subs_compression = {x_0: -f_c0, x_bar:-f_c,  y_bar: tau_bar-m*(-f_c0) }

import bmcs_utils.api as bu

class FDoubleCapExpr(bu.SymbExpr):
    # -------------------------------------------------------------------------
    # Symbolic derivation of variables
    # -------------------------------------------------------------------------

    x = x

    y = y

    # -------------------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------------------

    f_t, f_c, f_c0, tau_bar, m = f_t, f_c, f_c0,tau_bar, m

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------

    f_solved = sp.Piecewise(
        (f_cap_solved_.subs(subs_tension), f_cap_domain_.subs(subs_tension)),
        (f_cap_solved_.subs(subs_compression), f_cap_domain_.subs(subs_compression)),
        (f_mid_.subs(subs_shear), True)
    )

    #-------------------------------------------------------------------------
    # Declaration of the lambdified methods
    #-------------------------------------------------------------------------

    symb_model_params = ['f_t', 'f_c', 'f_c0', 'tau_bar', 'm']

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('f_solved', ('x', 'y')),
    ]

max_f_c = 100
max_f_t = 10
max_tau_bar = 10

class FDoubleCap(bu.InteractiveModel,bu.InjectSymbExpr):

    name = 'Threshold'

    symb_class = FDoubleCapExpr

    f_t = bu.Float(5, MAT=True)
    f_c = bu.Float(80, MAT=True)
    f_c0 = bu.Float(30, MAT=True)
    tau_bar = bu.Float(5, MAT=True)
    m = bu.Float(0.1, MAT=True)

    ipw_view = bu.View(
        bu.Item('f_t', editor=bu.FloatRangeEditor(low=1, high=max_f_t)),
        bu.Item('f_c', editor=bu.FloatRangeEditor(low=10, high=max_f_c)),
        bu.Item('f_c0', latex='f_{c0}', editor=bu.FloatRangeEditor(low=5, high=0.9*max_f_c)),
        bu.Item('tau_bar', latex=r'\bar{\tau}', editor=bu.FloatRangeEditor(low=1, high=max_tau_bar)),
        bu.Item('m', minmax=(0.0001, 0.5))
    )

    def subplots(self, fig):
        ax = fig.subplots(1, 1)
#        ax = fig.add_subplot(1, 1, 1, projection='3d')
        return ax

    def plot_3d(self, ax):
        X_a, Y_a = np.mgrid[-1.1*max_f_c:1.1*max_f_t:210j, -max_tau_bar:max_tau_bar:210j]
        Z_a = self.symb.get_f_solved(X_a, Y_a)
        #ax.contour(X_a, Y_a, Z_a, levels=8)
        Z_0 = np.zeros_like(Z_a)
        ax.plot_surface(X_a, Y_a, Z_a, rstride=1, cstride=1,
                        cmap='winter', edgecolor='none')
        ax.plot_surface(X_a, Y_a, Z_0, edgecolor='none')
        ax.set_title('threshold function');

    def plot_contour(self, ax):
        X_a, Y_a = np.mgrid[-1.05*max_f_c:1.1*max_f_t:210j, -max_tau_bar:max_tau_bar:210j]
        Z_a = self.symb.get_f_solved(X_a, Y_a)
        ax.contour(X_a, Y_a, Z_a, levels=0)
        ax.set_title('threshold function');


    def update_plot(self, ax):
        # Evaluate the threshold function within an orthogonal grid
        self.plot_contour(ax)