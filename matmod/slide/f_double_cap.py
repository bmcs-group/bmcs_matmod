#!/usr/bin/env python
# coding: utf-8

# # Derivation of the level set function for SLIDE interface model 3.2

# In[32]:


import sympy as sp
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv
import ipywidgets as ipw


# ![image.png](attachment:image.png)

# # Variables
# 
# **Stress variables**
# 
# $x$ corresponds to $\sigma_N$ and $y$ corresponds to $\sigma_T$

# In[33]:


x, y = sp.symbols('x y')


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
get_f_mid = sp.lambdify((x, y, y_bar, m), f_mid_)


# # Cap function
# 
# It is introduced in form of an ellipse centered at the position ($x_c, 0$)

# In[37]:


f_cap_ = sp.sqrt( (x-x_0-x_c)**2/a**2 + y**2/b**2 ) - c
f_cap_

# Construct the derivatives

# In[38]:

df_cap_dx_ = f_cap_.diff(x)
df_cap_dy_ = f_cap_.diff(y)

# # Value and continuity conditions specified in the figure above

# In[39]:

eq1 = sp.Eq( f_cap_.subs({x:x_bar, y:0}), 0 )
eq2 = sp.Eq( f_cap_.subs({x:x_0, y:y_bar}), 0)
eq3 = sp.Eq( ( -df_cap_dx_ / df_cap_dy_).subs({x:x_0, y:y_bar}), -m)

# Solve for $a, b$ and $x_c$

# In[40]:

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

# In[42]:

c_solved = sp.solve(eq4, c)[0]

# Substitute back the $f_\mathrm{cap}$ to obtain its resolved form

# In[43]:


f_cap_abxc_ = f_cap_abx_.subs(c, c_solved)

# # Test the results

# In[44]:

_x_0=-3
_x_bar=-5
_y_bar=3
_m=.3


# The value in the denominator must not be equal to 0

# In[45]:

m_limit_ = sp.solve(2*x_bar * m + y_bar, m)[0]
_m_limit = m_limit_.subs({x_bar:_x_bar, y_bar:_y_bar})
_m_limit


# In[46]:

if _m < _m_limit * sp.sign(_x_bar - _x_0):
    print('Take care')


# Test the obtained position of $x_c$

# In[47]:


x_c_ = abx_subs[x_c]
x_c_

# In[48]:

get_x_c = sp.lambdify((x_bar, y_bar, m, x_0), x_c_, 'numpy' )
_x_c = get_x_c(_x_bar, _y_bar, _m, _x_0)
_x_c

# # Domain separation for $f_\mathrm{cap}$ and $f_\mathrm{mid}$
# 
# Define the transition between cap and mid domains by defining a connection 
# line between [$x_c$,0]-[0,$\bar{\tau}$]. 

# In[49]:

y_trans_ = (-y_bar / (x_c) * (x - x_0 -x_c)).subs(x_c, x_c_)
f_cap_domain_ = sp.sign(x_bar-x_0) * sp.sign(-m) * (sp.Abs(y) - y_trans_)  > 0

# In[50]:

x_trans = np.linspace(_x_c, _x_0, 10)
get_y_trans = sp.lambdify((x, x_bar, y_bar, m, x_0), y_trans_)
y_trans = get_y_trans(x_trans, _x_bar, _y_bar, _m, _x_0)
x_trans, y_trans

# In[51]:

get_y_trans(_x_c, _x_bar, _y_bar, _m, _x_0)

# # Composed smooth level set function $f_\mathrm{full}(x,y)$

# In[52]:

f_full_ = sp.Piecewise(
    (f_cap_abxc_, f_cap_domain_),
    (f_mid_, True)
)

get_f_full = sp.lambdify( (x,y,x_bar,y_bar,m,x_0), f_full_, 'numpy')
f_full_

# # Visualization

f_t, f_c, f_c0, tau_bar = sp.symbols('f_t, f_c, f_c0, tau_bar')
subs_tension = {x_0:0, x_bar:f_t, y_bar:tau_bar}
subs_shear = {y_bar:tau_bar, x_0:0}
subs_compression = {x_0: -f_c0, x_bar:-f_c,  y_bar: tau_bar-m*(-f_c0) }


# In[71]:


# -------------------------------------------------------------------------
# Symbolic derivation of variables
# -------------------------------------------------------------------------
x = sp.Symbol(
    r'x', real=True,
)

max_f_c = 100
max_f_t = 10
max_tau_bar = 20

import bmcs_utils.api as bu
import traits.api as tr

class FDoubleCapExpr(bu.SymbExpr):
    # -------------------------------------------------------------------------
    # Symbolic derivation of variables
    # -------------------------------------------------------------------------

    x, y = x, y

    # -------------------------------------------------------------------------
    # Model parameters
    # -------------------------------------------------------------------------

    f_t, f_c, f_c0, tau_bar, m = f_t, f_c, f_c0,tau_bar, m

    # -------------------------------------------------------------------------
    # Expressions
    # -------------------------------------------------------------------------

    f_solved = sp.Piecewise(
        (f_cap_abxc_.subs(subs_tension), f_cap_domain_.subs(subs_tension)),
        (f_cap_abxc_.subs(subs_compression), f_cap_domain_.subs(subs_compression)),
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


class FDoubleCap(bu.InteractiveModel,bu.InjectSymbExpr):

    name = 'Threshold'

    symb_class = FDoubleCapExpr

    f_t = tr.Float(5, MAT=True)
    f_c = tr.Float(80, MAT=True)
    f_c0 = tr.Float(30, MAT=True)
    tau_bar = tr.Float(5, MAT=True)
    m = tr.Float(0.1, MAT=True)

    ipw_view = bu.View(
        bu.Item('f_t', minmax=(1, max_f_t)),
        bu.Item('f_c', minmax=(10, max_f_c)),
        bu.Item('f_c0', latex='f_{c0}', minmax=(5, 0.9 * max_f_c)),
        bu.Item('tau_bar', latex=r'\bar{\tau}', minmax=(1, max_tau_bar)),
        bu.Item('m', minmax=(0.0001, 0.5))
    )

    def update_plot(self, ax):
        # Evaluate the threshold function within an orthogonal grid

        # In[53]:

        X_a, Y_a = np.mgrid[-max_f_c:max_f_t:210j, -max_tau_bar:max_tau_bar:210j]
        Z_a = self.symb.get_f_solved(X_a, Y_a)

        # In[68]:
        ax.contour(X_a, Y_a, Z_a, levels=8)
