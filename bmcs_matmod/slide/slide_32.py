#!/usr/bin/env python
# coding: utf-8

# # Damage-plasticity SLIDE 3.2
# This notebook is a work in progress on an abstract and general implementation of time integration algorithm for general damage-plasticity modes. It serves for the development of a package that can be configured by specifying the ingredients of thermodynamically based model
# 
#  - Vector of state variables $\boldsymbol{\mathcal{E}}$
#  - Vector of thermodynamic streses $\boldsymbol{\mathcal{S}}$
#  - Helmholtz free energy $\psi(\boldsymbol{\mathcal{E}})$
#  - Threshold on thermodynamical forces  $f(\boldsymbol{\mathcal{S}},\boldsymbol{\mathcal{E}})$ / Yield condition
#  - Flow potential $\varphi(\boldsymbol{\mathcal{S}},\boldsymbol{\mathcal{E}})$
# 
# as symbolic equations using the sympy package. The time-stepping algorithm gets generated automatically within the thermodynamically framework. The derived  evolution equations and return-mapping to the yield surface is performed using Newton scheme.  

# In[1]:

import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# **Code generation** The derivation is adopted for the purpose of code generation both in Python and C utilizing the `codegen` package provided in `sympy`. The expressions that are part of the time stepping algorithm are transformed to an executable code directly at the place where they are derived. At the end of the notebook the C code can be exported to external files and applied in external tools. 

# In[2]:


# This code is needed to lambdify expressions named with latex symbols
# it removes the backslashes and curly braces upon before code generation.
from sympy.utilities.codegen import codegen
import re
def _print_Symbol(self, expr):
    CodePrinter = sp.printing.codeprinter.CodePrinter
    name = super(CodePrinter, self)._print_Symbol(expr)
    return re.sub(r'[\{^}]','_',re.sub(r'[\\\{\}]', '', name))
sp.printing.codeprinter.CodePrinter._print_Symbol = _print_Symbol

def ccode(cfun_name, sp_expr, cfile):
    '''Generate c function cfun_name for expr and directive name cfile 
    '''
    return codegen((cfun_name, sp_expr), 'C89', cfile + '_' + cfun_name)


# ## TODO
#  - The current implementation uses the threshold function linearized along $\lambda$ to represent the consistency condition. 
#  - As the next step include the fully linearized set of evolution equations and threshold functions. This might be important for SLIDE 3.x, 2D and 3D problems (von Mises, Drucker-Prager). In the current one-dimensional problem it probably has no effect - this feature will be included in SLIDE-core notebook which should be able to generate all the other problems by configuring the thermodynamic inputs symbolically.  
#  - Put the derived get_methods into a class and let the predictor-corrector implementation just access them. With this, the iteration scheme becomes completely generic. Any kind of model can be implemented then using the thermodynamic framework.
#  - Verification - check to see if the parameters controlling interaction between tension and shear have an effect and for which range of values
#  - Think of more flat profile of the f function to reduce the number of iterations during return mapping
#  - Visualize the iteration within the yield potential space for elementary loading scenarios.
#  - Check the reset of the interactive app upon change of the material parameters. Sometimes it is not replotted directly.

# &nbsp;<font color='blue'>
# **Naming conventions:**
#  - Variables with trailing underscore (e.g. `f_`, or `Sig_`) denote `sympy` expressions. 
#  - Variables denoting `sympy` symbols (e.g. `Sig` have no underscore at the end and have an the name which is close to the mathematical symbol
#  - Mathematical symbols defined as string in `sp.symbols(r'\tau^{\pi}')` use `latex` syntax to introduce Greek symbols, super and subindexes. This makes the pretty printing of expression possible.
#  - In an implemented algorithm at the end of the notebook, the Python variables containing the numerical values of the material parameters $E_b$, $\tau_\mathrm{Y}$, etc. are denoted with a leading underscore `_E_b` and `_tau_Y` to avoid name collisions within the notebook
# </font>

# ## Material parameters

# In[3]:


E_s = sp.Symbol('E_s', real=True, nonnegative=True)
gamma_s = sp.Symbol('gamma_s', real=True, nonnegative=True)
K_s = sp.Symbol('K_s', real=True)
S_s = sp.Symbol('S_s', real=True)
r_s = sp.Symbol('r_s', real=True)
c_s = sp.Symbol('c_s', real=True)
bartau = sp.Symbol(r'\bar{\tau}', real=True, nonnegative=True)
#bartau = sp.Symbol(r'tau_Y', real=True, nonnegative=True)

E_w = sp.Symbol('E_w', real=True, nonnegative=True)
S_w = sp.Symbol('S_w', real=True)
r_w = sp.Symbol('r_w', real=True)
c_w = sp.Symbol('c_w', real=True)
eta = sp.Symbol('eta', real=True, nonnegative=True)


# In[5]:



# ## State variables

# In[6]:


s_x, s_y = sp.symbols('s_x, s_y', real=True)
omega_s = sp.Symbol('omega_s', real=True, nonnegative=True)
s_pi_x, s_pi_y = sp.symbols(r's^{\pi}_x, s^{\pi}_y', real=True)
#s_pi_x, s_pi_y = sp.symbols(r's_pi_x, s_pi_y', real=True)
alpha_x, alpha_y = sp.symbols('alpha_x, alpha_y', real=True)
z = sp.Symbol('z', real=True)

w = sp.symbols('w', real=True)
omega_w = sp.Symbol('omega_w', real=True, nonnegative=True)
w_pi = sp.symbols(r'w^{\pi}', real=True)
#w_pi = sp.symbols(r'w_pi', real=True)


# In[7]:


Eps = sp.Matrix([s_pi_x, s_pi_y, w_pi, z, alpha_x, alpha_y, omega_s, omega_w])
Eps.T


# ## Helmholtz free energy

# The starting point in the thermodynamical representation of a process is a potential function of time dependent state variables. To describe the evolution of the state correctly describing the energy dissipation of the system the gradient of the potential function with respect to the state variables provides the generalized forces. The forces are constrained to characterize specific material properties, e.g. strength, hardening.

# In[8]:


rho_psi_s_ = sp.Rational(1,2)* ( 
    (1-omega_s)*E_s*(s_x-s_pi_x)**2 + 
    (1-omega_s)*E_s*(s_y-s_pi_y)**2 + 
    K_s * z**2 + 
    gamma_s * alpha_x**2 +
    gamma_s * alpha_y**2
)

rho_psi_w_ = sp.Rational(1,2) * (1 - omega_w) * E_w * (w - w_pi)**2


# In[9]:


rho_psi_ = rho_psi_s_ + rho_psi_w_
rho_psi_


# ## Thermodynamic forces

# In[10]:


tau_x, tau_y = sp.symbols('tau_x, tau_y', real=True)
tau_pi_x, tau_pi_y = sp.symbols(r'\tau^\pi_x, \tau^\pi_y', real=True)
#tau_pi_x, tau_pi_y = sp.symbols(r'tau_pi_x, tau_pi_y', real=True)
X_x, X_y = sp.symbols('X_x, X_y', real=True)
Z = sp.Symbol('Z', real=True, nonnegative=True)
Y_s = sp.Symbol('Y_s', real=True)

sig = sp.symbols('\sigma', real=True)
sig_pi = sp.symbols(r'\sigma^\pi', real=True)
#sig_pi = sp.symbols(r'sigma_pi', real=True)
Y_w = sp.Symbol('Y_w', real=True)


# In[11]:


Sig = sp.Matrix([tau_pi_x, tau_pi_y, sig_pi, Z, X_x, X_y, Y_s, Y_w])
Sig.T


# The introduce the thermodynamic forces we have to differentiate Hemholtz free energy
# with respect to the kinematic state variables
# \begin{align}
# \frac{\partial \rho \psi }{\partial \boldsymbol{\mathcal{E}}}
# \end{align}

# In[12]:


d_rho_psi_ = sp.Matrix([rho_psi_.diff(eps) for eps in Eps])
d_rho_psi_


# To obtain consistent signs of the Helmholtz derivatives we define a sign switch operator so that all generalized forces are defined as positive for the respective conjugate state variable $\boldsymbol{\Upsilon}$. 

# In[13]:


Sig_signs = sp.diag(-1,-1,-1,1,1,1,-1,-1)


# The constitutive laws between generalized force and kinematic variables then read
# \begin{align}
# \boldsymbol{\mathcal{S}} = \boldsymbol{\Upsilon}\frac{\rho \psi}{\partial\boldsymbol{\mathcal{E}}} 
# \end{align}

# In[14]:


Sig_ = Sig_signs * d_rho_psi_
Sig_.T


# **Executable code for** $\boldsymbol{\mathcal{S}}(s,\boldsymbol{\mathcal{E}})$

# In[15]:


# To derive the time stepping procedure we will need also the matrix of derivatives of the generalized stresses $\boldsymbol{\mathcal{S}}$ with respect to the kinematic variables $\boldsymbol{\mathcal{E}}$ 
# \begin{align}
# \frac{\partial \boldsymbol{S}}{\partial \boldsymbol{E}}
# \end{align}

# In[16]:


dSig_dEps_ = sp.Matrix([ 
    Sig_.T.diff(eps) for eps in Eps 
] ).T
dSig_dEps_


# **Executable Python code generation** $\displaystyle \frac{\partial }{\partial \boldsymbol{\mathcal{E}}}  \boldsymbol{\mathcal{S}}(s,\boldsymbol{\mathcal{E}})$

# In[17]:



# ## Threshold function

# To keep the framework general for different stress norms and hardening definitions let us first introduce a general function for effective stress. Note that the observable stress $\tau$ is identical with the plastic stress $\tau_\pi$ due to the performed sign switch in the definition of the thermodynamic forces.

# In[18]:


tau_eff_x = sp.Function(r'\tau^{\mathrm{eff}}_x')(tau_pi_x, omega_s)
tau_eff_y = sp.Function(r'\tau^{\mathrm{eff}}_y')(tau_pi_y, omega_s)
sig_eff = sp.Function(r'\sigma_{\mathrm{eff}}')(sig_pi, omega_w)
# tau_eff_x = sp.Function(r'tau_eff_x')(tau_pi_x, omega_s)
# tau_eff_y = sp.Function(r'tau_eff_y')(tau_pi_y, omega_s)
# sig_eff = sp.Function(r'sigma_eff')(sig_pi, omega_w)
Q_x = sp.Function('Q_x')(tau_eff_x,X_x)
Q_y = sp.Function('Q_y')(tau_eff_y,X_y)


# The stress norm is defined using the stress offset $X$, i.e. the kinematic hardening stress representing the shift of the origin of the yield locus.  

# In[19]:


norm_Q = sp.sqrt(Q_x*Q_x + Q_y*Q_y)


# Let us now introduce the back stress $X$ by defining the substitution for $Q = \tau^\mathrm{eff} - X$

# In[20]:


subs_Q = {Q_x: tau_eff_x - X_x, Q_y: tau_eff_y - X_y}
subs_Q


# Further substitution rule introduces the effective stress as a function of damage as

# In[21]:


subs_tau_eff = {tau_eff_x: tau_pi_x / (1-omega_s), tau_eff_y: tau_pi_y / (1-omega_s), sig_eff: sig_pi / (1-omega_w)}
subs_tau_eff


# After substitutions the yield function reads

# **Smooth yield function**

# In[24]:


# In[25]:

from bmcs_matmod.slide.f_double_cap import FDoubleCap
import bmcs_utils.api as bu

fdc = FDoubleCap()
f_t = fdc.symb.f_t
f_c = fdc.symb.f_c
f_c0 = fdc.symb.f_c0
m = fdc.symb.m
x = fdc.symb.x
y = fdc.symb.y

f_solved_ = fdc.symb.f_solved
f_1 = f_solved_.subs({x:sig_eff, y:norm_Q})
f_2 = f_1.subs(subs_Q)
f_3 = f_2.subs(subs_tau_eff)
f_ = f_3.subs(fdc.symb.tau_bar, (bartau+Z))

# **Executable code generation** $f(\boldsymbol{\mathcal{E}}, \boldsymbol{\mathcal{S}})$
# 
# Note that this is a function of both the forces and kinematic state variables

# The derivative of $f$ required for time-stepping $\frac{\partial f}{\partial \boldsymbol{\mathcal{S}}}$ is obtained as

df_dSig_ = f_.diff(Sig)
ddf_dEps_ = f_.diff(Eps)

phi_s_ext = (1-omega_s)**c_s * (Y_s**2 / S_s + eta * (Y_s * Y_w) / S_s)
phi_w_ext = (1-omega_w)**c_w * (Y_w**2 / S_w + eta * (Y_s * Y_w) / S_w) 

# The flow potential $\varphi(\boldsymbol{\mathcal{E}}, \boldsymbol{\mathcal{S}})$ reads

phi_ = f_ + phi_s_ext + phi_w_ext
phi_

# and the corresponding directions of flow given as a product of the sign operator $\Upsilon$ and of the derivatives with respect to state variables
# $\boldsymbol{\Upsilon} \, \partial_{\boldsymbol{\mathcal{S}}} \varphi$
# This renders following flow direction vector
# \begin{align}
# \boldsymbol{\Phi} = - \Upsilon \frac{\partial \varphi}{\partial \boldsymbol{\mathcal{S}}} 
# \end{align}

Phi_ = -Sig_signs * phi_.diff(Sig)

class Slide23Expr(bu.SymbExpr):

    # control and state variables
    s_x, s_y, w, Eps, Sig = s_x, s_y, w, Eps, Sig

    # model parameters
    E_s = E_s
    gamma_s = gamma_s
    K_s = K_s
    S_s = S_s
    r_s = r_s
    c_s = c_s
    bartau = bartau
    E_w = E_w
    S_w = S_w
    r_w = r_w
    c_w = c_w
    f_t = f_t
    f_c = f_c
    f_c0 = f_c0
    m = m
    eta = eta

    symb_model_params = [
        'E_s', 'gamma_s', 'K_s', 'S_s', 'c_s', 'bartau',
        'E_w', 'S_w', 'c_w', 'm', 'f_t', 'f_c', 'f_c0', 'eta'
    ]

    # expressions
    Sig_ = Sig_.T
    dSig_dEps_ = dSig_dEps_
    f_ = f_
    df_dSig_ = df_dSig_
    ddf_dEps_ = ddf_dEps_
    phi_ = phi_
    Phi_ = Phi_

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('Sig_', ('s_x', 's_y', 'w', 'Eps')),
        ('dSig_dEps_', ('s_x', 's_y', 'w', 'Eps')),
        ('f_', ('Eps', 'Sig')),
        ('df_dSig_', ('Eps', 'Sig')),
        ('ddf_dEps_', ('Eps', 'Sig')),
        ('phi_', ('Eps', 'Sig')),
        ('Phi_', ('Eps', 'Sig')),
    ]

#get_Sig_C = ccode('get_Sig',Sig_,'SLIDE1_3')
#get_dSig_dEps_C = ccode('get_dSig_dEps', dSig_dEps_, 'SLIDE1_3')
#get_f_C = ccode('get_f', f_, 'SLIDE1_3')
#get_df_dSig_C = ccode('get_df_dSig', df_dSig_, 'SLIDE1_3')
#get_ddf_dEps_C = ccode('get_df_dEps', ddf_dEps_, 'SLIDE1_3')
#get_Phi_C = ccode('get_Phi', Phi_, 'SLIDE1_3')

import traits.api as tr

class ConvergenceError(Exception):
    """ Inappropriate argument value (of correct type). """

    def __init__(self, message, state):  # real signature unknown
        self.message = message
        self.state = state
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} for state {self.state}'

class Slide32(bu.InteractiveModel,bu.InjectSymbExpr):

    name = 'Slide 3.2'
    symb_class = Slide23Expr

    E_s = bu.Float(28000, MAT=True)
    gamma_s = bu.Float(10, MAT=True)
    K_s = bu.Float(8, MAT=True)
    S_s = bu.Float(1, MAT=True)
    c_s = bu.Float(1, MAT=True)
    bartau = bu.Float(28000, MAT=True)
    E_w = bu.Float(28000, MAT=True)
    S_w =bu.Float(1, MAT=True)
    c_w = bu.Float(1, MAT=True)
    m = bu.Float(0.1, MAT=True)
    f_t = bu.Float(3, MAT=True)
    f_c = bu.Float(30, MAT=True)
    f_c0 = bu.Float(20, MAT=True)
    eta = bu.Float(0.5, MAT=True)

    ipw_view = bu.View(
        bu.Item('E_s', minmax=(0.5, 100)),
        bu.Item('S_s', minmax=(.00001, 100)),
        bu.Item('c_s', minmax=( 0.0001, 10)),
        bu.Item('gamma_s', latex=r'\gamma_\mathrm{s}', minmax=(-20, 20)),
        bu.Item('K_s', minmax=(-20, 20)),
        bu.Item('bartau', latex=r'\bar{\tau}', minmax=(0.5, 20)),
        bu.Item('E_w', minmax=(0.5, 100)),
        bu.Item('S_w', minmax=(0.0001, 100)),
        bu.Item('c_w', minmax=(0.0001, 10)),
        bu.Item('m', minmax=(0.0001, 0.4)),
        bu.Item('f_t', minmax=(0.1, 10)),
        bu.Item('f_c', latex=r'f_\mathrm{c}', minmax=(1, 200)),
        bu.Item('f_c0', latex=r'f_\mathrm{c0}', minmax=(1, 100)),
        bu.Item('eta', minmax=(0, 1))
    )

    def get_f_df(self, s_x_n1, s_y_n1, w_n1, Eps_k):
        Sig_k = self.symb.get_Sig_(s_x_n1, s_y_n1, w_n1, Eps_k)[0]
        dSig_dEps_k = self.symb.get_dSig_dEps_(s_x_n1, s_y_n1, w_n1, Eps_k)
        f_k = np.array([self.symb.get_f_(Eps_k, Sig_k)])
        df_dSig_k = self.symb.get_df_dSig_(Eps_k, Sig_k)
        ddf_dEps_k = self.symb.get_ddf_dEps_(Eps_k, Sig_k)
        df_dEps_k = np.einsum(
            'ik,ji->jk', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
        Phi_k = self.symb.get_Phi_(Eps_k, Sig_k)
        dEps_dlambda_k = Phi_k
        df_dlambda = np.einsum(
            'ki,kj->ij', df_dEps_k, dEps_dlambda_k)
        df_k = df_dlambda
        return f_k, df_k, Sig_k

    def get_Eps_k1(self, s_x_n1, s_y_n1, w_n1, Eps_n, lam_k, Eps_k):
        '''Evolution equations:
        The update of state variables
        for an updated $\lambda_k$ is performed using this procedure.
        '''

        Sig_k = self.symb.get_Sig_(s_x_n1, s_y_n1, w_n1, Eps_k)[0]
        Phi_k = self.symb.get_Phi_(Eps_k, Sig_k)
        Eps_k1 = Eps_n + lam_k * Phi_k[:, 0]
        return Eps_k1

    rtol = bu.Float(1e-3, ALG=True)
    '''Relative tolerance of the return mapping algorithm related 
    to the tensile strength
    '''

    def get_sig_n1(self, s_x_n1, s_y_n1, w_n1, Eps_n, k_max):
        '''Return mapping iteration:
        This function represents a user subroutine in a finite element
        code or in a lattice model. The input is $s_{n+1}$ and the state variables
        representing the state in the previous solved step $\boldsymbol{\mathcal{E}}_n$.
        The procedure returns the stresses and state variables of
        $\boldsymbol{\mathcal{S}}_{n+1}$ and $\boldsymbol{\mathcal{E}}_{n+1}$
        '''
        Eps_k = np.copy(Eps_n)
        lam_k = 0
        f_k, df_k, Sig_k = self.get_f_df(s_x_n1, s_y_n1, w_n1, Eps_k)
        f_k_norm = np.linalg.norm(f_k)
        f_k_trial = f_k[0]
        k = 0
        while k < k_max:
            if f_k_trial < 0 or f_k_norm < self.f_t * self.rtol:
                return Eps_k, Sig_k, k + 1
            dlam = np.linalg.solve(df_k, -f_k)
            lam_k += dlam
            Eps_k = self.get_Eps_k1(s_x_n1, s_y_n1, w_n1, Eps_n, lam_k, Eps_k)
            f_k, df_k, Sig_k = self.get_f_df(s_x_n1, s_y_n1, w_n1, Eps_k)
            f_k_norm = np.linalg.norm(f_k)
            k += 1
        else:
            raise ConvergenceError('no convergence for step', [s_x_n1, s_y_n1, w_n1])

    def plot_f_state(self, ax, Eps, Sig):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.bartau * 2
        upper_tau = self.bartau * 2
        tau_x, tau_y, sig = Sig[:3]
        tau = np.sqrt(tau_x**2 + tau_y**2)
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Eps_ts = np.zeros_like(Sig_ts)
        Sig_ts[0,...] = tau_x_ts
        Sig_ts[2,...] = sig_ts
        Sig_ts[3:,...] = Sig[3:,np.newaxis,np.newaxis]
        Eps_ts[...] = Eps[:,np.newaxis,np.newaxis]
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts)])
        ax.set_title('threshold function');
        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0)
        ax.plot(sig, tau, marker='H', color='red')
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)


    def plot_f(self, ax):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.bartau * 2
        upper_tau = self.bartau * 2
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Sig_ts[0,:] = tau_x_ts
        Sig_ts[2,:] = sig_ts
        Eps_ts = np.zeros_like(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts)])
        ax.set_title('threshold function');
        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0)
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)

    def update_plot(self, ax):
        self.plot_f(ax)


# # Time integration scheme

# ## Discrete yield condition
# In a continuous case we consistency condition to explicitly glue the state onto the yield surface 
# \begin{align}
# \dot{f}(\boldsymbol{\mathcal{S}}(s, \boldsymbol{\mathcal{E}(\lambda)}), \boldsymbol{\mathcal{E}(\lambda)} ) &= 0 \end{align}
# In discrete case, we relax this requirement. Indeed, by taking $f(s_{n+1}; \boldsymbol{\mathcal{E}_n}) $ as a first trial value we can obtain positive values.
# 
# &nbsp;<font color="green">
# **We allow for "trial" states which lie beyond the admissible domain $f \le 0$ during iteration. This allows us to construct a "return mapping" algorithm that iteratively approaches an admissible state on the yield surface.**</font>

# Given an inadmissible trial state $k$ with the yield condition $f_k > 0$, let us introduce a linearized approximation of its change along the plastic multiplier $\lambda$ around the state $k$. 
# \begin{align}
#  f_{k+1} &= f_{k} + \left. \frac{\partial f}{\partial \lambda} \right|_k \Delta \lambda
# \end{align}
# In this form, we can search for an admissible state $f_{n+1} = 0$ by iterating over $k$.
# Note that in initial iteration $k = 0$ the state from previous step is reused, i.e. 
# $f(s_{n+1}; \boldsymbol{\mathcal{E}_n}) $.

# In the linearized form, we can transform the yield condition to a recurrent formula
# \begin{align}
# \left. \frac{\mathrm{d} f}{\mathrm{d} \lambda}\right|_k \Delta \lambda &= -f_k,
# \hspace{1cm} f_k \rightarrow 0 \; \;\mathrm{for}\;\; k = 1\ldots\infty
# \end{align}
# This resembles the Newton method for iterative solution of a nonlinear equation. However, we need to consider the fact that the level of inadmissibility $f$ changes between iterations. 
# ![image.png](attachment:image.png)
# Note that the predictor is negative and $\Delta \lambda > 0$. In every step, the plastic multiplier is updated:
# \begin{align}
# \lambda_{k+1} &= \lambda_k + \Delta \lambda, \, \lambda_0 = 0 \nonumber \\ \nonumber
# \end{align}

# Two more questions must addressed to define a general numerical algorithm for plasticity:
# <font color="brown">
#  * **Update of state variables $\boldsymbol{\mathcal{E}}_{k+1}$ in each iteration**
#  * **Expression of the predictor $\mathrm{d} f / \mathrm{d} \lambda$ in terms of the state variables**
# </font>

# ## State update
# In every iteration step the state variables $\boldsymbol{\mathcal{E}}$ must be updated using the discrete evolution equations, i.e. 
# \begin{align}
# \boldsymbol{\mathcal{E}}_{k+1} &= 
# \boldsymbol{\mathcal{E}}_k + \lambda_{k+1} 
# \boldsymbol{\Phi}_k
# \label{eq:discrete_evolution}
# \end{align}
# Which is used in the calculation of the threshold function in the next step. Note that $\boldsymbol{\Phi}_k$ is evaluated in the state $k$ and not $k+1$.

# To reach an admissible state let us linearize the threshold function at an interim state $k$ as
# \begin{align}
# f_{k+1} = 
# f_k 
#  +
# \left.
# \frac
# {\partial f}
# {\partial \lambda}
# \right|_k
# \Delta \lambda
# \end{align}

# ## Predictor
# \begin{align}
# \left.
# \frac{\partial f}{\partial{\lambda}}  
# \right|_k 
# &=
# \left.
# \frac{\partial f}{\partial{\boldsymbol{ \mathcal{E}}}}  
# \right|_k 
# \left.
# \frac{\partial {\boldsymbol{ \mathcal{E}}}}{\partial \lambda}
# \right|_k =
# \left.
# \frac{\partial f}{\partial{\boldsymbol{ \mathcal{E}}}}  
# \right|_k 
# \boldsymbol{\Phi}_k \\
# \left.
# \frac{\partial f}{\partial{\boldsymbol{ \mathcal{E}}}}  
# \right|_k 
# &=
# \left. \frac{\partial f}{ \partial \boldsymbol{\mathcal{S}}}\right|_{k}
# \left. \frac{\partial \boldsymbol{\mathcal{S}}}{\partial \boldsymbol{\mathcal{E}}}\right|_{k}
# +
# \left. \frac{\partial^{\mathrm{dir}} f}{ \partial^{\mathrm{dir}} \boldsymbol{\mathcal{E}}}\right|_{k}
# \label{eq:df_dlambda}
# \end{align}

# **Remark 1:** The derivative $\partial^\mathrm{dir}$ denotes the direct derivative with respect to $\boldsymbol{\mathcal{E}}$

# **Remark 2:** Note that $\displaystyle \frac{\partial \boldsymbol{\mathcal{E}}}{\partial \lambda}$ in equation $\eqref{eq:df_dlambda}$ can be obtained from the evolution equations $\eqref{eq:discrete_evolution}$
# \begin{align}
# \boldsymbol{\mathcal{E}}_k = \boldsymbol{\mathcal{E}}_n + \lambda \, \boldsymbol{\Phi}_k\; \implies
# \left.
# \frac{\partial {\boldsymbol{ \mathcal{E}}}}{\partial \lambda}
# \right|_k = 
# \boldsymbol{\Phi}_k
# \end{align}

# Thus, by rewriting the linearized equation as a recurrence formula, the iteration algorithm is obtained
# \begin{align}
# &
# \left.
# \frac{\partial f}{\partial{\lambda}}  
# \right|_k 
# \Delta \lambda
# = - f^{(k)}\\
# & \lambda_{k+1} = \lambda_{k} + \Delta \lambda \\
# & \boldsymbol{\mathcal{E}}_{k+1} = \boldsymbol{\mathcal{E}}_{k} + 
#  \lambda_{k} \, 
# \frac{\partial {\boldsymbol{ \mathcal{E}}}}{\partial \lambda}
#  \\
# &k = k + 1
# \end{align}

# ## Implementation concept
# The gradient operators needed for the time-stepping scheme have been derived above and are now available for the implementation of the numerical algorithm both in `Python` and `C89` languages
# 
# <table style="width:50%">
# <tr>
# <th>Symbol</th>
# <th>Python</th>
# <th>C89</th>
# </tr>
# <tr>
# <td>$\mathcal{S}(s, \boldsymbol{\mathcal{E}}) $  
# </td>
# <td>get_Sig</td>
# <td>get_Sig_C</td>
# </tr>
# <tr>
# <td>$\partial_\boldsymbol{\mathcal{E}}  \boldsymbol{\mathcal{S}}(s, \boldsymbol{\mathcal{E}}) $</td>
# <td>get_dSig_dEps</td>
# <td>get_dSig_dEps_C</td>
# </tr>
# <tr>
# <td>$ f(\boldsymbol{\mathcal{S}}, \boldsymbol{\mathcal{E}})$</td>
# <td>get_f</td>
# <td>get_f_C</td>
# </tr>
# <tr>
# <td>$\partial_\boldsymbol{\mathcal{S}} f(\boldsymbol{\mathcal{S}}, \boldsymbol{\mathcal{E}}) $  
# </td>
# <td>get_df_dSig</td>
# <td>get_df_dSig_C</td>
# </tr>
# <tr>
# <td>$\partial_\boldsymbol{\mathcal{E}} f(\boldsymbol{\mathcal{S}}, \boldsymbol{\mathcal{E}}) $</td>
# <td>get_df_dEps</td>
# <td>get_df_dEps_C</td>
# </tr>
# <tr>
# <td>$\partial_\boldsymbol{\mathcal{S}} \varphi(\boldsymbol{\mathcal{S}}, \boldsymbol{\mathcal{E}}) $</td>
# <td>get_Phi</td>
# <td>get_Phi_C</td>
# </tr>
# </table>

# **Threshold and its derivatives:** To avoid repeated calculation of the same expressions, let us put the evaluation of $f$ and $\partial_\lambda f$ into a single procedure. The iteration loop can be constructed in such a way that the predictor for the next step is calculated along with the residuum. In case that the residuum is below the required tolerance, the overhead for an extra calculated derivative is negligible or, with some care, can be even reused in the next time step.  


# # Code generation
# See the docs for the code generation, the latexified  sympy symbols
# must be substituted such that they can act as standard C variable names.
# The issue with this substitution might have been partially fixed by the substitution
# code defined by this code.
#
# The methods
#  * `get_f_df()`, and
#  * `get_Eps_n1` must be rewritten in C.
#
# **How to transform `einsum` to C?**
#
# The state arrays `Eps` and `Sig` must be prepared by the callee, i.e. the function to be
# called from at a level of material point in a finite-element or lattice code.
# All the state variable matrices are flattened in the generated C code so that index access operators must be constructed with the correct convention, i.e.
# `i * n_row + j` or `i + n_col * j`.
#
# Let us consider the line
# ```Python
# df_dEps_k = np.einsum('ik,ji->jk', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
# ````
#
# To transform this systematically into the C loop it is proposed define
# index directives specifying the size of the first dimension. Then, the
# indexes from the `einsum` call can be systematically transferred to the
# multi-loop running over the indexes and respecting their order. No more
# thinking needed. The index operators below are prepared for arrays with
# 2, 3, and 4 dimensions.
# ```C
# #define IJ(N_I,I,J) (N_I * J + I)
# #define IJK(N_I,N_J,I,J,K) (N_I * IJ(N_J,J,K) + I)
# #define IJKL(N_I,N_J,N_K,I,J,K,L) (N_I * IJK(N_J,N_K,J,K,L) + I)
# ````
# to perform the matrix multiplication
# ```C
# int k=0;
# for(int i=0;i<N_I;i++)
#     for(int j=0;j<N_J;j++)
#         df_dEps_1[IJ(N_J,j,k)] += df_dSig_k[IJ(N_I(i,k)] * dSig_dEps_k[IJ(N_J,j,i)]
#     df_dEps[IJ(N_I,i,k)] = df_dEps_1[IJ(N_J,j,k)] + ddSig_dEps(IJ(N_I,i,k)];
# ````
#
# The whole material model is represented by the material method
# that mimics the internal par of the `get_response` function defined above
#
#  1. Call `get_f_df()` to get the trial state
#  2. Start the return mapping iteration loop
#  3. If admissibility criterion fulfilled - admissible state found - return stress
#  4. Evaluate delta of plastic multiplier and update it
#  5. Update state variables using evolution equations `get_Eps_n1`
#  6. Evaluate the residuum using `get_f_df()` and continuum with point 3.
#

# In[ ]:


# C_code = get_Sig_C, get_dSig_dEps_C, get_f_C, get_df_dSig_C, get_ddf_dEps_C, get_Phi_C

# In[ ]:


import os
import os.path as osp

def codegen():
    code_dirname = 'sympy_codegen'
    code_fname = 'SLIDE_1_3_2D'

    home_dir = osp.expanduser('~')
    code_dir = osp.join(home_dir, code_dirname)
    if not osp.exists(code_dir):
        os.makedirs(code_dir)

    code_file = osp.join(code_dir, code_fname)

    print('code_file', code_file)
    h_file = code_file + '.h'
    c_file = code_file + '.c'

    h_f = open(h_file, 'w')
    c_f = open(c_file, 'w')

    if True:
        for function_C in C_code:
            h_f.write(function_C[1][1])
            c_f.write(function_C[0][1])
    h_f.close()
    c_f.close()


# material_params = dict(
#     E_s=1, gamma_s=5, K_s=5, S_s=0.6, c_s=1, bartau=1,
#     E_w=1, S_w=0.6, c_w = 1, m = 0.01, f_t=1, f_c=-20, f_c0=-10, eta=0.5
# )
#
# slide = Slide32(**material_params)
#
# Eps_1 = np.zeros((8,), dtype=np.float_) + 0.1
# Eps_1[0] = 10
# Eps_1[2] = 0
#
# f_val = slide.symb.get_f_(Eps_1, np.zeros_like(Eps_1))
# print(f_val)