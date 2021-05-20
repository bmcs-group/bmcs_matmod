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

import sympy as sp
from bmcs_matmod.slide.cymbol import Cymbol, ccode
import traits.api as tr
import numpy as np
from bmcs_matmod.slide.f_double_cap import FDoubleCap
import bmcs_utils.api as bu
from ibvpy.tmodel import MATSEval

H_switch = Cymbol(r'H(\sigma^\pi)', real=True)
H = lambda x: sp.Piecewise( (0, x <=0 ), (1, True) )

# **Code generation** The derivation is adopted for the purpose of code generation both in Python and C utilizing the `codegen` package provided in `sympy`. The expressions that are part of the time stepping algorithm are transformed to an executable code directly at the place where they are derived. At the end of the notebook the C code can be exported to external files and applied in external tools. 

# This code is needed to lambdify expressions named with latex symbols
# it removes the backslashes and curly braces upon before code generation.
# from sympy.utilities.codegen import codegen
# import re
# def _print_Symbol(self, expr):
#     CodePrinter = sp.printing.codeprinter.CodePrinter
#     name = super(CodePrinter, self)._print_Symbol(expr)
#     return re.sub(r'[\{^}]','_',re.sub(r'[\\\{\}]', '', name))
# sp.printing.codeprinter.CodePrinter._print_Symbol = _print_Symbol

# ## Material parameters

E_T = Cymbol('E_T', real=True, nonnegative=True)
gamma_T = Cymbol('gamma_T', real=True, nonnegative=True)
K_T = Cymbol('K_T', real=True)
S_T = Cymbol('S_T', real=True)
c_T = Cymbol('c_T', real=True)
bartau = Cymbol(r'\bar{\tau}', real=True, nonnegative=True)

E_N = Cymbol('E_N', real=True, nonnegative=True)
S_N = Cymbol('S_N', real=True)
c_N = Cymbol('c_N', real=True)

eta = Cymbol('eta', real=True, nonnegative=True)

# ## State variables

s_x = Cymbol('s_x', real=True)
s_y = Cymbol('s_y', real=True)
omega_T = Cymbol('omega_T', real=True, nonnegative=True)
s_pi_x = Cymbol(r's^{\pi}_x', codename='s_pi_x', real=True)
s_pi_y = Cymbol(r's^{\pi}_y', codename='s_pi_y', real=True)
alpha_x = Cymbol('alpha_x', real=True)
alpha_y = Cymbol('alpha_y', real=True)
z = Cymbol('z', real=True)

w = Cymbol('w', real=True)
omega_N = Cymbol('omega_N', real=True, nonnegative=True)
w_pi = Cymbol(r'w^{\pi}', codename='w_pi', real=True)

Eps = sp.Matrix([s_pi_x, s_pi_y, w_pi, z, alpha_x, alpha_y, omega_T, omega_N])

tau_x = Cymbol('tau_x', real=True)
tau_y = Cymbol('tau_y', real=True)
tau_pi_x = Cymbol(r'\tau^\pi_x', codename='tau_pi_x', real=True)
tau_pi_y = Cymbol(r'\tau^\pi_y', codename='tau_pi_y', real=True)
X_x = Cymbol('X_x', real=True)
X_y = Cymbol('X_y', real=True)
Z = Cymbol('Z', real=True, nonnegative=True)
Y_T = Cymbol('Y_T', real=True)

sig = Cymbol(r'\sigma', real=True)
sig_pi = Cymbol(r'\sigma^\pi', codename='sig_pi', real=True)
Y_N = Cymbol('Y_N', real=True)

Sig = sp.Matrix([tau_pi_x, tau_pi_y, sig_pi, Z, X_x, X_y, Y_T, Y_N])

# ## Helmholtz free energy

rho_psi_T_ = sp.Rational(1,2)* (
    (1-omega_T)*E_T*(s_x-s_pi_x)**2 +
    (1-omega_T)*E_T*(s_y-s_pi_y)**2 +
    K_T * z**2 +
    gamma_T * alpha_x**2 +
    gamma_T * alpha_y**2
)

rho_psi_N_ = sp.Rational(1,2) * (1 - H(sig_pi) * omega_N) * E_N * (w - w_pi)**2

rho_psi_ = rho_psi_T_ + rho_psi_N_

# The introduce the thermodynamic forces we have to differentiate Hemholtz free energy
# with respect to the kinematic state variables
# \begin{align}
# \frac{\partial \rho \psi }{\partial \boldsymbol{\mathcal{E}}}
# \end{align}

d_rho_psi_ = sp.Matrix([rho_psi_.diff(eps) for eps in Eps])

# To obtain consistent signs of the Helmholtz derivatives we define a sign switch operator so that all generalized forces are defined as positive for the respective conjugate state variable $\boldsymbol{\Upsilon}$. 

Sig_signs = sp.diag(-1,-1,-1,1,1,1,-1,-1)

# The constitutive laws between generalized force and kinematic variables then read
# \begin{align}
# \boldsymbol{\mathcal{S}} = \boldsymbol{\Upsilon}\frac{\rho \psi}{\partial\boldsymbol{\mathcal{E}}} 
# \end{align}

Sig_ = Sig_signs * d_rho_psi_

# **Executable code for** $\boldsymbol{\mathcal{S}}(s,\boldsymbol{\mathcal{E}})$
# To derive the time stepping procedure we will need also the matrix of derivatives of the generalized stresses $\boldsymbol{\mathcal{S}}$ with respect to the kinematic variables $\boldsymbol{\mathcal{E}}$
# \begin{align}
# \frac{\partial \boldsymbol{S}}{\partial \boldsymbol{E}}
# \end{align}

dSig_dEps_ = sp.Matrix([
    Sig_.T.diff(eps) for eps in Eps 
] ).T

# **Executable Python code generation** $\displaystyle \frac{\partial }{\partial \boldsymbol{\mathcal{E}}}  \boldsymbol{\mathcal{S}}(s,\boldsymbol{\mathcal{E}})$
# ## Threshold function
# To keep the framework general for different stress norms and hardening definitions let us first introduce a general function for effective stress. Note that the observable stress $\tau$ is identical with the plastic stress $\tau_\pi$ due to the performed sign switch in the definition of the thermodynamic forces.

tau_eff_x = sp.Symbol(r'tau_eff_x')
tau_eff_y = sp.Symbol(r'tau_eff_y')
sig_eff = sp.Symbol(r'sigma_eff')
Q_x = sp.Function('Q_x')(tau_eff_x,X_x)
Q_y = sp.Function('Q_y')(tau_eff_y,X_y)

# The stress norm is defined using the stress offset $X$, i.e. the kinematic hardening stress representing the shift of the origin of the yield locus.  

norm_Q = sp.sqrt(Q_x*Q_x + Q_y*Q_y)

# Let us now introduce the back stress $X$ by defining the substitution for $Q = \tau^\mathrm{eff} - X$

subs_Q = {Q_x: tau_eff_x - X_x, Q_y: tau_eff_y - X_y}

tau_eff_x_ = tau_pi_x / (1-omega_T)
tau_eff_y_ = tau_pi_y / (1-omega_T)
sig_eff_ = sig_pi / (1- H(sig_pi) * omega_N)

subs_tau_eff = {tau_eff_x: tau_pi_x / (1-omega_T),
                tau_eff_y: tau_pi_y / (1-omega_T),
                sig_eff: sig_pi / (1- H_switch * omega_N)}

# After substitutions the yield function reads

# **Smooth yield function**

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

f_eff_ = f_2.subs(fdc.symb.tau_bar, (bartau+Z))

# **Executable code generation** $f(\boldsymbol{\mathcal{E}}, \boldsymbol{\mathcal{S}})$
# 
# Note that this is a function of both the forces and kinematic state variables

# The derivative of $f$ required for time-stepping $\frac{\partial f}{\partial \boldsymbol{\mathcal{S}}}$ is obtained as

df_dSig_ = f_.diff(Sig)
ddf_dEps_ = f_.diff(Eps)
r = sp.symbols(r'r', positive=True)

###########################################################################

# and the corresponding directions of flow given as a product of the sign operator $\Upsilon$ and of the derivatives with respect to state variables
# $\boldsymbol{\Upsilon} \, \partial_{\boldsymbol{\mathcal{S}}} \varphi$
# This renders following flow direction vector
# \begin{align}
# \boldsymbol{\Phi} = - \Upsilon \frac{\partial \varphi}{\partial \boldsymbol{\mathcal{S}}} 
# \end{align}

class Slide34Expr(bu.SymbExpr):

    # control and state variables
    s_x, s_y, w, Eps, Sig = s_x, s_y, w, Eps, Sig

    # model parameters
    E_T = E_T
    gamma_T = gamma_T
    K_T = K_T
    S_T = S_T
    c_T = c_T
    bartau = bartau
    E_N = E_N
    S_N = S_N
    c_N = c_N
    f_t = f_t
    f_c = f_c
    f_c0 = f_c0
    m = m
    eta = eta
    r = r
    H_switch = H_switch
    Sig_signs = Sig_signs

    ONE = Cymbol(r'I')
    ZERO = Cymbol(r'O')

    # expressions
    Sig_ = Sig_.T
    dSig_dEps_ = dSig_dEps_.subs(0,ZERO) * ONE
    f_ = f_
    df_dSig_ = df_dSig_.subs(0,ZERO) * ONE
    ddf_dEps_ = ddf_dEps_.subs(0,ZERO) * ONE

    phi_final_ = tr.Property()
    @tr.cached_property
    def _get_phi_final_(self):
        def ari(var1, var2):
            return (var1 + var2) / 2
        def geo(var1, var2):
            return np.sqrt(var1*var2)
        def max(var1, var2):
            return sp.Piecewise( (var1, var1 > var2),
                                 (var2, True))
        def avg(var1, var2):
            return ari(var1, var2)
        c_NT = avg(c_N, c_T)
        S_NT = avg(S_N, S_T)
        omega_NT = ari(omega_N, omega_T)
        phi_N = (1 - omega_N)**(c_N) * S_N/(r+1) * (Y_N/S_N)**(r+1) * H_switch
        phi_T = (1 - omega_T)**(c_T) * S_T/(r+1) * (Y_T/S_T)**(r+1)
        phi_NT  = (1 - omega_NT)**(c_NT) * S_NT/(r+1) * ((Y_N+Y_T)/S_NT)**(r+1)
        phi_ = f_ + (1 - eta) * (phi_N + phi_T) + eta * phi_NT
        return phi_.subs(r,1) # @TODO - fix the passing of the parameter - it damages the T response

    Phi_final_ = tr.Property()
    @tr.cached_property
    def _get_Phi_final_(self):
        return -self.Sig_signs * self.phi_final_.diff(self.Sig)

    H_sig_pi_ = H(sig_pi)

    tau_eff_x_ = tau_pi_x / (1 - omega_T)
    tau_eff_y_ = tau_pi_y / (1 - omega_T)
    sig_eff_ = sig_pi / (1 - H(sig_pi) * omega_N)

    symb_model_params = [
        'E_T', 'gamma_T', 'K_T', 'S_T', 'c_T', 'bartau',
        'E_N', 'S_N', 'c_N', 'm', 'f_t', 'f_c', 'f_c0', 'eta', 'r'
    ]

    # List of expressions for which the methods `get_`
    symb_expressions = [
        ('Sig_', ('s_x', 's_y', 'w', 'Sig', 'Eps')),
        ('dSig_dEps_', ('s_x', 's_y', 'w', 'Sig', 'Eps', 'ZERO', 'ONE')),
        ('f_', ('Eps', 'Sig', 'H_switch')),
        ('df_dSig_', ('Eps', 'Sig', 'H_switch', 'ZERO', 'ONE')),
        ('ddf_dEps_', ('Eps', 'Sig', 'H_switch', 'ZERO', 'ONE')),
        ('phi_final_', ('Eps', 'Sig', 'H_switch')),
        ('Phi_final_', ('Eps', 'Sig', 'H_switch', 'ZERO', 'ONE')),
        ('H_sig_pi_', ('Sig',))
    ]

class ConvergenceError(Exception):
    """ Inappropriate argument value (of correct type). """

    def __init__(self, message, state):  # real signature unknown
        self.message = message
        self.state = state
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} for state {self.state}'

class Slide34(MATSEval,bu.InjectSymbExpr):

    name = 'Slide 3.4'
    symb_class = Slide34Expr

    E_T = bu.Float(28000, MAT=True)
    gamma_T = bu.Float(10, MAT=True)
    K_T = bu.Float(8, MAT=True)
    S_T = bu.Float(1, MAT=True)
    c_T = bu.Float(1, MAT=True)
    bartau = bu.Float(28000, MAT=True)
    E_N = bu.Float(28000, MAT=True)
    S_N = bu.Float(1, MAT=True)
    c_N = bu.Float(1, MAT=True)
    m = bu.Float(0.1, MAT=True)
    f_t = bu.Float(3, MAT=True)
    f_c = bu.Float(30, MAT=True)
    f_c0 = bu.Float(20, MAT=True)
    eta = bu.Float(0.5, MAT=True)
    r = bu.Float(1, MAT=True)

    debug = bu.Bool(False)

    def C_codegen(self):

        import os
        import os.path as osp

        C_code = []
        for symb_name, symb_params in self.symb.symb_expressions:
            c_func_name = 'get_' + symb_name
            c_func = ccode(c_func_name, getattr(self.symb, symb_name), 'SLIDE33')
            C_code.append(c_func)

        code_dirname = 'sympy_codegen'
        code_fname = 'SLIDE33_3D'

        home_dir = osp.expanduser('~')
        code_dir = osp.join(home_dir, code_dirname)
        if not osp.exists(code_dir):
            os.makedirs(code_dir)

        code_file = osp.join(code_dir, code_fname)

        print('generated code_file', code_file)
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

    ipw_view = bu.View(
        bu.Item('E_T', latex='E_T', minmax=(0.5, 100)),
        bu.Item('S_T', minmax=(.00001, 100)),
        bu.Item('c_T', minmax=( 0.0001, 10)),
        bu.Item('gamma_T', latex=r'\gamma_\mathrm{T}', minmax=(-20, 20)),
        bu.Item('K_T', minmax=(-20, 20)),
        bu.Item('bartau', latex=r'\bar{\tau}', minmax=(0.5, 20)),
        bu.Item('E_N', minmax=(0.5, 100)),
        bu.Item('S_N', minmax=(0.0001, 100)),
        bu.Item('c_N', minmax=(0.0001, 10)),
        bu.Item('m', minmax=(0.0001, 0.4)),
        bu.Item('f_t', minmax=(0.1, 10)),
        bu.Item('f_c', latex=r'f_\mathrm{c}', minmax=(1, 200)),
        bu.Item('f_c0', latex=r'f_\mathrm{c0}', minmax=(1, 100)),
        bu.Item('eta', minmax=(0, 1)),
        bu.Item('r')
    )

    damage_interaction = tr.Enum('final', 'geometric','arithmetic')

    get_phi_ = tr.Property
    def _get_get_phi_(self):
        return self.symb.get_phi_final_

    get_Phi_ = tr.Property
    def _get_get_Phi_(self):
        return self.symb.get_Phi_final_

    def get_f_df(self, s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k):
        if self.debug:
            print('s_x_n1', s_x_n1.dtype, s_x_n1.shape)
            print('s_y_n1', s_y_n1.dtype, s_y_n1.shape)
            print('w_n1', w_n1.dtype,w_n1.shape)
            print('Eps_k', Eps_k.dtype, Eps_k.shape)
            print('Sig_k', Sig_k.dtype, Sig_k.shape)
        ONES = np.ones_like(s_x_n1, dtype=np.float_)
        if self.debug:
            print('ONES', ONES.dtype)
        ZEROS = np.zeros_like(s_x_n1, dtype=np.float_)
        if self.debug:
            print('ZEROS', ZEROS.dtype)
        Sig_k = self.symb.get_Sig_(s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k)[0]
        if self.debug:
            print('Sig_k', Sig_k.dtype, Sig_k.shape)
        dSig_dEps_k = self.symb.get_dSig_dEps_(s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k, ZEROS, ONES)
        if self.debug:
            print('dSig_dEps_k', dSig_dEps_k.dtype)
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_k)
        if self.debug:
            print('H_sig_pi', H_sig_pi.dtype)
        f_k = np.array([self.symb.get_f_(Eps_k, Sig_k, H_sig_pi)])
        if self.debug:
            print('f_k', f_k.dtype)
        df_dSig_k = self.symb.get_df_dSig_(Eps_k, Sig_k, H_sig_pi, ZEROS, ONES)
        if self.debug:
            print('df_dSig_k',df_dSig_k.dtype)
        ddf_dEps_k = self.symb.get_ddf_dEps_(Eps_k, Sig_k, H_sig_pi, ZEROS, ONES)
        if self.debug:
            print('ddf_dEps_k',ddf_dEps_k.dtype)
        df_dEps_k = np.einsum(
            'ik...,ji...->jk...', df_dSig_k, dSig_dEps_k) + ddf_dEps_k
        Phi_k = self.get_Phi_(Eps_k, Sig_k, H_sig_pi, ZEROS, ONES)
        dEps_dlambda_k = Phi_k
        df_dlambda = np.einsum(
            'ki...,kj...->ij...', df_dEps_k, dEps_dlambda_k)
        df_k = df_dlambda
        return f_k, df_k, Sig_k

    def get_Eps_k1(self, s_x_n1, s_y_n1, w_n1, Eps_n, lam_k, Sig_k, Eps_k):
        '''Evolution equations:
        The update of state variables
        for an updated $\lambda_k$ is performed using this procedure.
        '''
        ONES = np.ones_like(s_x_n1)
        ZEROS = np.zeros_like(s_x_n1)
        Sig_k = self.symb.get_Sig_(s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k)[0]
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_k)
        Phi_k = self.get_Phi_(Eps_k, Sig_k, H_sig_pi, ZEROS, ONES)
        Eps_k1 = Eps_n + lam_k * Phi_k[:, 0]
        return Eps_k1

    rtol = bu.Float(1e-3, ALG=True)
    '''Relative tolerance of the return mapping algorithm related 
    to the tensile strength
    '''

    Eps_names = tr.Property
    @tr.cached_property
    def _get_Eps_names(self):
        return [eps.codename for eps in self.symb.Eps]

    Sig_names = tr.Property
    @tr.cached_property
    def _get_Sig_names(self):
        return [sig.codename for sig in self.symb.Sig]

    state_var_shapes = tr.Property
    @tr.cached_property
    def _get_state_var_shapes(self):
        '''State variables shapes:
        variables are using the codename string in the Cymbol definition
        Since the same string is used in the lambdify method via print_Symbol
        method defined in Cymbol as well'''
        return {eps_name: () for eps_name in self.Eps_names + self.Sig_names}

    k_max = bu.Int(100, ALG=True)
    '''Maximum number of iterations'''

    def get_corr_pred(self, eps_Ema, t_n1, **state):
        '''Return mapping iteration:
        This function represents a user subroutine in a finite element
        code or in a lattice model. The input is $s_{n+1}$ and the state variables
        representing the state in the previous solved step $\boldsymbol{\mathcal{E}}_n$.
        The procedure returns the stresses and state variables of
        $\boldsymbol{\mathcal{S}}_{n+1}$ and $\boldsymbol{\mathcal{E}}_{n+1}$
        '''
        eps_aEm = np.einsum('...a->a...',eps_Ema)
        dim = len(eps_aEm)

        if dim == 2: # hack - only one slip considered - 2D version
            select_idx = (0, 2)
            s_x_n1, w_n1 = eps_aEm
            s_y_n1 = np.zeros_like(s_x_n1)
        else:
            select_idx = (0, 1, 2)
            s_x_n1, s_y_n1, w_n1 = eps_aEm

        ONES = np.ones_like(s_x_n1, dtype=np.float_)
        if self.debug:
            print('ONES', ONES.dtype)
        ZEROS = np.zeros_like(s_x_n1, dtype=np.float_)
        if self.debug:
            print('ZEROS', ZEROS.dtype)

        # Transform state to Eps_k and Sig_k
        Eps_n = np.array([ state[eps_name] for eps_name in self.Eps_names], dtype=np.float_)
        Eps_k = np.copy(Eps_n)
        #Sig_k = self.symb.get_Sig_(Eps_k)
        Sig_k = np.array([state[sig_name] for sig_name in self.Sig_names], dtype=np.float_)
        f_k, df_k, Sig_k = self.get_f_df(s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k)
        f_k, df_k = f_k[0,...], df_k[0,0,...]
        f_k_trial = f_k
        # indexes of inelastic entries
        L = np.where(f_k_trial > 0)
        # f norm in inelastic entries - to allow also positive values less the rtol
        f_k_norm_I = np.fabs(f_k_trial[L])
        lam_k = np.zeros_like(f_k_trial)
        k = 0
        while k < self.k_max:
            if self.debug:
                print('k', k)
            # which entries are above the tolerance
            I = np.where(f_k_norm_I > (self.f_t * self.rtol))
            if self.debug:
                print('f_k_norm_I', f_k_norm_I, self.f_t * self.rtol, len(I[0]))
            if (len(I[0]) == 0):
                # empty inelastic entries - accept state
                #return Eps_k, Sig_k, k + 1
                dSig_dEps_k = self.symb.get_dSig_dEps_(s_x_n1, s_y_n1, w_n1, Sig_k, Eps_k, ZEROS, ONES)
                ix1, ix2 = np.ix_(select_idx, select_idx)
                D_ = np.einsum('ab...->...ab',dSig_dEps_k[ix1, ix2, ...])
                sig_ = np.einsum('a...->...a',Sig_k[select_idx,...])
                # quick fix
                omega_T = Eps_k[..., 6]
                omega_N = Eps_k[..., 7]
                D_ = np.zeros(sig_.shape + (sig_.shape[-1],))
                D_[...,0,0] = self.E_T * (1 - omega_T)
                if dim == 2:
                    D_[...,1,1] = self.E_N * (1 - omega_N)
                else:
                    D_[...,1,1] = self.E_T * (1 - omega_T)
                    D_[...,2,2] = self.E_N * (1 - omega_N)
                return sig_, D_

            if self.debug:
                print('I', I)
                print('L', L)
            LL = tuple(Li[I] for Li in L)
            L = LL
            if self.debug:
                print('new L', L)
                print('f_k', f_k[L].shape,f_k[L].dtype)
                print('df_k', df_k[L].shape,df_k[L].dtype)
            # return mapping on inelastic entries
            dlam_L = -f_k[L] / df_k[L] # np.linalg.solve(df_k[I], -f_k[I])
            if self.debug:
                print('dlam_I',dlam_L,dlam_L.dtype)
            lam_k[L] += dlam_L
            if self.debug:
                print('lam_k_L',lam_k,lam_k.dtype, lam_k[L].shape)
            L_slice = (slice(None),) + L
            Eps_k_L = self.get_Eps_k1(s_x_n1[L], s_y_n1[L], w_n1[L],
                                      Eps_n[L_slice],
                                      lam_k[L], Sig_k[L_slice], Eps_k[L_slice])
            Eps_k[L_slice] = Eps_k_L
            f_k_L, df_k_L, Sig_k_L = self.get_f_df(s_x_n1[L], s_y_n1[L], w_n1[L],
                                                   Sig_k[L_slice], Eps_k_L)
            f_k[L], df_k[L] = f_k_L[0, ...], df_k_L[0, 0, ...]
            Sig_k[L_slice] = Sig_k_L
            if self.debug:
                print('Sig_k',Sig_k)
                print('f_k', f_k)
            f_k_norm_I = np.fabs(f_k[L])
            k += 1
        else:
            raise ConvergenceError('no convergence for entries', [L, s_x_n1[I], s_y_n1[I], w_n1[I]])
        # add the algorithmic stiffness
        # recalculate df_k and -f_k for a unit increment of epsilon and solve for lambda
        #

    def plot_f_state(self, ax, Eps, Sig):
        lower = -self.f_c * 1.05
        upper = self.f_t + 0.05 * self.f_c
        lower_tau = -self.bartau * 2
        upper_tau = self.bartau * 2
        lower_tau = 0
        upper_tau = 10
        tau_x, tau_y, sig = Sig[:3]
        tau = np.sqrt(tau_x**2 + tau_y**2)
        sig_ts, tau_x_ts  = np.mgrid[lower:upper:201j,lower_tau:upper_tau:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + tau_x_ts.shape)
        Eps_ts = np.zeros_like(Sig_ts)
        Sig_ts[0,...] = tau_x_ts
        Sig_ts[2,...] = sig_ts
        Sig_ts[3:,...] = Sig[3:,np.newaxis,np.newaxis]
        Eps_ts[...] = Eps[:,np.newaxis,np.newaxis]
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])

        #phi_ts = np.array([self.symb.get_phi_(Eps_ts, Sig_ts)])
        ax.set_title('threshold function');

        omega_N = Eps_ts[-1,:]
        omega_T = Eps_ts[-2,:]
        sig_ts_eff = sig_ts / (1 - H_sig_pi*omega_N)
        tau_x_ts_eff = tau_x_ts / (1 - omega_T)
        ax.contour(sig_ts_eff, tau_x_ts_eff, f_ts[0,...], levels=0, colors=('green',))

        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0, colors=('red',))
        #ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot(sig, tau, marker='H', color='red')
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)
        ax.set_ylim(ymin=0, ymax=10)

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
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        f_ts = np.array([self.symb.get_f_(Eps_ts, Sig_ts, H_sig_pi)])
        phi_ts = np.array([self.get_phi_(Eps_ts, Sig_ts, H_sig_pi)])
        ax.set_title('threshold function');
        ax.contour(sig_ts, tau_x_ts, f_ts[0,...], levels=0)
        ax.contour(sig_ts, tau_x_ts, phi_ts[0, ...])
        ax.plot([lower, upper], [0, 0], color='black', lw=0.4)
        ax.plot([0, 0], [lower_tau, upper_tau], color='black', lw=0.4)

    def plot_phi_Y(self, ax):
        lower_N = 0
        upper_N = 1
        lower_T = 0
        upper_T = 1
        Y_N, Y_T  = np.mgrid[lower_N:upper_N:201j,lower_T:upper_T:201j]
        Sig_ts = np.zeros((len(self.symb.Eps),) + Y_T.shape)
        Sig_ts[0,:] = Y_N
        Sig_ts[2,:] = Y_T
        Eps_ts = np.zeros_like(Sig_ts)
        H_sig_pi = self.symb.get_H_sig_pi_(Sig_ts)
        phi_ts = np.array([self.get_phi_(Eps_ts, Sig_ts, H_sig_pi)])
        ax.set_title('potential function');
        ax.contour(Y_N, Y_T, phi_ts[0,...]) #, levels=0)

    def update_plot(self, ax):
        self.plot_f(ax)
