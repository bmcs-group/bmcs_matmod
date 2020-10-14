
C40MA = dict(
    gamma_T = 100000,
    K_T = 10000.0,
    S_T = 0.005,
    r_T = 9.,
    e_T = 12.,
    c_T = 4.6,
    tau_pi_bar = 1.7,
    a = 0.003,
    Ad = 100.0,
    eps_0 = 0.00008,
    K_N = 10000.,
    gamma_N = 5000.,
    sigma_0 = 30.,
    # -------------------------------------------------------------------------
    # Cached elasticity tensors
    # -------------------------------------------------------------------------
    E = 35e+3,
    nu = 0.2
)

C80MA = dict()
C120MA = dict()
Tensile = dict()
Compressive = dict()
Biaxial = dict()
Paper_2D = dict()

if False:
    if concrete_type == 1:  # #   C80 MA

        gamma_T = 1000000.

        K_T = 30000.0

        S_T = 0.01

        r_T = 14.

        c_T = 6.

        e_T = 14.

        tau_pi_bar = 2.0

        a = 0.01

        Ad = 1000.0

        eps_0 = 0.0001

        K_N = 30000.

        gamma_N = 20000.

        sigma_0 = 60.

        E = 42e+3

        nu = 0.2

    if concrete_type == 2:  # PARAMETERS FOR C120 MA

        gamma_T = 2000000.

        K_T = 2200.0

        S_T = 0.015

        r_T = 17.5

        c_T = 8.

        e_T = 10.

        tau_pi_bar = 1.8

        a = 0.008

        Ad = 1000.0

        eps_0 = 0.0001

        K_N = 35000.

        gamma_N = 25000.

        sigma_0 = 90.

        E = 44e+3

        nu = 0.2

    if concrete_type == 3:      # TENSILE PARAMETRIC STUDY

        gamma_T = 80000.

        K_T = 10000.0

        S_T = 0.000000001

        r_T = 1.21

        e_T = 1.

        c_T = 1.85

        tau_pi_bar = 0.1

        a = 0.001

        Ad = 1500.0

        eps_0 = 0.00008

        K_N = 4000.

        gamma_N = 20000.

        sigma_0 = 180.

        E = 35e+3

        nu = 0.2

    # COMPRESSION
    if concrete_type == 4:

        gamma_T = 10000.

        K_T = 10000.0

        S_T = 0.000007

        r_T = 1.2

        e_T = 1.

        c_T = 1.25

        tau_pi_bar = 5.

        a = 0.001

        Ad = 1500.0

        eps_0 = 0.00008

        K_N = 10000.

        gamma_N = 10000.

        sigma_0 = 30.

        E = 35e+3

        nu = 0.2

    # BI-AXIAL ENVELOPE
    if concrete_type == 5:

        gamma_T = 10000.

        K_T = 10000.0

        S_T = 0.000007

        r_T = 1.2

        e_T = 1.

        c_T = 1.8

        tau_pi_bar = 5.

        a = 0.01

        Ad = 50000.0

        eps_0 = 0.00008

        K_N = 15000.

        gamma_N = 20000.

        sigma_0 = 30.

        E = 35e+3

        nu = 0.2

    if concrete_type == 6:  # # Paper 2D resdistribution

        gamma_T = 800000.

        K_T = 50000.0

        S_T = 0.029

        r_T = 13.

        e_T = 11.

        c_T = 8

        tau_pi_bar = 2.

        a = 0.012

        Ad = 1000.0

        eps_0 = 0.0001

        K_N = 80000.

        gamma_N = 100000.

        sigma_0 = 80.

        # -------------------------------------------------------------------------
        # Cached elasticity tensors
        # -------------------------------------------------------------------------

        E = 42e+3

        nu = 0.2
