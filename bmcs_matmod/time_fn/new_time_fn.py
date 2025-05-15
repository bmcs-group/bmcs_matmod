from traits.api import HasTraits, Float, Int, Array, Instance, Property, \
    cached_property, List, Str, Any
import numpy as np
import bmcs_utils.api as bu
import sympy as sp
sp.init_printing(use_unicode=True, wrap_line=False)

class SymbolRegistry:
    """
    A registry for managing and reusing symbolic variables.

    This class provides a mechanism to create and reuse symbolic variables
    (e.g., from SymPy) with specific names and attributes. It ensures that
    symbols with the same name and attributes are not duplicated, but instead
    reused from the registry.

    Attributes:
    -----------
    _registry : dict
        A class-level dictionary that stores created symbols. The keys are
        tuples of the symbol name and a frozenset of keyword arguments, and
        the values are the corresponding symbolic variables.

    Methods:
    --------
    get_symbol(name: str, **kwargs) -> sp.Symbol:
        Retrieves a symbolic variable with the specified name and attributes.
        If the symbol does not already exist in the registry, it is created
        and added to the registry. Otherwise, the existing symbol is returned.

        Parameters:
        -----------
        name : str
            The name of the symbolic variable.
        **kwargs : dict
            Additional attributes for the symbolic variable.

        Returns:
        --------
        sp.Symbol
            The symbolic variable with the specified name and attributes.
    """
    _registry = {}

    @classmethod
    def get_symbol(cls, name, **kwargs):
        key = (name, frozenset(kwargs.items()))
        if key not in cls._registry:
            print(f"Creating new symbol: {name} with kwargs: {kwargs}")
            cls._registry[key] = sp.Symbol(name, **kwargs)
        else:
            print(f"Reusing symbol: {name} with kwargs: {kwargs}")
        return cls._registry[key]

class TimeFnBase(bu.Model, SymbolRegistry):
    """
    TimeFnBase is a base class for defining time-dependent functions with symbolic and numerical capabilities. 
    It integrates traits-based modeling with symbolic computation using SymPy.

    Attributes:
        params (list): A class-level attribute that must be overridden in subclasses. 
                       It defines the list of parameters for the time-dependent function.
        t (Float): A trait representing the current time with symbolic assumptions 
                   (e.g., positive, real) and a LaTeX representation.

    Properties:
        symb_expr (Property): A symbolic expression representing the function. 
                              Subclasses should override this property if symbolic computation is required.
        symb_fn_lambdified (Property): A lambdified version of the symbolic expression 
                                       for numerical evaluation. It is automatically generated 
                                       from `symb_expr` if defined.

    Methods:
        __new__(cls, *args, **kwargs): Ensures that the class-level `params` attribute is defined 
                                       and initializes symbolic parameters based on traits.
        __init__(self, **traits): Initializes the instance with the provided traits.
        __call__(self, t): Evaluates the time-dependent function at a given time `t`. 
                           If `symb_fn_lambdified` is not defined, subclasses must implement this method.
        get_args(self): Returns a generator for the instance attributes corresponding to 
                        the parameters defined in `params` (excluding the first parameter).

    Notes:
        - Subclasses must define the `params` attribute as a list of parameter names.
        - If symbolic computation is required, subclasses should override the `symb_expr` property.
        - If `symb_expr` is not defined, subclasses must implement the `__call__` method directly.
    """
    """Base class for time-dependent functions."""

    params = []  # Class-level attribute to be overridden in subclasses

    def __new__(cls, *args, **kwargs):
        """Prepend the derivation of traits from class-level sp.Symbol definitions."""
        # Access the class-level 'params' attribute
        if not hasattr(cls, 'params') or not isinstance(cls.params, list):
            raise ValueError(f"Class '{cls.__name__}' must define a class-level 'params' attribute as a list.")

        class_traits = cls.class_traits()

        symb_params = []
        for trait_name in cls.params:
            trait = class_traits.get(trait_name, None)
            if trait is None:
                raise ValueError(f"Trait '{trait_name}' not found in class '{cls.__name__}'")
            symb_assumptions = getattr(trait, 'symbol', None)
            if symb_assumptions is not None:
                symb_name = symb_assumptions.get('latex', trait_name)
            else:
                symb_assumptions = {}
                symb_name = trait_name
            trait_symb_name = f"{trait_name}_sym"
            symbol = cls.get_symbol(symb_name, **symb_assumptions)
            setattr(cls, trait_symb_name, symbol)
            symb_params.append(symbol)

        cls.symb_params = tuple(symb_params)
        return super().__new__(cls)

    t = Float(0.0, symbol={'latex': r't', 'real': True, 'positive': True}, desc="Current time.")

    def __init__(self, **traits):
        """Standard initialization process."""
        # Initialize the base class with the provided traits
        super().__init__(**traits)

    def __call__(self, t):
        # If symb_fn_lambdified is None, fallback to child-defined __call__
        return self.symb_fn_lambdified(t, *self.get_args())

    def get_args(self):
        return (getattr(self, param) for param in self.params[1:])

    symb_expr = Property(depends_on='params')
    @cached_property
    def _get_symb_expr(self):
        return None  # Subclasses override if symbolic approach is needed

    # Factor out the symbolic lambdification in the base class
    symb_fn_lambdified = Property
    @cached_property
    def _get_symb_fn_lambdified(self):
        if self.symb_expr is None:
            raise NotImplementedError("Subclasses must implement __call__ or define symb_expr.")
        return sp.lambdify(self.symb_params, self.symb_expr, 'numpy')

class StepLoading(TimeFnBase):
    """Step loading scenario."""

    params = ['t', 't_s', 'val']  # Override the class-level attribute

    t_s = Float(1.0, symbol={'latex': r't_\mathrm{s}s', 'real': True, 'positive': True}, desc="Time at which the step occurs.")
    val = Float(1.0, symbol={'latex': r'C', 'real': True}, desc="Value after the step.")

    # symb_expr = Property
    # @cached_property
    def _get_symb_expr(self):
        t_sym, t_s_sym, val_sym = self.symb_params
        return sp.Piecewise((0, self.t_sym <= self.t_s_sym), (self.val_sym, True))  # Updated reference

class MonotonicAscending(TimeFnBase):
    """Monotonic ascending loading scenario."""

    params = ['t', 'rate']  # Override the class-level attribute

    rate = Float(1.0, symbol={'latex': r'\varrho', 'real': True, 'positive': True}, desc="Rate of increase.")

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        return self.rate_sym * self.t_sym

class TimeFnCyclicBase(TimeFnBase):
    symb_expr_at_zero_time = Property(depends_on='symb_expr')
    @cached_property
    def _get_symb_expr_at_zero_time(self):
        return self.symb_expr

class TimeFnCycleSinus(TimeFnCyclicBase):
    """Cyclic sinusoidal loading scenario with a single cycle."""

    params = ['t']  # Override the class-level attribute

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        return sp.Piecewise(
            (sp.sin(2 * sp.pi * self.t_sym), self.t_sym < 1),
            (0, True)
        )


class TimeFnCycleLinear(TimeFnCyclicBase):
    """Cyclic symmetric saw-tooth loading scenario starting at zero."""

    params = ['t']  # Override the class-level attribute

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        quarter_period = sp.Rational(1, 4)
        three_quarters_period = sp.Rational(3, 4)

        slope_up = 1 / quarter_period
        slope_down = -1 / quarter_period

        piecewise_expr = sp.Piecewise(
            (slope_up * self.t_sym, self.t_sym < quarter_period),
            (1 + slope_down * (self.t_sym - quarter_period), self.t_sym < three_quarters_period),
            (-1 + slope_up * (self.t_sym - three_quarters_period), self.t_sym < 1),
            (0, True)
        )
        return sp.simplify(piecewise_expr)


class TimeFnCycleWithRamps(TimeFnCyclicBase):
    """Single cycle of loading with ramps and horizontal branches defined by fractions of the period."""

    params = ['t', 'urf', 'mrf', 'lrf']  # Override the class-level attribute

    urf = Float(0.1, symbol={'latex': r'\theta_{\mathrm{upper}}', 'real': True, 'positive': True}, desc="Fraction of period spent relaxing at upper bound.")
    mrf = Float(0.1, symbol={'latex': r'\theta_{\mathrm{middle}}', 'real': True, 'positive': True}, desc="Fraction of period spent relaxing at zero.")
    lrf = Float(0.1, symbol={'latex': r'\theta_{\mathrm{lower}}', 'real': True, 'positive': True}, desc="Fraction of period spent relaxing at lower bound.")

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        t_sym, urf_sym, mrf_sym, lrf_sym = self.t_sym, self.urf_sym, self.mrf_sym, self.lrf_sym
        sum_f = urf_sym + mrf_sym + lrf_sym
        active_time = 1 * (1 - sum_f)  # Period is assumed to be 1

        t_a = active_time / 4
        slope_up = 1 / t_a
        slope_down = -1 / t_a

        t1 = t_a
        t2 = t1 + 1 * urf_sym
        t3 = t2 + t_a
        t4 = t3 + 1 * mrf_sym
        t5 = t4 + t_a
        t6 = t5 + 1 * lrf_sym
        t7 = t6 + t_a

        piecewise_expr = sp.Piecewise(
            (slope_up * t_sym, t_sym < t1),
            (1, t_sym < t2),
            (1 + slope_down * (t_sym - t2), t_sym < t3),
            (0, t_sym < t4),
            (slope_down * (t_sym - t4), t_sym < t5),
            (-1, t_sym < t6),
            (-1 + slope_up * (t_sym - t6), t_sym < t7),
            (0, True)
        )
        return sp.simplify(piecewise_expr)

    symb_expr_at_zero_time = Property(depends_on='symb_expr')
    @cached_property
    def _get_symb_expr_at_zero_time(self):
        """
        Determine the part of the symbolic expression relevant for calculating the slope at zero time.

        This property identifies and returns the portion of the symbolic expression (`symb_expr`) 
        needed to compute the slope at the initial time (zero time). While this is typically the 
        first derivative in most cases, for piecewise functions where the relevant part isn't 
        predefined, it explicitly extracts and returns the necessary subexpression.

        Returns:
            sympy.Expr: The relevant part of the symbolic expression for slope calculation.
        """
        t_sym, urf_sym, mrf_sym, lrf_sym = self.t_sym, self.urf_sym, self.mrf_sym, self.lrf_sym
        sum_f = urf_sym + mrf_sym + lrf_sym
        active_time = 1 * (1 - sum_f)

        t_a = active_time / 4
        slope_up = 1 / t_a
        return slope_up * self.t_sym

class PeriodicTimeFn(TimeFnBase):
    """Periodic time function stacking multiple cycles."""

    time_fn_cycle = bu.Instance(TimeFnBase, desc="Time function for a single cycle.")
    mean_value = Float(0.0, symbol={'latex': r'\mu', 'real': True, 'positive': True},desc="Mean value to shift the function.")
    amplitude = Float(1.0, symbol={'latex': r'a', 'real': True, 'positive': True},desc="Amplitude to scale the function.")
    period = Float(1.0, symbol={'latex': r'T', 'real': True, 'positive': True},desc="Period of the periodic function.")

    params = ['t', 'mean_value', 'amplitude', 'period']  # Override the class-level attribute

    dfn_dtime_at_zero_expr = Property(depends_on='time_fn_cycle.symb_expr')
    @cached_property
    def _get_dfn_dtime_at_zero_expr(self):
        t_sym = self.time_fn_cycle.symb_params
        t_sym, mean_value_sym, amplitude_sym, period_sym = self.symb_params
        scaled_and_shifted_expr = self.time_fn_cycle.symb_expr_at_zero_time.subs(t_sym, t_sym/period_sym) * amplitude_sym + mean_value_sym
        dfn_dtime_expr = sp.diff(scaled_and_shifted_expr, t_sym).subs(t_sym, 0)
        return dfn_dtime_expr

    t_start_cycling = Property(depends_on='mean_value, time_fn_cycle.symb_expr')
    @cached_property
    def _get_t_start_cycling(self):
        symb_params = self.symb_params + self.time_fn_cycle.symb_params[1:]
        get_dfn_dtime_at_zero = sp.lambdify(symb_params, self.dfn_dtime_at_zero_expr, 'numpy')
        args = tuple(self.get_args()) + tuple(self.time_fn_cycle.get_args())
        dfn_dtime_at_zero = get_dfn_dtime_at_zero(0, *args)
        if dfn_dtime_at_zero == 0:
            raise ValueError("Derivative at time zero is zero, cannot compute t_start_cycling.")
        print(dfn_dtime_at_zero, self.mean_value / dfn_dtime_at_zero)
        return self.mean_value / dfn_dtime_at_zero

    def __call__(self, t):
        cycle_period = self.period  # Use the specified period
        t_start = self.t_start_cycling

        # Initial ascending branch up to mean value
        if np.any(t < t_start):
            slope = self.mean_value / t_start
            initial_branch = slope * t[t < t_start]
        else:
            initial_branch = np.array([])

        # Periodic part
        t_mod = (t[t >= t_start] - t_start) % cycle_period
        scaled_t_mod = t_mod / cycle_period  # Scale time to match the cycle's period
        cycle_values = self.time_fn_cycle(scaled_t_mod)
        scaled_shifted_values = self.amplitude * cycle_values + self.mean_value

        # Combine initial branch and periodic part
        return np.concatenate((initial_branch, scaled_shifted_values))
