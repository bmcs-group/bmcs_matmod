from traits.api import HasTraits, Float, Int, Array, Instance, Property, \
    cached_property, List, Str, Any
import numpy as np
import bmcs_utils.api as bu
import sympy as sp
sp.init_printing(use_unicode=True, wrap_line=False)

from IPython.display import display, Math

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
            cls._registry[key] = sp.Symbol(name, **kwargs)
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

    vars = ['t'] # List of control variable names for symbolic representation
    params = []  # Class-level attribute to be overridden in subclasses

    def __new__(cls, *args, **kwargs):
        """Prepend the derivation of traits from class-level sp.Symbol definitions."""
        # Access the class-level 'params' attribute
        if not hasattr(cls, 'params') or not isinstance(cls.params, list):
            raise ValueError(f"Class '{cls.__name__}' must define a class-level 'params' attribute as a list.")

        class_traits = cls.class_traits()

        def build_symb_list(names):
            result = []
            for name in names:
                trait = class_traits.get(name, None)
                if trait is None:
                    raise ValueError(f"Trait '{name}' not found in class '{cls.__name__}'")
                symb_assumptions = getattr(trait, 'symbol', None)
                if symb_assumptions is not None:
                    symb_name = symb_assumptions.get('latex', name)
                else:
                    symb_assumptions = {}
                    symb_name = name
                trait_symb_name = f"{name}_sym"
                symbol = cls.get_symbol(symb_name, **symb_assumptions)
                setattr(cls, trait_symb_name, symbol)
                result.append(symbol)
            return tuple(result)

        cls.symb_vars = build_symb_list(getattr(cls, 'vars', []))
        cls.symb_params = build_symb_list(cls.params)
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
        return (getattr(self, param) for param in self.params)

    symb_expr = Property(depends_on='params')
    @cached_property
    def _get_symb_expr(self):
        return None  # Subclasses override if symbolic approach is needed

    def collect_symb_params(self):
        """
        Collect symbolic parameters from the class-level 'params' attribute.

        This is the default implementation, which simply returns the tuple of symbolic parameters
        as defined by the base class. Subclasses can override this method to adapt the collection
        of symbolic parameters if their parameter structure differs.

        Returns:
            list: A list of symbolic parameters.
        """
        return self.symb_params  # Exclude the first parameter (time)

    # Factor out the symbolic lambdification in the base class
    symb_fn_lambdified = Property
    @cached_property
    def _get_symb_fn_lambdified(self):
        if self.symb_expr is None:
            raise NotImplementedError("Subclasses must implement __call__ or define symb_expr.")
        symb_vars = self.symb_vars
        symb_params = self.collect_symb_params()
        return sp.lambdify(symb_vars + symb_params, self.symb_expr, 'numpy')

    name = Str('Time Function', desc="Name for plot titles.")

    def plot(self, ax, t, **plot_kwargs):
        """
        Plot the time function on the given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on.
        t : array-like
            The time array.
        plot_kwargs : dict
            Additional keyword arguments for ax.plot.
        """
        y = self(t)
        # Build label from class name and parameter values
        param_strs = []
        for param in self.params:
            val = getattr(self, param, None)
            if isinstance(val, float):
                param_strs.append(f"{param}={val:.3g}")
            elif isinstance(val, (np.ndarray, list)):
                param_strs.append(f"{param}=array")
            else:
                param_strs.append(f"{param}={val}")
        label = f"{self.__class__.__name__}({', '.join(param_strs)})"
        ax.plot(t, y, label=label, **plot_kwargs)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(self.name)
        ax.legend()
        ax.grid(True)

    def display_sym(self):
        """
        Display the symbolic expression in LaTeX format in a Jupyter notebook,
        prefixed by the class name and a double colon.
        """
        expr = self.symb_expr
        if expr is not None:
            latex_str = sp.latex(expr)
        else:
            latex_str = r"\text{No symbolic expression defined}"
        display(Math(f"{self.__class__.__name__}:: {latex_str}"))

class TimeFnStepLoading(TimeFnBase):
    """Step loading scenario."""
    name = Str('Step Loading')

    params = ['t_s', 'val']  # Removed 't'

    t_s = Float(2, symbol={'latex': r't_{\mathrm{s}}', 'real': True, 'positive': True}, desc="Step time.")
    val = Float(2, symbol={'latex': r'C', 'real': True}, desc="Step value.")

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        t_sym, t_s_sym, val_sym = self.t_sym, self.t_s_sym, self.val_sym
        return sp.Piecewise((0, t_sym <= t_s_sym), (val_sym, True))

class TimeFnMonotonicAscending(TimeFnBase):
    """Monotonic ascending loading scenario."""
    name = Str('Monotonic Ascending')

    params = ['rate']  # Removed 't'

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
    name = Str('Cyclic Sinusoidal')

    params = []  # Removed 't'

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        return sp.Piecewise(
            (sp.sin(2 * sp.pi * self.t_sym), self.t_sym < 1),
            (0, True)
        )

class TimeFnCycleLinear(TimeFnCyclicBase):
    """Cyclic symmetric saw-tooth loading scenario starting at zero."""
    name = Str('Cyclic Saw Tooth')

    params = []  # Removed 't'

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
    name = Str('Cycle With Ramps')

    params = ['urf', 'mrf', 'lrf']  # Removed 't'

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

class TimeFnPeriodic(TimeFnBase):
    """Periodic time function stacking multiple cycles."""
    name = Str('Periodic Time Function')

    time_fn_cycle = bu.Instance(TimeFnCyclicBase, desc="Time function for a single cycle.")
    mean_value = Float(0.0, symbol={'latex': r'\mu', 'real': True, 'positive': True},desc="Mean value to shift the function.")
    amplitude = Float(1.0, symbol={'latex': r'a', 'real': True, 'positive': True},desc="Amplitude to scale the function.")
    period = Float(1.0, symbol={'latex': r'T', 'real': True, 'positive': True},desc="Period of the periodic function.")

    params = ['mean_value', 'amplitude', 'period']  # Removed 't'

    dfn_dtime_at_zero_expr = Property(depends_on='time_fn_cycle.symb_expr')
    @cached_property
    def _get_dfn_dtime_at_zero_expr(self):
        t_sym = self.t_sym
        mean_value_sym, amplitude_sym, period_sym = self.symb_params
        scaled_and_shifted_expr = self.time_fn_cycle.symb_expr_at_zero_time.subs(t_sym, t_sym/period_sym) * amplitude_sym + mean_value_sym
        dfn_dtime_expr = sp.diff(scaled_and_shifted_expr, t_sym).subs(t_sym, 0)
        return dfn_dtime_expr

    t_start_cycling_expr = Property(depends_on='mean_value, time_fn_cycle.symb_expr')
    @cached_property
    def _get_t_start_cycling_expr(self):
        """
        Symbolic expression for the time at which cycling starts,
        i.e., mean_value / dfn_dtime_at_zero_expr.
        """
        mean_value_sym = self.mean_value_sym
        dfn_dtime_at_zero = self.dfn_dtime_at_zero_expr
        # Avoid division by zero symbolically
        return sp.Piecewise(
            (mean_value_sym / dfn_dtime_at_zero, dfn_dtime_at_zero != 0),
            (sp.oo, True)
        )

    symb_expr = Property(depends_on='time_fn_cycle.symb_expr')
    @cached_property
    def _get_symb_expr(self):
        """
        Symbolic expression for the periodic time function, combining:
        - Initial ascending branch up to mean_value (for t < t_start)
        - Periodic cyclic part (for t >= t_start), with explicit periodicity
        """
        # Symbols
        t_sym = self.t_sym
        mean_value_sym, amplitude_sym, period_sym = self.symb_params
        t_start_sym = self.t_start_cycling_expr

        # Slope for initial branch
        slope = mean_value_sym / t_start_sym

        # Scaled time for the cycle
        t_mod = sp.Mod(t_sym - t_start_sym, period_sym)
        scaled_t_mod = t_mod / period_sym

        # Cyclic part (scaled and shifted)
        cycle_expr = self.time_fn_cycle.symb_expr.subs(self.time_fn_cycle.t_sym, scaled_t_mod)
        periodic_part = amplitude_sym * cycle_expr + mean_value_sym

        # Piecewise definition
        expr = sp.Piecewise(
            (slope * t_sym, t_sym < t_start_sym),
            (periodic_part, t_sym >= t_start_sym)
        )
        return expr

    def collect_symb_params(self):
        """
        Collect symbolic parameters for the periodic function.

        This method overrides the base class implementation to include
        the symbolic parameters specific to the periodic function.
        """
        return self.symb_params + self.time_fn_cycle.collect_symb_params()

    def get_args(self):
        return tuple(super().get_args()) + tuple(self.time_fn_cycle.get_args())

class TimeFnStepping(TimeFnBase):
    """Piecewise constant (stepping) time function."""
    name = Str('Stepping Function')

    params = ['step_times', 'step_values']

    step_times = Array(dtype=float, symbol={'latex': r'\vec{t}_\mathrm{steps}'}, desc="Times at which steps occur.")
    step_values = Array(dtype=float, symbol={'latex': r'\vec{v}_\mathrm{steps}'}, desc="Values at each step.")

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        # Symbolic piecewise: for each interval, value is constant
        t_sym, step_times_sym, step_values_sym = self.t_sym, self.step_times_sym, self.step_values_sym
        # For symbolic, assume step_times and step_values are lists of symbols
        # For practical use, only numeric evaluation is supported
        # So, return None here and implement __call__ directly
        return None

    def __call__(self, t):
        # Numeric implementation: piecewise constant
        t = np.asarray(t)
        step_times = np.asarray(self.step_times)
        step_values = np.asarray(self.step_values)
        result = np.full_like(t, step_values[0], dtype=float)
        for i, st in enumerate(step_times):
            result[t >= st] = step_values[i]
        return result

class TimeFnOverlay(TimeFnBase):
    """Overlay (product) of two time functions."""
    name = Str('Overlay Function')

    params = []  # No explicit parameters; all come from fn1 and fn2

    fn1 = bu.Instance(TimeFnBase, desc="First time function.")
    fn2 = bu.Instance(TimeFnBase, desc="Second time function.")

    symb_expr = Property
    @cached_property
    def _get_symb_expr(self):
        # If both have symbolic expressions, return their product
        if self.fn1.symb_expr is not None and self.fn2.symb_expr is not None:
            return self.fn1.symb_expr * self.fn2.symb_expr
        return None

    def collect_symb_params(self):
        """Collect symbolic parameters from both fn1 and fn2."""
        params1 = self.fn1.collect_symb_params() if hasattr(self.fn1, 'collect_symb_params') else ()
        params2 = self.fn2.collect_symb_params() if hasattr(self.fn2, 'collect_symb_params') else ()
        return tuple(params1) + tuple(params2)

    def get_args(self):
        """Concatenate arguments from both fn1 and fn2."""
        args1 = self.fn1.get_args() if hasattr(self.fn1, 'get_args') else ()
        args2 = self.fn2.get_args() if hasattr(self.fn2, 'get_args') else ()
        return tuple(args1) + tuple(args2)

    def __call__(self, t):
        # Fallback to direct numerical computation if symbolic approach fails
        if self.symb_expr is None or self.symb_fn_lambdified is None:
            return self.fn1(t) * self.fn2(t)
        try:
            return super().__call__(t)
        except Exception:
            return self.fn1(t) * self.fn2(t)
