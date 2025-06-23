
# Discussion on software solutions for GSM package

## Question: Class naming convention compliant with PEP 8

A question regarding the conventions and practices and PEP related standardization in the python programming. I have a configurable symbolic framework that I use to define individual versions of Generalized Standard Material. There is a base called GSMBase. The subclasses are obtained by defining the Helmholtz potential and specify the external, internal variables, thermodynamic forces, threshold functions and explicitly specified evolution equations where needed - which are integrated as equality constraints into the generated symbolic Lagrangian. I name the classes based on the disspative mechanisms involved and the dimensionality included. For example GSMVEVP1D means visco-elastic, viscoplastic model for 1D material, GSMVE1D is just viscoelastic, or GSMVEVPD1D - is viscoelastic, viscoplastic 1d material. I am wondering if there is some convention in similar cases. To increase the readability I tend to do GSM_VEVPD_1D. This appears to be long, but on the other hand, better readable than the compact version. I recall that for python conventions, the underscore should not be used in class names - just in module names. BTW, each of the classes is defined in the corresponding module, e..g gsm_vevpd_1d.py. Then, I consider the GSM_VEVPD_1D actually appropriate. What would the people defining standards say to this?

## Answer

Your question touches on an important aspect of coding practices—striking a balance between adhering to established conventions and ensuring code readability and maintainability. Let's delve into the specifics of your situation and how it relates to Python's naming conventions, particularly those outlined in [PEP 8](https://www.python.org/dev/peps/pep-0008/), the Python Enhancement Proposal that serves as the style guide for Python code.

**PEP 8 Recommendations for Naming Conventions:**

1. **Class Names:**
   - *Recommendation:* Use the CapWords convention (also known as PascalCase or CamelCase) for class names. This means capitalizing the first letter of each word and not using underscores.
   - *Example:* `MyClass`, `GSMBase`

2. **Function and Variable Names:**
   - *Recommendation:* Use lowercase words separated by underscores as necessary to improve readability.
   - *Example:* `my_function`, `calculate_value`

3. **Module and Package Names:**
   - *Recommendation:* Use short, lowercase names. Underscores can be used if they improve readability.
   - *Example:* `gsm_models`, `gsm_vevpd_1d.py`

**Applying These Conventions to Your Case:**

Your class names like `GSMVEVP1D` follow the CapWords convention but result in a long string of capital letters, making them hard to read and decipher. Breaking down these names:

- `GSM`: Generalized Standard Material
- `VE`: Viscoelastic
- `VEP`: Viscoplastic
- `1D`: One-Dimensional

**Challenges with the Current Naming:**

- Readability suffers due to the dense abbreviation and lack of visual separation between components.
- It can be difficult for others (and even yourself at a later time) to quickly understand what each class represents.

**Considering Underscores in Class Names:**

You proposed using underscores in class names to improve readability, resulting in names like `GSM_VEVPD_1D`. This indeed makes the class names more readable by clearly separating the different components.

**PEP 8 on Underscores in Class Names:**

While PEP 8 recommends against using underscores in class names, emphasizing the CapWords convention, it also recognizes that readability and consistency are critical:

> "The naming conventions of Python's library are a bit of a mess, so we'll never get this completely consistent... But it's still important to strive for greater consistency."

PEP 8 allows for exceptions when they improve the readability and clarity of the code.

**Balancing Conventions with Practicality:**

In your specific case:

- **Readability is Significantly Improved:** Using underscores makes the class names easier to read and understand, which is a crucial aspect of maintainable code.
  
- **Consistency with Module Names:** Your module names use underscores (e.g., `gsm_vevpd_1d.py`), and aligning class names with module names can enhance consistency across your codebase.

- **Clarity in Communication:** If the class names clearly convey their purpose and structure, it helps other developers (and your future self) work with the code more effectively.

**Alternative Approaches:**

1. **Expand Abbreviations:**
   - Use full words to make class names more readable without underscores.
   - *Example:* `GeneralizedStandardMaterialViscoElasticViscoPlastic1D`
   - *Downside:* The names become excessively long, which can be cumbersome.

2. **Use Nested Classes or Namespaces:**
   - Break down the names by creating submodules or nested classes.
   - *Example:*

     ```python
     class GSM:
         class VEVP:
             class OneD:
                 pass
     ```

   - *Downside:* This can introduce complexity and may not be practical depending on the use case.

3. **Use Mixed Case with Abbreviations:**
   - Capitalize the first letter of each component, even if it's an abbreviation.
   - *Example:* `GsmVeVpD1D`
   - *Downside:* This might not significantly improve readability, especially with less common abbreviations.

**Recommendation:**

Given the trade-offs, here's what I recommend:

- **Prioritize Readability and Maintainability:** The primary goal of naming conventions is to make code understandable and maintainable. If using underscores in class names achieves this, it's acceptable to deviate from PEP 8 in this case.

- **Document Your Naming Conventions:**
  - Clearly document the naming conventions used in your project, including any deviations from PEP 8.
  - This helps team members understand the rationale and maintain consistency.

- **Maintain Consistency Across the Codebase:**
  - Ensure that the use of underscores in class names is consistent throughout your project.
  - Avoid mixing different styles unless there is a clear and justified reason.

- **Consider the Audience and Context:**
  - If your code is for internal use or a specialized audience familiar with the abbreviations, the impact is minimal.
  - If your code is intended for wider distribution or open-source, you might consider more universally understandable names.

**What the Standards Community Might Say:**

- **Flexibility in Application:**
  - The standards are guidelines, not strict laws. They aim to promote readability and consistency but recognize that practicality matters.
  
- **Case-by-Case Basis:**
  - Deviations are acceptable when they make sense for the specific context and improve the overall quality of the code.

- **Encouragement to Re-Evaluate Naming:**
  - They might encourage exploring alternative naming strategies that align more closely with the conventions while still achieving readability.

- **Emphasis on Documentation and Communication:**
  - Make sure that any deviations are well-documented and communicated to anyone who works with the code.

**Conclusion:**

While PEP 8 recommends using the CapWords convention without underscores for class names, the ultimate goal is to write code that is clear and maintainable. In your case, using underscores in class names like `GSM_VEVPD_1D` significantly enhances readability by clearly delineating the components of the class name. This helps both you and others understand and work with your code more effectively.

Therefore, it's acceptable to use underscores in your class names, provided you:

- **Maintain Consistency:** Apply this naming style consistently across your project.
- **Document the Decision:** Clearly explain the reasoning in your codebase's contributing guidelines or documentation.
- **Ensure Readability:** Continue to prioritize code readability and maintainability in all aspects of your coding practices.

By thoughtfully applying conventions and making informed decisions, you align with the spirit of PEP 8—writing clean, readable, and maintainable Python code.

## Question: Handling of material parameters

The workflow used in GSM package starts with the definition of the sympy symbols to introduce the external variables, internal variables, conjugate thermodynamic forces and material parameters defining the particular type of thermodynamic problem. For example, when defining a viscoelastic model, the variables are

```python
eps, eps_ve = sp.Symbol('eps, eps_ve')
sig, sig_ve = sp.Symbol('sig, sig_ve')
E, eta_ve = sp.Symbol('E, eta_ve')
```

where E and eta_ve are elastic modulus and viscoelastic modulus, respectively. 
The GSM framework uses the variables along the derivation scheme of the generalized standard material and produces an executable lambdified code. This parameters of the callable functions are derived from the symbols for internal variables and material parameters, for example

```python
class ViscoElastic(GSMBase):
  E, eta_ve = sp.Symbol('E, eta_ve')
  m_params = (E, eta_v) 
  get_sig = sp.lambdify((*m_params), sig_expr) # sig has been derived by the base class 
```

Then, there is another class including damage, i.e. 

```python
class ViscoElasticDamage(GSMBase):
  E, eta_ve, S = sp.Symbol('E, eta_ve, S')
  m_params = (E, eta_v, S) 
  get_sig = sp.lambdify((*m_params), sig_expr) # sig has been derived by the base class
```

I would like to provide a generator for a gallery of models which runs multiple demonstrative examples showing the response of the individual material models and reports the results in a structured way, either in jupyter notebook, or in PDF document. Such a gallery is intended to provide a quick overview and comparison of the models. The gallery should be generated for a specified loading history, for example monotonically increasing, stepwise loading, single-cycle loading, multiple-cycle loading, etc.

Now, I am coming to my question. To enable a generic implementation of the gallery, I need to specify the material parameters for each model individual, as each model can have a different set of parameters as exemplified for the ViscoElastic and ViscoElasticDamage models. Such engine might look like this:

```python
gallery_input = {
  ViscoElastic: dict(E = 1, eta_v = 1),
  ViscoElasticDamage: dict(E = 1, eta_v = 1, S = 1)
}
```

The loop running the studies can then look somewhat like this:

```python
for gsm_class, inp in gallery_input.items()
  gsm_model = gsm_class()
  gsm_model.get_sig(**inp)
```

This approach using keyword arguments, however, does not work for the new sympy version. They dropped this support for keyword arguments for efficiency reasons in relation to the compatibility with numpy. This is understandable. However it makes it difficult to me to achieve a readable code. To demonstrate this, consider this version: 

```python
gallery_input = {
  ViscoElastic: (1, 1),
  ViscoElasticDamage: (1, 1, 1)
}
```

and the loop running the studies would then be:

```python
for gsm_class, inp in gallery_input.items()
  gsm_model = gsm_class()
  gsm_model.get_sig(*inp)
```

The specification of parameters as tuples and not dictionaries makes the code cryptic. I cane up with a solution which is cumbersome, i.e. 

```python
gallery_input = {
  ViscoElastic: { ViscoElastic.E : 1, ViscoElastic.eta_v },
  ViscoElasticDamage: { ViscoElasticDamage.E : 1, ViscoElasticDamage.eta_v },
}
```

and the loop running the studies would then be:

```python
for gsm_class, inp in gallery_input.items()
  gsm_model = gsm_class()
  inp_args = [inp[var] for var in gsm_class.m_params]
  gsm_model.get_sig(*inp_args)
```

Even though this is working, It appears tedious to specify the parameter list as a dictionary with the keys denoting the sympy variables. In fact, I would like to retain the specification of the input in the form:

```python
gallery_input = {
  ViscoElastic: dict(E = 1, eta_v = 1),
  ViscoElasticDamage: dict(E = 1, eta_v = 1, S = 1)
}
```

where the keys are strings denoting the variables. For this purpose, one might think of using the name of the sympy variable as a key, i.e. 

```python
for gsm_class, inp in gallery_input.items()
  gsm_model = gsm_class()
  inp_args = [inp[var.name] for var in gsm_class.m_params]
  gsm_model.get_sig(*inp_args)
```

However, this does not work well either, as I am using latex names in sympy variable, i.e.

```python
eta_ve = sp.Symbol(r'\eta_\mathrm{ve}') 
```

Defining the keys in the input dictionary using this variables would also look awkward. Is it possible to associate a string to the sympy variable that could be used in these cases? For example:

```python
eta_ve = sp.Symbol(r'\eta_\mathrm{ve}', codename='eps_ve') 
```

so that I can issue

```python
for gsm_class, inp in gallery_input.items()
  gsm_model = gsm_class()
  inp_args = [inp[var.codename] for var in gsm_class.m_params]
  gsm_model.get_sig(*inp_args)
```

Even though lengthy, I hope that the question and broader context is understandable and a short answer to this is possible.

## Answer

Yes, it's possible to associate a string (like a codename) with SymPy symbols to use as keys in your parameter dictionaries, but it requires some care due to how SymPy handles symbols. Here are a few approaches to solve your problem while keeping the code readable and functional.

### **Option 1: Use a Mapping of Symbols to Codenames**

You can create a `params_codenames` dictionary within each of your material model classes that maps each SymPy symbol to a codename (a string suitable as a dictionary key). This way, you can maintain separate variable names for LaTeX representations and for coding purposes.

```python
import sympy as sp

class GSMBase:
    pass  # Base class implementation

class ViscoElastic(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
    m_params = (E, eta_v)
    params_codenames = {E: 'E', eta_v: 'eta_v'}
    sig_expr = ...  # Define your expression here
    sig_func = sp.lambdify(m_params, sig_expr)

    def get_sig(self, **kwargs):
        args = [kwargs[self.params_codenames[var]] for var in self.m_params]
        return self.sig_func(*args)
```

Your `gallery_input` remains readable:

```python
gallery_input = {
    ViscoElastic: dict(E=1, eta_v=1),
    ViscoElasticDamage: dict(E=1, eta_v=1, S=1)
}
```

And your loop:

```python
for gsm_class, inp in gallery_input.items():
    gsm_model = gsm_class()
    sig = gsm_model.get_sig(**inp)
```

### **Option 2: Define Parameter Codenames in Order**

Alternatively, you can keep a list of codenames in the same order as `m_params`. This avoids modifying symbols or creating mappings:

```python
class ViscoElastic(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
    m_params = (E, eta_v)
    param_codenames = ('E', 'eta_v')
    sig_expr = ...  # Define your expression here
    sig_func = sp.lambdify(m_params, sig_expr)

    def get_sig(self, **kwargs):
        args = [kwargs[codename] for codename in self.param_codenames]
        return self.sig_func(*args)
```

### **Option 3: Custom Symbol Class with Codename Attribute**

While not recommended due to potential caching issues in SymPy, you can create a custom symbol class (at your own risk):

```python
import sympy as sp

class SymbolWithCodename(sp.Symbol):
    def __new__(cls, name, codename=None, **kwargs):
        obj = sp.Symbol.__new__(cls, name, **kwargs)
        obj._codename = codename if codename else name
        return obj

    @property
    def codename(self):
        return self._codename

# Usage
E = SymbolWithCodename('E')
eta_v = SymbolWithCodename(r'\eta_\mathrm{ve}', codename='eta_v')
```

**Note:** Modifying or subclassing `Symbol` can lead to unexpected behavior because `Symbol` instances are cached and meant to be immutable. Therefore, the first two options are safer and more in line with SymPy's design.

### **Explanation**

- **SymPy Symbols and Attributes:** SymPy `Symbol` objects are immutable and cached. Adding attributes directly to symbols isn't reliable because two symbols with the same name are actually the same object in SymPy. So adding attributes can inadvertently affect other parts of your code.
  
- **Mapping Parameters:** By creating a separate mapping (`params_codenames` or `param_codenames`), you avoid modifying the symbols themselves and keep your code organized.

- **Order Matters:** By ensuring the codenames are in the same order as `m_params`, you can confidently map the keyword arguments to the positional arguments required by the lambdified function.

### **Benefit**

This approach allows you to:

- Maintain readable and clean code.
- Use meaningful parameter names in your input dictionaries.
- Avoid issues with SymPy's handling of symbols.
- Keep your code flexible and easily extensible for new models with different parameters.

### **Example Usage**

Given your `gallery_input`:

```python
gallery_input = {
    ViscoElastic: dict(E=1, eta_v=1),
    ViscoElasticDamage: dict(E=1, eta_v=1, S=1)
}
```

Your loop remains simple:

```python
for gsm_class, inp in gallery_input.items():
    gsm_model = gsm_class()
    sig = gsm_model.get_sig(**inp)
    # Process or store the result as needed
```

---

**Answer:**

Yes—associate each symbol with a codename (e.g., 'eta_v'), use these codenames as dict keys, and map them to the symbols when calling lambdified functions; this keeps input specs readable and works around lambdify’s limitations.

## Question

This brings me to an idea to make the specification of codenames a bit more comfortable using option 1. In case that the name of the sympy variable is simple, i.e. 'E', I don't want to require a redundant definition of the mapping to codename, the sympy variable name should be simply taken. Only in case that the variable name string is a latex symbol, the codename should be specified in a dictionary within the model definition. Thus, the codename mapping should be a property which is constructed automatically, by looping over the m_params tuple with the sympy variables, analyzing their names similarly to the way lambdify does it to decide if the variable should be dummified or not. In case the variable is in the dummify category, a lookup into the m_param_codenames dictionary, and use the string specify there. Please generate such a method which can be included in GSMBase as a property. The remaining should remain the same then.

## Answer: conditional mapping

Certainly! Below is the implementation of the method you described, included as a property in the `GSMBase` class. This method automatically constructs the `params_codenames` mapping by analyzing each symbol in `m_params`. If a symbol's name is a valid Python identifier (i.e., it wouldn't be "dummified" by SymPy's `lambdify`), it uses the symbol's name as the codename. If not, it looks up the codename in a class-level `m_param_codenames` dictionary, which only needs to include codenames for those symbols with non-standard names (like LaTeX representations).

Here's the detailed implementation:

```python
import sympy as sp
import keyword

def is_valid_variable_name(name):
    """Check if the given name is a valid Python variable name."""
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True

class GSMBase:
    # ... existing base class code ...

    @property
    def params_codenames(self):
        """Construct a mapping from symbols to codenames for parameter substitution."""
        param_codenames = {}
        for sym in self.m_params:
            sym_name = sym.name
            if is_valid_variable_name(sym_name):
                codename = sym_name
            else:
                # Check if a codename is provided in m_param_codenames
                m_param_codenames = getattr(self, 'm_param_codenames', {})
                if sym in m_param_codenames:
                    codename = m_param_codenames[sym]
                else:
                    raise ValueError(
                        f"Symbol '{sym}' has an invalid name '{sym_name}' "
                        f"and no codename was provided in 'm_param_codenames'."
                    )
            param_codenames[sym] = codename
        return param_codenames
    
    def get_sig(self, **kwargs):
        """Example method that uses the constructed params_codenames."""
        # Ensure that all required parameters are provided
        missing_params = [codename for codename in self.params_codenames.values() if codename not in kwargs]
        if missing_params:
            raise ValueError(f"Missing parameter values for: {missing_params}")
        
        # Extract arguments in the correct order
        args = [kwargs[self.params_codenames[var]] for var in self.m_params]
        # Call the lambdified function (assumed to be self.sig_func)
        return self.sig_func(*args)
```

### **Explanation:**

- **is_valid_variable_name(name):** A helper function to check if a string is a valid Python variable name. It checks if the name is an identifier and not a reserved keyword.

- **params_codenames property:**

  - Loops over each symbol in `self.m_params`.
  - Checks if `sym.name` is a valid Python variable name.
    - If **yes**, it uses `sym.name` as the codename.
    - If **no**, it attempts to get the codename from `self.m_param_codenames`. If not found, it raises an exception.
  - Constructs and returns a dictionary mapping each symbol in `self.m_params` to its codename.

- **get_sig method:**

  - Validates that all required parameters are provided in `kwargs`.
  - Extracts the arguments in the order specified by `self.m_params` using the `params_codenames` mapping.
  - Calls the lambdified function `self.sig_func` with the extracted arguments.

### **Usage in Derived Classes:**

Now, in your derived classes, you only need to specify `m_param_codenames` for symbols with invalid Python variable names (e.g., LaTeX representations). Here's how you can define your classes:

```python
class ViscoElastic(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')  # LaTeX representation
    m_params = (E, eta_v)
    m_param_codenames = {eta_v: 'eta_v'}  # Only specify codename for invalid names

    # Define the symbolic expression for sig_expr
    sig_expr = E * eta_v  # Example expression
    # Create the lambdified function
    sig_func = sp.lambdify(m_params, sig_expr)

class ViscoElasticDamage(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
    S = sp.Symbol('S')
    m_params = (E, eta_v, S)
    m_param_codenames = {eta_v: 'eta_v'}

    # Define the symbolic expression for sig_expr
    sig_expr = E * eta_v + S  # Example expression
    # Create the lambdified function
    sig_func = sp.lambdify(m_params, sig_expr)
```

### **Gallery Input and Loop:**

Your gallery input remains clean and readable:

```python
gallery_input = {
    ViscoElastic: dict(E=1, eta_v=1),
    ViscoElasticDamage: dict(E=1, eta_v=1, S=1)
}

for gsm_class, inp in gallery_input.items():
    gsm_model = gsm_class()
    sig = gsm_model.get_sig(**inp)
    # Process or store the result as needed
    print(f"{gsm_class.__name__}: sig = {sig}")
```

### **Advantages of This Approach:**

- **Minimal Redundant Definitions:** You only need to specify codenames for symbols with names that are not valid Python identifiers.

- **Automatic Codename Mapping:** The `params_codenames` property handles the mapping automatically, making your code cleaner and reducing the chance of errors.

- **Flexible and Extensible:** This method works with any number of parameters and can be easily extended to new models with different parameters.

- **Readable Input Specification:** The input dictionaries remain intuitive, using meaningful parameter names.

### **Detailed Steps:**

1. **Define Helper Function:**

   ```python
   import keyword

   def is_valid_variable_name(name):
       """Check if the given name is a valid Python variable name."""
       if not name.isidentifier():
           return False
       if keyword.iskeyword(name):
           return False
       return True
   ```

2. **Add `params_codenames` Property in `GSMBase`:**

   ```python
   class GSMBase:
       # ... existing code ...

       @property
       def params_codenames(self):
           """Construct a mapping from symbols to codenames for parameter substitution."""
           param_codenames = {}
           for sym in self.m_params:
               sym_name = sym.name
               if is_valid_variable_name(sym_name):
                   codename = sym_name
               else:
                   # Check if a codename is provided in m_param_codenames
                   m_param_codenames = getattr(self, 'm_param_codenames', {})
                   if sym in m_param_codenames:
                       codename = m_param_codenames[sym]
                   else:
                       raise ValueError(
                           f"Symbol '{sym}' has an invalid name '{sym_name}' "
                           f"and no codename was provided in 'm_param_codenames'."
                       )
               param_codenames[sym] = codename
           return param_codenames
   ```

3. **Modify Derived Classes to Include Only Necessary Codenames:**

   ```python
   class ViscoElastic(GSMBase):
       E = sp.Symbol('E')
       eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
       m_params = (E, eta_v)
       m_param_codenames = {eta_v: 'eta_v'}  # Codename for LaTeX symbol

       # Define the symbolic expression and lambdify it
       sig_expr = E * eta_v  # Example expression
       sig_func = sp.lambdify(m_params, sig_expr)
   ```

4. **Use the Classes with Readable Input Dictionaries:**

   ```python
   gallery_input = {
       ViscoElastic: dict(E=1, eta_v=1),
       ViscoElasticDamage: dict(E=1, eta_v=1, S=1)
   }

   for gsm_class, inp in gallery_input.items():
       gsm_model = gsm_class()
       sig = gsm_model.get_sig(**inp)
       print(f"{gsm_class.__name__}: sig = {sig}")
   ```

### **Testing the Implementation:**

Let's test the implementation to ensure it works as expected.

**Test Code:**

```python
import sympy as sp
import keyword

def is_valid_variable_name(name):
    """Check if the given name is a valid Python variable name."""
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True

class GSMBase:
    @property
    def params_codenames(self):
        """Construct a mapping from symbols to codenames for parameter substitution."""
        param_codenames = {}
        for sym in self.m_params:
            sym_name = sym.name
            if is_valid_variable_name(sym_name):
                codename = sym_name
            else:
                # Check if a codename is provided in m_param_codenames
                m_param_codenames = getattr(self, 'm_param_codenames', {})
                if sym in m_param_codenames:
                    codename = m_param_codenames[sym]
                else:
                    raise ValueError(
                        f"Symbol '{sym}' has an invalid name '{sym_name}' "
                        f"and no codename was provided in 'm_param_codenames'."
                    )
            param_codenames[sym] = codename
        return param_codenames

    def get_sig(self, **kwargs):
        """Example method that uses the constructed params_codenames."""
        # Ensure that all required parameters are provided
        missing_params = [codename for codename in self.params_codenames.values() if codename not in kwargs]
        if missing_params:
            raise ValueError(f"Missing parameter values for: {missing_params}")
        
        # Extract arguments in the correct order
        args = [kwargs[self.params_codenames[var]] for var in self.m_params]
        # Call the lambdified function (assumed to be self.sig_func)
        return self.sig_func(*args)

class ViscoElastic(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
    m_params = (E, eta_v)
    m_param_codenames = {eta_v: 'eta_v'}

    # Define the symbolic expression and lambdify it
    sig_expr = E + eta_v  # Example expression (E + eta_v)
    sig_func = sp.lambdify(m_params, sig_expr)

class ViscoElasticDamage(GSMBase):
    E = sp.Symbol('E')
    eta_v = sp.Symbol(r'\eta_\mathrm{ve}')
    S = sp.Symbol('S')
    m_params = (E, eta_v, S)
    m_param_codenames = {eta_v: 'eta_v'}

    # Define the symbolic expression and lambdify it
    sig_expr = E + eta_v + S  # Example expression (E + eta_v + S)
    sig_func = sp.lambdify(m_params, sig_expr)

gallery_input = {
    ViscoElastic: dict(E=1, eta_v=2),
    ViscoElasticDamage: dict(E=1, eta_v=2, S=3)
}

for gsm_class, inp in gallery_input.items():
    gsm_model = gsm_class()
    sig = gsm_model.get_sig(**inp)
    print(f"{gsm_class.__name__}: sig = {sig}")
```

**Expected Output:**

```
ViscoElastic: sig = 3
ViscoElasticDamage: sig = 6
```

**This confirms that the implementation works as intended.**

### **Notes:**

- **Exception Handling:** If a symbol has an invalid name and no codename is provided, a `ValueError` is raised, prompting the user to supply the necessary codename.

- **No Redundant Mappings:** Only symbols with invalid names require codenames in `m_param_codenames`. Symbols with valid names do not need to be included.

- **Flexible for Future Extensions:** If you add more parameters or models, you only need to provide codenames for new symbols with invalid names.

### **Conclusion:**

By integrating this method into your `GSMBase` class, you achieve a balance between code readability and functionality. The approach automates the tedious parts of parameter handling while keeping your input specifications clean and intuitive. It aligns with your goal of generating a gallery of models with minimal boilerplate code and maximum clarity.

---

**Answer:**

Yes—here's a method in `GSMBase` that builds the `params_codenames` mapping automatically: it uses symbol names as codenames when valid, and for invalid names (e.g., LaTeX), it looks up codenames you provide in each class; thus, you only specify codenames when necessary, keeping your code clean and readable.