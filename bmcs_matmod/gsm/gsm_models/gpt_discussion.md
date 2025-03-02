
# Discussion on Naming Conventions in Python

## Question

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
