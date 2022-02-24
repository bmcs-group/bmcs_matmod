
# SLIDE Interface Model

Name of the model 

Coupled sliding-decohesion-compression interface model (CSDCIM), or (SDCIM)

Coupled normal-tangential interface model (CoNTIM)

package name `contim`

Versions of the model - two classes 

 - `SConTIM` single point material model (scalar values for u_N and u_T)
 - `VConTIM` vectorized model (multi-dimensional arrays)



## Web-app SLIDE 3.2

 * Transfer the model components from the notebook to the 
   Python model components (Model) - [rch] 2 days
 
 * Implement the generic return mapping function in C  
   based on the Python code
   
   - order of parameters in the generated function
   
 * Compare the response for elementary loading conditions
 