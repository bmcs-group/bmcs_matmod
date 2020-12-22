
# Conventions to follow in coding material models

## General Python conventions

### Naming

 - File names 
   - only lower case letters
   - words connected by undersores
   
 - Class names
   - Capitalized words, e.g. **ClassName**

## Usage of traits

 - The `traits` is imported as follows
   `import traits.api as tr`

 - Classes must inherit directlo or indirectly from 
   `tr.HasTraits` or `tr.HasStrictTraits` 
   
## Definition of model components

 - Model components are ideally designed to 
   show the direct correspondence between inputs
   a the graphical representation of the model. Direct support
   for implementation of model components is provided
   in `bmcs_utils` package. Import as
   `import bmcs_utils.api as bu`
   
 - Model components should 
   - inherit from `bu.InteractiveModel`
     and define the set of paramaters as traits, i.e.
     `E = tr.Float(20)`
   - define an class atribute `ipw_view` specifying the 
     traits that can be manipulated interactively
     `ipw_view = View(Item('E'))` 
   - define a `plot` method visualizing the model
     `def plot(self,ax)`
   - optionally, they can define the method
     `def subplots(self,fig)`
     returning the layout for plotting provided via the matplotlib
     interface  
   
## Material models define
   - Material parameters as traits
   - `ipw_view` as `bu.View`
   - `get_corr_pred(eps, state)` - return the stress 
      and stiffness for given strain and state
   - `state_var_shape` - specify the size of arrays 
      for state variables

## Material model explorer
   - To verify and visualize the model functionality
     there is a class `MATSExplorer` that can control 
     the calculation - start, stop and shows the progress bar.
     
   - `MATSExplorer` also inherits from `InteractiveModel` and 
     defines the `plot` method.
     
     
@todo: provide a base class for 2D and 3D version of MS1

@todo: transform one of the scripts to an explorer

@todo: change the `Micro2Dplot.py` to a `plot`.