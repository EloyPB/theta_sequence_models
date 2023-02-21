**MODELS OF BEHAVIORAL-DEPENDENT SWEEPS**

This repository contains code for simulating 3 types of models of behavior-dependent sweeps and analyzing their results. 

**Usage** 

1. Choose which of the three models to use by commenting out the corresponding network class in *NetworkClass.py*.
``NetworkIntDriven`` produces internally generated theta sequences based on short term synaptic facilitation and depression, that then get used to map out space using a form of hebbian plasticity. 
``NetworkExtDriven`` learns the connections between cells that lead to theta sequences based on a form of behavioral timescale synaptic plasticity.
``NetworkIndep`` learns place fields using behavioral timescale synaptic plasticity and makes cells phase precess independently of each other.
2. Specify a path where pickles and results will be saved in *batch_config.py*
3. Run *batch_analysis.py*
4. Run *batch_plots.py*

The code is structured as a cascade of classes that depend on one another. Instantiated classes are stored as pickles so that they don't have to be generated every time they are needed.
They are only re-generated if either the code or the parameters of the class itself or one of its dependencies has changed. 
How this works is explained using an example at the bottom of *smart_sim.py*. 