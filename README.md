### Keras extract weights from model to python dictionary

The file extract_parameters.py

- takes Keras h5py model parameter file and outputs a python dictionary with option to output as pickle file
- extract_parameters.py --h5_path='file_path' --output_path='file_path'

#### Tested with: 

```
h5py  3.1.0
```

--------------------
Each Keras layer has a group that has members:  1 for the biases and 1 for the parameters of the layer.

---------------------
<b>Example</b>

Output for 3 layer neural network  - see dataset folder for input file.
