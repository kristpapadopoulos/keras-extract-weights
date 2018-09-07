'''
Extracting weights of Keras model layers from h5py file to csv

Parameters from layers are extracted in rows with biases at end of parameters

If layers have filters then parameters of filters are appended in the rows

Based on function for printing weights out:
https://github.com/keras-team/keras/issues/91

Args:
    weight_path (str) : Path to the file to analyze
      
    csv_path (str):  CSV file with parameters and biases per layer  

Date: July 2, 2018

Author: Krist Papadopoulos

'''

import argparse
import h5py
import pandas as pd

parser = argparse.ArgumentParser(description='Extract Model Weights to CSV')

parser.add_argument('--h5_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to parameter h5py file location (default: None)')

parser.add_argument('--csv_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to csv output file location (default: None)')

args = parser.parse_args()

    
f = h5py.File(args.h5_path)
    
try:
    if len(f.attrs.items()):
        print("Parameter File: {}".format(args.h5_path))
        print("Extracting Model Parameters to CSV File...")

    if len(f.items())==0:
        raise ValueError('No Layers with Paramters Found')

except ValueError as ve:
    print(ve)
    
else:
    weights = {}
    
    # each layer has a group
    for layer, group in f.items():
            weights[layer] = []
        
        # in the group the keys are the members: either the layer bias and/or parameters
        for p_name in group.keys():
            param = group[p_name]
            
            # if the group has 0 members then the layer has no biases or parameters
            if len(param) == 0:
                weights[layer].extend(None)
            else:
                
                # for each parameter in the layer added values to a list: order is first biases then parameters
                for k_name in param.keys():
                    weights[layer].extend(param[k_name].value[:].flatten().tolist())
       
finally:
    f.close()
             
csv_path = args.csv_path
        
weights_list = [[key] + value for key, value in weights.items()]
                
weights_df = pd.DataFrame(weights_list)
        
weights_df.to_csv(csv_path, index=False, header=False)
        
