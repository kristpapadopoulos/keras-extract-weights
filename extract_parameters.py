'''
Extracting weights of Keras model layers from h5py file to dictionary with option
to save to pickle file.

Based on function for printing weights out:
https://github.com/keras-team/keras/issues/91

Args:
    h5_path (str) : Path to the Keras model weights to analyze
    output_path (str):  pkl output file path 

Date: July 2, 2018 v0
Date: June 11, 2022 v1

Author: Krist Papadopoulos

'''

import argparse
import h5py
import pickle

parser = argparse.ArgumentParser(description='Extract model weights to dictionary')

parser.add_argument('--h5_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to parameter h5py file location (default: None)')

parser.add_argument('--output_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to parameter dictionary pkl output file location (default: None)')

args = parser.parse_args()
    
f = h5py.File(args.h5_path)
    
try:
    if len(f.attrs.items()):
        print("Parameter file: {}".format(args.h5_path))
        print("Extracting model parameters to parameter dictionary ...")

    if len(f.items())==0:
        raise ValueError('No Layers with paramters found')

except ValueError as ve:
    print(ve)
    
else:
    weights = {}
    
    # each layer has a group
    for layer, group in f.items():
        
        # in the group the keys are the members: either the layer bias and/or parameters
        for p_name in group.keys():
            param = group[p_name]
            
            # if the group has 0 members then the layer has no biases or parameters
            if len(param) == 0:
                weights[layer, None] = None
            
            else:
                # for each parameter in the layer output to dictionary
                for k_name in param.keys():
                    weights[layer, k_name] = param[k_name][()].tolist()
       
finally:
    f.close()

# save dictionary to pickle file
if args.output_path:
    with open(args.output_path, 'wb') as f:
        pickle.dump(weights, f)
              
