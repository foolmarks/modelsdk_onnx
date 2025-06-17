'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of  this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''

'''
Quantize and compile the ONNX model.
Usage from inside Palette docker: python run_modelsdk.py
'''

'''
Author: Mark Harvey
'''

import onnx
import os, sys
import argparse
import numpy as np
import logging
import cv2
import tarfile


# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, onnx_source
from afe.ir.tensor_type import ScalarType
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted
from afe.apis.defines import default_quantization,gen1_target, gen2_target


import config as cfg

DIVIDER=cfg.DIVIDER


# pre-processing
def _preprocessing(image):
  '''
  Normalize, Mean subtraction, div by std deviation
  Add batch dimension
  '''
  image = cfg.preprocess(image)
  image = image[np.newaxis, :, :, :]
  return np.float32(image)



def implement(args):

    enable_verbose_error_messages()


    # get filename from full path
    filename = (os.path.splitext(os.path.basename(args.model_path)))[0]

    # set an output path for saving results
    output_path=f'{args.build_dir}/{filename}'
   

    '''
    Interrogate ONNX model for input names, shapes
    '''
    model = onnx.load(args.model_path)
    input_names_list=[node.name for node in model.graph.input]
    input_shapes_list = [tuple(d.dim_value for d in _input.type.tensor_type.shape.dim) for _input in model.graph.input]
    print('Model inputs:')
    for n,s in zip(input_names_list,input_shapes_list):
        print(f' {n}  {s}')


    '''
    Load the floating-point ONNX model
    Refer to online documentation: https://developer.sima.ai/apps?id=22ef42b1-3652-4cc7-8019-16b86910ed53
    '''

    # input types & shapes are dictionaries
    # input types dictionary: each key,value pair is an input name (string) and a type
    # input shapes dictionary: each key,value pair is an input name (string) and a shape (tuple)
    input_shapes_dict={}
    input_types_dict={}
    for n,s in zip(input_names_list,input_shapes_list):
       input_shapes_dict[n]=s
       input_types_dict[n]=ScalarType.float32
       
    # importer parameters
    importer_params: ImporterParams = onnx_source(model_path=args.model_path,
                                                  shape_dict=input_shapes_dict,
                                                  dtype_dict=input_types_dict)
    
    '''
    load ONNX floating-point model into SiMa's LoadedNet format
    '''

    # choose DaVinci or Modalix as target
    target = gen2_target if args.generation == 2 else gen1_target
       
    loaded_net = load_model(importer_params,target=target)
    print(f'Loaded model from {args.model_path} targetting {target}')
       


    '''
    Set up calibration data - the calibration samples should be randomly chosen from the training dataset.
    The calibration data must be in NHWC format even if the original ONNX model is NCHW
    Each calibration data sample is supplied as a dictionary, key is model input name, value is preprocessed calibration data
    The dictionaries are appended to an iterable variable - a list is used in the example below
    '''
    with np.load(args.calib_data) as data:
        calib_images = data['x']
        print('Number of calibration images: ', calib_images.shape[0])

    # make a list of preprocessed calibration images
    calibration_data=[]
    for img in (calib_images):
        preproc_image = _preprocessing(img)
        calibration_data.append({input_names_list[0]:preproc_image})


    '''
    Quantize with default parameters
    Refer to online docs: https://developer.sima.ai/apps?id=22ef42b1-3652-4cc7-8019-16b86910ed53
    '''

    quant_model = loaded_net.quantize(calibration_data=length_hinted(len(calib_images),calibration_data),
                                      quantization_config=default_quantization,
                                      model_name=filename,
                                      log_level=logging.WARN)

    quant_model.save(model_name=filename, output_directory=output_path)
    print (f'Quantized and saved to {output_path}')



    '''
    Evaluate quantized model
    '''
    print('Evaluating quantized model...')
    
    # unpack test images and labels
    with np.load(args.test_data) as data:
        test_images = data['x']
        labels = data['y']
        print('Number of test images: ', test_images.shape[0])
    
    correct=0

    for i,img in enumerate(test_images):
        
        # preprocess
        img = _preprocessing(img)

        # dictionary key is name of input that preprocessed sample will be applied to
        test_data={input_names_list[0]: img }

        # emulate the quantized model
        prediction = quant_model.execute(test_data, fast_mode=True)

        # post-processing - argmax reduction
        prediction = np.argmax(prediction)

        if prediction == labels[i]:
            correct += 1
    accuracy = correct / len(labels) * 100
    print('Correct predictions: ', correct, ' Accuracy %:', accuracy)


    '''
    Compile
    Refer to online docs: https://developer.sima.ai/apps?id=22ef42b1-3652-4cc7-8019-16b86910ed53
    '''
    quant_model.compile(output_path=output_path,
                        batch_size=args.batch_size,
                        log_level=logging.INFO)  

    print(f'Compiled model written to {output_path}')


    '''
    untar the compiled model
    '''
    model = tarfile.open(f'{output_path}/{filename}_mpk.tar.gz')
    model.extractall(f'{output_path}')
    model.close() 

    return




def run_main():
  
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd', '--build_dir',   type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m',  '--model_path',  type=str, default='./onnx/resnet50.onnx', help='path to FP32 ONNX model')
    ap.add_argument('-b',  '--batch_size',  type=int, default=1, help="requested batch size of compiled model. Default is 1")
    ap.add_argument('-td', '--test_data',   type=str, default='test_data.npz', help='Path of test data numpy file. Default is test_data.npz')
    ap.add_argument('-cd', '--calib_data',  type=str, default='calib_data.npz', help='Path of calibration data numpy file. Default is calib_data.npz')
    ap.add_argument('-g',  '--generation',  type=int, choices=[1,2], default=1, help="Specify target platform, choices are 1 (MLSoC) or 2 (Modalix)")
    args = ap.parse_args()

    print('\n'+DIVIDER,flush=True)
    print('Model SDK version',get_model_sdk_version())
    print(sys.version,flush=True)
    print(DIVIDER,flush=True)

    implement(args)

if __name__ == '__main__':
    run_main()

