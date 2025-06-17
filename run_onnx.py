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
Evaluate floating-point ONNX model
'''


'''
Author: Mark Harvey
'''


import onnx
import onnxruntime as ort
import sys
import argparse
import numpy as np

import config as cfg
DIVIDER = cfg.DIVIDER


# pre-processing
def _preprocessing(image):
  '''
  Normalize, Mean subtraction, div by std deviation
  Add batch dimension and transpose to NCHW
  '''
  image = cfg.preprocess(image)
  image = image[np.newaxis, :, :, :]
  image = np.transpose(image, axes=[0, 3, 1, 2])
  return np.float32(image)



def implement(args):

    '''
    Interrogate ONNX model for input names,shapes
    '''
    model = onnx.load(args.model_path)
    print(f'Loaded model from {args.model_path}')
    input_names_list=[node.name for node in model.graph.input]
    input_shapes_list = [tuple(d.dim_value for d in _input.type.tensor_type.shape.dim) for _input in model.graph.input]
    print('Model inputs:')
    for n,s in zip(input_names_list,input_shapes_list):
      print(f' {n}  {s}')

    # this assumes that there is only one input of format NCHW
    height = input_shapes_list[0][2]
    width = input_shapes_list[0][3]

    '''
    Unpack test data & labels
    '''
    with np.load(args.test_data) as data:
        images = data['x']
        labels = data['y']
        print('Number of test images: ', images.shape[0])

    '''
    ONNX runtime session
    '''
    ort_sess = ort.InferenceSession(args.model_path)
    correct=0

    for i,img in enumerate(images):

        img = _preprocessing(img)

        # run inference - outputs NCHW format
        prediction = ort_sess.run(None, {input_names_list[0]: img})
        prediction = prediction[0]

        # argmax reduction
        prediction = np.argmax(prediction)

        if prediction == labels[i]:
            correct += 1
    accuracy = correct / len(labels) * 100
    print('Correct predictions: ', correct, ' Accuracy %:', accuracy)

    return






def run_main():
  
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-bd', '--build_dir',  type=str, default='./build', help='Path of build folder. Default is ./build')
  ap.add_argument('-td', '--test_data',  type=str, default='./test_data.npz', help='Path of test data numpy file. Default is ./test_data.npz')
  ap.add_argument('-m',  '--model_path', type=str, default='./onnx/resnet50.onnx', help='path to ONNX model')
  args = ap.parse_args()

  print('\n'+DIVIDER,flush=True)
  print(sys.version,flush=True)
  print(DIVIDER,flush=True)


  implement(args)


if __name__ == '__main__':
    run_main()