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
Author: Mark Harvey
'''


import os, sys
import numpy as np
import cv2
import argparse

import config as cfg
DIVIDER = cfg.DIVIDER


def pad_to_square(image):
    '''
    Pads the given image with a black border to make it square.
    '''
    # Get current height and width
    h, w = image.shape[:2]

    # If already square, return original
    if h == w:
        return image

    # Determine new square size
    size = max(h, w)

    # Compute padding amounts
    delta_w = size - w
    delta_h = size - h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    # Pad with black (zero) pixels
    color = [0, 0, 0]
    squared = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color
    )

    return squared




def images_to_npy(image_dir,label_file,classes,input_height,input_width,input_chans,compress,output_file):

  # open & read text file that lists all images and their labels
  if label_file:
    f = open(label_file, 'r') 
    listImages = f.readlines()
    f.close()
    # make labels placeholder array
    if(classes<=256):
      y = np.ndarray(shape=(len(listImages)), dtype=np.uint8, order='C')
    elif(classes<=65536):
      y = np.ndarray(shape=(len(listImages)), dtype=np.uint16, order='C')
    else:
      y = np.ndarray(shape=(len(listImages)), dtype=np.uint32, order='C')
  else:
    listImages = os.listdir(image_dir)


  # make image placeholder array
  x = np.ndarray(shape=(len(listImages),input_height,input_width,input_chans), dtype=np.uint8, order='C')


  # loop through all images & labels
  print(' Loading images & labels into memory')
  for i in range(len(listImages)):

    if label_file:
      image_name,label = listImages[i].split()
    else:
      image_name = listImages[i]

    # open image to numpy array and switch to RGB from BGR
    img = cv2.imread(os.path.join(image_dir,image_name))
    print(os.path.join(image_dir,image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # pad & resize
    img = pad_to_square(img)
    img = cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_AREA)

    x[i] = img
    
    # write label into placeholder array
    if label_file:
      y[i] = int(label)


  # report data types used
  print(' x shape:',x.shape)
  print(' x data type:',x[0].dtype)
  if label_file:
    print(' y shape:', y.shape)
    print(' y data type:', y[0].dtype)


  # write output file
  if label_file:
    if (compress):
      np.savez_compressed(output_file, x=x, y=y)
    else:
      np.savez(output_file, x=x, y=y)
  else:
    if (compress):
      np.savez_compressed(output_file, x=x)
    else:
      np.savez(output_file, x=x)    

  print(' Saved to',output_file+'.npz')


  return  



def run_main():
    
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument('-id','--image_dir',    type=str,   default='image_dir', help='Full path of folder containing images.')
  ap.add_argument('-l', '--label_file',   type=str,   default=None,        help='Full path of label file.')
  ap.add_argument('-c', '--classes',      type=int,   default=1000,        help='Number of classes.')
  ap.add_argument('-ih','--input_height', type=int,   default=224,         help='Input image height in pixels.')
  ap.add_argument('-iw','--input_width',  type=int,   default=224,         help='Input image width in pixels.')
  ap.add_argument('-ic','--input_chans',  type=int,   default=3,           help='Input image channels.')
  ap.add_argument('-cp','--compress',     action='store_true', help='Compress the output file if set, otherwise no compression. Default is no compression.')
  ap.add_argument('-o', '--output_file',  type=str,   default='dataset.npz', help='Full path of output file.')
  args = ap.parse_args()

  print('\n'+DIVIDER)
  print(sys.version)
  print(DIVIDER)
  print(' Command line options:')
  print('--image_dir     : ',args.image_dir    )
  print('--label_file    : ',args.label_file   )
  print('--classes       : ',args.classes      )
  print('--input_height  : ',args.input_height )
  print('--input_width   : ',args.input_width  )
  print('--input_chans   : ',args.input_chans  )
  print('--compress      : ',args.compress     )
  print('--output_file   : ',args.output_file  )
  print(DIVIDER)
  

  images_to_npy(args.image_dir, args.label_file, args.classes, args.input_height, \
  args.input_width, args.input_chans, args.compress, args.output_file)


if __name__ == '__main__':
  run_main()
