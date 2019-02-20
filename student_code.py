from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import random
import numpy as np

import cv2
import numbers
import collections

from utils import resize_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# These are helper functions or functions for demonstration
# You won't need to modify them
#################################################################################

class Compose(object):
  """Composes several transforms together.

  Args:
      transforms (list of ``Transform`` objects): list of transforms to compose.

  Example:
      >>> Compose([
      >>>     Scale(320),
      >>>     RandomSizedCrop(224),
      >>> ])
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img):
    for t in self.transforms:
      img = t(img)
    return img

  def __repr__(self):
    repr_str = ""
    for t in self.transforms:
      repr_str += t.__repr__() + '\n'
    return repr_str

class RandomHorizontalFlip(object):
  """Horizontally flip the given numpy array randomly 
     (with a probability of 0.5).
  """
  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be flipped.

    Returns:
        numpy array: Randomly flipped image
    """
    if random.random() < 0.5:
      img = cv2.flip(img, 1)
      return img
    return img

  def __repr__(self):
    return "Random Horizontal Flip"

#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
  """Rescale the input numpy array to the given size.

  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size, size * height / width)

      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      Pass None during testing: always use CV2.INTER_CUBIC
  """
  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
    assert (isinstance(size, int) 
            or (isinstance(size, collections.Iterable) 
                and len(size) == 2)
           )
    self.size = size
    # use bilinear if interpolation is not specified
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    """
    Args:
        img (numpy array): Image to be scaled.

    Returns:
        numpy array: Rescaled image
    """
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]

    # scale the image
    if isinstance(self.size, int):
      #################################################################################
      height, width = np.shape(img)[0:2:1]

      if width == height:
          new_size = (self.size, self.size)

      elif width > height:
          ratio = float(width) / float(height)
          new_width = ratio * self.size
          new_size = (int(np.floor(new_width)), self.size)

      elif height > width:
          ratio = float(height) / float(width)
          new_height = ratio * self.size
          new_size = (self.size, int(np.floor(new_height)))

      img = resize_image(img, new_size, interpolation)  
      #################################################################################
      return img
    else:
      #################################################################################
      img = resize_image(img, self.size, interpolation)
      #################################################################################
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Scale [Exact Size ({:d}, {:d})]".format(target_size[0], target_size[1])

class RandomSizedCrop(object):
  """Crop the given numpy array to random area and aspect ratio.

  A crop of random area of the original size and a random aspect ratio 
  of the original aspect ratio is made. This crop is finally resized to given size.
  This is widely used as data augmentation for training image classification models

  Args:
      size (sequence or int): size of target image. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int, 
          output size will be (size, size).
      interpolations (list of int, optional): Desired interpolation. 
      Default is ``CV2.INTER_NEAREST|CV2.INTER_LANCZOS|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
      area_range (list of int): range of the areas to sample from
      ratio_range (list of int): range of aspect ratio to sample from
      num_trials (int): number of sampling trials
  """

  def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS, 
               area_range=(0.25, 1.0), ratio_range=(0.8, 1.2), num_trials=10):
    self.size = size
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations
    self.num_trials = int(num_trials)
    self.area_range = area_range
    self.ratio_range = ratio_range

  def __call__(self, img):
    # sample interpolation method
    height, width = np.shape(img)[0:2:1]
    interpolation = random.sample(self.interpolations, 1)[0]

    for attempt in range(self.num_trials):

      # sample target area / aspect ratio from area range and ratio range
      area = img.shape[0] * img.shape[1]
      target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
      aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])
      # print('target_area',target_area)

      #################################################################################
      # Fill in the code here
      #################################################################################
      # if...

      new_long = np.floor(np.sqrt(target_area*aspect_ratio))
      new_short = np.floor(new_long/aspect_ratio)
      # print("edges",new_long,new_short)
      count = 0


      if (height > new_long) and (width > new_short):
        trim_h = int(np.floor((height-new_long)/2))
        trim_w = int(np.floor((width-new_short)/2))
        # print((trim_h,trim_w))
        img = img[trim_h:height-trim_h, trim_w:width-trim_w, :]  
        break
      elif (width > new_long) and (height > new_short):
        trim_h = int(np.floor((height-new_short)/2))
        trim_w = int(np.floor((width-new_long)/2))
        # print((trim_h,trim_w))
        img = img[trim_h:height-trim_h, trim_w:width-trim_w, :]        
        break
      else:
        count = count+1

      # compute the width and height
      # note that there are two possibilities
      # crop the image and resize to output size

    # Fall back
    if isinstance(self.size, int):
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      #################################################################################
      # Fill in the code here
      #################################################################################
      # with a square sized output, the default is to crop the patch in the center 
      # (after all trials fail)
      height, width = np.shape(img)[0:2:1]
      center = [height/2, width/2]
      if height>width:
        top = int(np.floor(center[0]-self.size/2))
        bottom = top+self.size
        img = img[top:bottom, :, :]
      else:
        top = int(np.floor(center[1]-self.size/2))
        bottom = top+self.size
        img = img[:,top:bottom,:]
      return img
   
    else:
      # with a pre-specified output size, the default crop is the image itself
      im_scale = Scale(self.size, interpolations=self.interpolations)
      img = im_scale(img)
      return img

  def __repr__(self):
    if isinstance(self.size, int):
      target_size = (self.size, self.size)
    else:
      target_size = self.size
    return "Random Crop" + \
           "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}%; Ratio {:.2f} - {:.2f}%]".format(
            target_size[0], target_size[1], 
            self.area_range[0], self.area_range[1],
            self.ratio_range[0], self.ratio_range[1])


class RandomColor(object):
  """Perturb color channels of a given image
  Sample alpha in the range of (-r, r) and multiple 1 + alpha to a color channel. 
  This is done independently for each channel.

  Args:
      color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
  """
  def __init__(self, color_range):
    self.color_range = color_range

  def __call__(self, img):
    #################################################################################
    for color in range(3):
      alpha = np.random.uniform(-self.color_range, self.color_range)
      jitter_ratio = 1 + alpha
      img[:,:,color] = np.floor(img[:,:,color]*jitter_ratio)
    #################################################################################
    return img

  def __repr__(self):
    return "Random Color [Range {:.2f} - {:.2f}%]".format(
            1-self.color_range, 1+self.color_range)


class RandomRotate(object):
  """Rotate the given numpy array (around the image center) by a random degree.

  Args:
      degree_range (float): range of degree (-d ~ +d)
  """
  def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
    self.degree_range = degree_range
    if interpolations is None:
      interpolations = [cv2.INTER_LINEAR]
    assert isinstance(interpolations, collections.Iterable)
    self.interpolations = interpolations

  def __call__(self, img):
    # sample interpolation method
    interpolation = random.sample(self.interpolations, 1)[0]
    # sample rotation
    degree = random.uniform(-self.degree_range, self.degree_range)
 
    # ignore small rotations
    if np.abs(degree) <= 1.0:
      return img
    else:
      perb = np.ones((np.shape(img)[0],np.shape(img)[1]))
  
      image_center = tuple(np.array(np.shape(img)[1::-1]) / 2)
      rot_mat = cv2.getRotationMatrix2D(image_center, degree, 1.0)
      img = cv2.warpAffine(img, rot_mat, np.shape(img)[1::-1], flags=cv2.INTER_LINEAR)
      # img1 = img

      perb_r = cv2.warpAffine(perb, rot_mat, np.shape(img)[1::-1], flags=cv2.INTER_LINEAR)

      mask = perb_r
      area = 0
      m = np.shape(mask)[0]
      n = np.shape(mask)[1]

      height = np.zeros(n)
      left = np.zeros(n)
      right = np.zeros(n)
      best = np.zeros(3)

      # dynamic programming
      for i in range(m):
        edge_l = 0 
        edge_r = n
        for j in range(n):
          if mask[i][j] == 0:
            height[j] = 0
          else:
            height[j] = height[j]+1
          if height[j] > 0:
            left[j] = max(left[j],edge_l)
          else:
            left[j] = 0
            edge_l = j+1
        for j in range(n-1,-1,-1):
          if height[j]>0:
            right[j] = min(right[j],edge_r)
          else:
            right[j] = n
            edge_r = j
        for j in range(n):
          new_area = (right[j] - left[j])*height[j]
          if new_area >area:
            area = new_area
            best = (left[j],right[j],height[j])
    
      h_pad = int(np.ceil((m - best[2]))/2)
      h_t = h_pad
      h_b = int(np.shape(img)[0])-h_pad
      w_l = int(best[0])
      w_r = int(best[1])

      img = img[(h_t):(h_b), (w_l):(w_r), :]

      
    #################################################################################
    # Fill in the code here
    #################################################################################
    # get the max rectangular within the rotated image
    return img

  def __repr__(self):
    return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range)