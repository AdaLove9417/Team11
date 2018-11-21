import os
#matrix math
import numpy as np
#read/write image data
import imageio
#visualize data
import matplotlib.pyplot as plt
#data preprocessing 
import pandas as pd
#deep learning
import torch
import TGSSaltDataset as tg
import UNET
import skimage.io as io
#just in case we need a backup datasets
from torch.utils import data

#visualization
def plot2x2Array(image, mask):
    #invoke matplotlib!
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(image)
    axarr[1].imshow(mask)
    axarr[0].grid()
    axarr[1].grid()
    axarr[0].set_title('Image')
    axarr[1].set_title('Mask')

#decoding
def rleToMask(rleString,height,width):
    #width heigh
    rows,cols = height,width
    try:
        #get numbers
        rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
        #get pairs
        rlePairs = np.array(rleNumbers).reshape(-1,2)
        #create an image
        img = np.zeros(rows*cols,dtype=np.uint8)
        #for each pair
        for index,length in rlePairs:
            #get the pixel value 
            index -= 1
            img[index:index+length] = 255
        
        
        #reshape
        img = img.reshape(cols,rows)
        img = img.T
    
    #else return empty image
    except:
        img = np.zeros((cols,rows))
    
    return img

def get_data(path,ids,train=True):
    """
    Getting image and masks arrays
    """
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    if train:
        return X, y
    else:
        return X

def add_random_gaussian_noise(image,mean,variance):
    """
    Generate noise to a given Image based on required noise type
    From: http://www.xiaoliangbai.com/2016/09/09/more-on-image-noise-generation
    
    Input parameters:
        image: ndarray (input image data)
       
    """
    row,col,ch= image.shape       
    mu = mean
    var = variance
    sigma = var**0.5
    gauss = np.array(image.shape)
    gauss = np.random.normal(mu,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy_img_clipped = np.clip(noisy, 0, 255)
    return noisy_img_clipped.astype('uint8')

def image_generator(X_train,y_train,batch_size):
    """
    Custom image generator that adds Gaussian noise and generates
    batches of images and masks arrays
    
    """
    mean = 0.0
    variance = np.random.uniform(0.01, 1.0)
    #X_train,y_train = get_data(path_train,train_ids,train=True) # Get arrays          
    batch_images = np.zeros((batch_size, 128,128,1))
    batch_masks = np.zeros((batch_size,128,128,1))
    while True:
      for i in range(batch_size):
          # choose random index in features
          index= np.random.choice(len(X_train),1)[0]
          batch_images[i] = add_random_gaussian_noise(X_train[index],mean,variance)
          batch_masks[i] = add_random_gaussian_noise(y_train[index],mean,variance)
      yield batch_images, batch_masks
    
#training path
path_train = "./"
path_test = "./"

#list of files
train_ids = next(os.walk(path_train+"images"))[2]
test_ids = next(os.walk(path_test+"test/images"))[2]


im_width = 128
im_height = 128
border = 5
im_chan = 1 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth

X, y = get_data(path_train, train_ids, train=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=2018)

bs= 16

model = U_Net()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # The mean_iou metrics seens to leak train and test values...

results = model.fit_generator(image_generator(X_train,y_train,bs), steps_per_epoch=round(len(X_train)/bs), epochs=100, callbacks=callbacks,validation_data=(X_valid, y_valid))
