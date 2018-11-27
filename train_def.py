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
import pdb
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
    im_width = 101
    im_height = 101
    im_chan = 3
    X = np.zeros((len(ids), im_height, im_width, im_chan), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width,1), dtype=np.float32)
    print('Getting images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # Load images
        if train:
           img = load_img(path + '/images/' + id_, grayscale=True)
           
        else:
           img = load_img(path + 'test/images/' + id_, grayscale=True) #Change path for your system

        x_img = img_to_array(img)
        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))

        # Save images
        X[n] = x_img
        
        if train:
            y[n] = mask
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
    gauss = np.random.normal(mu,sigma,(row,col,ch))*255
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    noisy_img_clipped = np.clip(noisy, 0, 255)
    return noisy_img_clipped

  
def horizontal_flip(image):
    """Flip image left-right"""
    image = cv2.flip(image, 1)
    return image

  
def vertical_flip(image): 
    """Flip image top-bottom"""
    image = cv2.flip(image, 0)
    return image
  
           
def image_generator(batch_size,train,augment):
    """
    Custom image generator that augments and generates
    batches of images and masks arrays
    if train = True, generates images and masks else only images
    if augment = noise, adds Gaussian noise
    if augment = lr, flips horizontally
    else, flips vertically
    
    """
    mean = 0.0
    variance = np.random.uniform(0.01, 1.0)
    border = 5
    
    path_train = "./"
    path_test = "./"
    
    #list of files
    train_ids = next(os.walk(path_train+"images"))[2]
    test_ids = next(os.walk(path_test+"test/images"))[2] # Change this for your system
    if not train:
       batch_size = 18000  #Fixed test size
    
    X = np.zeros((batch_size, 128, 128, 2), dtype=np.float32)
    if train:
        y = np.zeros((batch_size, 128, 128,1), dtype=np.float32)
        images,masks = get_data(path_train,train_ids,train=True) # Get arrays  
    else:
        images = get_data(path_train,test_ids,train=False) # Get arrays 
    
    batch_images = np.zeros((batch_size, 101,101,3))
    batch_masks = np.zeros((batch_size,101,101,1))
    print('Generating new images........')
    
    if batch_size < 4000:
      raise ValueError("You cannot input less than 4000")
    
    for i in range(batch_size):
        if train:
           n_samples = 4000 #train data size
            
        else:
           n_samples = 18000 #test data size
            
        if i<n_samples:
           # choose random index in features
           batch_images[i] = images[i]
           if train:
              batch_masks[i] = masks[i]

           x_image = resize(batch_images[i], (128, 128, 1), mode='constant', preserve_range=True)

           # Create cumsum x
           x_center_mean = x_image[border:-border, border:-border].mean()
           x_csum = (np.float32(x_image)-x_center_mean).cumsum(axis=0)
           x_csum -= x_csum[border:-border, border:-border].mean()
           x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

           # Load masks
           if train:
              mask = resize(batch_masks[i], (128, 128, 1), mode='constant', preserve_range=True)
              y[i] = mask / 255

           # Save images
           X[i,...,0] = x_image.squeeze() / 255
           X[i, ..., 1] = x_csum.squeeze()

           
            
        else:
           # choose random index in features
           index= np.random.choice(n_samples,1)[0]
           if augment == "noise":
              batch_images[i] = add_random_gaussian_noise(images[index],mean,variance)
           elif augment == "lr":
              batch_images[i] = horizontal_flip(images[index])
           else:
              batch_images[i] = vertical_flip(images[index])
              
           if train: # For training we need masks as well
              if augment == "noise":
                 batch_masks[i] = masks[index]
                  
              elif augment == "lr":
                 batch_masks[i] = resize(horizontal_flip(masks[index]), (101, 101, 1), mode='constant', preserve_range=True)
                  
              else:
                 batch_masks[i] = resize(vertical_flip(masks[index]), (101, 101, 1), mode='constant', preserve_range=True)

           x_image = resize(batch_images[i], (128, 128, 1), mode='constant', preserve_range=True)

           # Create cumsum x
           x_center_mean = x_image[border:-border, border:-border].mean()
           x_csum = (np.float32(x_image)-x_center_mean).cumsum(axis=0)
           x_csum -= x_csum[border:-border, border:-border].mean()
           x_csum /= max(1e-3, x_csum[border:-border, border:-border].std())

           # Load masks
           if train:
              mask = resize(batch_masks[i], (128, 128, 1), mode='constant', preserve_range=True)
              y[i] = mask / 255

           # Save images
           X[i,...,0] = x_image.squeeze() / 255
           X[i, ..., 1] = x_csum.squeeze()

           
    
    print('Done')
    if train:
       return X, y
    else:
       return X
    
    
 
im_width = 128
im_height = 128
border = 5
im_chan = 2 # Number of channels: first is original and second cumsum(axis=0)
n_features = 1 # Number of extra features, like depth

X, y = image_generator(8000,train=True,augment="noise")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)

save_str = 'model-tgs-salt-var-{0}--noise-doubled-'.format(v) + '-acc-{val_acc:.2f}-epoch-{epoch:02d}.h5'
callbacks = [
    EarlyStopping(patience=7, verbose=1), #was patience=3 for LRNon
    ReduceLROnPlateau(patience=5, verbose=1), #wast patience=3
    ModelCheckpoint(save_str, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_acc')
]

model = UNET.U_Net()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # The mean_iou metrics seens to leak train and test values...

results = model.fit({'img': X_train}, y_train, batch_size=16, epochs=100, callbacks=callbacks, validation_data=({'img': X_valid}, y_valid))  

# Hereon this code is predicting on test data and creating submission file
def iou_metric(y_true_in, y_pred_in, print_table=False):
    # src: https://www.kaggle.com/aglotero/another-iou-metric
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union
    
    
    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn
  
# Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)
  
def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)
    
def RLenc(img, order='F', format=True):
    """
    # Source https://www.kaggle.com/bguberfain/unet-with-depth
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e.
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
    
    
#Get best threshold
preds_val = model.predict({'img': X_valid}, verbose=1)
thres = np.linspace(0.25, 0.75, 20)
thres_iou = [iou_metric_batch(y_valid, np.int32(preds_val > t)) for t in tqdm(thres)]

best_thres = thres[np.argmax(thres_iou)]
best_thres, max(thres_iou)

X_test = image_generator(18000,train=False,augment="lr")  # For test batch_size and augment parameters do not matter

preds_test = model.predict(X_test,verbose=1)

path_test = "./"
    
#list of test files
test_ids = next(os.walk(path_test+"test/images"))[2] # Change this for your system

# Reshape predicted mask
preds_test_reshaped = []
for i in range(len(preds_test)):
    preds_test_reshaped.append(resize(np.squeeze(preds_test[i]), 
                                       (101, 101), 
                                       mode='constant', preserve_range=True))
pred_dict = {id_[:-4]:RLenc(np.round(preds_test_reshaped[i] > 0.2)) for i,id_ in tqdm(enumerate(test_ids))}

#Create submission file
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
