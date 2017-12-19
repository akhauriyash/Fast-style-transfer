from __future__ import print_function
from PIL import ImageGrab
import numpy as np
import time
import sys
import transform
import scipy.misc
from skimage.transform import resize
import tensorflow as tf

device_t = "/gpu:0"

g = tf.Graph()
#   Place process to secondary devices if GPU not available
soft_config = tf.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
#    batch_shape = (1,) + (540, 600, 3)
    batch_shape = (1,) + (1080, 1200, 3)
    img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name = 'img_placeholder')
    preds = transform.net(img_placeholder)
    X = np.zeros(batch_shape)
#    saver = tf.train.Saver()
    i= 0
#    saver.restore(sess, "udnie.ckpt")]
    sess.run(tf.global_variables_initializer())
    print("Attempting restoration of model")
    saver = tf.train.import_meta_graph('D:/Coding/Deep learning/fast-style-transfer-master/checkpoints/fns.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint("D:/Coding/Deep learning/fast-style-transfer-master/checkpoints/"))
    print("Success (Restoration)")
    while(1):
        i += 1
        a=time.time()
#        img = np.asarray(ImageGrab.grab(bbox=(288, 67, 888, 607)))
        img = np.asarray(ImageGrab.grab(bbox=(0, 0, 1200, 1080)))
        X[0] = img
        c = time.time()
        _preds = sess.run(preds, feed_dict = {img_placeholder: X})
        print("Style time: " + str(time.time() - c))
        print(_preds.shape)
        print(time.time() - a)
        img = np.clip(_preds[0], 0, 255).astype(np.uint8)
#        img = resize(img, (1080, 1200))
        print("Prediction shape: " + str(_preds.shape) + "  Input shape: " + str(img.shape))

#        scipy.misc.imsave(str(i) + ".jpg", img)
    
'''
    ######### Run time analysis:
        
    Objective is to maintain the aspect ratio while giving a reasonable output. 
    In transpose layer, if we use bigger stride, the output size increases in 
    proportion. Input ratio should be 1200/1080 (1.11111111) and output should
    maintain that in proportion.
    
    Perhaps 300x270 input could be a good place to start.
    
    If we take 100x90 input, we get 0.005s, which is 200 frames per second.
        We can use this with super resolution.
        
    From 0.2 seconds per stylization, we get to 0.05 seconds per stylization 
    But as we increase the number of strides, computation increases and we go 
    from 0.2 seconds to 0.16 per stylization. 
    
    Model Orig  : 0.2 seconds Input/Output = 1
    
    Model 1     : 0.16 seconds Input/Output = 1/4 (conv2 and conv3 strides = 1)
    
    Model 2     : 0.1 seconds Input/Output = 1/4  (Significant cutbacks, 
                                                   conv2 conv3 32 kernel depth
                                             residual block all to 32 from 128)
    We can trade:
        Model Depth
        Model strides
        Model kernel depth
    
    ######### Training time analysis:
    
    The network is simultaneously being trained for super resolution.
    We cannot thus backpropagate correctly unless we have the super resolution
    variant of the same image.
    
    Configuration below with residual channel depth at 64 gives
    0.027 style transfer time on average with 0.057s including everything.
    
        def net(image):
        conv1 = _conv_layer(image, 32, 9, 1)
        conv2 = _conv_layer(conv1, 64, 3, 2)
        conv3 = _conv_layer(conv2, 64, 3, 2)
        resid1 = _residual_block(conv3, 3)
        resid2 = _residual_block(resid1, 3)
        resid3 = _residual_block(resid2, 3)
        resid4 = _residual_block(resid3, 3)
        resid5 = _residual_block(resid4, 3)
        conv_t1 = _conv_tranpose_layer(resid5, 32, 3, 2)
        conv_t3 = _conv_layer(conv_t1, 3, 9, 1, relu=False)
        preds = tf.nn.tanh(conv_t3) * 150 + 255./2
        return preds

    

'''
