from __future__ import print_function
from PIL import ImageGrab
import numpy as np
import time
import sys
import os
sys.path.insert(0, 'src')
import transform
import scipy.misc
import tensorflow as tf
from tkinter import Tk, Label, Button, Entry

class MyFirstGUI:

    def __init__(self, master):
        self.master = master
        master.minsize(width=320, height=350)

        master.title("Art'em")
        
        self.L1 = Label(master, text="Time to stylize (in seconds)")
        self.e = Entry(master, bd=5)
        self.L2 = Label(master, text="H0")
        self.h0 = Entry(master, bd=5)
        self.L3 = Label(master, text="W0")
        self.w0 = Entry(master, bd=5)
        self.L4 = Label(master, text="H1")
        self.h1 = Entry(master, bd=5)
        self.L5 = Label(master, text="W1")
        self.w1 = Entry(master, bd=5)
        self.r1 = Button(master, text="udnie", command= lambda: self.start(1), height = 1, width = 20)
        self.r2 = Button(master, text="rain princess", command= lambda: self.start(2), height = 1, width = 20)
        self.r3 = Button(master, text="wreck", command= lambda: self.start(3), height = 1, width = 20)
        self.r4 = Button(master, text="scream", command= lambda: self.start(4), height = 1, width = 20)
        self.r5 = Button(master, text="la muse", command= lambda: self.start(5), height = 1, width = 20)
        self.close_button = Button(master, text="Close", command=master.quit, height = 1, width = 20)

        self.e.grid(row = 0, column = 1)
        self.L1.grid(row = 0, column = 0)
        self.h0.grid(row = 1, column = 1)
        self.L2.grid(row = 1, column = 0)
        self.w0.grid(row = 2, column = 1)
        self.L3.grid(row = 2, column = 0)
        self.h1.grid(row = 3, column = 1)
        self.L4.grid(row = 3, column = 0)
        self.w1.grid(row = 4, column = 1)
        self.L5.grid(row = 4, column = 0)
        self.r1.grid(row = 5, column = 1)
        self.r2.grid(row = 6, column = 1)
        self.r3.grid(row = 7, column = 1)
        self.r4.grid(row = 8, column = 1)
        self.r5.grid(row = 9, column = 1)
        self.close_button.grid(row = 10, column = 1)

    def start(self, k):
        device_t = "/gpu:0"
        if not os.path.exists('output_project'):
            os.makedirs('output_project')
        
        g = tf.Graph()
        timer = float(self.e.get())
        hs0 = int(self.h0.get())
        ws0 = int(self.w0.get())
        hs1 = int(self.h1.get())
        ws1 = int(self.w1.get())
        
        H = hs1 - hs0
        W = ws1 - ws0
            
        soft_config = tf.ConfigProto(allow_soft_placement=True)
        
        soft_config.gpu_options.allow_growth = True
        
        with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
            batch_shape = (1,) + (1080, 1200, 3)
            batch_shape = (1,) + (H, W, 3)
            img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name = 'img_placeholder')
            preds = transform.net(img_placeholder)
            X = np.zeros(batch_shape)
            saver = tf.train.Saver()
            i= 0
            if(k==1):
                saver.restore(sess, "udnie.ckpt")
            if(k==2):
                saver.restore(sess, "rain_princess.ckpt")
            if(k==3):
                saver.restore(sess, "wreck.ckpt")
            if(k==4):
                saver.restore(sess, "scream.ckpt")
            if(k==5):
                saver.restore(sess, "la_muse.ckpt")
            start_time = time.time()
            print('starting loop')
            print(timer)
            while((time.time() - start_time) < timer):
                i += 1
                a=time.time()
                img = np.asarray(ImageGrab.grab(bbox=(ws0, hs0, ws1, hs1)))
                X[0] = img
                c = time.time()
                _preds = sess.run(preds, feed_dict = {img_placeholder: X})
                print("Style time: " + str(time.time() - c))
                print(_preds.shape)
                print(time.time() - a)
                img = np.clip(_preds[0], 0, 255).astype(np.uint8)
                print("Prediction shape:" + str(_preds.shape) + "  Input shape: " + str(img.shape))
                scipy.misc.imsave("output_project/" + str(i) + ".jpg", img)
            print('loop ended')


root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()

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
