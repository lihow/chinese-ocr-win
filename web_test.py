# -*- coding:utf-8 -*-
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Keras
from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Flatten,BatchNormalization,Permute,TimeDistributed,Dense,Bidirectional,GRU
from keras.models import Model
from keras.layers import Lambda
from keras.optimizers import SGD
import numpy as np
#from PIL import Image
import keras.backend  as K
import ocr.keys as keys
import os

#from scipy import misc
from PIL import Image
from sklearn.externals import joblib

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

#crnn
import torch
import torch.utils.data
from torch.autograd import Variable
import crnn.dataset as dataset
import crnn.keys as keys1
import crnn.models.crnn as crnn
import crnn.util as util
GPU = False

#ctpn textdetector
import tensorflow as tf
import cv2
import sys
import time
from math import *

sys.path.append('ctpn')
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import  test_ctpn
from ctpn.ctpn.detectors import TextDetector
from ctpn.ctpn.other import draw_boxes
from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im


app = Flask(__name__)

#ctpn
def load_tf_model():
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state('ctpn/models/')
    saver.restore(sess, ckpt.model_checkpoint_path)
    return sess,saver,net

##init model
sess,saver,net = load_tf_model()

def text_detect(img):
    #ctpn
    scale, max_scale = Config.SCALE,Config.MAX_SCALE
    img,f = resize_im(img,scale=scale,max_scale=max_scale)
    scores, boxes = test_ctpn(sess, net, img)
    textdetector  = TextDetector()
    boxes = textdetector.detect(boxes,scores[:, np.newaxis],img.shape[:2])
    text_recs,tmp = draw_boxes(img, boxes, caption='im_name', wait=True,is_display=False)
    return text_recs,tmp,img

##OCR
#define model
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def get_model(height,nclass):
    rnnunit  = 256
    input = Input(shape=(height,None,1),name='the_input')
    m = Conv2D(64,kernel_size=(3,3),activation='relu',padding='same',name='conv1')(input)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool1')(m)
    m = Conv2D(128,kernel_size=(3,3),activation='relu',padding='same',name='conv2')(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,2),name='pool2')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv3')(m)
    m = Conv2D(256,kernel_size=(3,3),activation='relu',padding='same',name='conv4')(m)

    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool3')(m)

    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv5')(m)
    m = BatchNormalization(axis=1)(m)
    m = Conv2D(512,kernel_size=(3,3),activation='relu',padding='same',name='conv6')(m)
    m = BatchNormalization(axis=1)(m)
    m = ZeroPadding2D(padding=(0,1))(m)
    m = MaxPooling2D(pool_size=(2,2),strides=(2,1),padding='valid',name='pool4')(m)
    m = Conv2D(512,kernel_size=(2,2),activation='relu',padding='valid',name='conv7')(m)

    m = Permute((2,1,3),name='permute')(m)
    m = TimeDistributed(Flatten(),name='timedistrib')(m)

    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm1')(m)
    m = Dense(rnnunit,name='blstm1_out',activation='linear')(m)
    m = Bidirectional(GRU(rnnunit,return_sequences=True),name='blstm2')(m)
    y_pred = Dense(nclass,name='blstm2_out',activation='softmax')(m)

    basemodel = Model(inputs=input,outputs=y_pred)

    labels = Input(name='the_labels', shape=[None,], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    #model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    #model.summary()
    return model,basemodel

#load model
modelPath = os.path.join(os.getcwd(),"ocr/ocr0.2.h5")
height = 32
characters = keys.alphabet[:]
nclass = len(characters)
if os.path.exists(modelPath):
    model,basemodel = get_model(height,nclass+1)
    basemodel.load_weights(modelPath)


#ocr model
def decode(pred):
        charactersS = characters+u' '
        t = pred.argmax(axis=2)[0]
        length = len(t)
        char_list = []
        n = len(characters)
        for i in range(length):
            if t[i] != n and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(charactersS[t[i] ])
        return u''.join(char_list)

def predict(im):
    """
    
    """
    im = im.convert('L')
    scale = im.size[1]*1.0 / 32
    w = im.size[0] / scale
    w = int(w)
    im = im.resize((w,32))
    img = np.array(im).astype(np.float32)/255.0
    X  = img.reshape((32,w,1))
    X = np.array([X])
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:,2:,:]
    out    = decode(y_pred)##
    #out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :]

    #out = u''.join([characters[x] for x in out[0]])

    if len(out)>0:
        while out[0]==u'。':
            if len(out)>1:
               out = out[1:]
            else:
                break

    return out

def crnnSource():
    alphabet = keys1.alphabet
    converter = util.strLabelConverter(alphabet)
    if torch.cuda.is_available() and GPU:
       model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1).cpu()
    path = './crnn/samples/model_acc97.pth'
    model.eval()
    model.load_state_dict(torch.load(path))
    return model,converter
    
model,converter = crnnSource()
def crnnOcr(image):
       """
       crnn模型，ocr识别
       @@model,
       @@converter,
       @@im
       @@text_recs:text box

       """
       scale = image.size[1]*1.0 / 32
       w = image.size[0] / scale
       w = int(w)
       #print "im size:{},{}".format(image.size,w)
       transformer = dataset.resizeNormalize((w, 32))
       if torch.cuda.is_available() and GPU:
           image = transformer(image).cuda()
       else:
           image = transformer(image).cpu()

       image = image.view(1, *image.size())
       image = Variable(image)
       model.eval()
       preds = model(image)
       _, preds = preds.max(2)
       preds = preds.transpose(1, 0).contiguous().view(-1)
       preds_size = Variable(torch.IntTensor([preds.size(0)]))
       sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
       if len(sim_pred)>0:
          if sim_pred[0]==u'-':
             sim_pred=sim_pred[1:]

       return sim_pred


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """

    box = sorted(box,key=lambda x:sum([x[1],x[3],x[5],x[7]]))
    return box


def dumpRotateImage(img,degree,pt1,pt2,pt3,pt4):
    height,width=img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)


    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim,xdim = imgRotation.shape[:2]
    imgOut=imgRotation[max(1,int(pt1[1])):min(ydim-1,int(pt3[1])),max(1,int(pt1[0])):min(xdim-1,int(pt3[0]))]
    #height,width=imgOut.shape[:2]
    return imgOut

    
def crnnRec(im,text_recs,ocrMode='keras',adjust=False):
   """
   crnn模型，ocr识别
   @@model,
   @@converter,
   @@im:Array
   @@text_recs:text box
   
   """
   index = 0
   results = {}
   xDim ,yDim = im.shape[1],im.shape[0]
    
   for index,rec in enumerate(text_recs):
       results[index] = [rec,]
       xlength = int((rec[6] - rec[0])*0.1)
       ylength = int((rec[7] - rec[1])*0.2)
       if adjust:
           pt1 = (max(1,rec[0]-xlength),max(1,rec[1]-ylength))
           pt2 = (rec[2],rec[3])
           pt3 = (min(rec[6]+xlength,xDim-2),min(yDim-2,rec[7]+ylength))
           pt4 = (rec[4],rec[5])
       else:
           pt1 = (max(1,rec[0]),max(1,rec[1]))
           pt2 = (rec[2],rec[3])
           pt3 = (min(rec[6],xDim-2),min(yDim-2,rec[7]))
           pt4 = (rec[4],rec[5])
        
       degree =  degrees(atan2(pt2[1]-pt1[1],pt2[0]-pt1[0]))##图像倾斜角度

       partImg = dumpRotateImage(im,degree,pt1,pt2,pt3,pt4)

       image = Image.fromarray(partImg ).convert('L')
       if ocrMode=='keras':
            sim_pred = predict(image)
       else:
            sim_pred = crnnOcr(image)

       results[index].append(sim_pred)##识别文字
 
   return results

def model_predict(img):
    img = np.array(img.convert('RGB'))
    t = time.time()
    text_recs,tmp,img = text_detect(img)
    text_recs = sort_box(text_recs)
    #未提供pythroch版本 后续补充
    result = crnnRec(img,text_recs,ocrMode='pytorch',adjust=False)
    cost_t = time.time()-t
    return (cost_t, result)


'''
@app.route('/upload', methods=['POST'])
def upload():

    f = request.files['file']
    #im = misc.imread(f)
    img = Image.open(f)
    cost_t, result = model_predict(img)
    #img = im.reshape((1,784))

    #clf = joblib.load('model/ok.m')

    #l = clf.predict(img)

    #return 'predict: %s ' % (l[0])
'''

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        img = Image.open(f)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        #f.save(file_path)
        img.save(file_path)

        # Make prediction
        cost_t, result = model_predict(img)
        #preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        output = ''
        for key in result:
            output = output + result[key][1] + '\n'

        
        return output
    return None


@app.route('/')
def index():
    # Main page
    return render_template('index.html')


if __name__ == '__main__':
    #app.run(debug=True, port=7000)
    http_server = WSGIServer(('127.0.0.1', 1234), app)
    http_server.serve_forever()

