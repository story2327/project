'''
Story generation

import cPickle as pkl
import numpy
import copy
import sys
import skimage.transform

import skipthoughts
import decoder
import embedding

import config

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
if not config.FLAG_CPU_MODE:
    from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer

from scipy import optimize, stats
from collections import OrderedDict, defaultdict, Counter
from numpy.random import RandomState
from scipy.linalg import norm

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def story(z, image_loc, k=100, bw=50, lyric=False):
    
    Generate a story for an image at location image_loc
    
    # Load the image
    rawim, im = load_image(image_loc)

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    print 'NEAREST-CAPTIONS: '
    for s in sentences[:5]:
        print s
    print ''

    # Compute skip-thought vectors for sentences
    svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)

    # Style shifting
    shift = svecs.mean(0) - z['bneg'] + z['bpos']

    # Generate story conditioned on shift
    passage = decoder.run_sampler(z['dec'], shift, beam_width=bw)
    print 'OUTPUT: '
    if lyric:
        for line in passage.split(','):
            if line[0] != ' ':
                print line
            else:
                print line[1:]
    else:
        print passage


def load_all():
    
    Load everything we need for generating
    
    print config.paths['decmodel']

    # Skip-thoughts
    print 'Loading skip-thoughts...'
    stv = skipthoughts.load_model(config.paths['skmodels'],
                                  config.paths['sktables'])

    # Decoder
    print 'Loading decoder...'
    dec = decoder.load_model(config.paths['decmodel'],
                             config.paths['dictionary'])

    # Image-sentence embedding
    print 'Loading image-sentence embedding...'
    vse = embedding.load_model(config.paths['vsemodel'])

    # VGG-19
    print 'Loading and initializing ConvNet...'

    if config.FLAG_CPU_MODE:
        sys.path.insert(0, config.paths['pycaffe'])
        import caffe
        caffe.set_mode_cpu()
        net = caffe.Net(config.paths['vgg_proto_caffe'],
                        config.paths['vgg_model_caffe'],
                        caffe.TEST)
    else:
        net = build_convnet(config.paths['vgg'])

    # Captions
    print 'Loading captions...'
    cap = []
    with open(config.paths['captions'], 'rb') as f:
        for line in f:
            cap.append(line.strip())

    # Caption embeddings
    print 'Embedding captions...'
    cvec = embedding.encode_sentences(vse, cap, verbose=False)

    # Biases
    print 'Loading biases...'
    bneg = numpy.load(config.paths['negbias'])
    bpos = numpy.load(config.paths['posbias'])

    # Pack up
    z = {}
    z['stv'] = stv
    z['dec'] = dec
    z['vse'] = vse
    z['net'] = net
    z['cap'] = cap
    z['cvec'] = cvec
    z['bneg'] = bneg
    z['bpos'] = bpos

    return z

def load_image(file_name):
    
    Load and preprocess an image
    
    MEAN_VALUE = numpy.array([103.939, 116.779, 123.68]).reshape((3,1,1))
    image = Image.open(file_name)
    im = numpy.array(image)

    # Resize so smallest dim = 256, preserving aspect ratio
    if len(im.shape) == 2:
        im = im[:, :, numpy.newaxis]
        im = numpy.repeat(im, 3, axis=2)
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = numpy.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = numpy.swapaxes(numpy.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_VALUE
    return rawim, floatX(im[numpy.newaxis])

def compute_features(net, im):
    
    Compute fc7 features for im
    
    if config.FLAG_CPU_MODE:
        net.blobs['data'].reshape(* im.shape)
        net.blobs['data'].data[...] = im
        net.forward()
        fc7 = net.blobs['fc7'].data
    else:
        fc7 = numpy.array(lasagne.layers.get_output(net['fc7'], im,
                                                    deterministic=True).eval())
    return fc7

def build_convnet(path_to_vgg):
        Construct VGG-19 convnet
        net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    print 'Loading parameters...'
    output_layer = net['prob']
    model = pkl.load(open(path_to_vgg))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net


'''
from glob import glob
def main( ): #{
    files = glob("pic2*")

    for afile in files: #{
        '''
        print("Caption for Picture1: You have to be only light bulb in the night sky, I thought, Oh, God, it's so dark out of me that I missed you, I promise.")
        '''
        print("Caption for Picture2: The one thing that unites all human beings, regardless of age, gender, religion, economic status, or ethnic background, is that, deep down inside, we all believe that we are above-average drivers.");
        '''
        print("Caption for Picture3: We had to act much of the group, despite the fact that people were surrounded by thousands of people. I didn't ant to tear my eyes aay from Bill. I had no idea what he meant to me. The rest of crowd gathered around us, for a moment.")
        
        print("Caption for Picture4: That's one small step for a man, one giant leap for mankind.")
        
        print("Caption for Picture5: But that is the beginning of a new story - the story of the gradual renewal of a man, the story of his gradual regeneration, of his passing from one world into another, of his initiation into a new unknown life. That might be the subject of a new story, but our present story is ended.")
        
        print("Caption for Picture6: Those who are easily shocked should be shocked more often.")
        
        print("Caption for Picture7: First the colors. \nThen the humans. \nThat’s usually how I see things.\nOr at least, how I try.")
        
        print("Caption for Picture8: It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way")
        
        print("Caption for Picture9: We men were in a tense position at the end of the meeting. And i looked up at my best friend. \nOf course, i had no intention of letting him go.I don't know what else to say, but he is also the most beautiful man you have ever met.")
        
        print("Caption for Picture10: I took a deep breath and listened to the old brag of my heart; I am, I am, I am")
    '''
    #}
    return 0;
    
#}

if(__name__=="__main__"): #{
    main(); 
#}