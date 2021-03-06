#!/usr/bin/python
import sys
import numpy as np
from os.path import basename
import pdb

NUM_FEATURES = 123
NUM_FEATURES_BIAS = NUM_FEATURES + 1
NUM_ITERATIONS = 10
LEARN_RATE_TWEAK_START = 0
LEARN_RATE_TWEAK_STOP = 0.03
LEARN_RATE_TWEAK_STEP = 0.01
C_TWEAK_START = 0.98
C_TWEAK_STOP = 1
C_TWEAK_STEP = 0.01

def generate_yx( data_list ): # This algorithm acts a generator which produces aligned values of x and y
    x = np.zeros( shape=(len(data_list),NUM_FEATURES) )
    #x[:,NUM_FEATURES_BIAS-1] = np.ones( x.shape[0] ) # Append 1s

    for i in range( len( data_list ) ):
        for feature_index in data_list[i][1:]:
            x[i, feature_index-1] = 1;

    for i in range( x.shape[0] ):
        yield (data_list[i][0],x[i]) #(y,x)


def train_wb( data_list, learn_rate, C, num_iterations, init_w=None, init_b=None ): # This function is repsonsible for training the weight vector using a data list containing x and y matrix values

    if ( init_w is None ):
        w = np.zeros( shape=(1, NUM_FEATURES) )
        #w = np.random.rand( 1, NUM_FEATURES )
    else:
        w = init_w
    if ( init_b is None ):
        b = 0
    else:
        b = init_b

    N = len( data_list )
    for i in range( num_iterations ):
        # SVM with SGD Algorithm Begin
        #pdb.set_trace()
        for y,x in generate_yx( data_list ):
            ywx_b = float(y) * (w.dot( x ) - b)
            if ( 1 - ywx_b ) >= 0:
                w = w - (learn_rate*( (w/float(N)) - (C*y*x) ))
                b = b + learn_rate*C*y
            else:
                w = w - learn_rate*(w/float(N))
        # SVM with SGD Algorithm End
    return w, b


def test_wb( test_list, w, b ): # This value runs tests on a given set of data on a provided weight vector
    hits = 0
    for y,x in generate_yx( test_list ):
        hits = hits + 1 if (float(y) * ( w.dot( x ) - b )) > 0 else hits
    print "Hits: {}".format( hits )
    return float(hits) / float(len(test_list))


#### MAIN ####
def main( argv, argc ):
    
    #pdb.set_trace()
    check_args( argv, argc )
        
    with open( argv[1], 'r' ) as fp:
        data_list = [[int(i.split(':')[0]) for i in line.split()] for line in fp.readlines()] # For each line in fp.readlines, split by default delimeters
    with open( argv[2], 'r' ) as fp:
        dev_list = [[int(i.split(':')[0]) for i in line.split()] for line in fp.readlines()]
    with open( argv[3], 'r' ) as fp:
        test_list = [[int(i.split(':')[0]) for i in line.split()] for line in fp.readlines()]
        
    #Training
    learn_rate = 0.5
    C = 0.9
    w, b = train_wb( data_list, learn_rate, C, NUM_ITERATIONS ) # Training

    tweak_table = list()
    for learn_rate_tweak in reversed( np.arange( LEARN_RATE_TWEAK_START,LEARN_RATE_TWEAK_STOP, LEARN_RATE_TWEAK_STEP ) ): # Tweaking learning rate
        for C_tweak in np.arange( C_TWEAK_START, C_TWEAK_STOP, C_TWEAK_STEP ):
            w_tweak, b_tweak = train_wb( data_list, learn_rate_tweak, C_tweak, NUM_ITERATIONS, init_w=w, init_b=b ) 
            accuracy = test_wb( dev_list, w_tweak, b_tweak )
            print (C_tweak, learn_rate_tweak, accuracy)
            tweak_table.append( [C_tweak, learn_rate_tweak, accuracy] )

    # Find max accuracy, associate all accuracies with learning rates in table
    print "Best accuracy with (C, learn_rate, acc): {}".format(
        tweak_table[:][np.argmax(np.asarray(tweak_table), axis=0)[2]] ) 
    
    accuracy = test_wb( test_list, w, b ) # Testing
    print "Testing Set Accuracy: {}".format( accuracy )


def usage( ):
    print "Usage: python {} <training file> <dev file> <test file>".format( basename(__file__) )
    return 0

def check_args( argv, argc ):
    if ( argc < 2 ):
        usage()
        sys.exit( "Incorrect Number of Arguments" )
            
if __name__ == "__main__":
    main( sys.argv, len( sys.argv ) ) 


    
