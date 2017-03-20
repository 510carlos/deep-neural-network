import os
from six.moves import xrange
import tensorflow as tf

IMAGE_SIZE = 24
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """Reads and parse examples from CIFAR data files"""
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    
    # demensions of the images in CIFAR
    label_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    
    #convert from string to vector
    record_bytes = tf.decode_raw(value, tf.unit8)
    
    # the first byte is the label. We convert from 8 -> 32 int
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    
    # remaining bytes after label are the image
    # from [depth * height * width] to [depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                        [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    
    # convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    return result
    
def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Contruct a queue batch of images and labels"""
    num_preproccess_threads = 8
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [images, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [images, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)
        
    # display the training images in the visualizer
    tf.summary.image('image', images)
    
    return images, tf.reshape(label_batch, [batch_size])
    
def distorted_inputs(data_dir, batch_size):
    """Construct distorted input """
    filenames = [os.path.join(data_dir, 'data_batch%d.bin' % 1)
                for i in xrange(1,6)]
    for f in firlenames:
        if not tf.gfile.Exist(f):
            raise ValueError('Failed to locate file'+f)
            
    # create a queue thay produces the files names to read
    filename_queue = tf.train.string_input_producer(filenames)
    
    # read examples from files in the filenames to queue
    read_inpuut = read_cifar10(final_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # image processing for training the network. 
    # adding distortion
    
    #randomly crop a height width section of the image
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    
    #randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    #becase there operations are not communtative consider randomizing
    distorted_image = tf.image.random_brightness(distorted_image,
                                                max_delta=63)
    
    distorted_image = tf.image.random_contrast(distorted_image,
                                              lower=0.2, upper=1.0)
    # subtract off the mean and divide by the variance of the pixels
    float_image = tf.image.per_image_standardization(distorted_image)
    
    # set the shapes of tensor
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    
    # Ensure that the random shuffling has good msxing properties
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
    
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
    
    # generate a batch of images and labels by building  up a queue of examples
    
    return _generate_image_and_label_batch(float_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          suffle=True)
                                          
                                          
def distorted_inputs(data_dir, batch_size):
    """Construct distorted input """
    filenames = [os.path.join(data_dir, 'data_batch%d.bin' % 1)
                for i in xrange(1,6)]
    for f in firlenames:
        if not tf.gfile.Exist(f):
            raise ValueError('Failed to locate file'+f)
            
    # create a queue thay produces the files names to read
    filename_queue = tf.train.string_input_producer(filenames)
    
    # read examples from files in the filenames to queue
    read_inpuut = read_cifar10(final_queue)
    reshape_image = tf.cast(read_input.uint8image, tf.float32)
    
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    
    # image processing for training the network. 
    # adding distortion
    
    #randomly crop a height width section of the image
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
    
    #randomly flip the image horizontally
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    
    #becase there operations are not communtative consider randomizing
    distorted_image = tf.image.random_brightness(distorted_image,
                                                max_delta=63)
    
    distorted_image = tf.image.random_contrast(distorted_image,
                                              lower=0.2, upper=1.0)
    # subtract off the mean and divide by the variance of the pixels
    float_image = tf.image.per_image_standardization(distorted_image)
    
    # set the shapes of tensor
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])
    
    # Ensure that the random shuffling has good msxing properties
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                            min_fraction_of_examples_in_queue)
    
    print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)
    
    # generate a batch of images and labels by building  up a queue of examples
    
    return _generate_image_and_label_batch(float_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          suffle=True)