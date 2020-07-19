import os
import numpy as np
import random
import scipy.misc
import threading
from keras.preprocessing.sequence import pad_sequences

def sequence_generator(data_list, batch_size, input_shape, num_classes, secondary_data_list=None):
    '''
    Read sequence data of batch_size into memory
    :param data_list: The data generated by get_data_list
    :param batch_size:
    :param input_shape: tuple: the shape of numpy ndarray, e.g. (seq_len, 216, 216, 3) for sequence
                        or (216, 216, 18) for optical flow data
    :param num_classes: number of classes in the dataset
    :param secondary_data: The data generated by get_data_list for a second preprocessing method to 
                        be combine by max function. Must have > half the number of frames but not more frames
    :return:
    '''
    if isinstance(input_shape, tuple):
        x_shape = (batch_size,) + input_shape
    else:
        raise ValueError('Input shape is neither 1D or 3D')
    y_shape = (batch_size, num_classes)
    index = 0
    while True:
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            step = random.randint(1, len(data_list) - 1)  # approach a random-size step to get the next video sample
            index = (index + step) % len(data_list)
            clip_dir, clip_class = data_list[index]
            batch_y[i, clip_class - 1] = 1
            clip_dir = os.path.splitext(clip_dir)[0] + '.npy'
            
            if secondary_data_list != None: #using hybrid-DMD
                second_clip_dir, _ = secondary_data_list[index]
                second_clip_dir = os.path.splitext(second_clip_dir)[0] + '.npy'
                
            # avoid endless loop
            count = 0
            while not os.path.exists(clip_dir) or (secondary_data_list != None and os.path.exists(second_clip_dir)):
                print("couldn't find " + str(clip_dir))
                count += 1
                if count > 20:
                    raise FileExistsError('Too many file missing')
                index = (index + 1) % len(data_list)
                clip_dir, class_idx = data_list[index]
                clip_dir = os.path.splitext(clip_dir)[0] + '.npy'
                if secondary_data_list != None: #using hybrid-DMD
                    second_clip_dir, _ = secondary_data_list[index]
                    second_clip_dir = os.path.splitext(second_clip_dir)[0] + '.npy'
                    
            clip_data = np.load(clip_dir)
            if secondary_data_list != None: #using hybrid-DMD
                second_data = np.load(second_clip_dir)
                
            if clip_data.shape != batch_x.shape[1:]:
                #print('Truncating modes...')
                if len(clip_data.shape) == 4:
                    temp_clip = np.reshape(clip_data,(clip_data.shape[0]*clip_data.shape[1]*clip_data.shape[2], clip_data.shape[3]))
                    temp_clip = pad_sequences(temp_clip, maxlen=batch_x.shape[-1],padding='post',truncating='post',dtype=temp_clip.dtype)
                    clip_data = np.reshape(temp_clip,(clip_data.shape[0],clip_data.shape[1],clip_data.shape[2],batch_x.shape[-1]))
                else:
                    temp_clip = np.reshape(clip_data,(clip_data.shape[0]*clip_data.shape[2],clip_data.shape[1]))
                    temp_clip = pad_sequences(temp_clip, maxlen=batch_x.shape[2],padding='post',truncating='post',dtype=temp_clip.dtype)
                    clip_data = np.reshape(temp_clip,(clip_data.shape[0],batch_x.shape[2],clip_data.shape[2]))
                #temp_clip = temp_clip[:,:,:,0:batch_x.shape[-1]]
                
                #print('reshaped to ' + str(clip_data.shape))
            if clip_data.shape != batch_x.shape[1:]:
                print('shape should be '+str(batch_x.shape)+' but is actually '+str(clip_data.shape))
                raise ValueError('The number of time sequence is inconsistent with the video data')
            
            if secondary_data_list != None:
                extra_frames = clip_data.shape[-1] - second_data.shape[-1]
                new_secondary = np.ndarray(clip_data.shape)
                if extra_frames > 0:
                    offset=0
                    for i in range(second_data.shape[-1]):
                        if extra_frames < 1:
                            break
                        new_secondary[:,:,i+offset] = second_data[:,:,i]
                        offset += 1
                        new_secondary[:,:,i+offset] = second_data[:,:,i]
                        extra_frames -= 1
                
                clip_data = np.maximum(clip_data,new_secondary)
                
            batch_x[i] = clip_data
        yield batch_x, batch_y


def image_from_sequence_generator(data_list, batch_size, input_shape, num_classes):
    '''
        Read one frame in the sequence data into memory
        input_shape: (seq_len,) + img_size
    '''
    batch_image_shape = (batch_size,) + input_shape[1:]
    batch_image = np.ndarray(batch_image_shape)

    video_gen = sequence_generator(data_list, batch_size, input_shape, num_classes)

    while True:
        batch_video, batch_label = next(video_gen)
        for idx, video in enumerate(batch_video):
            sample_frame_idx = random.randint(0, input_shape[0] - 1)
            sample_frame = video[sample_frame_idx]
            batch_image[idx] = sample_frame

        yield batch_image, batch_label


def image_generator(listtxt_dir, frames_dir, batch_size, img_size, num_classes, mean_sub=True, normalization=True,
                    random_crop=True, horizontal_flip=True):
    '''
    
    :param listtxt_dir: 
    :param frames_dir: 
    :param batch_size: 
    :param img_size: 
    :param num_classes: 
    :param mean_sub: 
    :param normalization: 
    :return: 
    '''
    with open(listtxt_dir) as fo:
        lines = [line for line in fo]
    x_shape = (batch_size,) + img_size
    y_shape = (batch_size, num_classes)

    mean_dir = os.path.join(frames_dir, 'mean.npy')
    if os.path.exists(mean_dir) and mean_sub:
        mean = np.load(mean_dir)
        mean.astype(dtype='float16')
    elif mean_sub:
        raise FileExistsError('RGB mean file does not exist')
    else:
        mean = None
    while True:
        batch_x = np.zeros(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            line = random.choice(lines)
            clip_name, clip_index = line.split()
            clip_name = os.path.basename(clip_name)
            clip_name = clip_name[:clip_name.find('.')]
            clip_dir = os.path.join(frames_dir, clip_name)
            frames = os.listdir(clip_dir)
            frame = random.choice(frames)
            frame_dir = os.path.join(clip_dir, frame)
            frame = scipy.misc.imread(frame_dir)
            if random_crop:
                x = random.randrange(frame.shape[0] - img_size[0])
                y = random.randrange(frame.shape[1] - img_size[1])
                frame = frame[x:x + img_size[0], y:y + img_size[1], :]
            else:
                frame = scipy.misc.imresize(frame, img_size)
            if horizontal_flip and random.randrange(2) == 1:
                frame = frame[:, ::-1, :]
            frame = frame.astype(dtype='float16')
            if mean is not None:
                frame -= mean
            if normalization:
                frame /= 255

            batch_x[i] = frame
            batch_y[i, int(clip_index) - 1] = 1

        yield batch_x, batch_y


def two_stream3_generator(listtxt_dir, spatial_dir, temporal_dir, batch_size, img_size, num_classes, mean_sub=True,
                          normalization=True, random_crop=True, horizontal_flip=True):
    with open(listtxt_dir) as fo:
        lines = [line for line in fo]
    x_shape = (batch_size,) + img_size
    y_shape = (batch_size, num_classes)

    mean_dir = os.path.join(spatial_dir, 'mean.npy')
    if os.path.exists(mean_dir) and mean_sub:
        mean = np.load(mean_dir)
        mean.astype(dtype='float16')
    elif mean_sub:
        raise FileExistsError('RGB mean file does not exist')
    else:
        mean = None
    while True:
        spatial_x = np.zeros(x_shape)
        temporal_x = np.zeros(x_shape)
        two_stream_x = [spatial_x, temporal_x]
        two_stream_y = np.zeros(y_shape)
        for i in range(batch_size):
            line = random.choice(lines)
            clip_name, clip_index = line.split()
            clip_name = os.path.basename(clip_name)
            clip_name = clip_name[:clip_name.find('.')]
            for j, flow_dir in enumerate([spatial_dir, temporal_dir]):
                clip_dir = os.path.join(flow_dir, clip_name)
                frames = os.listdir(clip_dir)
                frame = random.choice(frames)
                frame_dir = os.path.join(clip_dir, frame)
                frame = scipy.misc.imread(frame_dir)
                if random_crop:
                    x = random.randrange(frame.shape[0] - img_size[0])
                    y = random.randrange(frame.shape[1] - img_size[1])
                    frame = frame[x:x + img_size[0], y:y + img_size[1], :]
                else:
                    frame = scipy.misc.imresize(frame, img_size)
                if horizontal_flip and random.randrange(2) == 1:
                    frame = frame[:, ::-1, :]
                frame = frame.astype(dtype='float16')
                if mean is not None and flow_dir == spatial_dir:
                    frame -= mean
                if normalization:
                    frame /= 255

                two_stream_x[j][i] = frame
                two_stream_y[i, int(clip_index) - 1] = 1

        yield two_stream_x, two_stream_y


def two_stream18_generator(list_dir, spatial_dir, temporal_dir, batch_size, spatial_shape, temporal_shape, num_classes):
    '''
    
    :param list_dir: '.../testlist.txt'
    :param spatial_dir: '.../test/'
    :param temporal_dir: '.../test/'
    :param batch_size: 
    :param input_shape: 
    :param num_classes: 
    :return: 
    '''
    with open(list_dir) as fo:
        test_list = [line for line in fo]

    spatial_x_shape = (batch_size,) + spatial_shape
    temporal_x_shape = (batch_size,) + temporal_shape
    y_shape = (batch_size, num_classes)

    while True:
        spatial_x = np.zeros(spatial_x_shape)
        temporal_x = np.zeros(temporal_x_shape)
        two_stream_x = [spatial_x, temporal_x]
        two_stream_y = np.zeros(y_shape)
        for i in range(batch_size):
            clip_name, clip_index = random.choice(test_list).split()
            clip_name = clip_name[:clip_name.find('.')] + '.npy'
            clip_spatial_dir = os.path.join(spatial_dir, clip_name)
            clip_temporal_dir = os.path.join(temporal_dir, clip_name)
            # read spatial data
            seq_data = np.load(clip_spatial_dir)
            spatial_x[i] = seq_data[random.randrange(seq_data.shape[0])]
            # read temporal data
            temporal_x[i] = np.load(clip_temporal_dir)

            two_stream_y[i][int(clip_index) - 1] = 1
        yield two_stream_x, two_stream_y


def get_data_list(list_dir, video_dir):
    '''
    Input parameters:
    list_dir: 'root_dir/data/ucfTrainTestlist'
    video_dir: directory that stores source train and test data

    Return value:
    test_data/train_data: list of tuples (clip_dir, class index)
    class_index: dictionary of mapping (class_name->class_index)
    '''
    train_dir = os.path.join(video_dir, 'train')
    test_dir = os.path.join(video_dir, 'test')
    testlisttxt = 'testlist.txt'
    trainlisttxt = 'trainlist.txt'

    testlist = []
    txt_path = os.path.join(list_dir, testlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            testlist.append(line[:line.rfind(' ')])

    trainlist = []
    txt_path = os.path.join(list_dir, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    class_index = dict()
    class_dir = os.path.join(list_dir, 'classInd.txt')
    with open(class_dir) as fo:
        for line in fo:
            class_number, class_name = line.split()
            class_number = int(class_number)
            class_index[class_name] = class_number

    train_data = []
    for i, clip in enumerate(trainlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(train_dir, clip)
        train_data.append((dst_dir, class_index[clip_class]))

    test_data = []
    for i, clip in enumerate(testlist):
        clip_class = os.path.dirname(clip)
        dst_dir = os.path.join(test_dir, clip)
        test_data.append((dst_dir, class_index[clip_class]))

    return train_data, test_data, class_index


if __name__ == '__main__':
    image_size = (216, 216, 3)

    data_dir = '/home/changan/ActionRecognition/data'
    list_dir = os.path.join(data_dir, 'ucfTrainTestlist')

    test_list = os.path.join(list_dir, 'testlist.txt')
    frames_dir = '/home/changan/ActionRecognition/data/UCF-Preprocessed-OF/test'
    flow_dir = '/home/changan/ActionRecognition/data/OF_data/test'
    BatchSize = 32
    N_CLASSES = 101
    generator = two_stream18_generator(test_list, frames_dir, flow_dir, BatchSize,
                                       (216, 216, 3), (216, 216, 18), N_CLASSES)

    for i in range(10):
        x, y = next(generator)
