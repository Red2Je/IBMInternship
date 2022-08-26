
import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt

def reduce(array4):
            #A function to make return an array containing the mean of a
            #2*2 array
            return np.array([np.mean(array4)])

def import_images(desiredoutputsize, global_path = 'MNIST/Reduced MNIST Data/Reduced Training data',imtype = "Training",numrange = 10, howmuch = 10, imageformat = 'jpg'):
    output = []
    output_label = []
    if imtype != "Training" and imtype != "Testing":
        raise Exception("Type of set not recognized, please choose between \"Training\" and \"Testing\"")
    for i in range(numrange):
        training_paths = pathlib.Path(global_path).glob(str(i)+'/*.'+imageformat)
        impath = list(training_paths)
        for path in impath[:howmuch]:
            read = imageio.imread(str(path))
            reduced = np.zeros((int(read.shape[0]/4),int(read.shape[1]/4)))
            for a in range(0,read.shape[0], 4):
                for o in range(0,read.shape[1],4):
                    reduced[int(a/4),int(o/4)] = reduce(read[a:a+4,o:o+4])
            reduced = reduced.reshape(reduced.shape[0]*reduced.shape[1],).tolist()
            #fill the end of the reduced list to match the amount of feature a quantum feature map has.
            #For example an EfficientSU2 feature map with 5 qubits and 4 repetitions can have 50 features encoded and an image of size 7*7 pixels has 49 features
            #The idea is then to fill the last value with another existing value.
            if len(reduced) < desiredoutputsize:
                for r in range(desiredoutputsize-len(reduced)):
                    reduced.append(reduced[r])
            reduced.append(reduced[0])
            output.append(reduced)
            output_label.append(i)
    return output,output_label

    
def import_singular_image(number_in_image,image_index = None, start_at = 0,global_path = 'MNIST/Reduced MNIST Data/Reduced Training data', imtype = "Testing", imageformat = 'jpg'):
    import random
    if imtype != "Training" and imtype != "Testing":
        raise Exception("Type of set not recognized, please choose between \"Training\" and \"Testing\"")
    path_list = list(pathlib.Path(global_path).glob(str(number_in_image)+'/*.'+imageformat))
    if image_index is not None : 
        read = imageio.imread(path_list[image_index])
    else:
        read = imageio.imread(random.choice(path_list[start_at:]))
    reduced = np.zeros((int(read.shape[0]/4),int(read.shape[1]/4)))
    for a in range(0,read.shape[0], 4):
        for o in range(0,read.shape[1],4):
            reduced[int(a/4),int(o/4)] = reduce(read[a:a+4,o:o+4])
    reduced = reduced.reshape(reduced.shape[0]*reduced.shape[1],).tolist()
    return reduced


def import_full_image(global_path = 'MNIST/Reduced MNIST Data/Reduced Training data',imtype = "Training",numrange = 10, howmuch = 10, imageformat = 'jpg'):
    output = []
    output_label = []
    if imtype != "Training" and imtype != "Testing":
        raise Exception("Type of set not recognized, please choose between \"Training\" and \"Testing\"")
    for i in range(numrange):
        training_paths = pathlib.Path(global_path).glob(str(i)+'/*.'+imageformat)
        impath = list(training_paths)
        for path in impath[:howmuch]:
            read = imageio.imread(str(path))
            read = np.array(read)
            output.append(read.reshape(read.shape[0]*read.shape[1],))
            output_label.append(i)
    return(output,output_label)

