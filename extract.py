import cv2
import numpy as np
import _pickle as pickle
import scipy
from scipy.misc import imread
import os
from annoy import AnnoyIndex
f = 40
t = AnnoyIndex(f, 'angular')
# Feature extractor
def extract_features(image_path, vector_size=32):
    #read image for the db
    image = imread(image_path,mode="RGB")
    try:
        alg = cv2.KAZE_create(threshold=0.0001)
        # finding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (vector_size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None

    return dsc

# extract feature from multiple images store in in an object and serialize
def batch_extractor(images_path, pickled_db_path="features_msrc.pickle"):
    # get image
    files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]

    result = {}
    print('extracting...')
    for f in files:
        # print('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        print(f"Extracting features from {name}")
        # dec of image and extracted image
        result[name] = extract_features(f)
    print(f"final data {result}")
    # saving all our feature vectors in pickled file




    with open(pickled_db_path, 'wb') as fp:
        pickle.dump(result, fp)

    # saving all our feature vector in file system
    with open('data.txt', 'w') as f:
        f.write(str(result))
        f.close()

    # Load saved features
    with open(pickled_db_path, 'rb') as fp:
        result = pickle.load(fp)
        for key, value in result:
            t.add_item(key, value)
        t.build(100)
        t.save('test.ann')
        u = AnnoyIndex(f, 'angular')
        u.load('test.ann')  # super fast, will just mmap the file
        print("p")
        print(u.get_nns_by_item(0, 1000))
        print("q")
    #
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        # print('Extracted image data')
        # print(data)
        # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')