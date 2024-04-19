import numpy as np
import dbow
import cv2
import os
from tqdm import tqdm

PRELOAD_IMAGE_PATH = './data/Images/'
class DBoW:
    def __init__(self):
        pass

if __name__ == '__main__':
    depth = 2
    clusters = 5
    orb = cv2.ORB_create()
    
    
    #develop vocabualry
    print("Loading Images")
    images = []
    for image_path in tqdm(os.listdir(PRELOAD_IMAGE_PATH)[:100]):
        images.append(cv2.imread(os.path.join(PRELOAD_IMAGE_PATH + image_path)))

    print("Building Vocabulary")
    vocab = dbow.Vocabulary(images, depth, clusters)
    
    print("Building Bag of Words")
    bag_of_words = []
    for image in tqdm(images):
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        bag_of_words.append(vocab.descs_to_bow(descs))
        
    print("Building Database")
    # build a database
    db = dbow.Database(vocab)
    for image in tqdm(images):
        kps, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        db.add(descs)
    
    print("Querying Database")
    # query database
    for image in tqdm(images):
        kp, descs = orb.detectAndCompute(image, None)
        descs = [dbow.ORB.from_cv_descriptor(desc) for desc in descs]
        scores = db.query(descs)
        match_bow = db[np.argmax(scores)]
        match_descs = db.descriptors[np.argmax(scores)]