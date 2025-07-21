import os
import numpy as np
import cv2
import scipy.io

def load_imdb_dataset(mat_path, images_path, max_images=10000):
    """
    Loads age and image data from the IMDB-WIKI .mat file
    """
    print("[INFO] Loading metadata from .mat...")
    data = scipy.io.loadmat(mat_path)
    wiki = data['imdb'][0][0]

    full_path = wiki['full_path'][0]
    dob = wiki['dob'][0]
    photo_taken = wiki['photo_taken'][0]

    X, y = [], []

    for i in range(min(max_images, len(full_path))):
        age = photo_taken[i] - (dob[i] // 365.25)
        if 0 <= age <= 100:
            img_path = os.path.join(images_path, full_path[i][0])
            if os.path.exists(img_path):
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (224, 224))
                    X.append(img)
                    y.append(age)
                except:
                    continue

    X = np.array(X, dtype="float32") / 255.0
    y = np.array(y, dtype="float32")
    return X, y
