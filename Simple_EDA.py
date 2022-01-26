# Import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import model as embedding
import model
from PIL import Image
import torchvision.transforms.functional as TF
import csv
import pandas as pd
import matplotlib.pyplot as plt
import torch

# load face detection model
protoPath = "./model_paths/deploy.prototxt.txt"
modelPath = "./model_paths/res10_300x300_ssd_iter_140000.caffemodel"

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load embedding model
embedder = embedding.InceptionResnetV1(pretrained='vggface2').eval()

currentDir = os.getcwd()

dataset = os.path.join(currentDir, "dataset")

datasetCrop = os.path.join(currentDir, "datasetCrop")

embeddingPickle = os.path.join(currentDir, "output/embeddings2.pickle")

# getting all images paths
imagePaths = list(paths.list_images(dataset))

#loop over the image paths
ImgPaths = []
names = []
imageIDs = []
boxs = []
embeddings = []

# initlize the total number of faces processed
total = 0

for (i, imagePath) in enumerate(imagePaths):
    
    #extract the person name from the image path
    
    name = imagePath.split(os.path.sep)[-2]
    imageID = imagePath.split(os.path.sep)[-1].split('.')[-2]
    
    image = cv2.imread(imagePath)
    (h,w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    detector.setInput(blob)
    detections = detector.forward()
    
    if len(detections) > 0:
        
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probalility
        
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        
        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        
        if confidence > 0.5:
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            
            (startX, startY, endX, endY) = box.astype("int")
            
            face = image[startY:endY , startX:endX]
            (fH , fW) = face.shape[:2]
            
            
            # ensure the facce width and height are sufficently large
            if fW < 20 or fH < 20:
                continue
                
            try:
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,(160, 160), (0, 0, 0), swapRB=True, crop=False)
            except:
                print("[Error] - Face size in Image not sufficent to get Embeddings : ", imagePath)
                continue
            
            faceTensor = torch.tensor(faceBlob)
            faceEmbed = embedder(faceTensor)
            flattenEmbed = faceEmbed.squeeze(0).detach().numpy()
            
            ImgPaths.append(imagePath)
            imageIDs.append(imageID)
            names.append(name)
            boxs.append(box)
            embeddings.append(flattenEmbed)
            total += 1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings ....".format(total))
data = {"paths":ImgPaths, "names":names, "imageIDs":imageIDs, "boxs":boxs, "embeddings":embeddings}
f = open(embeddingPickle , "wb")
f.write(pickle.dumps(data))
f.close()

import pandas as pd


picklefile = pd.read_pickle(embeddingPickle)
df = pd.DataFrame(picklefile)
df.head()


from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import seaborn as sns

X = np.array(picklefile["embeddings"])
Y = picklefile["names"]
print(X.shape)




from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

Embeddings = np.array(picklefile["embeddings"])
Labels = np.array(picklefile["names"])

# labels
le = LabelEncoder()
labels = le.fit_transform(Labels)

X_train, X_test, y_train, y_test = train_test_split(Embeddings, labels, test_size=0.4)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

knn = KNeighborsClassifier(n_neighbors=1 , metric='euclidean', weights="distance")
svc = LinearSVC()

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
y_pred_svc = svc.predict(X_test)
acc_svc = accuracy_score(y_test, y_pred_svc)

print(f'KNN accuracy = {acc_knn * 100}, SVM accuracy = {acc_svc * 100}')

Embeddings = np.array(picklefile["embeddings"])
Labels = labels

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(10,5))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));    

from sklearn.metrics import f1_score, accuracy_score

distances = [] # L2 distance
identical = [] #1 if same identity, 0 otherwise

num = len(Embeddings)

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(Embeddings[i], Embeddings[j]))
        identical.append(1 if labels[i] == labels[j] else 0)
        
distances = np.array(distances)
identical = np.array(identical)

minDist = min(distances)
maxDist = max(distances)

thresholds = np.arange(minDist, maxDist, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
#Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal accurecy
opt_acc = accuracy_score(identical, distances < opt_tau)

print("##### Best threshold = ", opt_tau)
print("##### Best F1-Score = ", f1_scores[opt_idx])
print("##### Best Accurecy = ", opt_acc)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend();
plt.savefig('save_as_a_png3.png')

dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend();

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend();