"""
Trine ML Model to Classify / Identify the person using extracted face embeddings
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


currentDir = os.getcwd()

# paths to embedding pickle file
#embeddingPickle = os.path.join(currentDir, "output/FinalEmbeddings.pickle")
embeddingPickle = os.path.join(currentDir, "output/FinalEmbeddingsStandard.pickle")

# path to save recognizer pickle file
#recognizerPickle = os.path.join(currentDir, "output/FinalRecognizer.pickle")
recognizerPickle = os.path.join(currentDir, "output/FinalRecognizerStandard.pickle")

# path to save labels pickle file
#labelPickle = os.path.join(currentDir, "output/FinalLabel.pickle")
labelPickle = os.path.join(currentDir, "output/FinalLabelStandard.pickle")

# loading embeddings pickle
data = pickle.loads(open(embeddingPickle, "rb").read())
data_table = pd.DataFrame(data)
#data_table.to_csv("dataset.csv")
data_table.to_csv("datasetStandard.csv")
#data_table.drop(columns = ["paths" , "names"] , inplace = True)
data_table.drop(columns = ["paths" , "imageIDs"] , inplace = True)
#data_table.to_csv("datatable.csv")
data_table.to_csv("datatableStandard.csv")


# encoding labels by names
label = LabelEncoder()
labels = label.fit_transform(data["names"])

# getting embeddings
Embeddings = np.array(data["embeddings"])



Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(Embeddings, labels, test_size=0.40, random_state=42)

print("Total number of embeddings : ", Embeddings.shape)
print("Total number of labels :", len(labels))


############ If you want to train SVM Classifier uncomment below code #########

## Hyper-parameter tuning
X_train, X_test, y_train, y_test = train_test_split(
    Embeddings, labels, test_size=0.40, random_state=42)

# using SVC Classifier to classify the facess

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    accsvc = accuracy_score(y_true, y_pred)
    print(classification_report(y_true, y_pred))
    print()

    # train a model on best parameters set

    # train the model used to accept the 512-d embeddings of the face and
# then produce the actual face recognition

#recognizer = KNeighborsClassifier(n_neighbors= 5, metric='euclidean', weights="distance")
bestParameters = clf.best_params_
bestParameters["probability"] = True

#recognizerSVC = SVC(**bestParameters)


recognizer = KNeighborsClassifier(n_neighbors=5)
recognizer.fit(Xknn_train, yknn_train)
prediction = recognizer.predict(Xknn_test)

accknn = accuracy_score(yknn_test, prediction)





# write the actual face recognition model to disk
f = open(recognizerPickle, "wb")
f.write(pickle.dumps(recognizer))
f.close()



# write the label encoder to disk
f = open(labelPickle,"wb")
f.write(pickle.dumps(label))
f.close()


#print("\nAccuracy: ", acc)
print("\nAccuracy for KNN: ", accknn)
print("\nAccuracy for SVC: ", accsvc)
print("[Info] : Models are saved successfully.")

