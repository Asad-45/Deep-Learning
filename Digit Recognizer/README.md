# <center>Digit Recognizer</center> 
Author: Asad Choudhary
***
## INTRODUCTION
_Purpose:_ To built a CNN Model for identifying digits present inside images for 'test.csv' file and saving the result in form of 'submission.csv'. 

_Data:_

- 'pixel0' to 'pixel783' columns inside train.csv and test.csv are the numbers of pixel of an Image, which has been stock into pandas.Dataframe as 1D vectors of 784 values.
- 'label' column is only present in train.csv, which refer to the digit present inside the corresponding image.

This pixels columns provide all the necessary information related to the image containing digit in it.Using this pixels columns data and label column in train.csv file, we will be building a CNN  Model for recognizing the digits present inside the images.

## PROJECT OUTCOME

![outcome](https://user-images.githubusercontent.com/62840804/108059715-eaf2c880-707b-11eb-86ff-77b3efb79994.png)

The final Outcome of this project is 'submission.csv' file which has the predict label columns generated from the pixel columns present in 'test.csv' file .

## PROJECT FLOW

![](https://user-images.githubusercontent.com/62840804/108062560-fba53d80-707f-11eb-97f3-6621903c39a8.png)


## HOW TO RUN THE  CODE
_This project runs in Jupyter Environment._

1. Ensure that all the necessary packages for this projects have been installed.
2. Download the .ipynb file and store in the Jupyter working directory.
3. Also download the train.csv and test.csv for reference link metion below.
4. The  program can be executed by running each and every cell inside Jupyter notebook or it can be run compeletely in one go by use the _run all_ button in the notebook.

## REFERENCE

The data are retrieved in the form of 'train.csv and 'test.csv' file format from the kaggle digit recognizer [Link](https://www.kaggle.com/c/digit-recognizer)

## REQUIREMENTS
The Python packages/libraries utilized for this project are given below:
- Importing Packages/libraries
``` Python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
%matplotlib inline
import itertools
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
```

## EXPLANATION OF THE CODE
**Data Preparation**
- Load Data

_Creating a Dataframe and Checking the shape of the Dataframe_
```Python
# Formulating the data file into pandas dataframe
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Checking the shape of the dataframe 
train_df.shape, test_df.shape
```

- _Creating 'y' variable for 'label' column  inside train_df_
``` Python
# Creating 'y' for label columns 
y = train_df['label'] 

# Show data as plot
plt.figure(figsize = (10,5))
sns.countplot(y)
plt.title('Frequency of classes in the data', fontsize = 14)
plt.xlabel('Classes', fontsize = 12)
plt.ylabel('Classes Frequency', fontsize = 12)
plt.show()
```
![Frequency data](https://user-images.githubusercontent.com/62840804/108059693-e4fce780-707b-11eb-8143-b4d6dd949c0d.png)
As shown in the figure majority of the digits are above 4000 counts.

- Check for Missing Value

_Checking missing values in train_df and test_df as follows_
``` Python
# for train_df
train_df.isnull().any().describe()
# for test_df
test_df.isnull().any().describe()
```

- Normalization

_Since pixels ranges from 0 to 255 and CNN converg faster on [0..1] data than on [0..255] , therefore noramlization is done._
``` Python
# normalizing the data
train_df = train_df.drop(['label'], axis=1)
train_df = train_df/255.0
test_df = test_df/255.0
```

- Reshaping

_train_df and test_df have image data in the form 1 dimension we have to reshape into  3 dimensions form i.e (h=28px , w=28px ,chanal=1)_
``` Python
train_df = train_df.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)
```

- Label Encoding

_'y' variable has the labels form 0 to 9 for 10 digits numbers. We need to encode these lables to one hot vectors  (example: 2> [0,0,1,0,0,0,0,0,0,0])._
``` Python
# Encode labels to one hot vectors 
y = to_categorical(y, num_classes = 10)
```

- Splitting into training and validation sets

_Split the train and the validation set for the fitting the CNN model and Checking the shape of split_
``` Python
# Split the train and the validation set for the fitting
X_train, X_val, y_train, y_val = train_test_split(train_df, y, test_size = 0.1, random_state=2)

# show shape of split
X_train.shape, y_train.shape, X_val.shape, y_val.shape
```
     
_Checking the image and its label after split_
``` Python
# checking image data and its label
plt.imshow(X_train[1][:,:,0])
plt.title(y_train[1].argmax());
```
![Image and it's label](https://user-images.githubusercontent.com/62840804/108059711-ea5a3200-707b-11eb-997e-3d6600375057.png)

The image above is perfectly match its digit from its label.

**Building a CNN Model**

- Define the Model

_Creating a CNN Model and checking it's summary as follows_
``` Python
# Creating the CNN model 
model = Sequential()

# Layer1
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Layer2
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

# Fully-Connected Layer
model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

# Check Summary
model.summary()
```

Model architechture is: _[[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out_

- Set Optimizer and Compile the Model

_Defining an optimizer and compiling the Model_
``` Python
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```

- Fit the Model
_Fitting the Model by using train and val data , and setting epochs size =10_
``` Python
model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_val, y_val))
```

**Evaluating the Model**

- Accuracy and Loss of Model

_Checking the accuracy and loss of CNN model for recognizing the digits_
``` Python
# Printing Accuracy and Loss
model_loss, Model_acc = model.evaluate(X_val, y_val)
model_loss, Model_acc
```

_Creating fuction for ploting Confusion Matrix_
``` Python
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
```

_Predicting 'X_val' datasets result into y_pred to match with y_val and ploting confusion matrix for it_
``` Python
# Predict the values from the validation dataset
y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 
y_pred = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 

# plot the confusion matrix
df_matrix = confusion_matrix(y_true, y_pred) 
plot_confusion_matrix(cf_matrix, classes = range(10)) 
```
![confusion matrix](https://user-images.githubusercontent.com/62840804/108059502-a36c3c80-707b-11eb-9c87-8c21b4289de1.png)

Above confusion matrix gave idea about how true and predict label are matched.

**Prediction and Submission**

- Predicting and Submitting the Result

_Creating 'result' variable for model to store the predicting for test_df and creating 'submission.csv' for submitting the result as follows_
``` Python
# predict results
results = model.predict(test_df)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)  
submission.to_csv("submission.csv",index=False)
```

- Saving the Model

_Creating 'digit_recognizer_model.h5' file for saving the CNN Model_
``` Pyhton
model.save('digit_recognizer_model.h5')
```

- Check submission file

_Creating submission_df to visualied the top 5 rows of dataframe_
``` Python
# formulating the submission.csv into dataframe
submission_df = pd.read_csv('submission.csv')
submission_df.head()
```

## RESULTS

Based on Identifying the digits inside the images our CNN model perform with the Accuracy of 99.38% and loss is estimated to 2.76%. Also, identifying digits for the test.csv is done and saved in the form of submission.csv.

## FUTURE SCOPE

This project focuses on recognizing the digit that are present inside images only. So, for any future recognizing problem this Model can be used or viewed as reference for future programming purpose.
