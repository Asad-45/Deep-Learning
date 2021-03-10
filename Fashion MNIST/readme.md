
# <center>Fashion MNIST</center> 
Author: Asad Choudhary
***
## INTRODUCTION
_Purpose:_ To built a CNN Model using keras, for fashion mnist dataset.

_Data:_

- 'pixel0' to 'pixel783' columns inside train.csv and test.csv are the numbers of pixel of an Image, which has been stock into pandas.Dataframe as 1D vectors of 784 values.
- 'label' column 
  Each training and test example is assigned to one of the following labels:

    | Label  |  Categories |
    |---|---|
    | 0 | T-shirt/top |
    | 1 | Trouser |
    | 2 | Pullover|
    | 3 | Dress |
    | 4 | Coat |
    | 5 | Sandal |
    | 6 | Shirt |
    | 7 | Sneaker |
    | 8 | Bag |
    |9 | Ankle boot |

This pixels columns provide all the necessary information related to the image .Using this pixels columns data and label column, we will be building a CNN  Model for fashion mnist dataset.

## PROJECT OUTCOME

The final Outcome of this project is a CNN Model build by using keras, for fashion mnist dataset which is saved as 'fashion_mnist_model.h5'.

## PROJECT FLOW

![Flow-chart](https://user-images.githubusercontent.com/62840804/110596348-3133d700-81a5-11eb-885b-e07155047768.png)


## HOW TO RUN THE  CODE
_This project runs in Jupyter Environment._

1. Ensure that all the necessary packages for this projects have been installed.
2. Download the .ipynb file and store in the Jupyter working directory.
3. Also download the train.csv and test.csv for reference link metion below.
4. The  program can be executed by running each and every cell inside Jupyter notebook or it can be run compeletely in one go by use the _run all_ button in the notebook.

## REFERENCE

The data are retrieved in the form of 'train.csv and 'test.csv' file format from the kaggle Fashion MNIST [Link](https://www.kaggle.com/zalando-research/fashionmnist)

## REQUIREMENTS
The Python packages/libraries utilized for this project are given below:
- Importing Packages/libraries
``` Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
from sklearn.metrics import confusion_matrix, classification_report
import itertools
%load_ext tensorboard
from keras import callbacks
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator 
from keras.callbacks import ReduceLROnPlateau 
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
import keras
```

## EXPLANATION OF THE CODE
**Data Pre-processing**
- Load Data

_Creating a Dataframe and Checking the shape of the Dataframe_
```Python
# Formulating the data file into pandas dataframe
train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')

```

- Check for Shape and Missing Values

_Checking for missing values in the dataframes_
``` Python
print(f"rows: {train_df.shape[0]}\ncolumns: {train_df.shape[1]}")
print(f"missing values: {train_df.isnull().any().sum()}")

print(f"rows: {test_df.shape[0]}\ncolumns: {test_df.shape[1]}")
print(f"missing values: {test_df.isnull().any().sum()}")
```

Visualization of label column from both the dataframes

_Creating a classes dictionary for mapping the labels_
``` Python
# creating classes variable for maping the label with respect to the naming conventions
classes = {0 : 'T-shirt/top',
           1 : 'Trouser',
           2 : 'Pullover',
           3 : 'Dress',
           4 : 'Coat',
           5 : 'Sandal',
           6 : 'Shirt',
           7 : 'Sneaker',
           8 : 'Bag',
           9 : 'Ankle boot'}
```

_Displaying the labels from train_df_
``` Python
# Creating  'label_names' column in train_df
train_df['label_names'] = train_df['label'].map(classes)

# Visualizing the 'label_names' with respect to their counts
plt.figure(figsize=(17,6))
sns.countplot(x = 'label_names', data = train_df)
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Training set images wrt classes')
```

![train label](https://user-images.githubusercontent.com/62840804/110596321-27aa6f00-81a5-11eb-98ea-e3627d845283.png)

_Displaying the labels from test_df_
``` Python
# Creating  'label_names' column in test_df
test_df['label_names'] = test_df['label'].map(classes)

# Visualizing the 'label_names' with respect to their counts
plt.figure(figsize=(17,6))
sns.countplot(x = 'label_names', data = test_df)
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Test set images wrt classes')
```
![test label](https://user-images.githubusercontent.com/62840804/110596344-2f6a1380-81a5-11eb-9572-cc087535235e.png)

- Train Test Split

_Creating X_train and y_train for training and X_test and y_test for testing the Model_
``` Python
X_train = train_df.drop(['label', 'label_names'], axis = 1)
X_test = test_df.drop(['label', 'label_names'], axis = 1)
y_train = train_df['label']
y_test = test_df['label']
```

- Normalization

_Since pixels ranges from 0 to 255 and CNN converg faster on [0..1] data than on [0..255] , therefore noramlization is done._
``` Python
# normalizing the data
X_train = X_train/255.0
X_test = X_test/255.0
```

- Reshaping

_train_df and test_df have image data in the form 1 dimension we have to reshape into  3 dimensions form i.e (h=28px , w=28px ,chanal=1)_
``` Python
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
```

- Label Encoding

_'y_trian'and 'y_test' variables has the labels form 0 to 9 for 10 fashion classes. We need to encode these lables to one hot vectors  (example: 2 ('Pullover') > [0,0,1,0,0,0,0,0,0,0])._
``` Python
# Encode labels to one hot vectors 
y_train = to_categorical(y_train, num_classes=10)
y_test  = to_categorical(y_test, num_classes=10)
```


_Displaying the Shapes for X_train, y_train, X_test and y_test_
``` Python
print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
```
     
_Checking the image and its label after split_
``` Python
# Plotting images with respect to their labels
plt.figure(figsize=(20,5))
for i in range(5):
    val=''
    for j in range(10):
        if y_train[i][j]==1:
            val = classes[j]
            
    plt.subplot(int(f"15{i+1}")), plt.imshow(X_train[i][:,:,0], cmap='gray'), plt.axis('off'), plt.title(val, size=14)
```
![image and its label](https://user-images.githubusercontent.com/62840804/110596726-a8696b00-81a5-11eb-91c5-6bb779eb6007.png)

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
# Learning-rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001
                                            )
```
- Data Augmentation

_Create datagen for Data Augmentation which helps to avoid overfitting problem and it makes existing dataset even larger_
``` Python
datagen = ImageDataGenerator(featurewise_center=False,  
                             samplewise_center=False,  
                             featurewise_std_normalization=False,  
                             samplewise_std_normalization=False, 
                             zca_whitening=False, 
                             rotation_range=5,  
                             zoom_range =0.1, 
                             width_shift_range=0.1, 
                             height_shift_range=0.1,  
                             horizontal_flip=False,  
                             vertical_flip=False  
)

datagen.fit(X_train)
```
    For the data augmentation, parameters are:
        - Randomly rotate some training images by 10 degrees
        - Randomly  Zoom by 10% some training images
        - Randomly shift images horizontally by 10% of the width
        - Randomly shift images vertically by 10% of the height

- Fit the Model

_Fitting the Model by using train and val data , and setting epochs size =10_
``` Python
model.fit(datagen.flow(X_train,y_train,batch_size=100), callbacks=[learning_rate_reduction], epochs=10,
                    validation_data=(X_test, y_test))
```
_Checking Accuracy and Loss_
``` Python
model_loss, model_acc = model.evaluate(X_test, y_test)
model_loss, model_acc
```
- Hyper-tuning the Model

_Creating function name 'build_model' for finding the optimal model_
``` Python
def build_model(hp):  
  model = keras.Sequential()
  model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), kernel_size=(5,5),padding='Same', 
                 activation ='relu', input_shape = (28,28,1)))
  model.add(MaxPool2D(pool_size=(2,2)))

  for i in range(hp.Int('layers', 1, 5)):
    model.add(Conv2D(hp.Int('cnn{name}'.format(name=i), min_value=32, max_value=256, step=32), kernel_size=(5,5),padding='Same', 
                 activation ='relu'))
   
  model.add(Flatten())
  model.add(Dense(256, activation = "relu"))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = "softmax")) 
  model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4])),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
  return model
```

_Creating 'tuner' variable for randomly searching Models_
``` Python
tuner = RandomSearch(build_model,
                    objective='val_accuracy',
                    max_trials=10,
                    executions_per_trial=2,
                    directory='output',
                    project_name="Mnist Fashion"
)
```
    The tuner variable it Randomly search for 10 models and store those 10 models in output directory inside 'Mnist Fashion'.

_Searching for the optimal model_
``` Python
tuner.search(datagen.flow(X_train, y_train, batch_size=100),
             epochs=3,
             validation_data=(X_test, y_test)
             )
```

_Creating 'opt_model' variable and assigning to the best model from tuner_
``` Python
opt_model=tuner.get_best_models(num_models=1)[0]
opt_model.summary()
```

_Creating tensorboard callbacks for visualization_
``` Python
# Creating path for tensorboard 
path = 'logs/fit'
tensorboard_callback = callbacks.TensorBoard(log_dir=path, histogram_freq=1)
```

_Fitting the 'opt_model' with 15 epoch_
``` Python
opt_model.fit(datagen.flow(X_train,y_train,batch_size=100), epochs=15, validation_data=(X_test, y_test), initial_epoch=3, callbacks=[tensorboard_callback])
```

_Visualization of TensorBoard_
``` Python
%tensorboard --logdir logs/fit
```

![Screenshot 2021-02-24 124637](https://user-images.githubusercontent.com/62840804/110596732-aa332e80-81a5-11eb-9a38-2b162c1ed601.png)

**Evaluating the Model**

- Accuracy and Loss of Model

_Checking the accuracy and loss of opt_model_
``` Python
# Printing Accuracy and Loss
model_loss, model_acc = opt_model.evaluate(X_test, y_test)
model_loss, model_acc
```

_Creating a 'y_pred' variable for storing the predicted labels_
``` Python
y_pred = opt_model.predict(X_test)
y_pred = np.argmax(y_pred, axis = 1) 
y_test = np.argmax(y_test, axis = 1)
```

_Visualization of predicted and actual label with respect to the images_
``` Python
fig, axis = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axis.flat):
    pred_val=y_pred[i].argmax()
            
    ax.imshow(X_test[i][:,:,0], cmap='gray'), ax.axis('off')
    ax.set(title = f"Real Class: {classes[y_test[i].argmax()]}\nPredict Class: {classes[pred_val]}")
```

![actual and predicted](https://user-images.githubusercontent.com/62840804/110596631-87087f00-81a5-11eb-9eac-2e1f3e96d2c6.png)

- Classification Report

_It helps us for better understanding of individual labels performance and accuracy _
``` Python
cr = classification_report(y_test, y_pred)
print(cr)
```

![classification report](https://user-images.githubusercontent.com/62840804/110596716-a43d4d80-81a5-11eb-8453-a8876c200a7e.png)

- Confusion Matrix

_Creating fuction for ploting Confusion Matrix and Visualizing Confusion Matrix_
``` Python
ddef plot_confusion_matrix(cm, classes,
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


# compute the confusion matrix
confusion_mtx = confusion_matrix(y_test, y_pred) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))
```

![confusion matrix](https://user-images.githubusercontent.com/62840804/110596711-a2738a00-81a5-11eb-8cf9-884d6aff596a.png)

Above confusion matrix gave idea about how true and predict label are matched.

- Error Visualization

_Displaying the errors with respect to its actual labels for better understanding of the model prediction performance_
``` Python
errors = (y_pred - y_test != 0)

y_pred_errors = y_pred[errors]
y_test_errors = y_test[errors]
X_test_errors = X_test[errors]

fig, axis = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axis.flat):
    pred_val=y_pred_errors[i]
    ax.imshow(X_test_errors[i][:,:,0], cmap='gray'), ax.axis('off')
    ax.set(title = f"Real Class: {classes[y_test_errors[i].argmax()]}\nPredict Class: {classes[pred_val]}")
```

![error](https://user-images.githubusercontent.com/62840804/110596353-32650400-81a5-11eb-8eec-af0bee3af9bb.png)

**Saving Model**

- Saving the Model

_Creating 'fashion_mnist_model.h5' file for saving the Optimized CNN Model_
``` Pyhton
opt_model.save('fashion_mnist_model.h5')
```


## RESULTS

Based on fashion recognition the model without optimizaton has the evaluation accuracy around 91%, for optimized CNN model it has the accuracy about 92%.a

## FUTURE SCOPE

This project focuses on recognizing the fashion that are present inside images only. So, for any future recognizing problem the Saved Model can be used.
