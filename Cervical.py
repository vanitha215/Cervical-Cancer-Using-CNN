#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as ts


# In[6]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[7]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = r"C:\Users\Vanitha\Desktop\Dataset\train"
valid_path = r"C:\Users\Vanitha\Desktop\Dataset\test"


# In[8]:


import tensorflow
resnet152V2 =tensorflow.keras.applications.ResNet152V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[9]:


for layer in resnet152V2.layers:
    layer.trainable = False


# In[10]:


folders = glob(r"C:\Users\Vanitha\Desktop\Dataset\train\*")


# In[11]:


# our layers - you can add more if you want
x = Flatten()(resnet152V2.output)


# In[12]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet152V2.input, outputs=prediction)


# In[13]:


model.summary()


# In[14]:


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[15]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)


# In[16]:


training_set = train_datagen.flow_from_directory(r"C:\Users\Vanitha\Desktop\Dataset\train",target_size = (224, 224),batch_size = 32,class_mode = 'categorical')


# In[17]:


test_set = test_datagen.flow_from_directory(r"C:\Users\Vanitha\Desktop\Dataset\test",target_size = (224, 224),batch_size = 32,class_mode = 'categorical')


# In[18]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit(training_set,validation_data=test_set,epochs=30,steps_per_epoch=len(training_set),validation_steps=len(test_set))


# In[19]:


import matplotlib.pyplot as plt


# In[21]:


# save it as a h5 file


from tensorflow.keras.models import load_model

model.save('model_resnet152V2.h5')


# In[22]:


y_pred = model.predict(test_set)


# In[23]:


y_pred


# In[24]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[25]:


y_pred


# In[26]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# In[27]:


model=load_model('model_resnet152V2.h5')


# In[28]:


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x=x/255
    x = np.expand_dims(x, axis=0)

    
    
    preds = model.predict(x)
    preds=np.argmax(preds, a
                    xis=1)
    if preds==0:
        preds="1"
    elif preds==1:
        preds="2"
    elif preds==2:
        preds="3"
    elif preds==3:
        preds="4"
    elif preds==4:
        preds="5"
        
    
    
    return preds


# In[29]:


model_predict(r"C:\Users\Vanitha\Desktop\Dataset\test\3\030.bmp",model)


# In[ ]:




