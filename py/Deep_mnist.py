# Import Package
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Mnits DataSets
mnit = keras.datasets.mnist

# Split Test and Train
(x_train,y_train),(x_test,y_test) = mnit.load_data()
index = 54
img = x_train[index]
print(y_train[index])
plt.imshow(img,cmap="gray")

#Model
model = keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=256,activation='relu'))
model.add(keras.layers.Dense(units=256,activation='relu'))
model.add(keras.layers.Dense(units=256,activation='relu'))
model.add(keras.layers.Dense(units=10,activation='softmax'))

# Compile
model.compile(optimizer='adam',loss=tf.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

# Summary
model.build(input_shape=(None,x_train.shape))
model.summary()

# Train Model
hist = model.fit(x_train,y_train,epochs=300,batch_size=200,validation_data=(x_test,y_test))

# Save
model.save("mnits.h5")

# Load Model
model_load = keras.models.load_model("mnits.h5")

# History
acc = hist.history['accuracy']
acc_val = hist.history['val_accuracy']
loss = hist.history['loss']
loss_val = hist.history['val_loss']

# accuracy
accuracy = model.evaluate(x_test,y_test)
accuracy

# Plot
plt.plot(acc,color="red",label="accuracy")
plt.plot(acc_val,color="green",label="validation accuracy")
plt.plot(loss,color="blue",label="loss")
plt.plot(loss_val,color="orange",label="validation loss")
plt.legend()
plt.show()