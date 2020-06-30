from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image


classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(256, 256, 3)))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))



classifier.add(Flatten())

classifier.add(Dense(activation = 'relu', units = 512))
classifier.add(Dense(activation = 'sigmoid', units = 1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.summary()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('/home/parrot/Studies/Image_Processing/Brain/',
                                                 target_size = (256,256),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/home/parrot/Studies/Image_Processing/Brain/',
                                            target_size = (256, 256),
                                            batch_size = 64,
                                            class_mode = 'binary')

history = classifier.fit_generator(training_set,
                         steps_per_epoch =20,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps =10)

print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Test set'], loc='upper left')
plt.show()

scores = classifier.evaluate_generator(test_set,624/16)
print("\nAccuracy:"+" %.2f%%" % ( scores[1]*100))


img =image.load_img('/home/parrot/Studies/Image_Processing/Brain/Benign/1Perfect.jpg', target_size=(256, 256))
plt.imshow(img,"Accent")
plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = classifier.predict_classes(images, batch_size=10)

img1 = image.load_img('/home/parrot/Studies/Image_Processing/Brain/Malignant/2.jpg', target_size=(256, 256))
plt.imshow(img,"Accent")
plt.show()
y = image.img_to_array(img1)
y = np.expand_dims(y, axis=0)
images = np.vstack([x, y])

prediction = classifier.predict(images)


i = 1

for things in prediction:
    if(things < 0.5):
        print('%d.Benign'%(i))
    else:
        print('%d.Malignant'%(i))
    i = i + 1

print(prediction)
