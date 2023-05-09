import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()

network.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
network.add(layers.Dense(10, activation="softmax"))

network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

network.fit(train_images, train_labels, epochs = 5, batch_size = 128)

test_loss, test_acc = network.evaluate(test_images, test_labels)

network.save('mnist.h5')
json_string = network.to_json()
network.save_weights('mnist_weights.h5')

print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

import cv2
from keras.models import load_model
model = load_model('mnist.h5')

tst = 255 - cv2.imread('Number_7_to_guess.png', 0)
tst = cv2.resize(tst, (28, 28))
tst = tst.reshape((1, 28 * 28))
tst = tst.astype('float32') / 255

pred = list(model.predict(tst)[0])
print(pred.index((max(pred))))


