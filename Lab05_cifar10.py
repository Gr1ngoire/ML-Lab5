import matplotlib.pyplot as plt
from keras import models, layers
from keras.datasets import cifar10
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()


x_train = x_train.reshape((50000, 32 * 32 * 3))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 32 * 32 * 3))
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

network = models.Sequential()

network.add(layers.Dense(1936, activation="relu", input_shape=(32 * 32 * 3,)))
network.add(layers.Dense(10, activation="softmax"))

network.compile(optimizer = 'rmsprop',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

network.fit(x_train, y_train, epochs = 5, batch_size = 128)

test_loss, test_acc = network.evaluate(x_train, y_train)

network.save('cifar10.h5')
json_string = network.to_json()
network.save_weights('cifar10_model_weights.h5')

print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

import cv2
from keras.models import load_model
model = load_model('cifar10.h5')

test_frog = 255 - cv2.imread('Frog_cifar10_test.jpg', 0)
test_frog = cv2.resize(test_frog, (32, 32 * 3))
test_frog = test_frog.reshape((1, 32 * 32 * 3))
test_frog = test_frog.astype('float32') / 255

pred = list(model.predict(test_frog)[0])
print(pred.index((max(pred))))


test_horse = 255 - cv2.imread('Buzefal_cifar10_test.jpg', 0)
test_horse = cv2.resize(test_horse, (32, 32 * 3))
test_horse = test_horse.reshape((1, 32 * 32 * 3))
test_horse = test_horse.astype('float32') / 255


pred = list(model.predict(test_horse)[0])
print(pred.index((max(pred))))


test_ship = 255 - cv2.imread('cater5_cifar10_test.jpg', 0)
test_ship = cv2.resize(test_ship, (32, 32 * 3))
test_ship = test_ship.reshape((1, 32 * 32 * 3))
test_ship = test_ship.astype('float32') / 255


pred = list(model.predict(test_ship)[0])
print(pred.index((max(pred))))

