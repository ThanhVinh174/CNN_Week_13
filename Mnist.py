(x_train_1,y_train_1),(x_test_1, y_test_1) = mnist.load_data()
for i in range(9):
  plt.subplot(330+i+1)
  plt.imshow(x_test_1[i])
print(x_train_1.shape, y_train_1.shape)
x_train_1 = x_train_1.reshape(60000,28,28,1)
x_test_1 = x_test_1.reshape(10000,28,28,1)
x_train_1 = x_train.astype('float32')
x_test_1 = x_test.astype('float32')
x_train_1 /=255
x_test_1 /=255
y_train_1 = np_utils.to_categorical(y_train_1)
y_test_1 = np_utils.to_categorical(y_test_1)
y_train_1.shape
model_mnist = Sequential()
model_mnist.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same',input_shape = (28,28,1)))
model_mnist.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_mnist.add(MaxPooling2D(2,2))

model_mnist.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_mnist.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_mnist.add(MaxPooling2D(2,2))

model_mnist.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_mnist.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_mnist.add(MaxPooling2D(2,2))

model_mnist.add(Flatten())
model_mnist.add(Dense(128, activation ='relu',kernel_initializer='he_uniform'))
model_mnist.add(Dense(10,activation ='softmax'))
model.summary()
opt1 = SGD(lr = 0.001, momentum = 0)
model_mnist.compile(optimizer = opt1, loss ='categorical_crossentropy', metrics = ['accuracy'])
his_mnist = model_mnist.fit(x_train_1, y_train_1, batch_size = 128, epochs = 100, validation_data = (x_test_1,y_test_1))
plot_history(his_mnist)
a = int(input('Phần tử muốn nhận dạng (từ 0 - 9999):'))
print(np.argmax(model_mnist.predict(x_test_1),axis=1)[a])
plt.imshow(x_test_1[a])
