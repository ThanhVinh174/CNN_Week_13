with open('data.pickle', 'rb') as f:
    (x_train_face, y_train_face) = pickle.load(f)


x_train_face = x_train_face[:194]
y_train_face = y_train_face[:194]

# x_train_face = x_train_face.reshape(x_train.shape[0],-1)

x_train_face = x_train_face.astype('float32')
x_train_face/=255

y_train_face = np_utils.to_categorical(y_train_face,2)
model_face = Sequential()
model_face.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same',input_shape = (150,150,3)))
model_face.add(Conv2D(32,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_face.add(MaxPooling2D(2,2))

model_face.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_face.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_face.add(MaxPooling2D(2,2))

model_face.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_face.add(Conv2D(64,(3,3),activation = 'relu',kernel_initializer='he_uniform',padding ='same'))
model_face.add(MaxPooling2D(2,2))

model_face.add(Flatten())
model_face.add(Dense(128, activation ='relu',kernel_initializer='he_uniform'))
model_face.add(Dense(2,activation ='softmax'))
model_face.summary()
opt = SGD(lr = 0.001, momentum =0)
model_face.compile(optimizer = opt, metrics = 'accuracy', loss = 'categorical_crossentropy')
his_face = model_face.fit(x_train_face,y_train_face, batch_size = 128, epochs = 50,validation_split = 0.2)
plot_history(his_face)
x_test_face = x_train_face[99]
x_test_face = x_test_face.reshape(1,150,150,3)
print(x_test_face.shape)
plt.imshow(x_train_face[99])
pre = np.argmax(model_face.predict(x_test_face))
if pre == 1 :
  print('This is Vinh')
else:
  print('This is another person')