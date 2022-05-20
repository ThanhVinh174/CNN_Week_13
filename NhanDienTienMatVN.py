from google.colab import files
upload = files.upload()
with open('data_money.pickle', 'rb') as f:
    (x_train_money, y_train_money) = pickle.load(f)

x_train_money, y_train_money = shuffle(x_train_money, y_train_money)
x_test_money = x_train_money[2400:3368]
y_test_money = y_train_money[2400:3368]

x_train_money_1 = x_train_money[:2400]
y_train_money_1 = y_train_money[:2400]
x_train_money_1 = x_train_money_1.astype('float32')
x_test_money = x_test_money.astype('float32')

x_train_money_1 /=255
x_test_money /=255

y_train_money_1 = np_utils.to_categorical(y_train_money_1,10)
y_test_money = np_utils.to_categorical(y_test_money,10)
model_money = Sequential()
model_money.add(Conv2D(32,(3,3), activation = 'relu',kernel_initializer ='he_uniform',padding ='same',input_shape=(150,150,3)))
model_money.add(Conv2D(32,(3,3),activation ='relu', kernel_initializer= 'he_uniform',padding = 'same'))
model_money.add(MaxPooling2D(2,2))

model_money.add(Conv2D(32,(3,3), activation = 'relu',kernel_initializer ='he_uniform',padding ='same'))
model_money.add(Conv2D(32,(3,3),activation ='relu', kernel_initializer= 'he_uniform',padding = 'same'))
model_money.add(MaxPooling2D(2,2))

model_money.add(Conv2D(32,(3,3), activation = 'relu',kernel_initializer ='he_uniform',padding ='same'))
model_money.add(Conv2D(32,(3,3),activation ='relu', kernel_initializer= 'he_uniform',padding = 'same'))
model_money.add(MaxPooling2D(2,2))

model_money.add(Flatten())
model_money.add(Dense(128, activation ='relu',kernel_initializer='he_uniform'))
model_money.add(Dense(10,activation ='softmax'))
model_money.summary()
opt = SGD(lr = 0.001, momentum =0)
model_money.compile(optimizer = opt, metrics = 'accuracy',loss = 'categorical_crossentropy')
his_money = model_money.fit(x_train_money_1, y_train_money_1, batch_size = 128, epochs = 40, validation_data = (x_test_money,y_test_money),validation_split = 0.2)
plot_history(his_money)
model_money.save('CNN_money.h5')
a = int(input('Du kien muon du doan: '))
label = ['1000','2000','5','5000','10.000','20.000','50.000','100.000','200.000','500.000']
print(label[np.argmax(model_money.predict(x_test_money),axis = 1)[a]])

plt.imshow(x_test_money[a])
plt.show()
