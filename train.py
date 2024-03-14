import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from cnn import x_train,y_train,x_test,y_test


# Reshape dữ liệu
x_train_reshaped = x_train.reshape(-1, 40*100)
x_train_reshaped = x_train_reshaped.astype('float32')

# Tạo mô hình
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(40*100,)))
model.add(Dense(1, activation='sigmoid'))

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train_reshaped, y_train, epochs=100)

model.save("my-model.keras")

# Đánh giá độ chính xác trên tập kiểm thử
x_test_reshaped = np.array(x_test)
y_test_reshaped = np.array(y_test).reshape(-1, 1) 

results = model.evaluate(x_test_reshaped, y_test_reshaped)
accuracy = results[1]
print(f"Độ chính xác trên tập kiểm thử: {accuracy * 100:.2f}%")

