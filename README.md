# ComVision 05주차 실습
# OpenCV 실습 과제

## 0501. 간단한 이미지 분류기 구현
- **설명**: 손글씨 숫자 이미지(MNIST 데이터셋)를 이용하여 간단한 이미지 분류기 구현
- **요구사항**:
  - tensorflow.keras.datasets에서 MNIST 데이터셋을 로드
  - 데이터를 train_data와 test_data로 분할
  - Sequential 모델과 Dense 레이어를 활용하여 간단한 신경망 모델 구축
  - 손글씨 숫자 이미지는 28X28 픽셀 크기의 흑백 이미지
- **코드**
  ```python
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Flatten
  
  # 1. MNIST 데이터셋 로드 및 분할
  # tensorflow.keras.datasets에서 제공하는 mnist 데이터를 불러옵니다.
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  
  # 2. 데이터 전처리
  # 픽셀 값을 0~255에서 0~1 사이의 값으로 정규화합니다.
  x_train, x_test = x_train / 255.0, x_test / 255.0
  
  # 3. 간단한 신경망 모델 구축
  model = Sequential([
      # 손글씨 숫자 이미지는 28x28 픽셀 크기의 흑백 이미지이므로 1차원 배열로 평탄화합니다.
      Flatten(input_shape=(28, 28)), 
      # Dense 레이어를 활용하여 은닉층과 출력층을 구성합니다.
      Dense(128, activation='relu'), 
      Dense(10, activation='softmax') # 0~9까지 10개의 클래스 분류
  ])
  
  # 4. 모델 컴파일 및 훈련
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  print("--- 모델 훈련 시작 ---")
  model.fit(x_train, y_train, epochs=5)
  # 5. 모델 정확도 평가
  print("\n--- 모델 정확도 평가 ---")
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  print(f"테스트 정확도(Accuracy): {test_acc:.4f}")
  ```
- **주요코드**
  ```python
  # 모델 구성
  model = Sequential([
    Flatten(input_shape=(28, 28)),     # 평탄화.
    Dense(128, activation='relu'),     # Dense net과 relu func
    Dense(10, activation='softmax')    # 0~9까지 10개의 클래스 분류
  ])

  #모델 학습
  model.fit(x_train, y_train, epochs=5)
  ```
- **결과물**:
<img width="1151" height="452" alt="image" src="https://github.com/user-attachments/assets/000c615f-10c4-43ca-86a3-7c57a90048c7" />





## 0502. CIFAR-10 데이터셋을 활용한 CNN 모델 구축
- **설명**: CIFAR-10 데이터셋을 활용하여 합성곱 신경망(CNN)을 구축  이미지 분류를 수행
- **요구사항**:
  - tensorflow.keras.datasets에서 MNIST 데이터셋을 로드
  - 데이터 전처리(정규화 등, 픽셀 값을 0~1 범위로 정규화 시 모델의 수렴이 빨라질 수 있음)를 수행
  - CNN 모델을 설계(Conv2D, MaxPooling, Flatten, Dense 레이어 활용)하고 훈련
- **코드**
  ```python
  import tensorflow as tf
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
  import numpy as np
  from tensorflow.keras.preprocessing import image
  import os
  
  # 1. CIFAR-10 데이터셋 로드
  # tensorflow.keras.datasets에서 CIFAR-10 데이터를 불러옵니다.
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  
  # 2. 데이터 전처리 (정규화)
  # 픽셀 값을 0~1 범위로 정규화하여 모델의 수렴 속도를 높입니다.
  x_train, x_test = x_train / 255.0, x_test / 255.0
  
  # CIFAR-10 클래스 이름 정의 (예측 결과 출력을 위함)
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                 'dog', 'frog', 'horse', 'ship', 'truck']
  
  # 3. CNN 모델 설계 
  # Conv2D, MaxPooling2D, Flatten, Dense 레이어를 활용하여 CNN 구성
  model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      Flatten(),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax')
  ])
  
  # 4. 모델 컴파일 및 훈련
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  
  print("--- CNN 모델 훈련 시작 ---")
  history = model.fit(x_train, y_train, epochs=10, 
                      validation_data=(x_test, y_test))
  
  # 5. 모델 성능 평가 
  print("\n--- 모델 성능 평가 ---")
  test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
  print(f"테스트 정확도(Accuracy): {test_acc:.4f}")
  
  # 6. 테스트 이미지(dog.jpg)에 대한 예측 수행 
  print("\n--- 테스트 이미지 예측 ---")
  img_path = 'dog.jpg'
  
  # 코드가 실행되는 폴더에 'dog.jpg' 파일이 있는지 확인
  if os.path.exists(img_path):
      # 이미지를 CIFAR-10 데이터 크기(32x32)에 맞게 로드
      img = image.load_img(img_path, target_size=(32, 32))
      img_array = image.img_to_array(img)
      img_array = np.expand_dims(img_array, axis=0) # 배치 차원 추가
      img_array = img_array / 255.0 # 정규화 적용
  
      # 예측 수행
      predictions = model.predict(img_array)
      predicted_class_idx = np.argmax(predictions[0])
      
      print(f"예측 결과: 해당 이미지는 '{class_names[predicted_class_idx]}' 입니다.")
  else:
      print("경고: 현재 폴더에 'dog.jpg' 파일이 없습니다. 예측 기능을 테스트하려면 사진을 준비해주세요.")
  ```
- **주요코드**
  ```python
  # 모델 구조
  model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
  ])

  # 모델 학습
  history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))
  ```
- **결과물**:
<img width="1814" height="541" alt="image" src="https://github.com/user-attachments/assets/ac80ff1f-5cd9-4625-9d0e-2d7d046c66f2" />

