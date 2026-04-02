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