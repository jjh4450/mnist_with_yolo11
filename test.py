import cv2
import torch
import numpy as np
from ultralytics import YOLO

# 학습된 모델 로드
model_path = "./runs/classify/train/weights/last.pt"
model = YOLO(model_path)  # 모델 로드

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # 웹캠 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # 1️⃣ 웹캠 프레임을 MNIST 스타일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    inverted = cv2.bitwise_not(gray)  # 📌 색상 반전 (MNIST 스타일)
    resized = cv2.resize(inverted, (32, 32))  # 32x32 크기로 조정

    # 2️⃣ 정규화 (0~1 범위)
    normalized = resized / 255.0

    # 3️⃣ 0.5 기준으로 반올림하여 이진화
    binary_image = (normalized >= 0.5).astype(np.float32)  # 0.5 이상이면 1, 아니면 0

    # 4️⃣ Grayscale → RGB 변환 (YOLO는 3채널 입력을 요구함)
    rgb_image = np.stack([binary_image] * 3, axis=-1)  # 1채널을 3채널로 확장

    # 5️⃣ 모델 입력 형식에 맞게 변환
    img_tensor = torch.tensor(rgb_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)

    # 6️⃣ YOLO 모델을 사용하여 예측 수행
    results = model.predict(img_tensor)

    # 7️⃣ 예측된 클래스 정보 가져오기
    if results and results[0].probs:
        predictions = results[0].probs.data  # 클래스별 확률값 (Tensor)
        class_id = torch.argmax(predictions).item()  # 가장 높은 확률의 클래스 ID
        confidence = predictions[class_id].item()  # 해당 클래스의 확률 값

        # 8️⃣ 화면에 클래스 및 확률 표시
        label = f"Class: {class_id}, Confidence: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 9️⃣ YOLO 모델이 보는 이미지(32x32)를 원본 화면에 추가
    overlay = cv2.resize((binary_image * 255).astype(np.uint8), (100, 100))  # 100x100 크기로 확대
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)  # 흑백을 다시 BGR 변환 (오버레이를 위해)

    # 오버레이를 화면의 우측 상단에 추가
    frame[10:110, -110:-10] = overlay  # (10,10)~(110,110) 위치에 오버레이

    # 🔟 프레임 출력
    cv2.imshow("YOLO Classification - Webcam", frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
