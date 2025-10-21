import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 초점 조절이 가능한지 확인
if cap.get(cv2.CAP_PROP_FOCUS) == -1:
    print("이 카메라는 초점 조절을 지원하지 않습니다.")
else:
    print("초점 조절을 시작합니다.")

# 윈도우 이름 설정
cv2.namedWindow('Camera')

# 초점 값을 변경하는 콜백 함수
def change_focus(val):
    # OpenCV에서 초점 값은 0.0 ~ 1.0 사이의 비율로 설정됨
    focus_value = val / 100  # 트랙바 값을 0~1로 변환
    cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    print(f"초점 값 설정: {focus_value}")

# 트랙바 생성
cv2.createTrackbar('Focus', 'Camera', 0, 100, change_focus)

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 카메라 화면 출력
    cv2.imshow('Camera', frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()