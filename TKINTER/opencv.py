#BASIC GREY-SCALING
import cv2
cap = cv2.VideoCapture(0)
while True:
    frame, ret = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('gray_frame', gray)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break
    cap.release()
    cv2.destroyAllWindows()