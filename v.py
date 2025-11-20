import cv2
import pyttsx3
import threading
from deepface import DeepFace
# Load video from webcam

last_emotion=None
cap = cv2.VideoCapture(0)

while True:
    key, img = cap.read()
    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    def spek_thread(text):
        engine=pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        
    # Display emotion on frame
    emotion = results[0]['dominant_emotion']
    if emotion and emotion !=last_emotion:
       t=threading.Thread(target=spek_thread,args=(emotion,))
       t.start()
       last_emotion=emotion
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()