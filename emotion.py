import cv2
from deepface import DeepFace
import pyttsx3
import threading
# Load video from webcam
cap = cv2.VideoCapture(0)
def speak_emotion(emotion):
    engine = pyttsx3.init()
    engine.say(emotion)
    engine.runAndWait()
while True:
    key, img = cap.read()
    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Display emotion on frame
    emotion = results[0]['dominant_emotion']
    threading.Thread(target=speak_emotion, args=(emotion,)).start()
    
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()