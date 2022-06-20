import face_recognition
import cv2
import os
import streamlit
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from faces import known_faces


cascPathface = "haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
known_names = ['Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny',
               'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny',
               'Evgeny', 'Evgeny', 'Evgeny', 'Evgeny', 'Mask', 'Mask', 'Mask', 'Mask', 'Mask', 'Nikita Kourov',
               'Nikita Kourov', 'Nikita Kourov', 'Nikita Kourov', 'Obabkov', 'Obabkov', 'Obabkov', 'Obabkov', 'Obabkov']
TOLERANCE = 0.6


def wr(word):
    streamlit.write(word)


class VideoProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        faces = faceCascade.detectMultiScale(cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY), 1.1, 3)
        encodings = face_recognition.face_encodings(frm, model='cnn')
        names = []
        for encoding in encodings:
            matches = face_recognition.compare_faces(known_faces,
                                                     encoding, TOLERANCE)
            name = "Unknown"
            if True in matches:
                name = known_names[matches.index(True)]
            names.append(name)
            for ((x, y, w, h), name) in zip(faces, names):
                cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frm, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(frm, format='bgr24')


wr("""
# Neural Paradise
""")
webrtc_streamer(key="key", video_processor_factory=VideoProcessor)#,
                #rtc_configuration=RTCConfiguration(
                #{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))