import cv2
import numpy as np
from fer import FER
import face_recognition

class VideoProcessor:
    def __init__(self):
        self.emotion_detector = FER(mtcnn=True)

    async def analyze_emotions(self, video_path):
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        emotions = []
        face_locations = []
        for i in range(0, frame_count, int(fps)):  # Analyze one frame per second
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                emotion = self.emotion_detector.detect_emotions(frame)
                if emotion:
                    emotions.append(emotion[0]['emotions'])
                    face_locations.append(self.detect_faces(frame))

        video.release()
        return {
            "duration": duration,
            "emotions": emotions,
            "face_locations": face_locations
        }

    async def analyze_video_frame(self, frame):
        emotions = self.emotion_detector.detect_emotions(frame)
        face_locations = self.detect_faces(frame)
        return {
            "emotions": emotions[0]['emotions'] if emotions else None,
            "face_locations": face_locations
        }

    def detect_faces(self, frame):
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        
        return face_locations

    def extract_facial_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        return face_landmarks_list

    def track_object(self, video_path, object_cascade_path):
        object_cascade = cv2.CascadeClassifier(object_cascade_path)
        video = cv2.VideoCapture(video_path)

        tracked_objects = []
        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects = object_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in objects:
                tracked_objects.append((x, y, w, h))

        video.release()
        return tracked_objects

    def optical_flow(self, video_path):
        video = cv2.VideoCapture(video_path)
        ret, first_frame = video.read()
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        # Add your optical flow implementation here
        # This is just a placeholder
        return "Optical flow analysis completed"