import cv2
import dlib

import distance_utils

from imutils import face_utils


class Detector:

    def __init__(self, is_presenting, is_recording):
        self.is_presenting = is_presenting
        self.is_recording = is_recording
        self.symptoms_dict = None
        self.init_symptoms_dict()
        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def init_symptoms_dict(self):
        """
        Init the symptoms' dict where the keys are the names of the symptoms to follows and the value is a dictionary
        where:
        1. ratio is the ratio between two possible events of a symptom
        2. current ratio saves the current ration of the symptom in the video
        3. frames counter count the number of frames that the symptoms' event is still occurring
        4. total is the total number the detector count that the symptom event occur.

        An event is only count when it occurs for minimal number of frames
        """
        self.symptoms_dict = {
            "blink": {
                "ratio": 0.25,
                "current_ratio": 0,
                "frames_counter": 0,
                "total": 0
            },
            "open mouth": {
                "ratio": 1,
                "current_ratio": 0,
                "frames_counter": 0,
                "total": 0
            },
            "chewing": {
                "ratio": 0.45,
                "current_ratio": 0,
                "frames_counter": 0,
                "total": 0
            },
            "head tilting": {
                "ratio": 20,
                "current_ratio": 0,
                "frames_counter": 0,
                "total": 0
            },
            "nodding": {
                "ratio": 0.5,
                "current_ratio": 0,
                "frames_counter": 0,
                "total": 0
            }
        }

    def run(self, video_source):
        """
        main function of Detector. Open the video source, counting symptoms and update the count of the video screen.
        :param video_source: the souce of the video for opencv. 0 is laptop camera.
        """
        cap = cv2.VideoCapture(video_source)
        if self.is_recording:
            out = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                  (int(cap.get(3)), int(cap.get(4))))

        while True:
            _, image = cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces_rectangles = self.face_detector(gray, 1)
            if len(faces_rectangles) > 0:
                face_structure = self._draw_face_rectangle_and_structure(faces_rectangles, gray, image)
                image = self._update_image_text(face_structure, image)
                if self.is_recording:
                    out.write(image)
                # image = cv2.resize(image, (165, 165))
                cv2.imshow("Image", image)
                k = cv2.waitKey(30)
                if k == ord("q"):
                    break

        # Release the VideoCapture object
        cap.release()
        if self.is_recording:
            out.release()
        cv2.destroyAllWindows()

        if self.is_recording:
            print("The video was successfully saved")

    def _draw_face_rectangle_and_structure(self, faces_rectangles, gray, image):
        face_rectangle = faces_rectangles[0]
        (x, y, w, h) = face_utils.rect_to_bb(face_rectangle)
        # draw the rectangle of the face detection
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        face_structure = self.shape_predictor(gray, face_rectangle)
        face_structure = face_utils.shape_to_np(face_structure)
        # draws the points of the face structure
        for i, (sX, sY) in enumerate(face_structure):
            cv2.circle(image, (sX, sY), 1, (0, 0, 255), -1)
        return face_structure

    def _update_image_text(self, face_structure, image):
        position = 300
        for symptom in self.symptoms_dict.keys():
            if face_structure is not None:
                self._update_symptoms_info(face_structure, symptom)
            image = self._put_update_text_on_image(position, image, symptom)
            position += 30
        return image

    def _put_update_text_on_image(self, position, image, symptom):
        if self.is_presenting:
            cv2.putText(image, f"Current {symptom} ratio: {self.symptoms_dict[symptom]['current_ratio']:.2f}",
                        (10, position), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 250, 250), 2, cv2.LINE_AA)

        cv2.putText(image, f'Symptom {symptom} : {self.symptoms_dict[symptom]["total"]}', (10, position - 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (250, 250, 250), 2, cv2.LINE_AA)
        return image

    def _update_symptoms_info(self, face_structure, symptom):
        self.symptoms_dict[symptom]['current_ratio'] = distance_utils.get_ratio(face_structure, symptom)
        if symptom == 'blink':
            symptom_level = self.symptoms_dict[symptom]['current_ratio'] <= self.symptoms_dict[symptom]['ratio']
        else:
            symptom_level = self.symptoms_dict[symptom]['current_ratio'] >= self.symptoms_dict[symptom]['ratio']
        if symptom_level:
            self.symptoms_dict[symptom]['frames_counter'] += 1
        elif self.symptoms_dict[symptom]['frames_counter'] >= 1:
            self.symptoms_dict[symptom]['total'] += 1
            self.symptoms_dict[symptom]['frames_counter'] = 0
