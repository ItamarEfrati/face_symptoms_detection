import math

import numpy as np

from scipy.spatial import distance as dist

top_lips_indices = list(map(lambda x: x - 49, [51, 52, 53, 62, 63, 64]))
bottom_lips_indices = list(map(lambda x: x - 49, [57, 58, 59, 66, 67, 68]))
EYE_AR_THRESH = 0.3


def get_jaw_angle(face_center, chin):
    vertical_vector = np.array([0, 1])
    chin_vector = chin - face_center

    unit1 = vertical_vector / np.linalg.norm(vertical_vector)
    unit2 = chin_vector / np.linalg.norm(chin_vector)
    dot = np.dot(unit1, unit2)
    angle = np.arccos(dot)
    return math.degrees(angle)


def get_mouth_height(top_lip, bottom_lip):
    sum = 0
    x = 2
    for i in [3, 4, 5]:
        # distance between two near points up and down
        distance = math.sqrt((top_lip[i][0] - bottom_lip[i + x][0]) ** 2 + (top_lip[i][1] - bottom_lip[i + x][1]) ** 2)
        sum += distance
        x -= 2
    return sum / 3


def get_lip_height(lip):
    sum = 0
    for i in [0, 1, 2]:
        # distance between two near points up and down
        distance = math.sqrt((lip[i][0] - lip[i + 3][0]) ** 2 + (lip[i][1] - lip[i + 3][1]) ** 2)
        sum += distance
    return sum / 3


def get_mouth_ratio(mouth_array: np.array):
    top_lip, bottom_lip = mouth_array[top_lips_indices, :], mouth_array[bottom_lips_indices, :],
    top_lip_height = get_lip_height(top_lip)
    bottom_lip_height = get_lip_height(bottom_lip)
    mouth_height = get_mouth_height(top_lip, bottom_lip)

    # if mouth is open more than lip height * ratio, return true.
    return mouth_height / min(top_lip_height, bottom_lip_height)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


def get_eye_area_ratio(left_eye, right_eye):
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    return ear


def chewing_ratio(mouth):
    vertical_distance = dist.euclidean(mouth[0], mouth[6])
    top_lips_length = dist.euclidean(mouth[3], mouth[14])
    lower_lips_length = dist.euclidean(mouth[9], mouth[18])
    horizontal_distance = top_lips_length + lower_lips_length
    return horizontal_distance / vertical_distance


def get_nodding_ratio(left_jaw, right_jaw, face_center):
    left_dist = dist.euclidean(left_jaw, face_center)
    right_dist = dist.euclidean(right_jaw, face_center)

    return min(left_dist, right_dist) / max(left_dist, right_dist)


def get_ratio(shape, symptom):
    if symptom in 'blink':
        return get_eye_area_ratio(left_eye=shape[36:42], right_eye=shape[42:48])
    elif symptom in "open mouth":
        return get_mouth_ratio(mouth_array=shape[48:68])
    elif symptom in "chewing":
        return chewing_ratio(shape[48:68])
    elif symptom in 'head tilting':
        return get_jaw_angle(chin=shape[8], face_center=shape[30])
    elif symptom in 'nodding':
        return get_nodding_ratio(left_jaw=shape[0], face_center=shape[27], right_jaw=shape[16])
    else:
        return -1
