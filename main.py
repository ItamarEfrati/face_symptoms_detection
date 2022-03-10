from Detector import Detector

VIDEO_SOURCE = 0 # 0 is laptop camera
PRESENT_FULL_DATA = True
RECORDING = False

if __name__ == '__main__':
    detector = Detector(is_presenting=PRESENT_FULL_DATA, is_recording=RECORDING)
    detector.run(VIDEO_SOURCE)
