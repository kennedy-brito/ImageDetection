import cv2

class VideoImageDetector:

  def __init__(self, model: str):

    self.classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + model)
    
    self.camera = cv2.VideoCapture(0)

  def detect_bounding_box(self, vid):
    
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    
    faces = self.classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return faces
    
  def run(self):
    """
    The main loop of the application
    """

    while True:

      result, video_frame = self.camera.read()  # read frames from the video
      if result is False:
          break 

      faces = self.detect_bounding_box(
          video_frame
      )  # apply the function we created to the video frame

      cv2.imshow(
          "My Face Detection Project", video_frame
      )  # display the processed frame in a window 

      if cv2.waitKey(1) & 0xFF == ord("q"): #quit after pressing the key 'q'
          break

    self.camera.release()
    cv2.destroyAllWindows()