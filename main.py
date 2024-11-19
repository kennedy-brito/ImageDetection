from video_face_detection import VideoImageDetector

plates = "haarcascade_russian_plate_number.xml"
faces = "haarcascade_frontalface_default.xml"

app  = VideoImageDetector(plates)


app.run()






#https://www.datacamp.com/tutorial/face-detection-python-opencv
#https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html