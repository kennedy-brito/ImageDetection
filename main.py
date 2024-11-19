from video_face_detection import VideoImageDetector
import argparse

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
                  prog='Image Detection',
                  description='A project for Face or Plates detection for the discipline of "Tópicos Especiais I"'
                  )
  parser.add_argument('-p', '--plate', action='store_true')
  parser.add_argument('-f', '--face', action='store_true')

  plates = "haarcascade_russian_plate_number.xml"
  faces = "haarcascade_frontalface_default.xml"

  args = parser.parse_args()
  
  if args.plate:
    model = plates
  else:
    model = faces

  app  = VideoImageDetector(model)


  app.run()
