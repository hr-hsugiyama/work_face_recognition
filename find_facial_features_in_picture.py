from PIL import Image, ImageDraw
import face_recognition
from scipy.spatial import distance

## https://github.com/ageitgey/face_recognition/blob/master/examples/find_facial_features_in_picture.py
# Load the jpg file into a numpy array
image = face_recognition.load_image_file("image_student.png")

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))


def calc_ear(eye):
    for index, e in enumerate(eye):
        d.point(e)
        d.text(e, 'p{}'.format(index))

    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    eye_ear = (A + B) / (2.0 * C)
    return round(eye_ear, 3)


# Create a PIL imagedraw object so we can draw on the picture
pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for index, face_landmarks in enumerate(face_landmarks_list):

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    left_eye_ear = calc_ear(face_landmarks["left_eye"])
    right_eye_ear = calc_ear(face_landmarks["right_eye"])
    print("left EAR {}, right EAR {}".format(left_eye_ear, right_eye_ear))

    top, right, bottom, left = face_locations[index]
    # print(
    #     "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right)
    # )
    outline = "white"
    ear_sum = left_eye_ear + right_eye_ear
    if (ear_sum) < 0.30:
        outline = "red"
    print(left_eye_ear + right_eye_ear)
    print(outline)
    d.rectangle([left, top, right, bottom], outline=outline, width=5)
    d.text((left, bottom), str(ear_sum), font_size=36)
    # Let's trace out each facial feature in the image with a line!
    # for facial_feature in face_landmarks.keys():
    #     if facial_feature in ('left_eye', 'right_eye'):
    #         d.line(face_landmarks[facial_feature], width=5)


# Show the picture
pil_image.show()
