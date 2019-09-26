import cv2
import numpy as np

# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.1, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.1,6)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None

    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]

        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        #cv2.rectangle(img,(x + int(w/2),y + int(h/2)),(x + int(w/2),y + int(h/2)),(0,255,0),2)
        
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

def pupil(img, threshold):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (7,7), 0)
    _, threshold = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY_INV)
    _, contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    img_h, img_w, z = img.shape
    vec_x = None
    vec_y = None
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        vec_x = x + w // 2 - img_w // 2
        vec_y = y + h // 2 - img_h // 2

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(img,(x + int(w/2),y + int(h/2)),(x + int(w/2),y + int(h/2)),(0,0,255),2)
        
    cv2.imshow('after', threshold)
    return img, (vec_x, vec_y)

def vecter(vec_list):
   
    image = np.zeros((500,500,3))
    g = 5
    print(vec_list)
    if vec_list[0] != None and vec_list[0][0] != None:
        print("L:x",vec_list[0][0])
        print("L:y",vec_list[0][1])
        cv2.arrowedLine(image, (250, 250), (g*vec_list[0][0]+250, g*vec_list[0][1]+250),(0, 255, 0), 2)


    if vec_list[1] != None and vec_list[1][0] != None:
        print("R:x",vec_list[1][0])
        print("R:y",vec_list[1][1])
        cv2.arrowedLine(image, (250, 250), (g*vec_list[1][0]+250, g*vec_list[1][1]+250),(0, 0, 255), 2)

    cv2.imshow('vecter', image)

def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

    while (cap.isOpened()):
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            
            vec_list = [None, None]
            for inx, eye in enumerate(eyes):
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    central_eye, vec_list[inx] = pupil(eye, threshold)
            vecter(vec_list)                            
    
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
