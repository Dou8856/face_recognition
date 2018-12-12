import numpy as np
import cv2
import datetime
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import numpy as np
import sys
import os

CAPTURE = False
RECOGNITION = False
CAP_NUM = 0
print(sys.argv)
if sys.argv[1] == "capture":
    print("Emily: capture mode")
    CAPTURE = True
    CAP_NUM = int(sys.argv[2])
    FOLDER = sys.argv[3]
    PATH_TO_SAVE = "./training_data/" + FOLDER
elif sys.argv[1] == "recognition":
        RECOGNITION = True
        

        
SIZE = 64

def draw_bounding_box():
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        print (cv2.contourArea(c))
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)


def network(x, test=False):
    # Input:x -> 3,64,64
    # ImageAugmentation
    h = F.image_augmentation(x, (3,64,64), (0,0), 1, 1, 0, 1, 0, False, False, 0, False, 1, 0.5, False, 0)
    # Convolution -> 16,60,60
    h = PF.convolution(h, 16, (5,5), (0,0), name='Convolution')
    # ReLU
    h = F.relu(h, True)
    # MaxPooling -> 16,30,30
    h = F.max_pooling(h, (2,2), (2,2))
    # Convolution_2 -> 30,28,28
    h = PF.convolution(h, 30, (3,3), (0,0), name='Convolution_2')
    # MaxPooling_2 -> 30,14,14
    h = F.max_pooling(h, (2,2), (2,2))
    # Tanh_2
    h = F.tanh(h)
    # Affine -> 150
    h = PF.affine(h, (150,), name='Affine')
    # ReLU_2
    h = F.relu(h, True)
    # Affine_2 -> 2
    h = PF.affine(h, (2,), name='Affine_2')
    # Softmax
    h = F.softmax(h)
    return h

x = nn.Variable((1,3,SIZE,SIZE))
y = network(x, test=True)
nn.load_parameters("./parameters.h5")
result_class = "can't identify"
cascPath = "../opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
print (faceCascade)

def pre_processing_frame(frame_in, size_to_be):
    #resize
    frame_resize = cv2.resize(frame_in, (size_to_be, size_to_be))
    #channel swap
    frame_resize = frame_resize[:,:,(2,1,0)]
    frame_resize = frame_resize.transpose(2,0,1)
    frame_resize = frame_resize*1.0/255
    return frame_resize

def classification(img_in):
    height, width, channel = img_in.shape
    x.d = img_in
    forward = y.forward()
    result_array = y.d[0]
    index = np.unravel_index(result_array.argmax(), result_array.shape)
    print("result array ", result_array)
    max_index = index[0]
    prob = result_array[max_index]
    print("result_array.argmax = ", result_array.argmax())
    if max_index == 0 and prob>0.9:
        result_class = "Hello Carlos"
        print(result_class)
    elif max_index == 1 and prob>0.9: 
        result_class = "Hello Emily"
        print(result_class)
    else: 
        result_class = "can't identify"
    return result_class, str(prob)


cap = cv2.VideoCapture(0)

while(True):
    #capture frame by frame
    ret, frame = cap.read()

    #Our operations on the frame come here
    #gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    font = cv2.FONT_ITALIC
    x_axis = 10 #position of text
    y_axis = 20 #position of text
   
    frame_to_process = frame.copy()

    if CAPTURE:
        for i in range(CAP_NUM):
            frame_to_process = cv2.resize(frame_to_process, (SIZE, SIZE))
            if os.path.exists(PATH_TO_SAVE) == False:
                os.mkdir(PATH_TO_SAVE)
            file_name = PATH_TO_SAVE + '_opencv'+str(i)+'.png'
            cv2.imwrite(file_name, frame_to_process)
            print("saved img to ", file_name)
        cap.release()
        cv2.destoryAllWindows()

    if RECOGNITION:
        frame_processed = pre_processing_frame(frame_to_process, SIZE)
        result = classification(frame_processed)
        faces = faceCascade.detectMultiScale(frame)
        # Draw a rectangle around the faces
        for (rec_x, rec_y, rec_w, rec_h) in faces:
            cv2.rectangle(frame, (rec_x, rec_y), (rec_x+rec_w, rec_y+rec_h), (0, 255, 0), 2)


        date_time =  datetime.datetime.now()
        datetime_str = date_time.strftime('%d %H:%M:%S')
        text_to_display = datetime_str + "\n " + result[0] + "\n probability: " + result[1]
        cv2.putText(frame, text_to_display, (x_axis,y_axis), font, 0.8, 255) 
    #Draw the text

    #display the reulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destoryAllWindows()

