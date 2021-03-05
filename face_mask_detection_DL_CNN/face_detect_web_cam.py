import cv2
import numpy as np
from tensorflow.keras.models import load_model
model=load_model('face_mask_detect.h5')
frontal_face_model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
while(True):
    success,img=cap.read()
    #print(img.shape)
    grey_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_coord=frontal_face_model.detectMultiScale(grey_img)
    for (x,y,w,h) in face_coord:
        face_img=img[y:y+h,x:x+w]
        face_img_resized=cv2.resize(face_img,(64,64))
        face_img_normalized=face_img_resized/255.0
        face_img_reshaped=np.reshape(face_img_normalized,(1,64,64,3))
        result = model.predict(face_img_reshaped)
        #print(round(result[0][0],2))
        if result[0][0]<0.1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
            cv2.putText(img,'With Mask',(x+3,y+3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        elif result[0][0]>0.8:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
            cv2.putText(img,'Without Mask',(x+3,y+3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
        else:
             cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
             cv2.putText(img,'Not Sure',(x+3,y+3),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
