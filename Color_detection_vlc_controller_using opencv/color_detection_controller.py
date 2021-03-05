import cv2
import numpy as np
import pyautogui
counter=0


def getPosition(x,y):
    if x<=300 and y<=225:
        return "UR"
    elif x>300 and y<=225:
        return "UL"
    elif x<=300 and y>225:
        return "LR"
    else:
        return "LL"

#space to play/pause video in vlc
#left to rewind video in vlc
#right to forward video in vlc
#up to volume increase in vlc
#down for volume decrease in vlc
def controlMedia(pos,color):
    if color=="blue":
        pyautogui.typewrite(['space'])
    elif color=="red":   
        if pos=="UL":
            pyautogui.typewrite(['left'])
        elif pos=="LL":
            pyautogui.typewrite(['right'])
        elif pos=="UR":
            pyautogui.typewrite(['up'])
        elif pos=="LR":
            pyautogui.typewrite(['down'])

def getContours(img,color):
    contours,hierarchy=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area=cv2.contourArea(cnt)
        #print(area)
        if area>500:
            #Drawing contours
            cv2.drawContours(output_img,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            #print(len(approx))
            x,y,w,h=cv2.boundingRect(approx)
            pos=getPosition(x,y)
            #Calling control media
            controlMedia(pos,color)


def cannyEdge(img):
    #Converting to grey image
    imgGrey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Blurring image
    imgBlur=cv2.GaussianBlur(imgGrey,(7,7),1)
    #Detecting edges
    imgCanny=cv2.Canny(imgBlur,50,50)
    return imgCanny

#Reading webcam
cap = cv2.VideoCapture(0)
while True:
    counter=counter+1
    if counter>10000000:
        counter=50000000
        _, frame=cap.read()
        output_img=frame.copy()
        #Converting image into HSV format
        hsv_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #For red color
        low_red_color=np.array([0,108,112])
        high_red_color=np.array([18,253,255])
        red_color_mask=cv2.inRange(hsv_frame,low_red_color,high_red_color)
        input_img_red=cv2.bitwise_and(frame,frame,mask=red_color_mask)
        #For blue color
        low_blue_color=np.array([69,132,255])
        high_blue_color=np.array([110,183,255])
        blue_color_mask=cv2.inRange(hsv_frame,low_blue_color,high_blue_color)
        input_img_blue=cv2.bitwise_and(frame,frame,mask=blue_color_mask)
        # For purple color
        '''low_purple_color = np.array([116,19,204])
        high_purple_color = np.array([126,192,255])
        purple_color_mask = cv2.inRange(hsv_frame, low_purple_color, high_purple_color)
        input_img_purple = cv2.bitwise_and(frame, frame, mask=purple_color_mask)'''
        
        '''cv2.imshow("img1", input_img_red)
        cv2.imshow("img2", input_img_blue)
        cv2.imshow("img3", input_img_purple)'''
        
        #Canny edge detector
        red_img=cannyEdge(input_img_red)
        blue_img=cannyEdge(input_img_blue)
        #purple_img=cannyEdge(input_img_purple)
        #Getting Contours
        getContours(red_img,"red")
        getContours(blue_img,"blue")
        #getContours(purple_img,"purple")
        cv2.imshow("output",output_img)
        #Press 'q' to stop the programq
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cv2.destroyAllWindows()
