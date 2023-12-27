#opencv'yi ve array oluştururken kullanacağımız numpyi import ediyoruz
  
import numpy as np 
import cv2 

  
#webcami açma
webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
  

while(True): 
      
   #webcami açtık ama şimdi okumamız lazım goruntuyu bu şekilde goruntuyu okuyoruz ve imageFrame değerine atıyoruz
    _, imageFrame = webcam.read() 
    #kameradan aldığımız BGR görüntüyü HSV'e çeviriyoruz RGB'e de çevrilebilir
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)     
    #spesifik renkleri belirlemek için renklerin hsv aralıklarını girmemiz gerekiyor böylelikle onları ayırt edebileceğiz kısaca masklayacağız
    red_lower = np.array([0, 100, 100], np.uint8) 
    red_upper = np.array([10, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
    #hsv değerleri genel olarak belli değil o yüzden dene-yanıl ile kendime en iyi değerleri bu şekilde buldum
    
    green_lower = np.array([50, 60,60]) 
    green_upper = np.array([70, 255, 255]) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 
    

    
    blue_lower = np.array([110, 60, 50], np.uint8) 
    blue_upper = np.array([130, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
    #5x5 matrix oluşturarak dilate ettiğimiz görüntüleri buraya aktarıyoruz
    kernel = np.ones((5, 5), np.uint8) 
    #görüntüleri dilate ederek pikselleri büyütüyoruz spesifikliği azalıyor ama renk detect için pürüzleri azaltıyor
    red_mask=cv2.dilate(red_mask,kernel)
    green_mask=cv2.dilate(green_mask,kernel)
    blue_mask=cv2.dilate(blue_mask,kernel)
    
    
    
    #
    contourred,hiearchy=cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    
    contourgreen,hiearchy=cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    contourblue,hiearchy=cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contourred)):
        if cv2.contourArea(contourred[i]) > 1000: 
            (x, y, w, h) = cv2.boundingRect(contourred[i])
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, f'Red {x} {y}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    for i in range(len(contourblue)):
        if cv2.contourArea(contourblue[i]) > 1000: 
            (x, y, w, h) = cv2.boundingRect(contourblue[i])
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, f'Blue {x} {y}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    for i in range(len(contourgreen)):
        if cv2.contourArea(contourgreen[i]) > 1000: 
            (x, y, w, h) = cv2.boundingRect(contourgreen[i])
            cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(imageFrame, f'Green{x} {y}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('RGB',imageFrame)
   
    
    ## bitwise = cv2.bitwise_and(imageFrame, imageFrame, mask = new_mask)   sadece kırmızı yeşil ve mavi renkli objeler gözükür bunu kullanmamalıyım
                                                          
    
    
    
    if cv2.waitKey(10) & 0xFF == ord('q'): 
        webcam.release() 
        cv2.destroyAllWindows() 
        break
        
  