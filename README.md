# IOT INTEGRATED SMART HOME SECURITY SYSTEM BASED ON FACIAL RECOGNITION USING TRANSFER LEARNING METHOD
This repo uses RetinaFace for the detection, FaceNet model for extraction of features and both k-Nearest Neighbor (kNN) and Support Vector Machine (SVM) methods for classification. Accuracy of detection for both cases has been above 96%. The hardware of the lock is constructed using a stepper motor, driver and 3D printed lock components modeled using SOLIDWORKS. NodeMCU and MQTT are used to interconnect the hardware to the internet, making it an IoT device. To make the system accessible to the general public, I built a simple and convenient user interface using PyQt5. The entirety of this system is able to automatically detect visitors and open a door within 7 seconds.
## Live Demo:

![alt text](https://github.com/JohayraPrithula/FacialRecognition/blob/ImplementingUI/authorized.gif?raw=true)
![alt text](https://github.com/JohayraPrithula/FacialRecognition/blob/ImplementingUI/unauthorized.gif?raw=true)


## Complete Algorithm 
![alt text](https://github.com/JohayraPrithula/FacialRecognition/blob/ImplementingUI/Picture1.jpg?raw=true)


## Facial Recognition
Detection: Retinaface
Extraction: Facenet
Recognition: kNN and SVM

#### After installation change the working directory to models/retinaface by using the following command:
```
cd models/retinaface
       
       Then run the following commands to execute each script in the following                  order:
       
       python box_utils.py
       
       python config.py
       
       python mobilev1.py
       
       python prior_box.py
       
       python py_cpu_nms.py
       
       python retinaface.py
       
       And finally run the face detection model by running:
       
       python detect.py
```

## Data Visualization

tsneGraph:



## Hardware Integration
![alt text](https://github.com/JohayraPrithula/FacialRecognition/blob/ImplementingUI/Picture2.jpg?raw=true)
![](https://github.com/JohayraPrithula/FacialRecognition/blob/ImplementingUI/Picture3.png?raw=true)

## MQTT Implementation
Modify MQTT_publish3.ino according to the network ip and password

## User Interface
Implemented using the pyqt5 library. Modify in file recognize to change.


