# IOT INTEGRATED SMART HOME SECURITY SYSTEM BASED ON FACIAL RECOGNITION USING TRANSFER LEARNING METHOD




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

## MQT T Implementation
Modify MQTT_publish3.ino according to the network ip and password

## User Interface
Implemented using the pyqt5 library. Modify in file recognize to change.


