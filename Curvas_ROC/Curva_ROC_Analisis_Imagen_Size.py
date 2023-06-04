# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:12:18 2023

@author: HP 690 -000B
"""

import os
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2

#Máscaras de color
def segmentarLimon(image):
    Lista_Imagenes_Limon = []
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  

    # Elegimos el umbral de verde en HSV
    umbral_verde__bajo = np.array([34,90,0])
    umbral_verde__alto = np.array([90,255,255])

    umbral_Amarillo_bajo = np.array([11,90,0])
    umbral_Amarillo_alto = np.array([33,255,255])

    rojoBajo1 = np.array([0, 90,0], np.uint8)
    rojoAlto1 = np.array([10, 255, 255], np.uint8)
    rojoBajo2 = np.array([120, 90, 0], np.uint8)
    rojoAlto2 = np.array([180, 255, 255], np.uint8)

    # Realizamos las máscaras de la imagen
    mascara_verde = cv2.inRange(image_hsv, umbral_verde__bajo, umbral_verde__alto)
    mascara_amarillo = cv2.inRange(image_hsv, umbral_Amarillo_bajo, umbral_Amarillo_alto)
    mascara_verde_amarillo =  cv2.add(mascara_verde, mascara_amarillo)

    mascara_Rojo1 = cv2.inRange(image_hsv, rojoBajo1, rojoAlto1)
    mascara_Rojo2 = cv2.inRange(image_hsv, rojoBajo2, rojoAlto2)
    mascara_Rojo =  cv2.add(mascara_Rojo1, mascara_Rojo2)

    # Macara final
    mascara_final =  cv2.add(mascara_verde_amarillo, mascara_Rojo)

    #Mascara en imagen
    res = cv2.bitwise_and(image, image, mask=mascara_final)
    Lista_Imagenes_Limon.append(mascara_final)
    Lista_Imagenes_Limon.append(res)
    
    return Lista_Imagenes_Limon

def getROI(Imagen):
    thr = Imagen[0]
    segmentada = Imagen[1]
    #encontrar contornos
    contours,hierarchy = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #opencv 3
    # Copias profundas de la imagen de entrada para dibujar resultados:
    minRectImage = segmentada.copy()
    polyRectImage = segmentada.copy()
    #variable para guardar la ROI final:
    croppedImg = 0 
    # Busque los cuadros delimitadores exteriores:
    for i, c in enumerate(contours):

        if hierarchy[0][i][3] == -1:

            # Get contour area:
            contourArea = cv2.contourArea(c)
            # Establecer el umbral de área mínima:
            minArea = 1000
            # Opción 2: Aproximar el contorno a un polígono:
            contoursPoly = cv2.approxPolyDP(c, 3, True)
            # Convierta el polígono en un rectángulo delimitador:
            boundRect = cv2.boundingRect(contoursPoly)
            # Establezca las dimensiones del rectángulo:
            rectangleX=boundRect[0]
            rectangleY=boundRect[1]
            rectangleWidth=boundRect[0] + boundRect[2]
            rectangleHeight=boundRect[1] + boundRect[3]
            
            # Draw the rectangle:
            cv2.rectangle(polyRectImage, (int(rectangleX), int(rectangleY)),(int(rectangleWidth), int(rectangleHeight)), (0, 255, 0), 2)

            # Recortar el ROI:
            croppedImg = segmentada[rectangleY:rectangleHeight, rectangleX:rectangleWidth]

            if croppedImg.shape[0]>100:
              return croppedImg

MM_PX =1434/146
data=np.zeros((4,4))
test_data=[]
test_labels=[]


print('--------------------------------- Tamaño pequeño -------------------------------------------------')
# Ubicación de las fotos
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Size\Test\Tomate_small'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_mediano'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_grande'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  # Procesar Imagen Limon
  Imagen_mascara = segmentarLimon(URL_Imagen) #retorna 2 imagenes. Primera es binaria y sda es rgb.
  #Obtener ROI del limon:
  roi_limon = getROI(Imagen_mascara)

  # Obteniendo el diámetro en mm del limón 1:
  anchopx = roi_limon.shape[1]
  DIAMETRO_LIMON = anchopx / MM_PX
  DIAMETRO_LIMON = round(DIAMETRO_LIMON, 2)
  
  test_labels.append(int(3))

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    x=[0,0,0,1]
    test_data.append(x)

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1
    x=[0,0,1,0]
    test_data.append(x)

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1
    x=[0,1,0,0]
    test_data.append(x)

  else:
    Tama_4=Tama_4+1
    x=[1,0,0,0]
    test_data.append(x)

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[0,0]=Tama_1
data[0,1]=Tama_2
data[0,2]=Tama_3
data[0,3]=Tama_4

print('--------------------------------- Tamaño mediano -------------------------------------------------')
# Ubicación de las fotos
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_small'
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Size\Test\Tomate_mediano'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_grande'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  # Procesar Imagen Limon
  Imagen_mascara = segmentarLimon(URL_Imagen) #retorna 2 imagenes. Primera es binaria y sda es rgb.
  #Obtener ROI del limon:
  roi_limon = getROI(Imagen_mascara)

  # Obteniendo el diámetro en mm del limón 1:
  anchopx = roi_limon.shape[1]
  DIAMETRO_LIMON = anchopx / MM_PX
  DIAMETRO_LIMON = round(DIAMETRO_LIMON, 2)
  
  test_labels.append(int(2))

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    x=[0,0,0,1]
    test_data.append(x)

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1
    x=[0,0,1,0]
    test_data.append(x)

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1
    x=[0,1,0,0]
    test_data.append(x)

  else:
    Tama_4=Tama_4+1
    x=[1,0,0,0]
    test_data.append(x)

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[1,0]=Tama_1
data[1,1]=Tama_2
data[1,2]=Tama_3
data[1,3]=Tama_4
print('--------------------------------- Tamaño grande -------------------------------------------------')
# Ubicación de las fotos
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_small'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_mediano'
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Size\Test\Tomate_grande'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0


for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  # Procesar Imagen Limon
  Imagen_mascara = segmentarLimon(URL_Imagen) #retorna 2 imagenes. Primera es binaria y sda es rgb.
  #Obtener ROI del limon:
  roi_limon = getROI(Imagen_mascara)

  # Obteniendo el diámetro en mm del limón 1:
  anchopx = roi_limon.shape[1]
  DIAMETRO_LIMON = anchopx / MM_PX
  DIAMETRO_LIMON = round(DIAMETRO_LIMON, 2)
  
  test_labels.append(int(1))

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    x=[0,0,0,1]
    test_data.append(x)

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1
    x=[0,0,1,0]
    test_data.append(x)

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1
    x=[0,1,0,0]
    test_data.append(x)

  else:
    Tama_4=Tama_4+1
    x=[1,0,0,0]
    test_data.append(x)
  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[2,0]=Tama_1
data[2,1]=Tama_2
data[2,2]=Tama_3
data[2,3]=Tama_4

print('-------------------------------- Tamaño extra grande -------------------------------------------')
# Ubicación de las fotos
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_small'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_mediano'
#URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Size\Test\Tomate_grande'
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  # Procesar Imagen Limon
  Imagen_mascara = segmentarLimon(URL_Imagen) #retorna 2 imagenes. Primera es binaria y sda es rgb.
  #Obtener ROI del limon:
  roi_limon = getROI(Imagen_mascara)

  # Obteniendo el diámetro en mm del limón 1:
  anchopx = roi_limon.shape[1]
  DIAMETRO_LIMON = anchopx / MM_PX
  DIAMETRO_LIMON = round(DIAMETRO_LIMON, 2)
  
  test_labels.append(int(0))

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    x=[0,0,0,1]
    test_data.append(x)

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1
    x=[0,0,1,0]
    test_data.append(x)

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1
    x=[0,1,0,0]
    test_data.append(x)

  else:
    Tama_4=Tama_4+1
    x=[1,0,0,0]
    test_data.append(x)

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[3,0]=Tama_1
data[3,1]=Tama_2
data[3,2]=Tama_3
data[3,3]=Tama_4

test_data=np.array(test_data)
test_labels=np.array(test_labels)
print(test_data.shape)
print(test_labels.shape)

y_test=to_categorical(test_labels) #Poner la información en vectores, es decir, en vez de aparecer el número 5 se posiciona un 1 en la posición 5 del arreglo de 10


classes=[1,2,3,4]
def plot_roc_curve_multiclass(y_true, y_score, classes):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)

    fpr = dict({0: [0]})
    tpr = dict({0: [0]})
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print(roc_auc[i])
        # plt.figure()
        # plt.plot(fpr[i], tpr[i], lw=2, label='AUC = %0.4f ' % (roc_auc[i]))

        # plt.plot([0, 1], [0, 1], 'k--', lw=2)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('Tasa de Falsos Positivos')
        # plt.ylabel('Tasa de Verdaderos Positivos')
        # plt.title('Curva ROC para la clase %s' % (classes[i]))
        # plt.legend(loc="lower right")
        # plt.show()      
    
    plt.figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.4f) for class %s' % (roc_auc[i], classes[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC multiclase')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve_multiclass(y_test, test_data, classes)