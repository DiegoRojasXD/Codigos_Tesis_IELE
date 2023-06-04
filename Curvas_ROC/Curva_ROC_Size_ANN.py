# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:53:20 2023

@author: da.rojass
"""

import os
import numpy as np
from PIL import Image
from skimage import color
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
longitud=19
clases=6

"""
--------------------------------- Data de entrenamiento -------------------------------------------------
"""

test_data=[]
test_labels=[]

"""
--------------------------------- Tamaño extragrande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  imagen=Image.open(URL_Imagen) #Abrir imagen
  imagen=imagen.resize([longitud,longitud])#Redimensionar imagen
  imgGray = color.rgb2gray(imagen)#Convertir imagen a blanco y negro
  imagen_arreglo=np.asarray(imgGray) #Convertir imagen en vector 
  #print(imagen_arreglo)
  #imagen_original=PIL.Image.fromarray(np.uint8(imgGray)) #Convertir vector en imagen
  #Image.Image.show(imagen_original)#mostrar imagen 
  
  test_data.append(imagen_arreglo)
  test_labels.append(int(0))

"""
--------------------------------- Tamaño grande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_grande'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  imagen=Image.open(URL_Imagen) #Abrir imagen
  imagen=imagen.resize([longitud,longitud])#Redimensionar imagen
  imgGray = color.rgb2gray(imagen)#Convertir imagen a blanco y negro
  imagen_arreglo=np.asarray(imgGray) #Convertir imagen en vector 
  #print(imagen_arreglo)
  #imagen_original=PIL.Image.fromarray(np.uint8(imgGray)) #Convertir vector en imagen
  #Image.Image.show(imagen_original)#mostrar imagen 
  
  test_data.append(imagen_arreglo)
  test_labels.append(int(1))

"""
--------------------------------- Tamaño mediano -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_mediano'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  imagen=Image.open(URL_Imagen) #Abrir imagen
  imagen=imagen.resize([longitud,longitud])#Redimensionar imagen
  imgGray = color.rgb2gray(imagen)#Convertir imagen a blanco y negro
  imagen_arreglo=np.asarray(imgGray) #Convertir imagen en vector 
  #print(imagen_arreglo)
  #imagen_original=PIL.Image.fromarray(np.uint8(imgGray)) #Convertir vector en imagen
  #Image.Image.show(imagen_original)#mostrar imagen 
  
  test_data.append(imagen_arreglo)
  test_labels.append(int(2))

"""
--------------------------------- Tamaño pequeño -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_small'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  imagen=Image.open(URL_Imagen) #Abrir imagen
  imagen=imagen.resize([longitud,longitud])#Redimensionar imagen
  imgGray = color.rgb2gray(imagen)#Convertir imagen a blanco y negro
  imagen_arreglo=np.asarray(imgGray) #Convertir imagen en vector 
  #print(imagen_arreglo)
  #imagen_original=PIL.Image.fromarray(np.uint8(imgGray)) #Convertir vector en imagen
  #Image.Image.show(imagen_original)#mostrar imagen 
  
  test_data.append(imagen_arreglo)
  test_labels.append(int(3))


test_data=np.array(test_data)
test_labels=np.array(test_labels)
print(test_data.shape)
print(test_labels.shape)

x_test=test_data.reshape((400,longitud*longitud))#Cambio de dimensiones de la data de 3D a 2D


y_test=to_categorical(test_labels) #Poner la información en vectores, es decir, en vez de aparecer el número 5 se posiciona un 1 en la posición 5 del arreglo de 10

modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_ANN\modelo.h5'
pesos_modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_ANN\pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)
array = model.predict(x_test)

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

plot_roc_curve_multiclass(y_test, array, classes)