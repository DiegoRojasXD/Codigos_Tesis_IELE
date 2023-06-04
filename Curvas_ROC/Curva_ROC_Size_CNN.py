# -*- coding: utf-8 -*-
"""
Created on Sat May 20 21:23:00 2023

@author: da.rojass
"""

import os
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.utils import load_img, img_to_array
longitud=64
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
  x = load_img(URL_Imagen, target_size=(longitud, longitud))
  x = img_to_array(x)
  
  test_data.append(x)
  test_labels.append(int(0))
  
"""
--------------------------------- Tamaño grande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_grande'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  x = load_img(URL_Imagen, target_size=(longitud, longitud))
  x = img_to_array(x)
  
  test_data.append(x)
  test_labels.append(int(1))

"""
--------------------------------- Tamaño mediano -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_mediano'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  x = load_img(URL_Imagen, target_size=(longitud, longitud))
  x = img_to_array(x)
  
  test_data.append(x)
  test_labels.append(int(2))

"""
--------------------------------- Tamaño pequeño -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_small'
Nombre_imagenes=os.listdir(URL)
for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  x = load_img(URL_Imagen, target_size=(longitud, longitud))
  x = img_to_array(x)
  
  test_data.append(x)
  test_labels.append(int(3))
  
test_data=np.array(test_data)
test_labels=np.array(test_labels)
print(test_data.shape)
print(test_labels.shape)



y_test=to_categorical(test_labels) #Poner la información en vectores, es decir, en vez de aparecer el número 5 se posiciona un 1 en la posición 5 del arreglo de 10

modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\modelo.h5'
pesos_modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)
array = model.predict(test_data)

classes=[1,2,3,4]
def plot_roc_curve_multiclass(y_true, y_score, classes):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=classes)
    threshold = 0.5
    y_scores_adjusted = [1 if score >= threshold else 0 for score in y_score]

    fpr = dict({0: [0]})
    tpr = dict({0: [0]})
    roc_auc = dict()

    for i in range(n_classes):
        
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_adjusted[:, i])
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

plot_roc_curve_multiclass(y_test, array/2, classes)