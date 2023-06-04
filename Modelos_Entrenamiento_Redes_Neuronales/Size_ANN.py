# -*- coding: utf-8 -*-
"""
Created on Fri May 19 18:31:03 2023

@author: Usuario
"""

import time
start_time = time.time()
import os
import numpy as np
from PIL import Image
from skimage import color
from keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf
import psutil

longitud=19
clases=4

    
"""
--------------------------------- Data de entrenamiento -------------------------------------------------
"""

train_data=[]
train_labels=[]

"""
--------------------------------- Tamaño extragrande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Entrenamiento\Tomate_extra_grande'
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
  
  train_data.append(imagen_arreglo)
  train_labels.append(int(0))

"""
--------------------------------- Tamaño grande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Entrenamiento\Tomate_grande'
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
  
  train_data.append(imagen_arreglo)
  train_labels.append(int(1))

"""
--------------------------------- Tamaño mediano -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Entrenamiento\Tomate_mediano'
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
  
  train_data.append(imagen_arreglo)
  train_labels.append(int(2))

"""
--------------------------------- Tamaño pequeño -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Entrenamiento\Tomate_small'
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
  
  train_data.append(imagen_arreglo)
  train_labels.append(int(3))

train_data=np.array(train_data)
train_labels=np.array(train_labels)
print(train_data.shape)
print(train_labels.shape)
"""
--------------------------------- Data de validación -------------------------------------------------
"""

validacion_data=[]
validacion_labels=[]

"""
--------------------------------- Tamaño extragrande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Validacion\Tomate_extra_grande'
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
  
  validacion_data.append(imagen_arreglo)
  validacion_labels.append(int(0))

"""
--------------------------------- Tamaño grande -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Validacion\Tomate_grande'
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
  
  validacion_data.append(imagen_arreglo)
  validacion_labels.append(int(1))

"""
--------------------------------- Tamaño mediano -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Validacion\Tomate_mediano'
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
  
  validacion_data.append(imagen_arreglo)
  validacion_labels.append(int(2))

"""
--------------------------------- Tamaño pequeño -------------------------------------------------
"""
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Validacion\Tomate_small'
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
  
  validacion_data.append(imagen_arreglo)
  validacion_labels.append(int(3))


  
validacion_data=np.array(validacion_data)
validacion_labels=np.array(validacion_labels)
print(validacion_data.shape)
print(validacion_labels.shape)

x_train=train_data.reshape((3200,longitud*longitud))#Cambio de dimensiones de la data de 3D a 2D
x_validacion=validacion_data.reshape((400,longitud*longitud))#Cambio de dimensiones de la data de 3D a 2D


y_train=to_categorical(train_labels) #Poner la información en vectores, es decir, en vez de aparecer el número 5 se posiciona un 1 en la posición 5 del arreglo de 10
y_validacion=to_categorical(validacion_labels)


model = models.Sequential() # Crear red neuronal base
model.add(layers.Dense(
    units=512, 
    input_shape=(longitud*longitud,)
))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(1024, activation='relu'))    
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))      
model.add(layers.Dropout(0.375))
model.add(layers.Dense(clases,activation='softmax')) #10 neuronas porque son las 10 salodas de los numeros de 0 a 9
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(0.0001),
    metrics=[
            tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
            tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
            tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None),
            tf.keras.metrics.Accuracy()
        ]) 

model.fit(x_train,y_train,epochs=20,batch_size=128,
          validation_data = (x_validacion, y_validacion)) #Entrenar la red con 5 interaciones  y lotes de 128 unidades

# -*- coding: utf-8 -*-
target_dir = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_ANN'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
model.save('E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_ANN\modelo.h5')
model.save_weights('E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_ANN\pesos.h5')

# hacemos las grafica 2 perdidas
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Perdidas del modelo')
plt.ylabel('pérdidas')
plt.xlabel('épocas')
plt.legend(['Entrenamiento', 'test'], loc='upper rigth')
plt.show()

end_time = time.time()
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")
process = psutil.Process(os.getpid())
print("Memoria RAM usada en MB:", process.memory_info().rss / 1024 / 1024)
process = psutil.Process()
print("Número de núcleos de procesador utilizados:", len(process.cpu_affinity()))  