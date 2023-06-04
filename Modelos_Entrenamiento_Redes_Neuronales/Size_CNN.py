# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:05:01 2023

@author: da.rojass
"""

import time

start_time = time.time()
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.layers import  Convolution2D, MaxPooling2D
import psutil

"""
Datos
"""
data_entrenamiento = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Entrenamiento'
data_validacion = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Validacion'

"""
Parametros
"""
epocas=20
longitud, altura = 112,112
batch_size = 256
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 4
lr = 0.01

"""
Preparamos nuestras imagenes
"""

entrenamiento_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
        data_entrenamiento,
        target_size=(altura, longitud),
        batch_size=batch_size,
        class_mode='categorical')

validacion_generador = test_datagen.flow_from_directory(
        data_validacion,
        target_size=(altura, longitud),
        batch_size=batch_size,
        class_mode='categorical')



"""
Red Neuronal Convolucional
"""
cnn = Sequential()
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))
cnn.add(MaxPooling2D(pool_size=tamano_pool))
cnn.add(Flatten())
cnn.add(Dense(2048, activation='relu'))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.375))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(optimizer='Adam',
loss='categorical_crossentropy',
metrics=[
        tf.keras.metrics.Precision(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
        tf.keras.metrics.Recall(thresholds=None, top_k=None, class_id=None, name=None, dtype=None),
        tf.keras.metrics.TruePositives(thresholds=None, name=None, dtype=None),
        tf.keras.metrics.Accuracy()
    ]) #Compilar la red con un optimizador, función de pérdida y una metrica de éxito

cnn.fit_generator(
    entrenamiento_generador,
    epochs=epocas,
    validation_data=validacion_generador)


"""
Guardar Datos
"""
target_dir = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN'
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\modelo.h5')
cnn.save_weights('E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\pesos.h5')

# hacemos las grafica 2 perdidas
plt.plot(cnn.history.history['loss'])
plt.plot(cnn.history.history['val_loss'])
plt.title('Perdidas del modelo')
plt.ylabel('pérdidas')
plt.xlabel('épocas')
plt.legend(['Entrenamiento', 'test'], loc='upper right')
plt.show()
 
end_time = time.time()
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")
process = psutil.Process(os.getpid())
print("Memoria RAM usada en MB:", process.memory_info().rss / 1024 / 1024)
process = psutil.Process()
print("Número de núcleos de procesador utilizados:", len(process.cpu_affinity()))