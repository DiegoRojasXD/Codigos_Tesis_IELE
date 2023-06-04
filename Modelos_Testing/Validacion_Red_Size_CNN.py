# -*- coding: utf-8 -*-
"""
Created on Fri May 19 19:30:53 2023

@author: da.rojass
"""
import time
start_time = time.time()
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.utils import load_img, img_to_array
from keras.models import load_model
import psutil

longitud, altura = 64, 64
modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\modelo.h5'
pesos_modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Size_final_CNN\pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  return answer

data=np.zeros((4,4))


print('--------------------------------- Tamaño pequeño -------------------------------------------------')
# Ubicación de las fotos
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_small'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  answer=predict(URL_Imagen)
  if answer == 0:
    Tama_1=Tama_1+1
  elif answer == 1:
    Tama_2=Tama_2+1
  elif answer == 2:
    Tama_3=Tama_3+1
  elif answer == 3:
    Tama_4=Tama_4+1  
    
Error=((60-Tama_4)/60)*100

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[0,0]=Tama_4
data[0,1]=Tama_3
data[0,2]=Tama_2
data[0,3]=Tama_1

print('--------------------------------- Tamaño mediano -------------------------------------------------')
# Ubicación de las fotos
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_mediano'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  answer=predict(URL_Imagen)

  if answer == 0:
    Tama_1=Tama_1+1
  elif answer == 1:
    Tama_2=Tama_2+1
  elif answer == 2:
    Tama_3=Tama_3+1
  elif answer == 3:
    Tama_4=Tama_4+1
  
Error=((60-Tama_3)/60)*100
  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[1,0]=Tama_4
data[1,1]=Tama_3
data[1,2]=Tama_2
data[1,3]=Tama_1
print('--------------------------------- Tamaño grande -------------------------------------------------')
# Ubicación de las fotos
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  answer=predict(URL_Imagen)

  if answer == 0:
    Tama_1=Tama_1+1
  elif answer == 1:
    Tama_2=Tama_2+1
  elif answer == 2:
    Tama_3=Tama_3+1
  elif answer == 3:
    Tama_4=Tama_4+1
  
Error=((60-Tama_2)/60)*100
  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[2,0]=Tama_4
data[2,1]=Tama_3
data[2,2]=Tama_2
data[2,3]=Tama_1

print('-------------------------------- Tamaño extra grande -------------------------------------------')
# Ubicación de las fotos
URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Size\Test\Tomate_extra_grande'
Nombre_imagenes=os.listdir(URL)

Tama_1=0
Tama_2=0
Tama_3=0
Tama_4=0

#Cantidad_Verde=[]
#Cantidad_Amarillo=[]
#Cantidad_Rojo=[]
for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 
  answer=predict(URL_Imagen)

  if answer == 0:
    Tama_1=Tama_1+1
  elif answer == 1:
    Tama_2=Tama_2+1
  elif answer == 2:
    Tama_3=Tama_3+1
  elif answer == 3:
    Tama_4=Tama_4+1
  
Error=((60-Tama_1)/60)*100

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[3,0]=Tama_4
data[3,1]=Tama_3
data[3,2]=Tama_2
data[3,3]=Tama_1

"""
Evaluacion
"""
VP=[]
suma_total=0
for i in range(len(data)):
    for j in range(len(data)):
        suma_total=suma_total+data[i,j]
        if i==j:
            VP.append(data[i,j])
VN=[]
for k in range(len(data)):
    sum_1=0
    for i in range(len(data[k, :])):
        sum_1=sum_1+data[k,i]
        
    for i in range(len(data[:, k])):
        sum_1=sum_1+data[i,k]
    sum_1=sum_1-data[k,k]
    
    VN.append(suma_total-sum_1)
    
FP=[]
for k in range(len(data)):
    sum_2=0
    for i in range(len(data[k, :])):
        sum_2=sum_2+data[i,k]
    sum_2=sum_2-data[k,k]
    
    FP.append(sum_2)
    
FN=[]
for k in range(len(data)):
    sum_3=0
    for i in range(len(data[:, k])):
        sum_3=sum_3+data[k,i]
    sum_3=sum_3-data[k,k]
    
    FN.append(sum_3)
    
print("----------------- Metricas por clase---------------")    
sensibilidad=[]
especificidad=[]
exactitud=[]
precision=[]

for l in range(len(VP)):
    sen=VP[l]/(VP[l]+FN[l])
    esp=VN[l]/(VN[l]+FP[l])
    exa=(VP[l]+VN[l])/(VP[l]+FN[l]+VN[l]+FP[l])
    pre=VP[l]/(VP[l]+FP[l])
    
    sensibilidad.append(sen)
    especificidad.append(esp)
    exactitud.append(exa)
    precision.append(pre)
    
print("El sensibilidad del modelo es: "+str(sensibilidad))
print("El especificidad del modelo es: "+str(especificidad))
print("La exactitud del modelo es: "+str(exactitud))
print("La precision del modelo es: "+str(precision))


print("----------------- Metricas totales---------------")

sen_to=sum(VP)/(sum(VP)+sum(FN))
esp_to=sum(VN)/(sum(VN)+sum(FP))
exa_to=(sum(VP)+sum(VN))/(sum(VP)+sum(FN)+sum(VN)+sum(FP))
pre_to=sum(VP)/(sum(VP)+sum(FP))

print("El sensibilidad total del modelo es: "+str(sen_to))
print("El especificidad total del modelo es: "+str(esp_to))
print("La exactitud total del modelo es: "+str(exa_to))
print("La precision total del modelo es: "+str(pre_to))

#Mapa de Calor
def heatmap(data, row_labels, col_labels, ax = None,
            cbar_kw = {}, cbarlabel = "", **kwargs):

    if not ax:
      ax = plt.gca()

    im = ax.imshow(data, **kwargs)

    cbar = ax.figure.colorbar(im, ax = ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation = -90, va = "bottom")

    ax.set_xticks(np.arange(data.shape[1]), labels = col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels = row_labels)

    ax.tick_params(top = True, bottom = False,
                   labeltop = True, labelbottom = False)

    plt.setp(ax.get_xticklabels(), rotation = -30, ha = "right",
             rotation_mode = "anchor")

    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor = True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor = True)
    ax.grid(which = "minor", color = "w", linestyle = '-', linewidth = 3)
    ax.tick_params(which =  "minor", bottom = False, left = False)

    return im, cbar


def annotate_heatmap(im, data = None, valfmt="{x:.2f}",
                     textcolors = ("black", "white"),
                     threshold = None, **textkw):

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    kw = dict(horizontalalignment = "center",
              verticalalignment = "center")
    kw.update(textkw)

    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# Etiquetas
labs = ["TP", "TM", "TG", "TEG"]
         
# Mapa de calor
fig, ax = plt.subplots()
im, cbar = heatmap(data, row_labels = labs, col_labels = labs,
                   ax = ax, cmap = "YlGn", cbarlabel = "Cantidad de Imagenes")
texts = annotate_heatmap(im, valfmt = "{x:.1f}")
plt.title('Mapa de calor del modelo CNN')
plt.show()

print("----------------- recursos computacionales---------------")
end_time = time.time()
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")
process = psutil.Process(os.getpid())
print("Memoria RAM usada en MB:", process.memory_info().rss / 1024 / 1024)
process = psutil.Process()
print("Número de núcleos de procesador utilizados:", len(process.cpu_affinity()))