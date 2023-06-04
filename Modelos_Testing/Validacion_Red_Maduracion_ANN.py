# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:27:17 2023

@author: da.rojass
"""
import time
start_time = time.time()
import os
import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
import matplotlib
from keras.models import load_model
import psutil

longitud=32
modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Maduracion_final_ANN\modelo.h5'
pesos_modelo = 'E:\Tesis_IMEC_Diego\Tesis_IELE\Modelos finales\modelo_Maduracion_final_ANN\pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)

def predict(file):
  imagen=Image.open(file) #Abrir imagen
  imagen=imagen.resize([longitud,longitud])#Redimensionar imagen
  imgGray = color.rgb2gray(imagen)#Convertir imagen a blanco y negro
  imagen_arreglo=np.asarray(imgGray) #Convertir imagen en vector  
  x_test=imagen_arreglo.reshape((1,longitud*longitud))#Cambio de dimensiones de la data de 3D a 2D  
  array = model.predict(x_test)
  result = array[0]
  answer = np.argmax(result)
  return answer

data=np.zeros((6,6))

"""
--------------------------------- Data de test -------------------------------------------------
"""

data=np.zeros((6,6))


print('--------------------------------- Grado 1 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_1'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_1)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[0,0]=Grado_1
data[0,1]=Grado_2
data[0,2]=Grado_3
data[0,3]=Grado_4
data[0,4]=Grado_5
data[0,5]=Grado_6

print('--------------------------------- Grado 2 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_2'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_2)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[1,0]=Grado_1
data[1,1]=Grado_2
data[1,2]=Grado_3
data[1,3]=Grado_4
data[1,4]=Grado_5
data[1,5]=Grado_6

print('--------------------------------- Grado 3 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_3'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_3)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[2,0]=Grado_1
data[2,1]=Grado_2
data[2,2]=Grado_3
data[2,3]=Grado_4
data[2,4]=Grado_5
data[2,5]=Grado_6

print('--------------------------------- Grado 4 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_4'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_4)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[3,0]=Grado_1
data[3,1]=Grado_2
data[3,2]=Grado_3
data[3,3]=Grado_4
data[3,4]=Grado_5
data[3,5]=Grado_6

print('--------------------------------- Grado 5 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_5'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_5)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[4,0]=Grado_1
data[4,1]=Grado_2
data[4,2]=Grado_3
data[4,3]=Grado_4
data[4,4]=Grado_5
data[4,5]=Grado_6

print('--------------------------------- Grado 6 -------------------------------------------------')

URL='E:\Tesis_IMEC_Diego\Tesis_IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_6'
Nombre_imagenes=os.listdir(URL)

Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

for Nombre_imagenes in Nombre_imagenes:
  URL_Imagen=URL+"/"+Nombre_imagenes
  answer=predict(URL_Imagen)

  if answer == 0:
    Grado_1=Grado_1+1
  elif answer == 1:
    Grado_2=Grado_2+1
  elif answer == 2:
    Grado_3=Grado_3+1
  elif answer == 3:
    Grado_4=Grado_4+1
  elif answer == 4:
    Grado_5=Grado_5+1
  elif answer == 5:
    Grado_6=Grado_6+1
    
Error=((60-Grado_6)/60)*100

print('La cantidad de Imagenes clasificadas como Grado 1 son: '+ str(Grado_1))
print('La cantidad de Imagenes clasificadas como Grado 2 son: '+ str(Grado_2))
print('La cantidad de Imagenes clasificadas como Grado 3 son: '+ str(Grado_3))
print('La cantidad de Imagenes clasificadas como Grado 4 son: '+ str(Grado_4))
print('La cantidad de Imagenes clasificadas como Grado 5 son: '+ str(Grado_5))
print('La cantidad de Imagenes clasificadas como Grado 6 son: '+ str(Grado_6))
print(str(Error)+' %')

data[5,0]=Grado_1
data[5,1]=Grado_2
data[5,2]=Grado_3
data[5,3]=Grado_4
data[5,4]=Grado_5
data[5,5]=Grado_6

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
        sum_2=sum_2+data[k,i]
    sum_2=sum_2-data[k,k]
    
    FP.append(sum_2)
    
FN=[]
for k in range(len(data)):
    sum_3=0
    for i in range(len(data[:, k])):
        sum_3=sum_3+data[k,i]
    sum_3=sum_3-data[k,k]
    
    FN.append(sum_3)
    
sensibilidad=[]
especificidad=[]
exactitud=[]
precision=[]
sum_pre_to=0
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

sen_to=np.sum(VP)/(np.sum(VP)+np.sum(FN))
esp_to=np.sum(VN)/(np.sum(VN)+np.sum(FP))
exa_to=(np.sum(VP)+np.sum(VN))/(np.sum(VP)+np.sum(FN)+np.sum(VN)+np.sum(FP))
pre_to=np.sum(VP)/(np.sum(VP)+np.sum(FP))

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
labs = ["G1", "G2", "G3", "G4",
         "G5", "G6"]
         
# Mapa de calor
fig, ax = plt.subplots()
im, cbar = heatmap(data, row_labels = labs, col_labels = labs,
                   ax = ax, cmap = "YlGn", cbarlabel = "Cantidad de Imagenes")
texts = annotate_heatmap(im, valfmt = "{x:.1f}")
plt.title('Mapa de calor del modelo ANN')
plt.show()

print("----------------- recursos computacionales---------------")
end_time = time.time()
execution_time = end_time - start_time
print("Tiempo de ejecución:", execution_time, "segundos")
process = psutil.Process(os.getpid())
print("Memoria RAM usada en MB:", process.memory_info().rss / 1024 / 1024)
process = psutil.Process()
print("Número de núcleos de procesador utilizados:", len(process.cpu_affinity()))