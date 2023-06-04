# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:22:27 2023

@author: HP 690 -000B
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

MM_PX =p[f]/146
data=np.zeros((4,4))



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

#Cantidad_Verde=[]
#Cantidad_Amarillo=[]
#Cantidad_Rojo=[]
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

  #show(roi_limon, "ROI limón 1 - "+str(roi_limon.shape))
  #print("El ancho del cuadro que contiene el tomate es de : "+str(anchopx)+"px") 
  #print("Diámetro estimado tomate 1: "+str(DIAMETRO_LIMON)+"mm")
  #print("Diámetro tomate real: 95,8mm")

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    #print("Es un tomate de tamaño pequeño")

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1

    #print("Es un tomate de tamaño mediano")

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1

    #print("Es un tomate de tamaño grande")

  else:
    Tama_4=Tama_4+1

    #print("Es un tomate de tamaño extra grande")

  
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

#Cantidad_Verde=[]
#Cantidad_Amarillo=[]
#Cantidad_Rojo=[]
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

  #show(roi_limon, "ROI limón 1 - "+str(roi_limon.shape))
  #print("El ancho del cuadro que contiene el tomate es de : "+str(anchopx)+"px") 
  #print("Diámetro estimado tomate 1: "+str(DIAMETRO_LIMON)+"mm")
  #print("Diámetro tomate real: 95,8mm")

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    #print("Es un tomate de tamaño pequeño")

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1

    #print("Es un tomate de tamaño mediano")

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1

    #print("Es un tomate de tamaño grande")

  else:
    Tama_4=Tama_4+1

    #print("Es un tomate de tamaño extra grande")

  
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

#Cantidad_Verde=[]
#Cantidad_Amarillo=[]
#Cantidad_Rojo=[]
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

  #show(roi_limon, "ROI limón 1 - "+str(roi_limon.shape))
  #print("El ancho del cuadro que contiene el tomate es de : "+str(anchopx)+"px") 
  #print("Diámetro estimado tomate 1: "+str(DIAMETRO_LIMON)+"mm")
  #print("Diámetro tomate real: 95,8mm")

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    #print("Es un tomate de tamaño pequeño")

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1

    #print("Es un tomate de tamaño mediano")

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1

    #print("Es un tomate de tamaño grande")

  else:
    Tama_4=Tama_4+1

    #print("Es un tomate de tamaño extra grande")

  
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

#Cantidad_Verde=[]
#Cantidad_Amarillo=[]
#Cantidad_Rojo=[]
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

  #show(roi_limon, "ROI limón 1 - "+str(roi_limon.shape))
  #print("El ancho del cuadro que contiene el tomate es de : "+str(anchopx)+"px") 
  #print("Diámetro estimado tomate 1: "+str(DIAMETRO_LIMON)+"mm")
  #print("Diámetro tomate real: 95,8mm")

  if DIAMETRO_LIMON<59:
    Tama_1=Tama_1+1
    #print("Es un tomate de tamaño pequeño")

  elif DIAMETRO_LIMON>=59 and DIAMETRO_LIMON<64:
    Tama_2=Tama_2+1

    #print("Es un tomate de tamaño mediano")

  elif DIAMETRO_LIMON>=64 and DIAMETRO_LIMON<71:
    Tama_3=Tama_3+1

    #print("Es un tomate de tamaño grande")

  else:
    Tama_4=Tama_4+1

    #print("Es un tomate de tamaño extra grande")

  
print("El número de tomates de tamaño pequeño: "+str(Tama_1))
print("El número de tomates de tamaño mediano: "+str(Tama_2))
print("El número de tomates de tamaño grande: "+str(Tama_3))
print("El número de tomates de tamaño extra grande: "+str(Tama_4))

data[3,0]=Tama_1
data[3,1]=Tama_2
data[3,2]=Tama_3
data[3,3]=Tama_4

"""
Evaluacion
"""

print("----------------- Metricas por clase---------------")

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
    
    sum_pre_to=sum_pre_to+VP[l]


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
plt.title('Mapa de calor del modelo por análisis de imagen con f de: '+str(f)+'y precisión: '+str(pre_to))
plt.show()