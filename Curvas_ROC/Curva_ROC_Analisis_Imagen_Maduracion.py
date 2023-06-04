# -*- coding: utf-8 -*-
"""
Created on Sat May 20 22:42:01 2023

@author: HP 690 -000B
"""

import time

start_time = time.time()

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
"""
Funciones de las máscaras de color
"""

def Mascara_Verde(URL):
    listImg = []
    image = cv2.imread(URL)

    #Cambiar de espectro de color 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # De BGR a RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #De RGB a HSV 

    # Elegimos el umbral de verde en HSV (Color, Saturación, Profundidad)
    umbral_bajo = np.array([34,50,50])
    umbral_alto = np.array([90,255,255])

    # Realizamos la máscara de la imagen
    mascara = cv2.inRange(image_hsv, umbral_bajo, umbral_alto)

    #Aplicar la máscara a la imagen original
    Imagen_resultante = cv2.bitwise_and(image, image, mask=mascara)

    #Guardamos los resultados de la máscara y la imagen final
    listImg.append(mascara)
    listImg.append(Imagen_resultante)
    
    return listImg

def Mascara_Amarillo(URL):
    listImg = []
    image = cv2.imread(URL)

    #Cambiar de espectro de color 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # De BGR a RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #De RGB a HSV 

    # Elegimos el umbral de amarillo en HSV (Color, Saturación, Profundidad)
    umbral_bajo = np.array([11,50,50])
    umbral_alto = np.array([33,255,255])

    # Realizamos la máscara de la imagen
    mascara = cv2.inRange(image_hsv, umbral_bajo, umbral_alto)

    #Aplicar la máscara a la imagen original
    Imagen_resultante = cv2.bitwise_and(image, image, mask=mascara)

    #Guardamos los resultados de la máscara y la imagen final
    listImg.append(mascara)
    listImg.append(Imagen_resultante)
    
    return listImg
def Mascara_Rojo(URL):
    listImg = []
    image = cv2.imread(URL)

    #Cambiar de espectro de color 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # De BGR a RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #De RGB a HSV 

    # Elegimos el umbral de rojo en HSV (Color, Saturación, Profundidad)
    rojoBajo1 = np.array([0, 50, 50], np.uint8)
    rojoAlto1 = np.array([10, 255, 255], np.uint8)
    rojoBajo2 = np.array([120, 50, 150], np.uint8)
    rojoAlto2 = np.array([180, 255, 255], np.uint8)

    # Realizamos la máscara de la imagen
    mascara_Rojo1 = cv2.inRange(image_hsv, rojoBajo1, rojoAlto1)
    mascara_Rojo2 = cv2.inRange(image_hsv, rojoBajo2, rojoAlto2)
    mascara_Rojo =  cv2.add(mascara_Rojo1, mascara_Rojo2)
    #mascara_Rojo = cv2.bitwise_or(mask1, mask2)

    #Aplicar la máscara a la imagen original
    Imagen_resultante = cv2.bitwise_and(image, image, mask=mascara_Rojo)

    #Guardamos los resultados de la máscara y la imagen final
    listImg.append(mascara_Rojo)
    listImg.append(Imagen_resultante)
    
    return listImg

def Contar_Area(Mascara):
    contorno1, _ = cv2.findContours(Mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total=0

    for i in range (np.size(contorno1,0)):
      cnt = contorno1[i]
      area = cv2.contourArea(cnt)
      area_total=area_total+area
    #Area del contorno/Area total de la imagen
    extension = float(area_total)/(np.size(Mascara,0)*np.size(Mascara,1))
        
    return extension



data=np.zeros((6,6))
test_data=[]
test_labels=[]

print('--------------------------------- Grado 1 -------------------------------------------------')

URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_1'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(0))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)


Error=((100-Grado_1)/100)*100

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
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_2'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(1))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)
      
Error=((100-Grado_2)/100)*100
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
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_3'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(2))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)
    
Error=((100-Grado_3)/100)*100
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
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_4'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(3))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)
      
Error=((100-Grado_4)/100)*100
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
URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_5'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(4))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)
    
Error=((100-Grado_5)/100)*100
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

URL='D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos_Totales\Maduracion\Test\Tomate_Grado_6'
Nombre_imagenes=os.listdir(URL)
Grado_1=0
Grado_2=0
Grado_3=0
Grado_4=0
Grado_5=0
Grado_6=0

imagen1=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_5\IMG_20230219_155527.jpg')
imagen2=cv2.imread('D:\OneDrive - Universidad de los Andes\GoodNotes\9-Noveno Semestre\Tesis IELE\Fotos\Maduracion\Test\Tomate_Grado_6\IMG_20230219_215415.jpg')

Cantidad_Verde_hist_dif=[]
Cantidad_Azul_hist_dif=[]
Cantidad_Rojo_hist_dif=[]

for Nombre_imagenes in Nombre_imagenes:
  #Ubicación de foto a foto
  URL_Imagen=URL+"/"+Nombre_imagenes 

  #Porcentaje de color Verde en cada foto
  Imagen_Final_verde=Mascara_Verde(URL_Imagen)
  Proporcion_Area_verde=Contar_Area(Imagen_Final_verde[0])

  #Porcentaje de color Amarillo en cada foto
  Imagen_Final_Amarillo=Mascara_Amarillo(URL_Imagen)
  Proporcion_Area_Amarilla=Contar_Area(Imagen_Final_Amarillo[0])

  #Porcentaje de color Rojo en cada foto
  Imagen_Final_Rojo=Mascara_Rojo(URL_Imagen)
  Proporcion_Area_Roja=Contar_Area(Imagen_Final_Rojo[0])
  test_labels.append(int(5))
  
  img = cv2.imread(URL_Imagen)
  if np.shape(img)[0]==3060:
      img=np.reshape(img,(4080,3060,3))
      
  difference1=cv2.absdiff(imagen1,img)
  Conv_hsv_Gray=cv2.cvtColor(difference1, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference1[mask != 255]=[0,0,255]
  
  difference2=cv2.absdiff(imagen2,img)
  Conv_hsv_Gray=cv2.cvtColor(difference2, cv2.COLOR_BGR2GRAY)
  ret, mask=cv2.threshold(Conv_hsv_Gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  difference2[mask != 255]=[0,0,255]
  
  color = ('b','g','r')
  can_color_dif_4=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([difference1], [i], None, [256], [0, 256])
      can_color_dif_4.append(max(hist))
      
  can_color_dif_5=[]
  for i, c in enumerate(color):
        hist = cv2.calcHist([difference2], [i], None, [256], [0, 256])
        can_color_dif_5.append(max(hist))
      
  

  color = ('b','g','r')
  can_color=[]
  for i, c in enumerate(color):
      hist = cv2.calcHist([img], [i], None, [256], [0, 256])
      can_color.append(max(hist))
      
  if Proporcion_Area_verde>0:
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
      else:
          Grado_1=Grado_1+1
          x=[1,0,0,0,0,0]
          test_data.append(x)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1
          x=[0,0,1,0,0,0]
          test_data.append(x)
      else:
          Grado_2=Grado_2+1
          x=[0,1,0,0,0,0]
          test_data.append(x)
  elif Proporcion_Area_Roja>0:
      if Proporcion_Area_Amarilla>0.0028:
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              x=[0,0,1,0,0,0]
              test_data.append(x)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1
              x=[0,0,0,1,0,0]
              test_data.append(x)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              x=[0,0,0,0,1,0]
              test_data.append(x)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_3=Grado_3+1
                  x=[0,0,1,0,0,0]
                  test_data.append(x)
      elif Proporcion_Area_Amarilla<0.0028:
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  x=[0,0,0,0,1,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  x=[0,0,0,1,0,0]
                  test_data.append(x)
              else:
                  Grado_6=Grado_6+1
                  x=[0,0,0,0,0,1]
                  test_data.append(x)
              
      else:
          Grado_6=Grado_6+1
          x=[0,0,0,0,0,1]
          test_data.append(x)
  else:
      Grado_3=Grado_3+1
      x=[0,0,1,0,0,0]
      test_data.append(x)
      
Error=((100-Grado_6)/100)*100
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

test_data=np.array(test_data)
test_labels=np.array(test_labels)
print(test_data.shape)
print(test_labels.shape)

y_test=to_categorical(test_labels) #Poner la información en vectores, es decir, en vez de aparecer el número 5 se posiciona un 1 en la posición 5 del arreglo de 10


classes=[1,2,3,4,5,6]
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