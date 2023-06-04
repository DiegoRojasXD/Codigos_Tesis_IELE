# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:59:23 2023

@author: HP 690 -000B
"""
import time

start_time = time.time()

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import psutil
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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)


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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)
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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)
    
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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)
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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)
    
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
      
  #print('--------------')
  if Proporcion_Area_verde>0:
      #print(1)
      if round(Proporcion_Area_verde,3)<0.002:
          Grado_2=Grado_2+1
          #print(2)
      else:
          Grado_1=Grado_1+1
          #print(3)
  elif round(Proporcion_Area_Roja,3)<0.004 and round(Proporcion_Area_verde,3)==0:
      #print(4)
      if can_color[0] < 416000 and Proporcion_Area_Amarilla<0.062:
          Grado_3=Grado_3+1 
          #print(5)
      else:
          Grado_2=Grado_2+1
          #print(6)
  elif Proporcion_Area_Roja>0:
      #print(7)
      if Proporcion_Area_Amarilla>0.0028:
          #print(8)
          if can_color[0] < 416000:
              Grado_3=Grado_3+1 
              #print(9)
          elif can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
              Grado_4=Grado_4+1  
              #print(10)
          elif can_color_dif_4[0]<=652062 and can_color_dif_4[1]<=703855 and can_color_dif_4[2]<=614544:
              Grado_5=Grado_5+1
              #print(15)
          else:                  
              if can_color_dif_4[1]<3356013:
                  Grado_4=Grado_4+1 
              else:
                  Grado_3=Grado_3+1 
              #print(11)
      elif Proporcion_Area_Amarilla<0.0028:
          #print(12)
          if Proporcion_Area_Roja<0.03 and can_color[2]>460000 :
              #print(13)
              if can_color[0]<449835 and round(Proporcion_Area_Roja,3)<0.017 and round(Proporcion_Area_Roja,3)>0.003:
                  Grado_4=Grado_4+1
                  #print(14)
                  
              elif can_color_dif_5[0]<2901535 and can_color_dif_5[1]<2880870 and can_color_dif_5[2]<2778247:
                  Grado_5=Grado_5+1
                  #print(16)
              else:
                  Grado_6=Grado_6+1
                  #print(17)
          else:
              if can_color[0]<418000 and can_color[0]>713000:
                  Grado_4=Grado_4+1
                  #print(18)
              else:
                  Grado_6=Grado_6+1
                  #print(19)
              
      else:
          Grado_6=Grado_6+1
          #print(20)
  else:
      Grado_3=Grado_3+1
      #print(21)
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


"""
Evaluación modelo total
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



pre_to=sum_pre_to/(60*6)

print("El sensibilidad del modelo es: "+str(sensibilidad))
print("El especificidad del modelo es: "+str(especificidad))
print("La exactitud del modelo es: "+str(exactitud))
print("La precision del modelo es: "+str(precision))

sen_to=sum(VP)/(sum(VP)+sum(FN))
esp_to=sum(VN)/(sum(VN)+sum(FP))
exa_to=(sum(VP)+sum(VN))/(sum(VP)+sum(FN)+sum(VN)+sum(FP))
pre_to=sum(VP)/(sum(VP)+sum(FP))

print("El sensibilidad total del modelo es: "+str(sen_to))
print("El especificidad total del modelo es: "+str(esp_to))
print("La exactitud total del modelo es: "+str(exa_to))
print("La precision total del modelo es: "+str(pre_to))


print("La precision general del modelo es: "+str(pre_to))

plt.stem(especificidad,sensibilidad)
plt.title('Curva ROC')
plt.xlabel('Especificidad')
plt.ylabel('Sensibilidad')
plt.show()


x=[0,0.2,0.4,0.6,0.8,1]
sensibilidad.sort()
plt.stem(x,sensibilidad)
plt.plot(x,x)
plt.title('Curva ROC')
plt.xlabel('Especificidad')
plt.ylabel('Sensibilidad')
plt.show()


sensibilidad.sort()
especificidad.sort()    
plt.stem(especificidad,sensibilidad)
plt.title('Curva ROC')
plt.xlabel('Especificidad')
plt.ylabel('Sensibilidad')
plt.show()


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
plt.title('Mapa de calor del modelo CNN')
plt.show()

end_time = time.time()
execution_time = end_time - start_time

print("Tiempo de ejecución:", execution_time, "segundos")
process = psutil.Process(os.getpid())
print("Memoria RAM usada en MB:", process.memory_info().rss / 1024 / 1024)
process = psutil.Process()
print("Número de núcleos de procesador utilizados:", len(process.cpu_affinity()))
