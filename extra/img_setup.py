import cv2
import numpy as np

### Al meter una imagen en este script lo convierte en una imagen compatible con mi modelo CNN
### La imagen que tomo con el celular tiene resolucion y tamaño diferente a las del dataset
### El objetivo era que fueran lo mas parecidas posibles para una mejor deteccion

def seleccionar_area(imagen):
    # Mostrar la imagen original
    cv2.namedWindow("Imagen Original", cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen Original", imagen)

    # Permitir al usuario seleccionar un area rectangular
    r = cv2.selectROI("Imagen Original", imagen)

    # Extraer las coordenadas del area seleccionada
    x, y, w, h = r

    # Recortar la imagen al area seleccionada
    area_seleccionada = imagen[y:y+h, x:x+w]

    return area_seleccionada

def ajustar_tamano_resolucion(imagen, tamano, resolucion, profundidad_bits):
    # Redimensionar la imagen al tamaño deseado
    imagen_redimensionada = cv2.resize(imagen, tamano)

    # Ajustar la resolucion de la imagen
    altura, ancho = imagen_redimensionada.shape[:2]
    imagen_ajustada = cv2.resize(imagen_redimensionada, (int(ancho * resolucion / 96), int(altura * resolucion / 96)), interpolation=cv2.INTER_AREA)

    """
    # Convertir la imagen a la profundidad de bits deseada
    if profundidad_bits == 24:
        imagen_ajustada = cv2.cvtColor(imagen_ajustada, cv2.COLOR_BGR2BGR)
    """

    return imagen_ajustada

# Cargar la imagen original
imagen_original = cv2.imread("radish/img4.jpeg")

# Seleccionar un area de la imagen
area_seleccionada = seleccionar_area(imagen_original)

# Ajustar el tamaño y la resolucion del area seleccionada
area_ajustada = ajustar_tamano_resolucion(area_seleccionada, (256, 256), 96, 24)

# Guardar la imagen ajustada en formato JPEG
cv2.imwrite("radish/img_adj4.jpg", area_ajustada)


