import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

### A partir de una carpeta con cultivos buenos y otra con cultivos malos, este script
### busca usar dataAugmentation para convertir 1 imagen en 20 (o 'n' cantidad) imagenes para una CNN mas robusta

#pip install tensorflow keras

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

#buenos_dir = 'Cultivos_buen_estado'
#buenos_files = os.listdir(buenos_dir)
ungrown_dir = 'RadishVillage/grown'
ungrown_files = os.listdir(ungrown_dir)

for j, file in enumerate(ungrown_files, start=1):
    file = file.split('.')
    #print(file[0])
    nombre=file[0]

    img = load_img('RadishVillage/grown/'+nombre+'.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='RadishVillage/grown/', save_prefix=f'{nombre}_{i}', save_format='jpg'):
        i += 1
        if i >= 20:
            break 



"""


#buenos_dir = 'Cultivos_buen_estado'
#buenos_files = os.listdir(buenos_dir)
malos_dir = 'Cultivos_mal_estado'
malos_files = os.listdir(malos_dir)

for j, file in enumerate(malos_files, start=1):
    file = file.split('.')
    #print(file[0])
    nombre=file[0]

    img = load_img('Cultivos_mal_estado/'+nombre+'.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='preview', save_prefix=nombre, save_format='jpg'):
        i += 1
        if i >= 1:
            break 

            """