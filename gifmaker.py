from PIL import Image
import os

# Carpeta donde están las imágenes PNG
folder_path = "activation_graphs/model"
# Nombre del GIF que vas a crear
output_gif = "model.gif"

# Listar y ordenar los archivos PNG
images = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".png")]
images.sort()

# Abrir todas las imágenes y guardarlas en una lista
frames = [Image.open(image) for image in images]

# Guardar la secuencia de imágenes como un GIF animado
frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=500, loop=0)