from PIL import Image
from pylab import *
from pca import pca
from imtools import get_image_list
import pickle

im_list = get_image_list("a_thumbs")
print(im_list[0])

im = array(Image.open((im_list[0])))  # открыть одно изображение для получения размера

m, n = im.shape[0:2]  # Получить размер изображения
im_nbr = len(im_list)  # Получить число изображений

# Создать матрицу для хранения всех линеаризованных изображений
im_matrix = array([array(Image.open(im)).flatten() for im in im_list], 'f')

# Выполнить метод главных компонент
V, S, im_mean = pca(im_matrix)

# Показать несколько изображений (среднее и первые 7 мод)
figure()
gray()
subplot(2, 4, 1)
imshow(im_mean.reshape(m, n))
for i in range(7):
    subplot(2, 4, i + 2)
    imshow(V[i].reshape(m, n))

show()

# Сохранить среднее изображение и главные компоненты (сериализация)
with open('font_pca_model.pkl', 'wb') as f:
    pickle.dump(im_mean, f)
    pickle.dump(V, f)


# Загрузить среднее изображение и главные компоненты (сериализация)
with open('font_pca_model.pkl', 'rb') as f:
    im_mean = pickle.load(f)
    V = pickle.load(f)
