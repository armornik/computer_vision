import os
from PIL import Image
from pylab import *


def get_image_list(folder):
    """Возвращает список имён всех jpg-файлов в каталоге."""
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.jpg')]


def im_resize(im, sz):
    """Изменить размер массива с помощью PIL."""
    pil2_im = Image.fromarray(uint8(im))
    return array(pil2_im.resize(sz))


def histeq(im, nbr_bins=256):
    """Выравнивание гистограммы полутонового изображения (нормировка яркости, контрастность)"""

    # получить гистограмму изображения
    im_hist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    cdf = im_hist.cumsum()  # Функция распределения
    cdf = 255 * cdf / cdf[-1]  # Нормировать

    # Использовать линейную интерполяцию cdf для нахождения значений новых пикселей
    # numpy flatten - преобразовать в одномерный массив
    im2 = interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf


def compute_average(im_list):
    """Вычислить среднее списка изображений (усреднение изображений)."""

    # Открыть первое изображение и преобразовать в массив типа float
    average_im = array(Image.open(im_list[0]), 'f')

    for im_name in im_list[1:]:
        try:
            average_im += array(Image.open(im_name))
        except:
            print(f'{im_name} ...пропущено')

    average_im /= len(im_list)

    # Вернуть среднее в виде массива значений типа uint8
    return array(average_im, 'uint8')


pil_im = array(Image.open('img//5.jpg').convert("L"))
my_im, cdf2 = histeq(pil_im)
my_im = Image.fromarray(uint8(my_im))
my_im.show()

my_list = get_image_list('img')
print(my_list)

pil_im = Image.open('img//4.jpg')
pil_im_black = Image.open('img//4.jpg').convert("L")  # сделать изображение полутоновым
# pil_im_save = pil_im_black.save('4_black.jpg')  # сохранить ч/б изображение
pil_mini = Image.open(r'C:\Users\Костя\PycharmProjects\computer_vision\img\4.jpg')
MAX_SIZE = (100, 100)
pil_mini.thumbnail(MAX_SIZE)  # создать миниатюру с сохранением пропорций с максимальной шириной или высотой 100 рх.
# pil_mini.save('4_mini.jpg')
# pil_mini.show()

# Обрезка (кадрирование) изображения методом crop
box = (100, 100, 400, 400)  # (левая, верхняя, правая, нижняя). Начало координат (0, 0) - в левом верхнем углу
region = pil_im.crop(box)
# region.show()
region = region.transpose(Image.ROTATE_180)  # Повернуть вырезанное изаброжение на 180 градусов
# region.show()
pil_im.paste(region, box)  # Вставить вырезанное изображение на место
# pil_im.show()
out = pil_im.resize((128, 128))  # Изменить размер до нужных размеров
# out.show()
out2 = pil_im.rotate(45)  # Повернуть изображение
out2.show()
