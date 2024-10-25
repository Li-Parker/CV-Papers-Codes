import os
from PIL import Image
import pandas

lowest_common_multiple = 32


def crop_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    width_min = 1024
    height_min = 1024
    width_max = 1024
    height_max = 1024
    dic = []
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的文件格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size
            if width < height:
                temp = width
                width = height
                height = temp
            dic.append((width,height,filename))
    dic = pandas.DataFrame(dic)
    dic.to_csv('data_info_1.csv')
    return width_min, height_min, width_max, height_max






input_folder = './base'
output_folder = './process/base'
# crop_images(input_folder, output_folder)
input_folder = './extended'
output_folder = './process/extended'
print(crop_images(input_folder, output_folder))