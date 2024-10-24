import os
from PIL import Image

lowest_common_multiple = 32


def crop_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的文件格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size

            if width == 1024:
                # 根据宽度裁剪
                new_width = width
                new_height = height - (height % lowest_common_multiple)  # 确保高度能被lowest_common_multiple整除
                img_cropped = img.crop((0, 0, new_width, new_height))
            elif height == 1024:
                # 根据高度裁剪
                new_height = height
                new_width = width - (width % lowest_common_multiple)  # 确保宽度能被lowest_common_multiple整除
                img_cropped = img.crop((0, 0, new_width, new_height))
            else:
                continue  # 不处理不满足条件的图像

            # 保存裁剪后的图像
            img_cropped.save(os.path.join(output_folder, filename))




def crop_images_item(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的文件格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size

            if width == 1024:
                # 根据宽度裁剪
                new_width = width
                new_height = height - (height % lowest_common_multiple)  # 确保高度能被lowest_common_multiple整除
                img_cropped = img.crop((0, 0, new_width, new_height))
            elif height == 1024:
                # 根据高度裁剪
                new_height = height
                new_width = width - (width % lowest_common_multiple)  # 确保宽度能被lowest_common_multiple整除
                img_cropped = img.crop((0, 0, new_width, new_height))
            else:
                continue  # 不处理不满足条件的图像

            # 保存裁剪后的图像
            img_cropped.save(os.path.join(output_folder, filename))


input_folder = './base'
output_folder = './process/base'
crop_images(input_folder, output_folder)
input_folder = './extended'
output_folder = './process/extended'
crop_images(input_folder, output_folder)