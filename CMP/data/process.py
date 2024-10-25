import math
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

def crop_images_v2(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    base_url = output_folder+"/base"
    extended_url = output_folder+"/extended"
    meta_size = 512
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):  # 支持的文件格式
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)
            width, height = img.size
            min_edge = min(width, height)
            if min_edge < meta_size:
                img.save(os.path.join(extended_url, filename))
            else:
                if width == 1024:
                    for i_x in range(3):
                        for i_y in range(math.ceil(height / meta_size)):
                            left = i_x * (meta_size // 2)
                            right = left + meta_size
                            if i_y == 0:
                                upper = 0
                                lower = upper + meta_size
                            elif i_y == 1:
                                lower = height
                                upper = lower - meta_size
                            else:
                                continue
                            img_cropped = img.crop((left, upper, right, lower))
                            img_cropped.save(os.path.join(base_url, str(i_x) + str(i_y) + "_" + filename))
                elif height == 1024:
                    for i_y in range(3):
                        for i_x in range(math.ceil(width / meta_size)):
                            upper = i_y * (meta_size // 2)
                            lower = upper + meta_size
                            if i_x == 0:
                                left = 0
                                right = left + meta_size
                            elif i_x == 1:
                                right = width
                                left = right - meta_size
                            else:
                                continue
                            img_cropped = img.crop((left, upper, right, lower))
                            img_cropped.save(os.path.join(base_url, str(i_x) + str(i_y) + "_" + filename))
                else:
                    continue  # 不处理不满足条件的图像


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


# input_folder = './base'
# output_folder = './process/base'
# crop_images(input_folder, output_folder)
# input_folder = './extended'
# output_folder = './process/extended'
# crop_images(input_folder, output_folder)


input_folder = './total'
output_folder = './process_v2'
crop_images_v2(input_folder, output_folder)