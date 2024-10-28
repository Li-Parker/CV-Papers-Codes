import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 读取图像
image_path = r"../data/base/cmp_b0001.png"
# image = plt.imread(image_path)[:,:,0]
image = np.array(Image.open(image_path))
image[0][0] = 12
print(np.max(image))
print(image.shape)

# 显示图像
plt.imshow(image)
plt.axis('off')
plt.show()