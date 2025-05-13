from PIL import Image
import numpy as np

# 图像路径
input_path = '/home/zifanw/rl_robot/LidarSensor/real_lidar_red.png'
output_path = '/home/zifanw/rl_robot/LidarSensor/real_lidar_white.png'

# 打开图像
img = Image.open(input_path)
img_array = np.array(img)

print(f"图像形状: {img_array.shape}")

# 处理图像 - 将黑色像素转换为白色
if len(img_array.shape) == 2:  # 灰度图像
    black_pixels = img_array == 0
    img_array[black_pixels] = 255  # 转换为白色
else:  # 彩色图像
    # 检查图像是否为RGBA (4通道)
    if img_array.shape[2] == 4:
        # 仅比较RGB三个通道是否为黑色
        black_pixels = np.all(img_array[:,:,:3] == [0, 0, 0], axis=2)
        # 创建白色像素，保留原始alpha值
        white_pixels = np.zeros_like(img_array)
        white_pixels[:,:,:3] = 255
        white_pixels[:,:,3] = img_array[:,:,3]  # 保留原始alpha值
        # 应用更改
        img_array[black_pixels] = white_pixels[black_pixels]
    else:
        # 处理常规RGB图像
        black_pixels = np.all(img_array == [0, 0, 0], axis=2)
        img_array[black_pixels] = [255, 255, 255]  # 转换为白色

# 转换回图像并保存
processed_img = Image.fromarray(img_array)
processed_img.save(output_path)

print(f"图像处理完成，已保存到 {output_path}")