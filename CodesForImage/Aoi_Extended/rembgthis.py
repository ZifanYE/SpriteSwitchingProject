import os
from rembg import remove
from PIL import Image

# 文件夹路径（同级）
input_folder = "with_bg"
output_folder = "no_bg"

# 如果输出文件夹不存在，就创建
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹的所有文件
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)

    # 只处理图片文件（可根据需求扩展）
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

        # 打开图片并去背景
        with Image.open(input_path) as img:
            result = remove(img)
            result.save(output_path)

        print(f"✅ Processed: {filename} -> {output_path}")

print("🎉 All images processed!")
