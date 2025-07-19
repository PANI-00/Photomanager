import clip
import torch
from PIL import Image
import requests
from io import BytesIO

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备图像数据集（可以替换为自己的图像URL列表）
image_urls = [
    'https://globalimg.sucai999.com/preimg/AF76E3/700/AF76E3/25/250355ed900c6c9a0e1887ef9502fe04.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/6D28AC/700/6D28AC/201/5b4a759c5788ca45d4395211521bf2c.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/DBC456/700/DBC456/103/2c1f0905d93e64af2799d80c67df5.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/8AA05E/700/8AA05E/103/95e47c5820ab2b738f0e9e2ddc6ca8.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/E625C8/700/E625C8/148/546aeb60b2446bf0c4aa247454d32d50.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/E625C8/700/E625C8/111/f436fc2891516b652564367cd2d859a7.jpg?x-oss-process=image/resize,w_320/format,webp',
    'https://globalimg.sucai999.com/preimg/123D82/700/123D82/198/d849caf1496cf78e70e08a2a9265e762.jpg?x-oss-process=image/resize,w_320/format,webp'

]

# 下载并预处理所有图像
images = []
for url in image_urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(preprocess(img))

# 将所有图像转为batch形式并移动到指定设备
image_input = torch.stack(images).to(device)

# 计算所有图像的特征
with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)


# 主程序入口
def main():
    print("请选择操作：")
    print("1. 图片分类")
    print("2. 文本检索")

    choice = input("请输入选项（1 或 2）: ")

    if choice == '1':
        zero_shot_classification()
    elif choice == '2':
        cross_modal_retrieval()
    else:
        print("无效选项，请重新运行并选择正确的选项。")


# 零样本分类函数
def zero_shot_classification():
    exam1 = str(input('输入你的图片链接:\n'))
    image_response = requests.get(exam1)
    image = Image.open(BytesIO(image_response.content)).convert("RGB")
    image_input_single = preprocess(image).unsqueeze(0).to(device)

    text_labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car",
                   "a photo of a house","a photo of gundam"]
    text_inputs = clip.tokenize(text_labels).to(device)

    with torch.no_grad():
        image_features_single = model.encode_image(image_input_single)
        text_features = model.encode_text(text_inputs)

        # 归一化特征向量
        image_features_single /= image_features_single.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity = 100.0 * image_features_single @ text_features.T
        k = min(5, similarity.size(-1))

        # 结果
        values, indices = similarity[0].topk(k=k, dim=-1)

        print("\n 图像分类预测结果：")
        for value, index in zip(values, indices):
            print(f"{text_labels[index]}: {value.item():.2f}%")


# 跨模态检索函数
def cross_modal_retrieval():
    query_texts = []
    query_texts.append(str(input('输入你的文本(English):\n')))
    query_inputs = clip.tokenize(query_texts).to(device)

    with torch.no_grad():
        query_features = model.encode_text(query_inputs)
        query_features /= query_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        similarity_scores = (100.0 * query_features @ image_features.T).softmax(dim=1)

        # 打印结果
        for i, query in enumerate(query_texts):
            print(f"\nQuery '{query}' 匹配得分:")
            values, indices = similarity_scores[i].topk(k=5, dim=-1)
            for value, index in zip(values, indices):
                print(f"Image URL: {image_urls[index]} with score: {value.item():.2f}")


if __name__ == "__main__":
    main()