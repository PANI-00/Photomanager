import clip
import torch
from PIL import Image
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

exam1 = str(input('输入你的图片链接:\n'))
image_url = exam1
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(device)

text_labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird", "a photo of a car", "a photo of a house"]
text_inputs = clip.tokenize(text_labels).to(device)

#相似度计算
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# 归一特征向量
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

#计算相似度
    similarity = 100.0 * image_features @ text_features.T
    k = min(5, similarity.size(-1))

#结果
values, indices = similarity[0].topk(k=k, dim=-1)

print(" 图像分类预测结果：")
for value, index in zip(values, indices):
    print(f"{text_labels[index]}: {value.item():.2f}%")

query_texts = ["a blue cat", "a red car", "a cute animal", "a building", "a bird flying"]
query_inputs = clip.tokenize(query_texts).to(device)

with torch.no_grad():
    query_features = model.encode_text(query_inputs)
    query_features /= query_features.norm(dim=-1, keepdim=True)

    # 文本与图像的相似度
    retrieval_scores = (100.0 * query_features @ image_features.T).softmax(dim=0)

print("\n 跨模态检索得分（越高越匹配）：")
for i, score in enumerate(retrieval_scores):
    print(f"Query '{query_texts[i]}' 匹配得分: {score.item():.2f}")