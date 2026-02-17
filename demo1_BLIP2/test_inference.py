# THIS IS FOR INDEPENDENTLY INFERENCE.
# EXAMPLE RUN: python test_inference.py --image logs_and_outputs/20260216_185709/adversarial/adv_A_standard_cat.png

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import argparse
import os

# 配置颜色输出
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

def verify_image(model_name, image_path, device="cuda"):
    print(f"Loading Model: {model_name}...")
    
    try:
        # 1. 加载模型
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16
        ).to(device)
        model.eval()
    except Exception as e:
        print(f"{RED}模型加载失败: {e}{RESET}")
        return

    if not os.path.exists(image_path):
        print(f"{RED}错误: 找不到文件 {image_path}{RESET}")
        return
    
    print(f"Testing Image: {image_path}")
    raw_image = Image.open(image_path).convert('RGB')
    
    # 2. 预处理图片 (获取 pixel_values)
    # 这一步只负责把图片转成 tensor，不负责生成 input_ids
    inputs = processor(images=raw_image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs["pixel_values"]

    print("Generating (Manual Forward Pass)...")
    
    # 3. 手动推理流程 (完全复刻 BLIP-2 的内部逻辑，绕过 generate 的 bug)
    # 我们像拆解机器一样，手动执行它的三个组件，这样绝对不会报错
    with torch.no_grad():
        # A. Vision Model: 图片 -> 图片特征
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state

        # B. Q-Former: 图片特征 -> 查询特征
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state

        # C. Projection: 查询特征 -> 语言模型输入
        language_model_inputs = model.language_projection(query_output)

        # D. LLM Generate: 直接让语言模型生成
        # 我们直接把“翻译”好的视觉特征喂给 OPT 语言模型
        inputs_embeds = language_model_inputs
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        # 直接调用语言模型生成，没有任何中间商赚差价
        generated_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False  # 关闭随机采样，保证结果确定性
        )

    # 4. 解码输出
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 5. 输出结果
    print("-" * 50)
    print(f"图片文件: {os.path.basename(image_path)}")
    print(f"模型输出: {GREEN}{generated_text}{RESET}")
    print("-" * 50)

    # 6. 验证结果
    target_keywords = ["chicken", "dinner"]
    hits = [k for k in target_keywords if k in generated_text.lower()]
    
    if len(hits) == 2:
        print(f"结论: {GREEN}攻击成功！(PNG有效){RESET}")
    elif len(hits) > 0:
        print(f"结论: {GREEN}部分成功 (包含 '{hits}'){RESET}")
    else:
        print(f"结论: {RED}攻击失败 (未发现关键词){RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立验证 BLIP-2 对抗样本")
    parser.add_argument("--image", type=str, required=True, help="对抗样本图片的路径")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    
    verify_image(MODEL_NAME, args.image, args.device)