import gradio as gr
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1. 加载模型 (只需加载一次)
print("正在启动 Web UI，请稍候...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Salesforce/blip2-opt-2.7b"

try:
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("模型加载完成！")
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 2. 定义推理函数
def predict(image):
    if image is None:
        return "请先上传图片"
    
    # 预处理
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    pixel_values = inputs["pixel_values"]

    # 手动推理流程 (复刻 inference.py 的成功逻辑)
    with torch.no_grad():
        vision_outputs = model.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs.last_hidden_state
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs = model.language_projection(query_output)
        
        inputs_embeds = language_model_inputs
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        generated_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=50,
            do_sample=False
        )

    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response

# 3. 搭建界面
with gr.Blocks(title="BLIP-2 对抗攻击演示") as demo:
    gr.Markdown("# 🛡️ VisInject 对抗攻击演示")
    gr.Markdown("上传一张对抗样本图片，看看 BLIP-2 模型会说什么。")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="上传图片 (Drag & Drop)")
            run_btn = gr.Button("开始推理", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="模型输出", lines=4, elem_id="output")
    
    # 绑定事件
    run_btn.click(fn=predict, inputs=input_img, outputs=output_text)

# 4. 启动
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)