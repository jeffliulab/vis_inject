"""
模型加载器：BLIP-2端到端攻击 + 验证
参考AAAI 2024论文方法，手动构造embedding序列实现梯度传播
"""

import torch
import torch.nn.functional as F
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BLIP2AttackModel:
    """BLIP-2统一模型：攻击 + 验证"""
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
        self.device = device
        
        logger.info(f"加载BLIP-2模型: {model_name}")
        print(f"加载BLIP-2模型: {model_name}")
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # 直接加载到GPU，不使用device_map
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # 冻结所有参数（只优化图像像素）
        self.model.requires_grad_(False)
        
        # 启用gradient checkpointing减少显存
        if hasattr(self.model.language_model, 'gradient_checkpointing_enable'):
            self.model.language_model.gradient_checkpointing_enable()
            logger.info("已启用OPT gradient checkpointing")
        
        # 获取组件引用
        self.vision_model = self.model.vision_model
        self.qformer = self.model.qformer
        self.language_projection = self.model.language_projection
        self.language_model = self.model.language_model
        self.query_tokens = self.model.query_tokens
        
        # 显存信息
        if device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"模型显存占用: {mem:.1f}GB")
            print(f"✓ 模型加载完成，显存: {mem:.1f}GB")
        else:
            print(f"✓ 模型加载完成")
    
    def _preprocess_pil_to_tensor(self, image):
        """将PIL Image转换为模型输入tensor（和compute_attack_loss完全相同的路径）"""
        import numpy as np
        # PIL → numpy → tensor（和utils.pil_to_tensor一致）
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # 和compute_attack_loss完全相同的预处理
        resized = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
        pixel_values = (resized.half() - mean) / std
        return pixel_values
    
    def generate(self, image_path, question=""):
        """标准生成 - 使用和攻击完全相同的预处理路径"""
        logger.info(f"[Generate] 图片: {image_path}")
        image = Image.open(image_path).convert('RGB')
        
        try:
            # 关键：用和compute_attack_loss完全相同的预处理！
            pixel_values = self._preprocess_pil_to_tensor(image)
            
            with torch.no_grad():
                # 手动走BLIP-2的pipeline（和compute_attack_loss一样的路径）
                vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
                image_embeds = vision_outputs.last_hidden_state
                
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_outputs = self.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    return_dict=True
                )
                language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
                
                # 用image_embeds作为prefix，让OPT自回归生成
                attention_mask = torch.ones(language_model_inputs.shape[:2], dtype=torch.long, device=self.device)
                
                # 关闭gradient checkpointing以使用generate
                gc_enabled = getattr(self.language_model, 'gradient_checkpointing', False)
                if gc_enabled:
                    self.language_model.gradient_checkpointing_disable()
                
                out = self.language_model.generate(
                    inputs_embeds=language_model_inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=80,
                    do_sample=False
                )
                
                # 恢复gradient checkpointing
                if gc_enabled:
                    self.language_model.gradient_checkpointing_enable()
            
            response = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
            logger.info(f"[Generate] 输出: '{response}'")
            return response
        except Exception as e:
            logger.error(f"[Generate] 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""
    
    def compute_attack_loss(self, image_tensor, target_text):
        """
        端到端攻击loss计算
        
        目标：只给图像，让模型输出target_text
        和generate验证时的路径完全一致：image_embeds → OPT → 生成文本
        
        流程：
        pixel → vision_model → Q-Former → language_projection → image_embeds
        image_embeds → OPT(inputs_embeds=image_embeds, labels=target_ids) → loss
        """
        # 1. 图像预处理（保持梯度）
        resized = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
        pixel_values = (resized.half() - mean) / std
        
        # 2. 通过vision model（保持梯度）
        vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        
        # 3. 通过Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            return_dict=True
        )
        query_output = query_outputs.last_hidden_state
        
        # 4. 投影到语言模型空间
        language_model_inputs = self.language_projection(query_output)  # [1, 32, 2560]
        
        # 5. 准备目标文本的token
        target_tokens = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=20,
            truncation=True
        )
        target_ids = target_tokens.input_ids.to(self.device)
        
        # 6. 构造输入：image_embeds + target_text_embeds
        # 自回归模型：位置i的logits预测位置i+1的token
        # 要让generate第一个输出就是target的第一个token
        # 需要让最后一个image_token的logits → 预测target[0]
        # 所以inputs = [image_embeds, target_embeds]
        # labels =     [-100...(image),  target_ids...,  -100(最后一个)]
        # 这样：
        #   image最后一个位置的logits → 学习预测target[0]（"chicken"）
        #   target[0]位置的logits → 学习预测target[1]（"dinner"）
        
        target_embeds = self.language_model.get_input_embeddings()(target_ids)
        inputs_embeds = torch.cat([language_model_inputs, target_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        
        num_img = language_model_inputs.shape[1]
        target_len = target_ids.shape[1]
        
        # labels构造：
        # [  -100, -100, ..., -100,   target[0], target[1], ..., target[-1] ]
        #   ← image tokens (忽略) →   ← 最后一个img位置开始预测target →
        # 注意：自回归中labels向左移一位，所以：
        # 位置 num_img-1 的logits应该预测labels[num_img] = target[0]
        image_labels = torch.full((1, num_img), -100, dtype=torch.long, device=self.device)
        text_labels = target_ids.clone()
        labels = torch.cat([image_labels, text_labels], dim=1)
        
        # 8. 通过语言模型计算loss
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        # 监控：看最后一个image token之后的预测
        with torch.no_grad():
            logits = outputs.logits
            # 最后一个image token的预测 = generate的第一个输出
            pred_ids = torch.argmax(logits[0, num_img-1:num_img-1+target_len], dim=-1)
            pred_text = self.processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
        
        return loss, pred_text
    
    def generate_from_tensor(self, image_tensor):
        """直接从tensor生成（不经过保存/加载，避免量化损失）"""
        with torch.no_grad():
            resized = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device, dtype=torch.float16).view(1, 3, 1, 1)
            pixel_values = (resized.half() - mean) / std
            
            vision_outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                return_dict=True
            )
            language_model_inputs = self.language_projection(query_outputs.last_hidden_state)
            attention_mask = torch.ones(language_model_inputs.shape[:2], dtype=torch.long, device=self.device)
            
            gc_enabled = getattr(self.language_model, 'gradient_checkpointing', False)
            if gc_enabled:
                self.language_model.gradient_checkpointing_disable()
            
            out = self.language_model.generate(
                inputs_embeds=language_model_inputs,
                attention_mask=attention_mask,
                max_new_tokens=80,
                do_sample=False
            )
            
            if gc_enabled:
                self.language_model.gradient_checkpointing_enable()
        
        response = self.processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
        logger.info(f"[GenerateTensor] 输出: '{response}'")
        return response
    
    def save_adversarial_image_and_test(self, adv_image_tensor, question, save_path):
        """保存对抗图像并多方式验证"""
        import numpy as np
        from utils import tensor_to_pil
        
        base_path = save_path.rsplit('.', 1)[0]
        
        # 保存1：8-bit PNG
        adv_image_8bit = tensor_to_pil(adv_image_tensor)
        adv_image_8bit.save(save_path)
        
        # 保存2：16-bit PNG
        png16_path = base_path + '_16bit.png'
        tensor_np = adv_image_tensor.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).float().numpy()
        img_16bit = (tensor_np * 65535).astype(np.uint16)
        from PIL import Image as PILImage
        # PIL不直接支持16bit RGB PNG，用raw方式
        # 改用保存为numpy
        npy_path = base_path + '.npy'
        np.save(npy_path, tensor_np)
        
        # 保存3：tensor (.pt)
        pt_path = base_path + '.pt'
        torch.save(adv_image_tensor.cpu(), pt_path)
        
        logger.info(f"[保存] 8bit-PNG: {save_path}")
        logger.info(f"[保存] NPY: {npy_path}")
        logger.info(f"[保存] PT: {pt_path}")
        
        # === 验证 ===
        results = {}
        
        # 验证1：直接tensor（无损，基准）
        results['direct'] = self.generate_from_tensor(adv_image_tensor)
        logger.info(f"[验证-direct] '{results['direct']}'")
        
        # 验证2：8-bit PNG
        results['png8'] = self.generate(save_path, question)
        logger.info(f"[验证-png8] '{results['png8']}'")
        
        # 验证3：从npy加载（完整精度）
        loaded_npy = np.load(npy_path)
        loaded_tensor = torch.from_numpy(loaded_npy).permute(2, 0, 1).unsqueeze(0).to(self.device)
        results['npy'] = self.generate_from_tensor(loaded_tensor)
        logger.info(f"[验证-npy] '{results['npy']}'")
        
        # 验证4：从pt加载
        loaded_pt = torch.load(pt_path, weights_only=True).to(self.device)
        results['pt'] = self.generate_from_tensor(loaded_pt)
        logger.info(f"[验证-pt] '{results['pt']}'")
        
        # 返回direct结果
        return results['direct']
    
    def __call__(self, image_path, question=""):
        return self.generate(image_path, question)


def load_model(model_name="Salesforce/blip2-opt-2.7b", device="cuda"):
    return BLIP2AttackModel(model_name=model_name, device=device)
