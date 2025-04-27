# 集成GRPO训练器和NeuSymRAG
import os, sys
from typing import Dict, List, Any, Optional, Tuple
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.frameworks.grpo_agent_integration import initialize_agent, integrate_grpo_with_agent, prepare_database_prompts

class GRPONeuSymIntegration:
    """
    将GRPO训练器与NeuSymRAG智能体集成的类
    
    这个类处理GRPO训练器与NeuSymRAG智能体之间的交互，使GRPO生成的文本可以通过智能体进行处理
    """
    
    def __init__(self, args, grpo_trainer=None):
        """
        初始化集成器
        
        Args:
            args: 命令行参数
            grpo_trainer: GRPO训练器实例
        """
        self.args = args
        self.grpo_trainer = grpo_trainer
        
        # 初始化智能体和环境
        self.agent, self.env, self.llm = initialize_agent(args)
        
        # 准备数据库和向量存储提示
        self.database_prompt, self.vectorstore_prompt = prepare_database_prompts(args)
    
    def process_sample(self, data, completion_text, output_path=None, **kwargs):
        """
        处理单个样本
        
        Args:
            data: 数据样本
            completion_text: GRPO生成的文本
            output_path: 输出路径
            **kwargs: 其他参数
            
        Returns:
            result: 处理结果
            logits_seq: 概率序列
        """
        return integrate_grpo_with_agent(
            self.agent, 
            data, 
            self.database_prompt, 
            self.vectorstore_prompt,
            completion_text=completion_text,
            dataset=self.args.dataset,
            window_size=self.args.window_size,
            model=self.args.llm, 
            temperature=self.args.temperature, 
            top_p=self.args.top_p, 
            max_tokens=self.args.max_tokens,
            output_kwargs={'output_format': self.args.output_format}, 
            output_path=output_path
        )
    
    def generate_with_grpo(self, data, prompt_text):
        """
        使用GRPO生成文本
        
        Args:
            data: 数据样本
            prompt_text: 提示文本
            
        Returns:
            completion_text: 生成的文本
        """
        if self.grpo_trainer is None:
            raise ValueError("GRPO训练器未初始化")
        
        # 这里是GRPO训练器生成文本的具体实现
        # 使用GRPO的completions_text作为智能体的输入
        # 实际实现中需要替换为对GRPO训练器的调用
        
        # 示例代码，实际使用中需替换
        # 从GRPO的_generate_and_score_completions方法获取结果
        prompts_text = [prompt_text]
        if hasattr(self.grpo_trainer, 'processing_class'):
            prompt_inputs = self.grpo_trainer.processing_class(
                text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            prompt_inputs = self.grpo_trainer._prepare_inputs(prompt_inputs)
            
            # 生成文本完成
            with torch.no_grad():
                outputs = self.grpo_trainer.model.generate(
                    input_ids=prompt_inputs["input_ids"],
                    attention_mask=prompt_inputs["attention_mask"],
                    max_new_tokens=self.grpo_trainer.max_completion_length,
                    do_sample=True,
                    temperature=self.grpo_trainer.temperature,
                    top_p=self.grpo_trainer.top_p
                )
                
            # 解码生成的文本
            completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]
            completion_text = self.grpo_trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)[0]
            return completion_text
        
        # 如果无法使用GRPO生成，返回空字符串
        return ""
    
    def close(self):
        """关闭智能体和环境"""
        if hasattr(self, 'agent') and self.agent is not None:
            self.agent.close() 