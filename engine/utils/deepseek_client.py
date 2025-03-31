import os
import json
import requests
from typing import List, Dict, Any, Tuple, Union, Optional
import time
import glob
import hashlib
from engine.constants import DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL

class DeepSeekClient:
    def __init__(self, model: str = DEEPSEEK_MODEL, api_key: str = DEEPSEEK_API_KEY, api_base: str = DEEPSEEK_API_BASE):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.cache_dir = os.environ.get("DEEPSEEK_CACHE_DIR", ".cache/deepseek")
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, prompt_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{prompt_hash}.json")
    
    def _compute_hash(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # 创建一个不可变的哈希对象
        config_str = json.dumps({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            **{k: v for k, v in kwargs.items() if k not in ["skip_cache"]}
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def generate(
        self, 
        user_prompt: Union[str, List[Dict[str, str]]], 
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 40960,
        num_completions: int = 1,
        skip_cache: bool = False,
        **kwargs
    ) -> Tuple[List[str], List[List[str]]]:
        prompt_hash = self._compute_hash(system_prompt, user_prompt, temperature=temperature, 
                                        max_tokens=max_tokens, **kwargs)
        cache_path = self._get_cache_path(prompt_hash)
        
        # 如果存在缓存且不跳过缓存，则直接读取
        if not skip_cache and os.path.exists(cache_path):
            print(f"[INFO] 从缓存中加载 DeepSeek 响应: {cache_path}")
            with open(cache_path, 'r') as f:
                cached_data = json.load(f)
                return cached_data["prompts"], cached_data["completions"]
        
        completions = []
        
        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        if isinstance(user_prompt, str):
            messages.append({"role": "user", "content": user_prompt})
        else:
            # 处理多模态输入
            content = []
            for item in user_prompt:
                if item["type"] == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]}
                    })
            messages.append({"role": "user", "content": content})
        
        # 调用 DeepSeek API
        for _ in range(num_completions):
            try:
                response = requests.post(
                    f"{self.api_base}/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        **kwargs
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    completion = result["choices"][0]["message"]["content"]
                    completions.append([completion])
                else:
                    print(f"[ERROR] DeepSeek API 返回了意外的结果: {result}")
                    completions.append(["API 错误: 没有找到完成结果"])
            
            except Exception as e:
                print(f"[ERROR] DeepSeek API 调用失败: {e}")
                completions.append([f"API 错误: {str(e)}"])
            
            # 避免超过 API 限制，添加请求之间的延迟
            if _ < num_completions - 1:
                time.sleep(0.5)
        
        # 缓存结果
        cache_data = {
            "prompts": [user_prompt] * num_completions,
            "completions": completions
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
            
        return [user_prompt] * num_completions, completions

def setup_deepseek():
    """设置并返回 DeepSeek 客户端实例"""
    return DeepSeekClient()
