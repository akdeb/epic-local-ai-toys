from typing import Generator, Optional
from llama_cpp import Llama


class LLMService:
    def __init__(self):
        self.llm: Optional[Llama] = None
        # self.model_repo = "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF"
        # self.model_file = "llama-3.2-3b-instruct-q4_k_m.gguf"
        self.model_repo = "mistralai/Ministral-3-3B-Instruct-2512-GGUF"
        self.model_file = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"
    
    def initialize_llm(self, model_repo: str = None, model_file: str = None) -> None:
        if self.llm is not None:
            return
        
        repo = model_repo or self.model_repo
        filename = model_file or self.model_file
        
        print(f"Initializing LLM from {repo}...")
        self.llm = Llama.from_pretrained(
            repo_id=repo,
            filename=filename,
            verbose=False,
            n_gpu_layers=-1,
            # n_ctx=2048,
            # chat_format="llama-3",
        )
        print("LLM initialized successfully!")
    
    def generate_stream(self, prompt: str, system_prompt: str = None) -> Generator[str, None, None]:
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Call initialize_llm() first.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
    
    def generate(self, prompt: str, system_prompt: str = None) -> str:
        if self.llm is None:
            raise RuntimeError("LLM not initialized. Call initialize_llm() first.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        return response["choices"][0]["message"]["content"]
    
    def is_initialized(self) -> bool:
        return self.llm is not None

    def unload(self) -> None:
        if self.llm is not None:
            print("Unloading LLM...")
            del self.llm
            self.llm = None
            import gc
            gc.collect()
            print("LLM unloaded successfully!")


llm_service = LLMService()
