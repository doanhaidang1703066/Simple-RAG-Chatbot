# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama

class LLMModel:
    def __init__(self, model_name="qwen2.5:3b", temperature=0.1):
        """
        Initiate connection with the model running in background
        """
        self.model_name = model_name
        self.temperature = temperature

        self.llm = ChatOllama(
            model=self.model_name,
            temperature=self.temperature
        )

    def create_llm(self):
        """
        return LLM object to LangChain
        """
        return self.llm


        
'''
if __name__ == "__main__":
    my_llm_setup = LLMModel(model_name="qwen2.5:3b")
    llm = my_llm_setup.create_llm()
    
    print("Thinking...\n")
    response = llm.invoke("Hãy giải thích ngắn gọn IncoTerms là gì?")
    
    print("AI trả lời:")
    print(response.content)
'''