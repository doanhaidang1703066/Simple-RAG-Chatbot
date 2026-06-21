'''
in this file, we construct a chain from LangChain to connect parts of RAG together
including: using retriever to take context --> build prompt --> pass to LLM --> get answer
'''
import re
# pyrefly: ignore [missing-import]
from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_core.runnables import RunnablePassthrough
# pyrefly: ignore [missing-import]
from langchain_core.output_parsers import StrOutputParser

class Str_OutputParser(StrOutputParser):
    def __init__(self):
        super().__init__()
    
    def parse(self, text_response: str) -> str:
       return self.extract_answer(text_response)

    def extract_answer(self,
                        text_response: str,
                        pattern = r"Answer:\s*(.*)") -> str:

        '''
        this function aims to extract the answer from the text response of the LLM
        the reason for this function is that the prompt sometimes forces the LLM to output 
        the answer in a specific format, for example:
        "Answer: 
        "
        '''
        #we use re.DOTALL so that . can match with newline characters (important)
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            return text_response

class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        template = template = """You are a corporate AI assistant. Your task is to answer the question ONLY based on the provided Context below.
        
        STRICT RULES:
        1. ABSOLUTELY DO NOT use your internal knowledge, general knowledge, or prior training data to answer.
        2. If the provided Context does not contain the necessary information to answer the question, or if the Context is empty, you MUST reply with exactly: "I'm sorry, the current document does not contain information on this topic."
        3. Do not attempt to fabricate, guess, or deduce the answer outside of the given text.

        Context:
        {context}

        User Question: {question}

        Answer:"""
        self.prompt = PromptTemplate.from_template(template)
        self.parser = Str_OutputParser()
    
    def get_chain(self, retriever):
        '''
        explain flow:
        user type in query as input data
        then the retriever will retrieve related docs --> format_docs --> feed in the context
        simultaneously, the query will run through RunnablePassthrough() --> feed in the question
        context, question in the input_data dictionary will be passed into the prompt pulled from rag-prompt
        after that, the prompt will be passed to LLM --> then output pass through StrOutputParser to extract the answer
        '''
        input_data = {
            "context": retriever | self.format_docs,
            "question": RunnablePassthrough()
        }

        rag_chain = (
            input_data
            | self.prompt
            | self.llm
            | self.parser
        )
        return rag_chain
    
    def format_docs(self,docs):
        return "\n\n".join(doc.page_content for doc in docs)