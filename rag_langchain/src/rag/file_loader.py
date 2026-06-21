import multiprocessing
from tqdm import tqdm
import multiprocessing
from typing import List, Tuple, Union, Literal
import glob
# pyrefly: ignore [missing-import]
from langchain_community.document_loaders import PyPDFLoader

# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

#clean messy pdf text
def remove_non_utf8_characters(text):
    return "".join(char for char in text if ord(char) < 128)

#load pdf file
def load_pdf(pdf_file):
    docs = PyPDFLoader(pdf_file, extract_images = True).load()
    for doc in docs:
        doc.page_content = remove_non_utf8_characters(doc.page_content)
    return docs

#get number of cpu
def get_num_cpu():
    return multiprocessing.cpu_count()

#classes to load pdf
#abstract class
class BaseLoader:
    def __init__(self) -> None:
        self.num_processes = get_num_cpu()

    def __call__(self, files: List[str], **kwargs): #**kwargs meaning keyword argument, allowing optional argument
        pass

class PDFLoader(BaseLoader):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pdf_files: List[str], **kwargs):
        self.num_processes = min(self.num_processes, kwargs["workers"])
        
        # Mở Pool
        with multiprocessing.Pool(processes = self.num_processes) as pool:
            doc_loaded = []
            total_files = len(pdf_files)
            
            # Đưa tqdm vào BÊN TRONG Pool để đảm bảo Pool vẫn mở khi vòng for chạy
            with tqdm(total=total_files, desc="Loading PDF files", unit="file") as pbar:
                for result in pool.imap_unordered(load_pdf, pdf_files):
                    doc_loaded.extend(result) 
                    pbar.update(1) 

        # Trả về kết quả sau khi cả hai khối with đã hoàn thành
        return doc_loaded

class TextSplitter:
    def __init__(self, 
                separators: List[str]  = ["\n\n", "\n", " ",""],
                chunk_size: int = 300,
                chunk_overlap: int = 0
    ) -> None:

        self.splitter = RecursiveCharacterTextSplitter(
            separators = separators,
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )
    
    def __call__(self, documents):
        return self.splitter.split_documents(documents)


class Loader:
    def __init__(self, file_type: str = Literal["pdf"],
                    split_kwargs: dict = {
                        "chunk_size": 300,
                        "chunk_overlap": 0
                    }) -> None:
        assert file_type in ["pdf"], "File type must be pdf"
        self.file_type = file_type
        if self.file_type == "pdf":
            self.loader = PDFLoader()
        else:
            raise ValueError("File type not supported")
        #split_kwargs have to be in dictionary form
        self.splitter = TextSplitter(**split_kwargs)

    def load(self, pdf_files: Union[str, List[str]], workers: int = 1):
        if isinstance(pdf_files, str):
            pdf_files = [pdf_files]
        documents = self.loader(pdf_files, workers = workers)
        documents = self.splitter(documents)
        return documents

    def load_dir(self, dir_path: str, workers: int = 1):
        if self.file_type == "pdf":
            pdf_files = glob.glob(f"{dir_path}/*.pdf")
            assert len(pdf_files) > 0, f"No pdf files found in {dir_path}"
        else:
            raise ValueError("File type must be pdf")

        return self.load(pdf_files, workers = workers)
        
        
    
    