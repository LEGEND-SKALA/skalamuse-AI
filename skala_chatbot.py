import os
import shutil
import openai
import requests
import json
import pypdf
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.document_loaders import PdyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from get_lunch_menu import lunch
os.environ["OPENAI_API_KEY"] = ""
client = openai.OpenAI()

class Chatbot:
    def __init__(self):
        """Chatbot 클래스 초기화"""
        self.system_prompt = "기본 시스템 프롬프트입니다."
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        self.vector_stores = {}

    def set_system_prompt(self, prompt):
        """운영자가 System Prompt 설정"""
        self.system_prompt = prompt

    def initialize_chroma(self, db_name):
        """ChromaDB 초기화 (운영 및 교육 DB 생성)"""
        chroma_db_path = f"./chroma_db_{db_name}"
        
        if os.path.exists(chroma_db_path):
            shutil.rmtree(chroma_db_path)
            print(f"🗑 기존 ChromaDB 데이터 삭제 완료! (DB: {db_name})")
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_stores[db_name] = Chroma(embedding_function=embeddings, persist_directory=chroma_db_path)

    def is_pdf_encrypted(self, pdf_path):
        """PDF 파일이 암호화되어 있는지 확인"""
        try:
            with open(pdf_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                return reader.is_encrypted
        except Exception as e:
            print(f":x: PDF 암호화 확인 중 오류 발생: {e}")
            return False

    def process_pdf_dir_to_chroma(self, directory, db_name):
        """디렉토리 내 모든 PDF 파일을 ChromaDB에 저장"""
        if db_name not in self.vector_stores:
            self.initialize_chroma(db_name)

        if not os.path.exists(directory):
            print(f":x: {directory} 경로가 존재하지 않습니다.")
            return

        pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]

        if not pdf_files:
            print(f":warning: {directory}에 PDF 파일이 없습니다.")
            return

        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            print(f":page_facing_up: PDF 처리 중: {pdf_file}")

            try:
                if self.is_pdf_encrypted(pdf_path):
                    print(f":x: {pdf_file}은(는) 암호화되어 있어 처리할 수 없습니다.")
                    continue
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
            except Exception as e:
                print(f":x: PDF 파일 로드 중 오류 발생: {e}")
                continue

            corpus = Document(page_content="".join([page.page_content + "\n---\n" for page in pages]))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents([corpus])
            self.vector_stores[db_name].add_documents(chunks)
            print(f":white_check_mark: {pdf_file}에서 {len(chunks)}개 문서가 {db_name} ChromaDB에 저장되었습니다!")

    def scrape_and_store_skala(self, db_name):
        """SKALA 웹사이트에서 데이터 크롤링 후 ChromaDB에 저장 (운영 관련)"""
        if db_name not in self.vector_stores:
            self.initialize_chroma(db_name)

        url = "https://skala.co.kr/"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f":x: 웹사이트 크롤링 오류: {e}")
            return

        soup = BeautifulSoup(response.text, "html.parser")

        sections = {
            "about": soup.find("section", {"class": "one_page_section _one_page_section one_page_section_1"}),
            "curriculum": soup.find("section", {"class": "one_page_section _one_page_section one_page_section_2"}),
            "faq": soup.find("section", {"class": "one_page_section _one_page_section one_page_section_3"}),
        }

        scraped_data = {
            key: section.get_text(separator="\n", strip=True) if section else "Not Found"
            for key, section in sections.items()
        }

        documents = [
            Document(page_content=value, metadata={"section": key})
            for key, value in scraped_data.items() if value != "Not Found"
        ]

        if documents:
            self.vector_stores[db_name].add_documents(documents)
            print(f":white_check_mark: 크롤링 데이터 {len(documents)}개가 {db_name} ChromaDB에 저장되었습니다!")

    async def get_lunch_menu(self):
        return get_lunch_menu()

    def classify_query(self, user_prompt):
        """LLM을 이용해 사용자의 질문을 자동으로 분류"""
        classification_prompt = ChatPromptTemplate.from_template("""
        사용자의 질문을 다음 중 하나로 분류하세요: general, operation, education, lunch

        질문: {question}
        
        답변 형식: [분류]
        """)

        classification_chain = LLMChain(llm=self.llm, prompt=classification_prompt)
        result = classification_chain.run(question=user_prompt)
        
        category = result.strip().lower().replace("[", "").replace("]", "")
        print(f":pushpin: 질문 분류 결과: {category}")
        return category

    def query_rag(self, user_prompt, db_name):
        """운영 및 교육 관련 질문을 ChromaDB + LangChain RAG로 검색"""
        if db_name not in self.vector_stores:
            print(f":x: {db_name} ChromaDB가 존재하지 않습니다. 먼저 데이터 구축 후 시도하세요.")
            return

        retriever = self.vector_stores[db_name].as_retriever(search_kwargs={"k": 5})

        prompt_template = ChatPromptTemplate.from_template(f"""
        {self.system_prompt}
        
        문서 내용:
        {{context}}

        사용자 질문:
        {{question}}

        답변을 제공하세요.
        """)

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template},
        )

        response = qa_chain.invoke(user_prompt)
        return response["result"]

    def handle_user_query(self, user_prompt):
        """사용자의 질문을 분류 후 적절한 방식으로 답변"""
        query_type = self.classify_query(user_prompt)

        if query_type == "general":
            return self.llm.invoke(f"{self.system_prompt}\n\n사용자 질문: {user_prompt}")
        elif query_type == "operation":
            return self.query_rag(user_prompt, "operation")
        elif query_type == "education":
            return self.query_rag(user_prompt, "education")
        elif query_type == "lunch":
            return self.query_rag(user_prompt, "lunch")
        else:
            return ":x: 질문을 올바르게 분류할 수 없습니다. 다시 입력해주세요."


# 실행 코드 예시
if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.set_system_prompt("당신은 전문적인 AI 비서입니다.")

    chatbot.process_pdf_dir_to_chroma("./operation", "operation")
    chatbot.process_pdf_dir_to_chroma("./education", "education")

    user_prompt = input("사용자 질문을 입력하세요: ").strip()
    response = chatbot.handle_user_query(user_prompt)

    print("\n:memo: 답변:")
    print(response)