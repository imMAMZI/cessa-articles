#!/usr/bin/env python3
"""
02_langchain_rag.py
===================
پیاده‌سازی مدرن RAG با استفاده از LangChain و سینتکس مدرن LCEL (LangChain Expression Language).

در این پایپ‌لاین:
  ۱. اسناد دانشگاهی به صورت Document تعریف می‌شوند.
  ۲. یک Retriever سفارشی برای بازیابی تکه‌های مرتبط متن ساخته می‌شود.
  ۳. یک قالب پرامپت (ChatPromptTemplate) برای تزریق کانتکست ایجاد می‌شود.
  ۴. زنجیره RAG با سینتکس بیانی `|` متصل می‌شود:
     chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
"""

import os
from pathlib import Path
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


from pydantic import ConfigDict


class SimpleSemanticRetriever(BaseRetriever):
    """یک بازیاب مبتنی بر شباهت معنایی برای لانگ‌چین"""

    documents: List[Document]
    vectorizer: object
    doc_vectors: object
    top_k: int = 2

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, docs: List[Document], top_k: int = 2):
        vectorizer = TfidfVectorizer()
        texts = [doc.page_content for doc in docs]
        doc_vectors = vectorizer.fit_transform(texts)
        return cls(
            documents=docs,
            vectorizer=vectorizer,
            doc_vectors=doc_vectors,
            top_k=top_k,
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_indices = scores.argsort()[::-1][: self.top_k]
        return [self.documents[i] for i in top_indices]


def format_docs(docs: List[Document]) -> str:
    """چسباندن تکه‌های بازیابی‌شده به همدیگر برای قرارگیری در پرامپت"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "نامشخص")
        formatted.append(f"[منبع {i} - {source}]:\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_llm():
    """ایجاد LLM واقعی (در صورت وجود API KEY) یا ماک هوشمند آموزشی"""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        except Exception as e:
            print(f"⚠️ اتصال به OpenAI ناموفق بود: {e}")

    # مدل شبیه‌ساز آفلاین برای تست بدون هزینه و اینترنت
    def mock_llm_inference(prompt_value):
        text = prompt_value.to_string() if hasattr(prompt_value, "to_string") else str(prompt_value)
        return (
            "بر اساس ماده ۱۴ آیین‌نامه آموزشی و سیلابس درس:\n"
            "۱. حضور در تمام جلسات کلاس الزامی است.\n"
            "۲. حداکثر غیبت مجاز دانشجو ۳ جلسه در طول ترم می‌باشد.\n"
            "۳. در صورت ثبت غیبت چهارم (حتی اگر موجه باشد)، درس شما به دلیل «غیبت بیش از حد» "
            "حذف آموزشی شده و متاسفانه نمره صفر در کارنامه منظور می‌گردد!"
        )

    return RunnableLambda(mock_llm_inference)


def main():
    print("=" * 65)
    print("🦜🔗 اجرای پایپ‌لاین مدرن RAG با LangChain (LCEL)")
    print("=" * 65)

    # ۱. خواندن فایل‌های متنی دانشگاهی
    base_dir = Path(__file__).parent / "mock_data"
    os_file = base_dir / "os_course_syllabus.txt"
    faq_file = base_dir / "university_faq.txt"

    docs = []
    for file_path, source_name in [(os_file, "سیلابس سیستم‌عامل"), (faq_file, "آیین‌نامه آموزشی")]:
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # شکستن بخش‌ها بر اساس تیترها
                sections = [s.strip() for s in content.split("## ") if s.strip()]
                for s in sections:
                    docs.append(Document(page_content=s, metadata={"source": source_name}))

    print(f"📄 تعداد {len(docs)} بخش آموزشی در پایگاه دانش بارگذاری شد.")

    # ۲. ساخت بازیاب (Retriever)
    retriever = SimpleSemanticRetriever.from_documents(docs, top_k=2)

    # ۳. تعریف تمپلیت پرامپت (Prompt Template)
    template = """شما مشاور آموزشی دانشگاه هستید. با تکیه بر اطلاعات رسمی زیر به سوال دانشجو پاسخ دقیق دهید.
اگر پاسخ در منابع زیر نبود، صراحتاً اعلام کنید که اطلاع ندارید و از حدس زدن پرهیز کنید.

اطلاعات رسمی:
{context}

سوال دانشجو:
{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # ۴. ساخت زنجیره به کمک LCEL
    llm = build_llm()
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ۵. اجرای زنجیره روی سوال دانشجو
    user_query = "اگر من ۴ جلسه در کلاس سیستم‌های عامل غیبت کنم چه بلایی سرم میاد؟ حذف میشم؟"
    print(f"\n❓ پرسش دانشجو:\n   {user_query}\n")

    print("🔎 اسناد مرتبط استخراج‌شده توسط LangChain Retriever:")
    relevant_docs = retriever.invoke(user_query)
    for i, doc in enumerate(relevant_docs, 1):
        print(f"   [{i}] منبع: {doc.metadata.get('source')} | چکیده: {doc.page_content[:90]}...")

    print("\n💬 پاسخ تولید شده توسط زنجیره RAG:")
    response = rag_chain.invoke(user_query)
    print(f"{response}\n")
    print("=" * 65)


if __name__ == "__main__":
    main()
