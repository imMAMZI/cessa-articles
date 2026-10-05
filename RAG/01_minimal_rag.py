#!/usr/bin/env python3
"""
01_minimal_rag.py
=================
یک پیاده‌سازی مینی‌مال و مفهومی از RAG در پایتون بدون نیاز به فریم‌ورک‌های پیچیده.
این اسکریپت فازهای اصلی را قدم‌به‌قدم نشان می‌دهد:
  ۱. قطعه‌بندی متن (Chunking)
  ۲. برداری‌سازی (Vector Embedding) با فضای برداری و شباهت کسینوسی (Cosine Similarity)
  ۳. بازیابی مرتبط‌ترین بخش‌ها (Retrieval) بر اساس سوال
  ۴. تقویت پرامپت (Prompt Augmentation)
  ۵. تولید پاسخ نهایی (Generation)
"""

import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MinimalVectorStore:
    """یک پایگاه داده برداری سبک و مفهومی برای ذخیره و جستجوی چانک‌ها"""

    def __init__(self):
        self.chunks = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def add_documents(self, text: str, chunk_size: int = 300, overlap: int = 50):
        """قطعه‌بندی متن اسناد و برداری‌سازی آن‌ها"""
        words = text.split()
        raw_chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            raw_chunks.append(chunk)
            i += chunk_size - overlap

        self.chunks.extend(raw_chunks)
        # محاسبه ماتریس بردارها (Embeddings)
        self.vectors = self.vectorizer.fit_transform(self.chunks)
        print(f"✅ تعداد {len(self.chunks)} چانک با موفقیت ایندکس و برداری شد.")

    def search(self, query: str, top_k: int = 2):
        """بازیابی تکه‌های با بالاترین شباهت کسینوسی به سوال کاربر"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.vectors).flatten()

        # استخراج اندیس‌های برتر با بیشترین امتیاز شباهت
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], scores[idx]))
        return results


def generate_answer(query: str, retrieved_chunks: list) -> str:
    """
    تزریق کانتکست به پرامپت و تولید پاسخ
    اگر کلید OPENAI_API_KEY تنظیم شده باشد، با مدل واقعی صدا می‌زند،
    در غیر این صورت فرآیند را به صورت شبیه‌سازی‌شده و دقیق نمایش می‌دهد.
    """
    context = "\n---\n".join([chunk for chunk, score in retrieved_chunks])
    augmented_prompt = f"""شما یک دستیار آموزشی هوشمند دانشگاه هستید.
لطفاً بر اساس اطلاعات موثق زیر به سوال دانشجو پاسخ دقیق دهید. اگر پاسخ در اطلاعات نبود، بگویید «در منابع موجود اطلاعاتی پیدا نشد»:

[محتوای بازیابی شده]:
{context}

[سوال دانشجو]:
{query}
"""
    # در صورت وجود کلید OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": augmented_prompt}],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ خطا در فراخوانی OpenAI API: {e}")

    # حالت استنتاج آفلاین / آموزشی
    return (
        "💡 [پاسخ شبیه‌سازی‌شده مبتنی بر کانتکست بازیابی‌شده]:\n"
        "بر اساس سیلابس رسمی درس سیستم‌های عامل:\n"
        "• بارم آزمون میان‌ترم: ۶ نمره\n"
        "• بارم آزمون پایان‌ترم: ۸ نمره\n"
        "• پروژه‌های برنامه‌نویسی و آزمایشگاه: ۵ نمره (شامل الگوریتم‌های زمان‌بندی، همگام‌سازی و مدیریت حافظه)\n"
        "• کوئیزهای کلاسی و حضور منظم: ۱ نمره\n"
        "(مجموع: ۲۰ نمره کامل)"
    )


def main():
    print("=" * 60)
    print("🚀 اجرای نمونه ساده و مفهومی RAG (Minimal Python RAG)")
    print("=" * 60)

    # ۱. بارگذاری سند نمونه از mock_data
    mock_file = Path(__file__).parent / "mock_data" / "os_course_syllabus.txt"
    if not mock_file.exists():
        print("❌ فایل نمونه یافت نشد!")
        return

    with open(mock_file, "r", encoding="utf-8") as f:
        document_text = f.read()

    # ۲. ساخت پایگاه برداری و ایندکس کردن
    store = MinimalVectorStore()
    store.add_documents(document_text, chunk_size=80, overlap=15)

    # ۳. سوال نمونه دانشجو
    user_query = "بارم‌بندی درس سیستم عامل چطوریه و پروژه‌ها چند نمره دارند؟"
    print(f"\n❓ سوال دانشجو (Query): {user_query}\n")

    # ۴. بازیابی (Retrieval)
    results = store.search(user_query, top_k=2)
    print("🔍 تکه‌های بازیابی‌شده از دیتابیس برداری (Retrieved Chunks):")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  [{i}] امتیاز شباهت: {score:.4f}")
        print(f"      متن تکه: {chunk[:120]}...\n")

    # ۵. تولید پاسخ (Generation)
    print("🤖 خروجی نهایی مدل زبانی (RAG Generated Response):")
    answer = generate_answer(user_query, results)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
