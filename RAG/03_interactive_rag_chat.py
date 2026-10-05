#!/usr/bin/env python3
"""
03_interactive_rag_chat.py
==========================
چت‌بات تعاملی در ترمینال برای پرسش و پاسخ درباره قوانین دانشگاه و سیلابس درس سیستم عامل.
می‌توانید هر سوالی بپرسید تا سیستم چانک‌های مرتبط را جستجو و پاسخ دهد.

اجرا:
    python3 03_interactive_rag_chat.py
"""

import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_knowledge_base():
    """بارگذاری و قطعه‌بندی فایل‌های متنی دانشگاهی"""
    base_dir = Path(__file__).parent / "mock_data"
    files = [
        (base_dir / "os_course_syllabus.txt", "سیلابس سیستم‌عامل"),
        (base_dir / "university_faq.txt", "آیین‌نامه آموزشی"),
    ]

    chunks = []
    metadata = []

    for file_path, source in files:
        if not file_path.exists():
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # تکه‌تکه کردن بر اساس بخش‌های مارک‌داون
            raw_sections = [s.strip() for s in content.split("## ") if s.strip()]
            for sec in raw_sections:
                chunks.append(sec)
                metadata.append(source)

    return chunks, metadata


def main():
    print("=" * 60)
    print("🎓 به دستیار هوشمند دانشجویی (RAG Chatbot) خوش آمدید!")
    print("=" * 60)
    print("این چت‌بات مجهز به RAG است و به اسناد زیر دسترسی دارد:")
    print("  • سیلابس و بارم‌بندی درس سیستم‌های عامل")
    print("  • آیین‌نامه و سوالات متداول آموزشی دانشگاه\n")

    chunks, metadata = load_knowledge_base()
    if not chunks:
        print("❌ فایل‌های داده در پوشه mock_data یافت نشدند!")
        return

    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(chunks)

    print(f"✨ پایگاه داده با موفقیت با {len(chunks)} بخش آموزشی بارگذاری شد.")
    print("💡 برای خروج عبارت 'exit' یا 'خروج' را تایپ کنید.\n")

    suggested_questions = [
        "بارم‌بندی و پروژه‌های درس سیستم‌های عامل چطوریه؟",
        "شرایط مشروطی و حداقل معدل در دانشگاه چیه؟",
        "حداکثر چند جلسه غیبت در کلاس مجاز هست؟",
        "شرایط و زمان اخذ واحد کارآموزی چیه؟",
    ]
    print("چند نمونه سوال پیشنهادی که می‌توانید بپرسید:")
    for q in suggested_questions:
        print(f"  👉 {q}")
    print("-" * 60)

    # اگر کاربر در محیط غیرتعاملی یا تستی اجرا کرد:
    import sys

    while True:
        try:
            if not sys.stdin.isatty():
                # حالت تست غیرتعاملی
                query = "شرایط مشروطی چیست؟"
                print(f"\n[حالت تست خودکار] سوال: {query}")
                is_demo = True
            else:
                query = input("\n💬 سوال شما: ").strip()
                is_demo = False

            if not query:
                continue

            if query.lower() in ["exit", "quit", "q", "خروج"]:
                print("👋 خداحافظ! موفق و پیروز باشید.")
                break

            # ۱. بازیابی تکه‌های مرتبط
            query_vec = vectorizer.transform([query])
            scores = cosine_similarity(query_vec, doc_vectors).flatten()
            top_indices = scores.argsort()[::-1][:2]

            print("\n🔎 [فاز ۱ - بازیابی]: تکه‌های مرتبط یافت‌شده:")
            retrieved_texts = []
            for rank, idx in enumerate(top_indices, 1):
                score = scores[idx]
                source = metadata[idx]
                text = chunks[idx]
                retrieved_texts.append(f"[{source}]:\n{text}")
                preview = text.replace("\n", " ")[:110]
                print(f"  ({rank}) منبع: {source} | امتیاز شباهت: {score:.3f}")
                print(f"      خلاصه: {preview}...")

            # ۲. تزریق و تولید پاسخ
            print("\n🤖 [فاز ۲ و ۳ - غنی‌سازی و پاسخ هوشمند]:")
            context = "\n\n".join(retrieved_texts)

            # بررسی کلید OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key)
                    prompt = f"با توجه به متن زیر، به سوال دانشجو پاسخ دقیق بده:\n{context}\n\nسوال: {query}"
                    res = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                    )
                    print(res.choices[0].message.content)
                except Exception as e:
                    print(f"خطا در OpenAI: {e}")
            else:
                # پاسخ با استخراج دقیق از کانتکست
                best_chunk = chunks[top_indices[0]]
                print(f"بر اساس {metadata[top_indices[0]]}:\n{best_chunk}")

            print("-" * 60)

            if is_demo:
                break

        except (KeyboardInterrupt, EOFError):
            print("\n👋 خداحافظ!")
            break


if __name__ == "__main__":
    main()
