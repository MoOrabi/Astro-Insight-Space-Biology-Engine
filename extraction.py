import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import nltk
import tiktoken
from sentence_transformers import SentenceTransformer
import chromadb

# 1. تحميل الموديل من HuggingFace (مجاني وسريع)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


nltk.download("punkt")
nltk.download("punkt_tab")

tokenizer = tiktoken.get_encoding("cl100k_base")


headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/119.0.0.0 Safari/537.36"
}


def fetch_text(url):
    try:
        response = requests.get(url, headers= headers, timeout=20)
        response.raise_for_status()
        return response.text  # assumes link returns plain text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def clean_text(text):
    # remove page numbers
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    # remove multiple spaces/newlines
    # text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(text, max_tokens=512, overlap=50):
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0

    for sent in sentences:
        sent_tokens = tokenizer.encode(sent)
        if current_len + len(sent_tokens) > max_tokens:
            # خزّن chunk
            chunks.append(" ".join(current_chunk))
            # ابدأ chunk جديد مع overlap
            overlap_sentences = current_chunk[-overlap:] if overlap else []
            current_chunk = overlap_sentences + [sent]
            current_len = sum(len(tokenizer.encode(s)) for s in current_chunk)
        else:
            current_chunk.append(sent)
            current_len += len(sent_tokens)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


df = pd.read_csv("SB_publication_PMC-test.csv")

df["raw_text"] = df["Link"].apply(fetch_text)

df["clean_text"] = df["raw_text"].dropna().apply(clean_text)

soup = BeautifulSoup(df["clean_text"][0], "html.parser")

# 1. هات كل section id بيبدأ بـ s
sections = soup.select("section[id^=s]")

# 2. من جوه كل سكشن هات الـ p فقط
paragraphs = []
for sec in sections:
    ps = [p.get_text(strip=True) for p in sec.find_all("p")]
    paragraphs.extend(ps)

# 3. النص النهائي
clean_text = "\n".join(paragraphs)

chunks = chunk_text(clean_text, max_tokens=30, overlap=2)

# for i, c in enumerate(chunks, 1):
#     print(f"Chunk {i}: {c}\n")

embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)

client = chromadb.PersistentClient(path="chroma_db")

collection = client.create_collection(name="research_papers")

# 6. إضافة البيانات (chunks + embeddings)
collection.add(
    embeddings=embeddings,
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)


