# Fake_News_Detection


achived a 0.9962205574677735 % accuracy score 



# what id tfid 
Bahut badhiya 🔥 tu sahi jagah pe focus kar raha hai — TF-IDF samajh gaya toh pura text ML easy lagne lagega.
Chalo step-by-step **simple numeric example** se samjhte hain 👇

---

## 🧩 Step 1: Text data le lein

Maan le 3 chhote sentences hain:

```python
docs = [
    "I love mango",
    "I love apple",
    "Mango and apple"
]
```

---

## 🧱 Step 2: Vocabulary banana (unique words)

Sab words le lo aur unique bana do (lowercase karne ke baad):
👉 ['i', 'love', 'mango', 'apple', 'and']

Ye **5 words = 5 dimensions** ban gaye.
Ab har sentence ka ek vector banega of length 5.

| Word Index | Word  |
| ---------- | ----- |
| 0          | i     |
| 1          | love  |
| 2          | mango |
| 3          | apple |
| 4          | and   |

---

## 🧮 Step 3: TF (Term Frequency)

TF = word kitni baar aaya / total words in that sentence

| Sentence          | i   | love | mango | apple | and |
| ----------------- | --- | ---- | ----- | ----- | --- |
| "I love mango"    | 1/3 | 1/3  | 1/3   | 0     | 0   |
| "I love apple"    | 1/3 | 1/3  | 0     | 1/3   | 0   |
| "Mango and apple" | 0   | 0    | 1/3   | 1/3   | 1/3 |

So far yeh **term frequency** hai.
Abhi tak sabhi words barabar important lag rahe hain.
Lekin “I” aur “love” har jagah hain — toh unka importance kam hona chahiye.

---

## ⚖️ Step 4: IDF (Inverse Document Frequency)

IDF measure karta hai:

> “Yeh word kitne documents mein aata hai — jitne zyada, utni kam importance.”

Formula:
[
IDF = \log\left(\frac{N}{n_i}\right)
]

* N = total documents = 3
* ( n_i ) = number of docs jisme word aaya

| Word  | n_i | IDF = log(3 / n_i) |
| ----- | --- | ------------------ |
| i     | 2   | log(3/2) = 0.176   |
| love  | 2   | 0.176              |
| mango | 2   | 0.176              |
| apple | 2   | 0.176              |
| and   | 1   | log(3/1) = 1.098   |

So “and” sabse rare hai ⇒ sabse **zyada important** word.

---

## 🧠 Step 5: TF × IDF = TF-IDF Value

Ab har cell mein multiply kar do TF × IDF:

| Sentence          | i                 | love              | mango             | apple | and               |
| ----------------- | ----------------- | ----------------- | ----------------- | ----- | ----------------- |
| "I love mango"    | 0.333×0.176=0.058 | 0.333×0.176=0.058 | 0.333×0.176=0.058 | 0     | 0                 |
| "I love apple"    | 0.058             | 0.058             | 0                 | 0.058 | 0                 |
| "Mango and apple" | 0                 | 0                 | 0.058             | 0.058 | 0.333×1.098=0.366 |

---

## 📊 Step 6: Final TF-IDF Vectors

| Sentence          | TF-IDF Vector               |
| ----------------- | --------------------------- |
| "I love mango"    | [0.058, 0.058, 0.058, 0, 0] |
| "I love apple"    | [0.058, 0.058, 0, 0.058, 0] |
| "Mango and apple" | [0, 0, 0.058, 0.058, 0.366] |

Ab dekho:

* “and” ka weight **highest** (rare word)
* “I”, “love” ka weight **low** (common words)
* Is tarah har sentence numeric vector ban gaya jo meaning preserve karta hai.

---

## 🧰 Step 7: Model ke liye ready

Ab ye vectors LinearSVC ko diye jaate hain:

```
"I love mango" → [0.058, 0.058, 0.058, 0, 0]
```

Model in numbers se seekhta hai ki kis tarah ke pattern positive, negative, spam, etc. hota hai.

---

## 🔍 Chhota Python demo (samajhne ke liye)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = ["I love mango", "I love apple", "Mango and apple"]
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(docs)

print(tfidf.get_feature_names_out())
print(X.toarray())
```

Output kuch aisa hoga:

```
['and' 'apple' 'i' 'love' 'mango']
[[0.     0.     0.577  0.577  0.577]
 [0.     0.577  0.577  0.577  0.    ]
 [0.707  0.5    0.     0.     0.5  ]]
```

Yaani har row = ek sentence ka TF-IDF vector
Aur har column = ek unique word ka weight.

---

Aise hi `TfidfVectorizer()` **automatically TF aur IDF calculate karke** har text ko numeric matrix bana deta hai.

Chaahe lakhon sentences ho — sabka vector ban jaata hai efficiently ⚡

---

Chahe toh main tujhe ek aur **visual explanation** bana ke dikha du (diagram type) jisme TF aur IDF flow dikhega — chahiye kya?

