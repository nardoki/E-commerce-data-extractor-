import pandas as pd
import re
import random

# ------------------ Keywords ------------------
location_keywords = set([
    "መገናኛ", "ቦሌ", "ሜክሲኮ", "መሰረት", "ደፋርሞል", "ፕላዛ", "መዳህኒዓለም",
    "ሁለተኛፎቅ", "ቢሮ", "ቁ", "ቁጥር", "አድራሻ", "ሲቲ", "አዳማ", "ፎቅ", "ቤተክርስቲያን",
    "Addis", "mexico", "bole", "ዛምሞል", "ቄራ", "ሞል", "ህንፃ"
])

price_keywords = ["ብር", "ዋጋ", "የዋጋ", "በ", "ተቀናጀ", "በነፃ"]
promo_keywords = ["በረፍት", "ሱቅ", "ሱቃችን", "እንኳን", "እንገኛለን", "ቅናሽ", "Eid"]
delivery_keywords = ["ዲሊቨሪ", "Free", "እናደርሳለን", "በሞተረኞች"]

# ------------------ Helper Functions ------------------
def is_amharic(token):
    return bool(re.search(r'[\u1200-\u137F]', token))

def is_price(token):
    return is_amharic(token) and ('ብር' in token or 'ዋጋ' in token)

def is_location(token):
    return is_amharic(token) and any(loc in token for loc in location_keywords)

def is_promo_or_delivery(token):
    return is_amharic(token) and any(k in token for k in promo_keywords + delivery_keywords)

def is_possible_product(token):
    return is_amharic(token) and not is_location(token) and not is_price(token)

# ------------------ Labeling Logic ------------------
def label_tokens(message):
    tokens = message.strip().split()
    labels = []
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if not is_amharic(token):
            i += 1
            continue  # Skip non-Amharic tokens

        if is_promo_or_delivery(token):
            labels.append((token, "O"))
        elif is_price(token):
            labels.append((token, "B-PRICE"))
            i += 1
            while i < len(tokens) and re.match(r'[\d,]+', tokens[i]):
                if is_amharic(tokens[i]):
                    labels.append((tokens[i], "I-PRICE"))
                i += 1
            continue
        elif is_location(token):
            labels.append((token, "B-LOC"))
            i += 1
            while i < len(tokens) and is_location(tokens[i]):
                labels.append((tokens[i], "I-LOC"))
                i += 1
            continue
        elif is_possible_product(token):
            labels.append((token, "B-Product"))
            i += 1
            while i < len(tokens) and is_possible_product(tokens[i]):
                labels.append((tokens[i], "I-Product"))
                i += 1
            continue
        else:
            labels.append((token, "O"))

        i += 1

    return labels

# ------------------ Load & Process Dataset ------------------
df = pd.read_csv("scraped_data.csv")  # or 'public_telegram_messages.csv'
text_col = "text" if "text" in df.columns else "message"
messages = df[text_col].dropna().tolist()
random.seed(42)
messages = random.sample(messages, min(len(messages), 50))

# ------------------ Write to CoNLL Format ------------------
with open("labeled_conll.txt", "w", encoding="utf-8") as f:
    for msg in messages:
        pairs = label_tokens(msg)
        if not pairs:
            continue
        for word, tag in pairs:
            f.write(f"{word} {tag}\n")
        f.write("\n")  # Blank line between messages

print("✅ Labeled Amharic-only tokens to CoNLL format → labeled_conll.txt")
