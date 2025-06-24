import pandas as pd
import re
import os

def clean_amharic_text(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u1200-\u137F\w\s።፡፣፤]', '', text)
    return text.strip()

def preprocess_amharic_csv(input_file='scraped_data.csv', output_file='preprocessed_amharic.csv'):
    if not os.path.exists(input_file):
        print(f"❌ File '{input_file}' not found.")
        return

    df = pd.read_csv(input_file)
    print("📄 Columns found:", df.columns.tolist())

    message_col = None
    for col in df.columns:
        if 'message' in col.lower() or 'text' in col.lower():
            message_col = col
            break

    channel_col = None
    for col in df.columns:
        if 'channel' in col.lower():
            channel_col = col
            break

    if not message_col or not channel_col:
        print("❌ Could not find a column with 'message' or 'channel' in the name.")
        return

    # Clean message column safely
    df['clean_message'] = df[message_col].apply(clean_amharic_text)

    output_cols = [channel_col, 'clean_message']
    if 'timestamp' in df.columns:
        output_cols.insert(1, 'timestamp')
    elif 'date' in df.columns:
        output_cols.insert(1, 'date')

    df[output_cols].to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_amharic_csv()
