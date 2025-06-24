import asyncio
import os
import re
import pandas as pd
from datetime import datetime
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest

# Create directories
os.makedirs("media", exist_ok=True)

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[^\u1200-\u137F\sA-Za-z\d.,!?@]', '', text)  # Keep Amharic, Latin, digits, punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

    #Connects to Telegram and scrapes a defined number of messages from each specified channel.
async def fetch_channel_messages(api_id, api_hash, phone, channels, limit=100):
    async with TelegramClient("user_session", api_id, api_hash) as client:
        await client.start(phone=phone)
        all_data = []

        for channel in channels:
            print(f"\n Scraping channel: @{channel}")
            try:
                entity = await client.get_entity(channel)
                history = await client(GetHistoryRequest(
                    peer=entity,
                    limit=limit,
                    offset_date=None,
                    offset_id=0,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))

                for msg in history.messages:
                    text = clean_text(msg.message)
                    if not text:
                        continue

                    image_path = ""
                    if msg.media and hasattr(msg.media, 'photo'):
                        image_path = f"media/{channel}_{msg.id}.jpg"
                        await client.download_media(msg, file=image_path)

                    all_data.append({
                        "channel": channel,
                        "message_id": msg.id,
                        "text": text,
                        "timestamp": msg.date.isoformat(),
                        "views": msg.views or 0,
                        "media_path": image_path
                    })

            except Exception as e:
                print(f"❌ Failed to scrape @{channel}: {e}")

        return all_data

if __name__ == "__main__":
    print("🔐 Telegram API Login")
    api_id = int(input("Enter your api_id: "))
    api_hash = input("Enter your api_hash: ").strip()
    phone = input("Enter your phone number (with country code): ").strip()

     # List of Telegram channels to scrape
    channels = [
        'Shageronlinestore',
        'aradabrand2',
        'gebeyaadama',
        'ethio_brand_collection',
        'Fashiontera'
    ]

    data = asyncio.run(fetch_channel_messages(api_id, api_hash, phone, channels))

    df = pd.DataFrame(data)
    df.to_csv("scraped_data.csv", index=False, encoding="utf-8-sig")
    print(f"\n Scraping finished. {len(df)} messages saved to scraped_data.csv")
