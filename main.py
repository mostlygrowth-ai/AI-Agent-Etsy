import os
import openai
from PIL import Image
import base64
import requests
from datetime import datetime
import time
from dotenv import load_dotenv

# Įkrauname API raktus iš .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# Folderis dizainams išsaugoti
OUTPUT_FOLDER = "designs"
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Funkcija: sugeneruoti raktažodžius
def generate_keywords(topic):
    prompt = f"Sugeneruok 5-10 raktažodžių, kurie tiktų Etsy produktui tema: {topic}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.7
    )
    keywords = response['choices'][0]['message']['content']
    return keywords.strip().split(", ")

# Funkcija: generuoti dizainą per DALL·E
def generate_design(prompt, filename):
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    image_base64 = response['data'][0]['b64_json']
    image_data = base64.b64decode(image_base64)
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(image_data)
    print(f"Dizainas išsaugotas: {filepath}")

# Pagrindinė agento funkcija
def main():
    topic_list = ["minimalist sarkazmas vasara", "funny quote", "summer vibes"]  # pradinis promptų sąrašas
    while True:
        for topic in topic_list:
            # 1️⃣ Generuojame raktažodžius
            keywords = generate_keywords(topic)
            print(f"Raktažodžiai {topic}: {keywords}")

            # 2️⃣ Generuojame dizainą
            design_prompt = f"{topic}, {', '.join(keywords)}, minimalist style"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{topic.replace(' ', '_')}_{timestamp}.png"
            generate_design(design_prompt, filename)

            # 3️⃣ Palaukti, kad neperspaustume API limitų
            time.sleep(10)  # gali koreguoti pagal savo poreikį

        # 4️⃣ Laukiame valandą iki kito ciklo (arba pagal norą)
        print("Ciklas baigtas. Laukiame 1 valandą...")
        time.sleep(3600)

if __name__ == "__main__":
    main()
