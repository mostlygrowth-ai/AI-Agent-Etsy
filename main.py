import os
import base64
from dotenv import load_dotenv

# Tikrinam Python versiją
import sys
if sys.version_info < (3, 10):
    print("Rekomenduojama Python 3.10+")

# Stable Diffusion
from diffusers import StableDiffusionPipeline
import torch

# OpenAI GPT
from openai import OpenAI

# 1️⃣ Įkeliam .env kintamuosius
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY nerastas. Patikrink .env failą!")

client = OpenAI(api_key=api_key)
print("🤖 AI agentas paleistas...")

# 2️⃣ Funkcija idėjoms normalizuoti
def normalize_idea(idea_text):
    """
    Trumpina idėją iki aiškaus objekto + emocijos/veiksmo.
    Pašalina ilgus aprašymus, fone esančius elementus, nebūtinas frazes.
    """
    keywords_to_remove = [
        "minimalist", "line art", "vector", "t-shirt print",
        "illustration", "background", "centered", "composition",
        "high contrast", "simple"
    ]
    
    idea_clean = idea_text.lower()
    for word in keywords_to_remove:
        idea_clean = idea_clean.replace(word, "")
    
    # Pašalina simbolius
    idea_clean = idea_clean.replace(".", "").replace(",", "")
    
    # Sutrumpina iki ~5–7 žodžių
    words = idea_clean.strip().split()
    short_idea = " ".join(words[:7])
    
    # Pirmoji raidė didžioji
    return short_idea.capitalize()

# 3️⃣ Promptas idėjoms generuoti
prompt = """
Sugeneruok idėją minimalistinių marškinėlių dizainui.
Be pasikartojimų, be frazių "less is more" ar "minimal", be žinomų citatų.

"""

# 4️⃣ Generuojame naujas idėjas su GPT
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Tu esi minimalistinių dizainų asistentas."},
        {"role": "user", "content": prompt}
    ]
)

raw_ideas = [line.strip() for line in response.choices[0].message.content.splitlines() if line.strip() != ""]
naujos_idejos = [normalize_idea(i) for i in raw_ideas]

# 5️⃣ Įrašome naujas idėjas į rezultatai.txt
rezultatai_file = "rezultatai.txt"
senos_idejos = set()
if os.path.exists(rezultatai_file):
    with open(rezultatai_file, "r", encoding="utf-8") as f:
        senos_idejos = set(line.strip() for line in f if line.strip() != "")

# Filtruojam tik naujas
final_ideas = [i for i in naujos_idejos if i not in senos_idejos]

with open(rezultatai_file, "a", encoding="utf-8") as f:
    for id in final_ideas:
        f.write(id + "\n")

print(f"📝 {len(final_ideas)} naujų idėjų įrašyta į {rezultatai_file}")
for i in final_ideas:
    print("🎯", i)

# 6️⃣ Stable Diffusion pipeline (256x256)
print("🎨 Kuriu Stable Diffusion pipeline...")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
sd_pipe = sd_pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# 7️⃣ Generuojame paveikslėlius
if not os.path.exists("images"):
    os.makedirs("images")

for idx, idea in enumerate(final_ideas, start=1):
    print(f"🖼️ Generuoju paveikslėlį {idx}: {idea}")
    try:
        prompt_image = f"""
Minimalist black line art illustration, white background, simple continuous black lines,
flat vector style, no background, no scenery, no shading, no gradients,
high contrast, centered composition, t-shirt print design.
Subject: {idea}
"""
        negative_prompt = """
landscape, scenery, sky, grass, nature, background, colorful, colors,
realistic, photo, painting, shadows, texture, gradient
"""
        image = sd_pipe(
            prompt_image,
            negative_prompt=negative_prompt,
            height=256,
            width=256,
            guidance_scale=7.5
        ).images[0]

        filename = f"images/{idea[:10].replace(' ', '_')}_{idx}.png"
        image.save(filename)
    except Exception as e:
        print(f"❌ Klaida generuojant paveikslėlį '{idea}': {e}")

print("✅ Nauji paveikslėliai sugeneruoti ir įrašyti į images/ folderį")
