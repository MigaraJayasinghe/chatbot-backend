# generate_questions.py

from pdf_reader import extract_text_from_pdf, split_text_into_chunks
import google.generativeai as genai

# === Set up Google Gemini ===
API_KEY = "YOUR_GEMINI_API_KEY"  # Replace with your real API key
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# === PDF File Path ===
pdf_path = "pdfs\අත්වැල 01.pdf"  # Update this!

# === Step 1: Extract and Split Text ===
full_text = extract_text_from_pdf(pdf_path)
chunks = split_text_into_chunks(full_text, max_length=700)  # Larger chunks for better question generation

# === Step 2: Generate Questions ===
all_questions = []

for idx, chunk in enumerate(chunks):
    print(f"🧩 Generating questions from chunk {idx + 1}/{len(chunks)}...")

    prompt = f"""
    Based on the following text, generate 5 exam-style questions in Sinhala, English, or Tamil (based on the text language):

    --- TEXT ---
    {chunk}
    --- END TEXT ---

    Only return the questions.
    """

    try:
        response = gemini_model.generate_content(prompt)
        questions = response.text.strip()
        all_questions.append(questions)
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        continue

# === Step 3: Save or Print Questions ===
with open("generated_questions.txt", "w", encoding="utf-8") as f:
    for q in all_questions:
        f.write(q + "\n\n")

print("✅ Questions generated and saved to 'generated_questions.txt'!")
