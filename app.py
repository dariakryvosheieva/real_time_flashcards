# === real_time_flashcards/app.py ===

import os
import re
import json
import queue
import threading
from datetime import datetime

import numpy as np
import sounddevice as sd
import whisper
from flask import Flask, render_template, request, redirect, url_for
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

nltk.download('punkt')

# === Globals ===
samplerate = 16000
block_duration = 5
recording_queue = queue.Queue()
model = whisper.load_model("base")
sections = {}
section_filenames = []
current_section_index = 0
transcriber_thread_running = False
audio_stream = None

# === Flask factory ===
def create_app():
    app = Flask(__name__)

    @app.route('/')
    def index():
        global current_section_index
        print("📘 Route accessed. Sections available:", section_filenames)
        if not section_filenames:
            return "❌ No sections loaded.", 500
        section_name = section_filenames[current_section_index]
        with open(os.path.join('book_sections', section_name), 'r', encoding='utf-8') as f:
            html = f.read()
        return render_template('page.html', content=html, section=section_name)

    @app.route('/next')
    def next_page():
        global current_section_index
        if current_section_index < len(section_filenames) - 1:
            current_section_index += 1
        return redirect(url_for('index'))

    @app.route('/prev')
    def prev_page():
        global current_section_index
        if current_section_index > 0:
            current_section_index -= 1
        return redirect(url_for('index'))

    return app

# === Flashcard and audio logic ===
def load_sections_from_folder(folder_path):
    result = {}
    print(f"📂 Checking folder: {folder_path}")
    for filename in sorted(os.listdir(folder_path)):
        print(f"🔍 Found file: {filename}")
        if filename.endswith('.html'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            result[filename] = sent_tokenize(text)
    return result

def highlight_word(sentence, word):
    pattern = re.compile(re.escape(word), re.IGNORECASE)
    return pattern.sub(r'<mark>\g<0></mark>', sentence, count=1)

def save_flashcard(word, sentence, section_name, path='flashcards.json'):
    flashcard = {
        "word": word,
        "sentence": sentence,
        "section": section_name,
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = {"flashcards": []}

    data["flashcards"].append(flashcard)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_spoken_text(text, sentences, section_name):
    spoken_words = word_tokenize(text)
    for word in spoken_words:
        for sentence in sentences:
            if word.lower() in [w.lower() for w in word_tokenize(sentence)]:
                highlighted = highlight_word(sentence, word)
                save_flashcard(word, highlighted, section_name)
                print(f"✅ Flashcard saved for '{word}': {highlighted}")
                return

def transcriber_thread():
    global current_section_index
    while True:
        audio_chunk = recording_queue.get()
        audio_np = audio_chunk.flatten().astype(np.float32)
        result = model.transcribe(audio_np, fp16=False)
        spoken_text = result['text']
        print("🗣️", spoken_text.strip())

        section_name = section_filenames[current_section_index]
        sentences = sections[section_name]
        process_spoken_text(spoken_text, sentences, section_name)

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio Status:", status)
    recording_queue.put(indata.copy())

def start_background_threads():
    global transcriber_thread_running, audio_stream
    if not transcriber_thread_running:
        threading.Thread(target=transcriber_thread, daemon=True).start()
        audio_stream = sd.InputStream(
            channels=1,
            samplerate=samplerate,
            callback=audio_callback,
            blocksize=int(samplerate * block_duration)
        )
        audio_stream.start()
        transcriber_thread_running = True

# === App entry point ===
if __name__ == '__main__':
    sections = load_sections_from_folder('book_sections')
    section_filenames = list(sections.keys())
    start_background_threads()
    app = create_app()
    app.run(debug=True)
