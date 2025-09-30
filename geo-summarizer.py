# =========================
# Install dependencies
# =========================
# pip install transformers spacy geopy pandas beautifulsoup4 folium gradio torch --quiet

import spacy
import time
import pandas as pd
import json
from geopy.geocoders import Nominatim
from transformers import pipeline
from bs4 import BeautifulSoup
import folium
from folium.plugins import MarkerCluster
import gradio as gr
import torch
import os
from rouge_score import rouge_scorer

# =========================
# Ensure spaCy model is available
# =========================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =========================
# Geocoding Setup
# =========================
geolocator = Nominatim(user_agent="geo_app")
_geocache_path = "geocache.json"

# Load or initialize geocache
if os.path.exists(_geocache_path):
    try:
        with open(_geocache_path, "r", encoding="utf-8") as f:
            _geocache = json.load(f)
            if not isinstance(_geocache, dict):
                _geocache = {}
    except json.JSONDecodeError:
        _geocache = {}
else:
    _geocache = {}

def save_cache():
    with open(_geocache_path, "w", encoding="utf-8") as f:
        json.dump(_geocache, f, ensure_ascii=False, indent=2)

def geocode_location(name, pause=1):
    if not name:
        return None
    if name in _geocache:
        return _geocache[name]
    try:
        loc = geolocator.geocode(name, timeout=10)
        time.sleep(pause)
        if loc:
            res = {"lat": loc.latitude, "lon": loc.longitude, "raw": loc.address}
            _geocache[name] = res
            save_cache()
            return res
    except Exception:
        _geocache[name] = None
        save_cache()
    return None

# =========================
# NER for Locations
# =========================
def extract_locations(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC", "FAC")]

# =========================
# Summarization & QA Models
# =========================
device = 0 if torch.cuda.is_available() else -1

# Pegasus XSum for abstractive summarization
summarizer = pipeline("summarization", model="google/pegasus-xsum", device=device)

# DistilBERT QA
qa_pipeline = pipeline(
    "question-answering",
    model="distilbert-base-uncased-distilled-squad",
    tokenizer="distilbert-base-uncased-distilled-squad",
    device=device
)

# =========================
# Helpers
# =========================
def chunk_text(text, max_tokens=500):
    words = text.split()
    for i in range(0, len(words), max_tokens):
        yield " ".join(words[i:i+max_tokens])

def summarize_full_text(article):
    chunks = list(chunk_text(article, max_tokens=500))
    summaries = []
    for chunk in chunks:
        try:
            s = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            summaries.append(s)
        except:
            continue
    return " ".join(summaries)

def hybrid_answer_question(article_text, summary_text, question, confidence_threshold=0.8):
    try:
        result = qa_pipeline(question=question, context=summary_text)
        if result['score'] >= confidence_threshold:
            return result['answer']
        result_full = qa_pipeline(question=question, context=article_text)
        return result_full['answer']
    except Exception:
        return "Could not generate answer."

def generate_map(locations):
    if not locations:
        return "<p>No locations found</p>"

    first = locations[0]
    m = folium.Map(location=[first['lat'], first['lon']], zoom_start=4)
    cluster = MarkerCluster().add_to(m)

    # Add markers
    for loc in locations:
        folium.Marker(
            location=[loc['lat'], loc['lon']],
            popup=f"{loc['location']}<br>Lat: {loc['lat']:.4f}, Lon: {loc['lon']:.4f}"
        ).add_to(cluster)

    # Overlay box with all coords
    all_coords = "<br>".join([f"{loc['location']}: ({loc['lat']:.4f}, {loc['lon']:.4f})" for loc in locations])
    m.get_root().html.add_child(folium.Element(f'''
        <div style="position: fixed;
                    bottom: 10px; left: 10px; width: 250px; height: auto;
                    background-color: rgba(255,255,255,0.8);
                    padding: 8px; border-radius: 5px; z-index:999;">
            <b>Coordinates:</b><br>
            {all_coords}
        </div>
    '''))

    return m._repr_html_()

# =========================
# Main processing for Gradio
# =========================
def process_article(article_text, question=None):
    summary = summarize_full_text(article_text)

    locations = extract_locations(article_text)
    coords = []
    for loc in locations:
        g = geocode_location(loc)
        if g:
            coords.append({"location": loc, "lat": g["lat"], "lon": g["lon"]})

    answer = None
    if question:
        answer = hybrid_answer_question(article_text, summary, question)

    map_html = generate_map(coords) if coords else "<p>No locations to map</p>"

    return summary, map_html, answer

# =========================
# Batch Processing Function for CSV
# =========================
def process_dataset(input_csv, output_csv, max_rows=5251, progress_interval=50):
    df = pd.read_csv(train_csv).head(max_rows)
    
    if 'summary' not in df.columns:
        df['summary'] = ""
    if 'locations' not in df.columns:
        df['locations'] = ""
    if 'lat_lon' not in df.columns:
        df['lat_lon'] = ""
    
    total_rows = len(df)
    for idx, row in df.iterrows():
        article_text = row.get('article_text', "")
        
        # Summarize
        summary = summarize_full_text(article_text)
        df.at[idx, 'summary'] = summary
        
        # Locations & geocoding
        locations = extract_locations(article_text)
        coords = []
        for loc in locations:
            g = geocode_location(loc)
            if g:
                coords.append({"location": loc, "lat": g["lat"], "lon": g["lon"]})
        
        df.at[idx, 'locations'] = [loc['location'] for loc in coords]
        df.at[idx, 'lat_lon'] = [{"lat": loc['lat'], "lon": loc['lon']} for loc in coords]
        
        # Show progress
        if (idx + 1) % progress_interval == 0 or (idx + 1) == total_rows:
            print(f"Processed {idx+1}/{total_rows} rows")
            df.to_csv(output_csv, index=False)  # save intermittently
    
    # Final save
    df.to_csv(output_train, index=False)
    print(f"Finished processing {total_rows} rows. Output saved to {output_train}")



## Rouge Scores
# Load CSV
df = pd.read_csv("/content/drive/MyDrive/geo_summarizer/output_train.csv")
# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
# Store scores
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
for _, row in df.iterrows():
    reference = str(row['reference'])      # or a longer reference text if available
    summary = str(row['summary'])
    scores = scorer.score(reference, summary)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)
# Add scores to dataframe
df['rouge1'] = rouge1_scores
df['rouge2'] = rouge2_scores
df['rougeL'] = rougeL_scores
# Calculate average ROUGE scores
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average ROUGE-2: {avg_rouge2:.4f}")
print(f"Average ROUGE-L: {avg_rougeL:.4f}")

# =========================
# Gradio Interface
# =========================
iface = gr.Interface(
    fn=process_article,
    inputs=[
        gr.Textbox(label="Article Text", lines=15, placeholder="Paste news article here..."),
        gr.Textbox(label="Question", placeholder="Ask a question about the article")
    ],
    outputs=[
        gr.Textbox(label="Article Summary", lines=15),
        gr.HTML(label="Map with Locations"),
        gr.Textbox(label="Answer to your question", lines=10)
    ],
    title="Text Summarization & Geo-Tagged QA Pipeline",
    allow_flagging="never"
)

# Launch interface
iface.launch(share=True)
