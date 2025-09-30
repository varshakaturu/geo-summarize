# Text Summarization & Geo-Tagged QA Pipeline

[🌐 Geo-Summarizer Demo](https://huggingface.co/spaces/varshakaturu/geo-summarizer)

This project processes raw news articles to:
1. **Summarize** them into short abstractive summaries (using Pegasus).
2. **Answer user questions** about the article (using DistilBERT QA).
3. **Extract locations** (via spaCy NER + geopy) and plot them on an interactive **map** with coordinates.

All features are combined in an interactive **Gradio web interface**.

---

## Features

- **Abstractive Summarization**  
  Generates human-like summaries instead of extractive snippets.

- **Question Answering - Hybrid Method**  
  Users can ask specific questions about the article.  
  → Uses summary first (for efficiency), falls back to full article for accuracy.

- **Geo-Tagging & Visualization**  
  - Extracts all mentioned locations from the article text.  
  - Geocodes them into latitude/longitude.  
  - Displays them on a **Folium map** with markers and a coordinate overlay.
 
- **Batch Processing (CSV → CSV)**  
  Summarized and geo-tagged **5,000+ rows from `train.csv`**, producing an **`output.csv`** with:
  - `id` → original row ID
  - `reference` → original article/text
  - `summary` → abstractive summary  
  - `locations` → detected places  
  - `lat`, `lon` → geographic coordinates (latitude and longitude)
    
- **Rouge Scores**  
  Gives Rouge 1, Rouge 2 and Rouge L scores.

- **Interactive Web Interface**  
  Built with Hugging Face - Gradio.

---

## Tech Stack
- **Python 3.10**  
- **Transformers (Hugging Face)** – Summarization & QA  
- **spaCy** – Named Entity Recognition (location extraction)  
- **Geopy** – Geocoding (lat/lon)  
- **Folium** – Interactive maps  
- **Gradio** – Web-based UI  
- **Pandas** – Data handling  
- **BeautifulSoup4** – Text cleaning  


---
## Interface

### Input

- Paste a news article in the Article Text box.

- Optionally, enter a question in the Question box.

### Output

- Summary: Abstractive summary of the article.

- Map with Locations: Interactive Folium map with detected places.

- QA Answer: Direct answer to the user’s question.

- Rouge Scores: Accuracy of the model.

---

## Future Improvements

- Support for multiple QA models (e.g., GPT-style).

- Multilingual support for non-English articles.

- Batch processing for CSV/JSON datasets with progress tracking.

- Integration with PowerBI or dashboards for visualization.

---

## Ethical AI & Bias Considerations
- **Summarization Bias:** Automated summarizers may unintentionally omit important context or emphasize certain details, leading to biased interpretations of news articles.  
- **Named Entity Recognition (NER):** Location extraction relies on pretrained models, which may misidentify or fail to recognize non-Western or underrepresented place names.  
- **Geocoding Risks:** Mapping real-world locations from text could potentially expose sensitive information if applied carelessly.  
- **Mitigation:**  
  - This project is intended for research and educational use only.  
  - Summaries should be treated as *assistance*, not as authoritative sources.  
  - Users should verify extracted locations and summaries against original sources.  


---
[Kaggle Dataset](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-)
---

## Contributing

Pull requests are welcome! Please fork the repo and create a feature branch.




