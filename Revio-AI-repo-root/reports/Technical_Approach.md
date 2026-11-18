# Technical Approach Report — Revio AI

**Project:** Revio AI – Intelligent Customer Review Analyzer & Response Generator  
**Prepared for:** Cynbit Technologies internship task  
**Candidate:** Kratika Jain  
**Date:** 18 November 2026

---

## 1. Project overview

Revio AI is a Streamlit prototype that turns unstructured customer reviews into structured insight and a draft response. It is designed as a practical extension to Cynbit's existing **Revio** product: review management, sentiment intelligence and AI-generated replies.

The prototype is deliberately runnable without API keys. It uses a pretrained Hugging Face sentiment model when the package and model weights are available, and a transparent rule/lexicon fallback otherwise. Complaint detection and reply generation are implemented explicitly so reviewers can inspect every decision.

---

## 2. Problem statement

A review operations team needs to answer four questions quickly:

1. Is this review positive, negative or neutral, and how confident are we?
2. What is the customer actually complaining about?
3. How urgent is it?
4. What safe, professional reply can be sent?

Doing this manually is slow and inconsistent; doing it with a black-box model alone is risky. The prototype therefore combines a pretrained classifier with explainable rules and a human-visible editable reply.

---

## 3. Technology stack

| Component | Technology used in this repo | Why |
| --- | --- | --- |
| Programming | Python 3.10+ | Fits Cynbit's Python / AI stack |
| Frontend | Streamlit | Fastest way to build a data/ML app without React or Node.js |
| Data processing | Pandas | CSV handling, aggregation, filtering and display |
| Sentiment | Hugging Face Transformers (`distilbert-base-uncased-finetuned-sst-2-english`) | Strong general-purpose sentiment baseline, no training from scratch |
| Fallback NLP | Custom lexicon + regex preprocessing | Keeps the demo functional offline and explains decisions |
| NLP preprocessing | `utils/preprocessing.py` | Cleaning, tokenisation, stop-word removal, keyword extraction, language detection |
| Response generation | Rule/template engine + optional OpenAI-compatible hook | Safe, deterministic fallback; upgrade path to LLM |
| Visualisation | Plotly | Interactive dashboard charts |
| Database | SQLite | Zero-setup local persistence |
| Version control | Git / GitHub | Required submission workflow |
| Deployment | Streamlit Community Cloud | Requested deployment target |

Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `REVIEW_SENTIMENT_BACKEND` | `auto` (default), `hf` or `lexicon` |
| `REVIEW_SENTIMENT_MODEL` | Override the Hugging Face model ID |
| `REVIO_USE_LLM` | Set to `true` to attempt an LLM-generated reply |
| `OPENAI_API_KEY` | Required only for the optional LLM hook |
| `REVIO_LLM_MODEL` | Defaults to `gpt-4o-mini` |

---

## 4. Model selection

### Chosen model

`distilbert-base-uncased-finetuned-sst-2-english`

### Why this model

- **Pretrained and task-aligned:** fine-tuned on sentiment classification, so no from-scratch training is needed.
- **Small and fast:** DistilBERT is lighter than BERT while retaining strong performance on English sentiment.
- **Good baseline:** suitable for a first version before collecting client-specific labelled data.
- **Simple integration:** Hugging Face `pipeline("sentiment-analysis")` returns a label and probability.

### Why not train from scratch

A small internship prototype does not have enough labelled, client-specific data. Training from scratch would add cost and risk without improving the demonstration. The approach is: start with a strong pretrained baseline, then fine-tune later with real data.

### Why a fallback is included

If `transformers`, weights or internet access are unavailable, the app still works. The fallback:

- scores positive and negative lexicon terms,
- handles common Hindi / Hinglish words,
- partially accounts for negation,
- blends a small star-rating signal,
- returns the same output contract as the transformer path.

This is important for reviewers opening the app on a restricted network.

### Known limitation

SST-2 is English-centric. Hindi and Hinglish reviews are handled by the fallback cues in this prototype. A production upgrade would use a multilingual model such as `XLM-RoBERTa` or an Indic language model fine-tuned on review data. This is **proposed**, not claimed as implemented.

---

## 5. Data collection and preprocessing

### Data source

`data/sample_reviews.csv` contains 45 synthetic, hand-written reviews. Fields:

| Field | Description |
| --- | --- |
| `review_id` | Stable identifier, e.g. `REV-1001` |
| `review_text` | Customer review in English, Hindi or Hinglish |
| `rating` | 1–5 stars |
| `source` | Google, Instagram, Website, Zomato, etc. |
| `created_at` | Timestamp used for trend charts |

No real customer data or Cynbit client data is included.

### Preprocessing steps

`utils/preprocessing.py` implements:

1. URL, handle and hashtag removal.
2. Zero-width / invisible character cleanup.
3. Whitespace normalisation.
4. Unicode-aware tokenisation, including Devanagari range `U+0900–U+097F`.
5. Stop-word removal for English and common romanised Hindi terms.
6. Keyword frequency extraction for “Frequently mentioned issues”.
7. Heuristic language detection: Hindi / Hinglish / English.

Sentiment cues are intentionally preserved: words such as “late”, “rude”, “refund”, “खराब” and “बहुत अच्छा” carry meaning and must not be removed as generic stop words.

### Persistence

On first run the app creates `data/revio.db`, seeds it from the CSV, then analyzes unprocessed rows. New reviews entered in the UI are also stored in SQLite with the analysis result and any edited reply.

---

## 6. System architecture

```text
                         ┌──────────────────────────┐
                         │  CSV batch / UI review   │
                         └────────────┬─────────────┘
                                      │
                                      v
                         ┌──────────────────────────┐
                         │ preprocessing.py         │
                         │ clean / tokenise / detect │
                         └────────────┬─────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  v                                       v
   ┌──────────────────────────────┐      ┌──────────────────────────────┐
   │ sentiment_model.py           │      │ response_generator.py        │
   │ HF transformer / lexicon     │      │ issue detection + priority    │
   │ sentiment + confidence       │      │ template / optional LLM reply │
   └──────────────┬───────────────┘      └──────────────┬───────────────┘
                  └───────────────────┬───────────────────┘
                                      v
                         ┌──────────────────────────┐
                         │ SQLite: reviews table    │
                         │ original + analysis +    │
                         │ edited reply             │
                         └────────────┬─────────────┘
                                      v
        ┌────────────────────────────────────────────────────────────┐
        │ Streamlit pages                                            │
        │ Dashboard | Review Analyzer | AI Response Generator |      │
        │ Review Insights                                            │
        └────────────────────────────────────────────────────────────┘
```

---

## 7. AI pipeline

### Step 1 — Ingestion

- A reviewer pastes one review into the **Review Analyzer**, or the app seeds all CSV rows at startup.
- Each review has text, rating, source and timestamp.

### Step 2 — Sentiment classification

`SentimentAnalyzer.analyze(text, rating)` returns:

```python
{
    "sentiment": "Negative",
    "confidence": 0.856,
    "model": "lexicon-fallback (Hugging Face unavailable)",
    "negative_feedback": True,
    "scores": {"Positive": 0.166, "Negative": 0.834, "Neutral": 0.0},
    "evidence": ["disappointing", "late"]
}
```

The transformer label is mapped to Positive / Negative / Neutral. A small rating signal is blended so star ratings and text broadly agree.

### Step 3 — Complaint detection

`detect_issues(text)` scores an issue taxonomy using weighted phrase matching:

- Delivery delay
- Poor customer support
- Product quality
- Billing / refund
- Staff behavior
- App / website issue
- Cleanliness / hygiene
- Appointment / scheduling
- Safety concern
- Pricing / value

The highest-scoring issue becomes the **main issue**; the next becomes the **secondary issue**.

### Step 4 — Priority

`determine_priority(sentiment, main_issue, rating)`:

- Negative + high-severity issue (or rating ≤ 2) → **High**
- Negative + other issue → **Medium**
- Neutral → **Low**
- Positive → **Low**

Safety, refund, delivery and support failures are intentionally treated as high severity.

### Step 5 — Response generation

`ResponseGenerator.generate()` tries the optional LLM hook first when enabled, then falls back to a brand-safe template. Templates vary by:

- sentiment,
- main and secondary issue,
- priority,
- selected tone (`empathetic`, `professional`, `concise`).

Safety concerns receive an escalation sentence. The reply is always shown in an editable text area; nothing is auto-published.

### Step 6 — Analytics

The Dashboard and Review Insights pages aggregate SQLite data into:

- total reviews,
- positive / negative / neutral percentages,
- high-priority count,
- sentiment donut chart,
- priority and rating bars,
- most common complaints,
- frequently mentioned terms,
- rolling 3-day satisfaction trend.

---

## 8. Implementation

### Repository structure

```text
Revio-AI/
├── app.py
├── data/
│   └── sample_reviews.csv
├── models/
│   ├── __init__.py
│   └── sentiment_model.py
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── response_generator.py
├── reports/
│   ├── AI_Opportunity_Report.md
│   └── Technical_Approach.md
├── requirements.txt
├── README.md
└── .gitignore
```

### Streamlit sections

| Section | Implemented behaviour |
| --- | --- |
| **Dashboard** | KPI cards, sentiment distribution, priority bar, ratings, common complaints, recent reviews table |
| **Review Analyzer** | Text + rating + source form, sentiment/confidence/issue/priority table, confidence gauge, save to SQLite |
| **AI Response Generator** | Use last analysis or paste a new review, choose tone, generate reply, edit, download or save edited reply |
| **Review Insights** | Filter by sentiment/source, complaint chart, keyword chart, satisfaction trend, detailed table and CSV export |

### Runtime behaviour

- `data/revio.db` is created automatically and ignored by Git.
- The sentiment model object is cached with `@st.cache_resource`.
- Pending CSV rows are analyzed once on startup; subsequent runs reuse stored results.
- The sidebar always shows the active sentiment engine and expands the implemented/proposed checklist.

---

## 9. Testing and evaluation

### Manual validation cases

| Input | Expected sentiment | Expected main issue | Expected priority |
| --- | --- | --- | --- |
| “The delivery was very late, and customer support did not respond to my calls. Very disappointing experience.” | Negative | Delivery delay | High |
| “Excellent service! The team was friendly and the product arrived ahead of schedule.” | Positive | Staff behavior / Delivery delay | Low |
| “Average experience. The service was okay but nothing special.” | Neutral | General service | Low |
| “The app crashes every time I try to log in. Please fix this bug.” | Negative | App / website issue | Medium/High depending on rating |
| “बहुत अच्छा अनुभव था। टीम विनम्र थी और डिलीवरी समय पर हुई।” | Positive | Staff behavior | Low |
| “I had a safety concern with the product and need someone to contact me urgently.” | Negative | Safety concern | High |

### How to verify locally

```bash
python -m compileall app.py models utils
python -c "from models.sentiment_model import SentimentAnalyzer; from utils.response_generator import detect_issues; a=SentimentAnalyzer(); t='The delivery was very late, and customer support did not respond to my calls.'; print(a.analyze(t,1)); print(detect_issues(t)[:2])"
streamlit run app.py
```

Then open the URL printed by Streamlit (usually `http://localhost:8501`) and test all four sections.

### Evaluation approach

- **Sentiment:** label vs prediction on a held-out labelled set; report accuracy, precision, recall and F1 per class.
- **Issue extraction:** exact-match of main issue against annotations; report precision/recall per issue type.
- **Priority:** agreement with an operations rubric.
- **Response quality:** human rubric for empathy, relevance, factual safety and tone consistency.
- **Reliability:** verify the app still runs with the lexicon backend by setting `REVIEW_SENTIMENT_BACKEND=lexicon`.

**Current status:** the repo includes the evaluation harness design and manual test cases above. Formal metrics on real client data are **not yet available** and are proposed for the next phase.

### Known limitations

- SST-2 is English-first; Hindi/Hinglish handling in the transformer path is limited.
- Rule-based issue detection can miss sarcasm or unusual phrasing.
- The lexicon fallback is explainable but not a substitute for a fine-tuned production model.
- The LLM hook is optional and untested without a key; the template fallback is the default.
- No live platform connectors or auto-publishing are implemented.

---

## 10. Future improvements

All items below are **proposed**, not implemented in this submission.

1. **Multilingual fine-tuning:** use labelled English/Hindi/Hinglish reviews to fine-tune XLM-R or IndicBERT; evaluate per language.
2. **Platform connectors:** pull reviews from Google Business Profile, Instagram, Facebook and Zomato through approved APIs or approved vendor integrations.
3. **RAG over brand policies:** retrieve each client's refund, delivery and escalation rules before drafting, so replies never invent promises.
4. **Human approval workflow:** role-based queue where an agent approves, edits and audits every reply before publication.
5. **Auto-publishing:** push approved replies back to the source platform with a full audit log.
6. **Active learning:** send low-confidence or ambiguous reviews for annotation, then retrain on the new labels.
7. **Drift monitoring:** track language mix, score distribution and complaint taxonomy changes over time.
8. **A/B testing:** compare template vs LLM replies on response time, customer sentiment and escalation rate.
9. **Scalable storage:** move from SQLite to PostgreSQL when review volume and concurrent users grow.
10. **Integration with Revio:** expose the pipeline as a FastAPI service that the main Revio product can call.

---

## Summary

This prototype intentionally separates **what is built** from **what is proposed**. It already delivers a working end-to-end review intelligence loop with a real ML model, deterministic fallback, persistent storage and a business dashboard. The roadmap explains exactly how the same design would become a production-grade Revio module.
