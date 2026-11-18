# Revio AI — Intelligent Customer Review Analyzer & Response Generator

Streamlit proof of concept built for **Cynbit Technologies**. It extends Cynbit's existing **Revio** review-management product with sentiment analysis, complaint detection, priority assignment, response drafting and business analytics.

> **Status:** This repository contains a working local prototype. Some capabilities are deliberately listed as future work in the reports. Nothing is auto-published to review platforms.

---

## 1. What is implemented

- [x] Streamlit app with four working sections: **Dashboard**, **Review Analyzer**, **AI Response Generator**, **Review Insights**
- [x] CSV sample dataset seeded into **SQLite**
- [x] Hugging Face Transformers sentiment model with automatic **lexicon fallback** for offline use
- [x] Sentiment classification: **Positive / Negative / Neutral** + confidence score
- [x] Main and secondary complaint detection from a 10-category issue taxonomy
- [x] Priority assignment: **High / Medium / Low**
- [x] Professional template-based reply generation
- [x] Optional LLM reply hook (disabled by default; falls back safely)
- [x] Editable replies, reply download and save-back to SQLite
- [x] Plotly dashboard: sentiment mix, priority, ratings, common complaints, trends
- [x] Filters, keyword frequencies and CSV export in Review Insights
- [x] Two written reports with explicit implemented vs proposed sections

### Proposed, not implemented

- Live Google / Instagram / Zomato / Facebook review connectors
- Fine-tuned multilingual transformer on real client data
- RAG over brand refund / callback policies
- Human approval queue and role-based access
- Auto-posting approved replies to review platforms
- Drift monitoring, A/B testing and scheduled retraining

See [`reports/AI_Opportunity_Report.md`](reports/AI_Opportunity_Report.md) and [`reports/Technical_Approach.md`](reports/Technical_Approach.md).

---

## 2. Repository structure

```text
Revio-AI/
├── app.py                         # Main Streamlit application
├── data/
│   └── sample_reviews.csv         # 45 synthetic demo reviews
├── models/
│   ├── __init__.py
│   └── sentiment_model.py         # HF model + lexicon fallback
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py           # Cleaning, tokens, keywords, language
│   └── response_generator.py      # Issue detection, priority, replies
├── reports/
│   ├── AI_Opportunity_Report.md
│   └── Technical_Approach.md
├── requirements.txt
├── README.md
└── .gitignore
```

`data/revio.db` is created automatically on first run and is ignored by Git.

---

## 3. Quick start

### 3.1 Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> The first run with `transformers` installed will download the sentiment model weights. If that fails, the app automatically switches to the built-in lexicon fallback and still works.

### 3.3 Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal, usually:

```text
http://localhost:8501
```

### 3.4 Force the offline model (optional)

```bash
# Windows PowerShell
$env:REVIEW_SENTIMENT_BACKEND="lexicon"
streamlit run app.py
```

```bash
# macOS / Linux
REVIEW_SENTIMENT_BACKEND=lexicon streamlit run app.py
```

---

## 4. Example input and output

**Input**

> “The delivery was very late, and customer support did not respond to my calls. Very disappointing experience.”

| Feature | Output |
| --- | --- |
| Sentiment | Negative |
| Confidence | ~0.86 in fallback mode |
| Main Issue | Delivery delay |
| Secondary Issue | Poor customer support |
| Priority | High |
| Suggested Reply | “We sincerely apologize for the inconvenience. We understand your concern regarding the delivery delay and customer support. Our team will look into this issue and get back to you shortly.” |

The app also shows total reviews, positive/negative/neutral percentages, common complaints and satisfaction trends.

---

## 5. Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `REVIEW_SENTIMENT_BACKEND` | `auto` | `auto`, `hf` or `lexicon` |
| `REVIEW_SENTIMENT_MODEL` | DistilBERT SST-2 | Override the Hugging Face model ID |
| `REVIO_USE_LLM` | `false` | Set `true` to attempt an LLM-generated reply |
| `OPENAI_API_KEY` | unset | Needed only when the LLM hook is enabled |
| `REVIO_LLM_MODEL` | `gpt-4o-mini` | Model used by the optional hook |

To test the optional LLM hook:

```bash
# Windows PowerShell
$env:REVIO_USE_LLM="true"
$env:OPENAI_API_KEY="your-key-here"
streamlit run app.py
```

If the API call fails, the app automatically uses the template reply.

---

## 6. Push to GitHub

Your required repository name is:

```text
Revio-AI-Review-Intelligence
```

From inside the `Revio-AI` folder:

```bash
git init
git add .
git commit -m "Revio AI review intelligence prototype"
git branch -M main
git remote add origin https://github.com/<your-username>/Revio-AI-Review-Intelligence.git
git push -u origin main
```

Or create the repository on GitHub first and use **Add files → Upload files**.

---

## 7. Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to <https://share.streamlit.io/> and sign in with GitHub.
3. Click **New app**.
4. Select:
   - Repository: `<your-username>/Revio-AI-Review-Intelligence`
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**.
6. Copy the generated URL and paste it below.

**Live demo:** _Add your Streamlit Community Cloud link here after deployment._

Optional secrets for the LLM hook can be added under **App settings → Secrets** as:

```toml
REVIO_USE_LLM = "true"
OPENAI_API_KEY = "your-key-here"
REVIO_LLM_MODEL = "gpt-4o-mini"
```

Leave these unset if you only want the default template-based replies.

---

## 8. Notes for review

- All review data is synthetic; no real Cynbit client data is included.
- Public sources used for the reports: `cynbit.in` (home, services, products, case studies), Cynbit LinkedIn and public Clutch listings.
- Impact numbers in the opportunity report are labelled assumptions, not measured results.
- The browser-free Python modules can be tested independently:

```bash
python -m compileall app.py models utils
python -c "from models.sentiment_model import SentimentAnalyzer; from utils.response_generator import detect_issues; a=SentimentAnalyzer(); t='The delivery was very late, and customer support did not respond to my calls.'; print(a.analyze(t,1)); print(detect_issues(t)[:2])"
```

---

Built for Cynbit Technologies — Kratika Jain, November 2026.
