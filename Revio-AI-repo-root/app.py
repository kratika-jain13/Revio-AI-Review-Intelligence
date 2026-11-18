"""Revio AI — Intelligent Customer Review Analyzer & Response Generator.

Streamlit prototype for Cynbit Technologies.

Implemented in this prototype:
  - CSV ingestion into SQLite
  - Hugging Face sentiment analysis with transparent lexicon fallback
  - Sentiment classification + confidence score
  - Main / secondary complaint detection and priority assignment
  - Template-based response generation with an optional LLM hook
  - Editable replies, dashboard, review analyzer and review insights

Proposed for future development (see reports):
  - Live Google / Instagram / Zomato review connectors
  - Fine-tuned multilingual transformer and RAG over brand policy documents
  - Human approval workflow and auto-publishing to review platforms
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from models.sentiment_model import SentimentAnalyzer  # noqa: E402
from utils.preprocessing import clean_text, detect_language, get_keywords  # noqa: E402
from utils.response_generator import (  # noqa: E402
    ResponseGenerator,
    determine_priority,
    detect_issues,
)

DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "revio.db"
CSV_PATH = DATA_DIR / "sample_reviews.csv"

st.set_page_config(
    page_title="Revio AI | Review Desk",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --ink: #1e2a32;
        --muted: #6b7882;
        --line: #dde3e6;
        --accent: #1d6b58;
        --bg: #f6f7f5;
      }

      .stApp { background: var(--bg); }

      .block-container {
        padding-top: 1.6rem;
        padding-bottom: 3rem;
        max-width: 1480px;
      }

      section[data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
      }

      section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
      }

      .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.1rem 0 1rem;
      }

      .sidebar-mark {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        background: var(--accent);
        color: #fff;
        display: grid;
        place-items: center;
        font-weight: 700;
        font-size: 0.86rem;
        letter-spacing: 0.04em;
      }

      .sidebar-title {
        font-size: 0.98rem;
        font-weight: 650;
        color: var(--ink);
        line-height: 1.15;
      }

      .sidebar-sub {
        font-size: 0.72rem;
        color: var(--muted);
        margin-top: 0.15rem;
      }

      .page-head {
        border-bottom: 1px solid var(--line);
        padding-bottom: 0.9rem;
        margin-bottom: 1.25rem;
      }

      .page-eyebrow {
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        font-weight: 700;
      }

      .page-title {
        font-size: 1.75rem;
        margin: 0.35rem 0 0.3rem;
        color: var(--ink);
        font-weight: 650;
      }

      .page-sub {
        font-size: 0.9rem;
        color: var(--muted);
        margin: 0;
        max-width: 780px;
      }

      div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        box-shadow: none;
      }

      div[data-testid="stMetricLabel"] {
        font-size: 0.76rem;
        color: var(--muted);
      }

      div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        color: var(--ink);
      }

      .stButton > button,
      .stFormSubmitButton > button {
        border-radius: 8px;
        border-color: var(--accent);
      }

      .stButton > button[kind="primary"],
      .stFormSubmitButton > button[kind="primary"] {
        background: var(--accent);
        color: #fff;
      }

      .stButton > button:hover,
      .stFormSubmitButton > button:hover {
        background: #175744;
        color: #fff;
      }

      .stExpander {
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #fff;
      }

      .stCaption { color: var(--muted); }

      [data-testid="stAppDeployButton"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached model objects
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


@st.cache_resource(show_spinner=False)
def get_generator(tone: str) -> ResponseGenerator:
    return ResponseGenerator(tone=tone)


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------
def init_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT UNIQUE NOT NULL,
            review_text TEXT NOT NULL,
            rating INTEGER,
            source TEXT,
            created_at TEXT,
            sentiment TEXT,
            confidence REAL,
            model TEXT,
            main_issue TEXT,
            secondary_issue TEXT,
            priority TEXT,
            suggested_reply TEXT,
            edited_reply TEXT,
            analyzed_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def seed_sample_data(conn: sqlite3.Connection) -> None:
    if not CSV_PATH.exists():
        return
    sample = pd.read_csv(CSV_PATH)
    for row in sample.itertuples(index=False):
        conn.execute(
            """INSERT OR IGNORE INTO reviews
               (review_id, review_text, rating, source, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (str(row.review_id), str(row.review_text), int(row.rating), str(row.source), str(row.created_at)),
        )
    conn.commit()


def analyze_review(
    text: str,
    rating: int,
    analyzer: SentimentAnalyzer,
    generator: ResponseGenerator,
) -> dict:
    cleaned = clean_text(text)
    sentiment_result = analyzer.analyze(cleaned, rating=rating)
    sentiment = sentiment_result["sentiment"]
    main_issue, secondary_issue, ranked = detect_issues(cleaned)
    priority = determine_priority(sentiment, main_issue, rating)
    response = generator.generate(
        cleaned, sentiment, main_issue, secondary_issue, priority, rating
    )
    return {
        "sentiment": sentiment,
        "confidence": sentiment_result["confidence"],
        "model": sentiment_result["model"],
        "scores": sentiment_result["scores"],
        "negative_feedback": sentiment_result["negative_feedback"],
        "evidence": sentiment_result["evidence"],
        "main_issue": main_issue,
        "secondary_issue": secondary_issue,
        "ranked_issues": ranked,
        "priority": priority,
        "suggested_reply": response["reply"],
        "response_source": response["source"],
        "language": detect_language(cleaned),
    }


def analyze_pending(conn: sqlite3.Connection, analyzer: SentimentAnalyzer, generator: ResponseGenerator) -> int:
    pending = conn.execute(
        "SELECT review_id, review_text, rating FROM reviews WHERE sentiment IS NULL"
    ).fetchall()
    if not pending:
        return 0
    progress = st.progress(0.0, text="Analyzing stored reviews…")
    for index, (review_id, text, rating) in enumerate(pending, start=1):
        result = analyze_review(text, int(rating or 3), analyzer, generator)
        conn.execute(
            """UPDATE reviews SET sentiment=?, confidence=?, model=?, main_issue=?,
               secondary_issue=?, priority=?, suggested_reply=?, analyzed_at=?
               WHERE review_id=?""",
            (
                result["sentiment"], result["confidence"], result["model"],
                result["main_issue"], result["secondary_issue"], result["priority"],
                result["suggested_reply"], datetime.now().isoformat(timespec="seconds"),
                review_id,
            ),
        )
        progress.progress(index / len(pending))
    conn.commit()
    progress.empty()
    return len(pending)


def insert_review(
    conn: sqlite3.Connection,
    text: str,
    rating: int,
    source: str,
    analyzer: SentimentAnalyzer,
    generator: ResponseGenerator,
) -> dict:
    result = analyze_review(text, rating, analyzer, generator)
    review_id = f"REV-USR-{int(time.time() * 1000)}"
    conn.execute(
        """INSERT INTO reviews
           (review_id, review_text, rating, source, created_at, sentiment,
            confidence, model, main_issue, secondary_issue, priority,
            suggested_reply, analyzed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            review_id, clean_text(text), rating, source,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["sentiment"], result["confidence"], result["model"],
            result["main_issue"], result["secondary_issue"], result["priority"],
            result["suggested_reply"], datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    result["review_id"] = review_id
    return result


def fetch_reviews(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM reviews ORDER BY created_at DESC", conn)


def save_edited_reply(conn: sqlite3.Connection, review_id: str, reply: str) -> None:
    conn.execute(
        "UPDATE reviews SET edited_reply=? WHERE review_id=?", (reply, review_id)
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
PRIORITY_COLOURS = {"High": "#b4523c", "Medium": "#c58a2c", "Low": "#1d6b58"}
SENTIMENT_COLOURS = {"Positive": "#1d6b58", "Negative": "#b4523c", "Neutral": "#8a949c"}


def metric_card(container, label: str, value: str, help_text: str | None = None) -> None:
    container.metric(label, value, help=help_text)


def page_header(section: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-eyebrow">Revio AI / {section}</div>
          <div class="page-title">{title}</div>
          <div class="page-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_table(result: dict) -> None:
    rows = [
        {"Feature": "Sentiment", "Output": result["sentiment"]},
        {"Feature": "Confidence", "Output": f"{result['confidence']:.0%}"},
        {"Feature": "Main Issue", "Output": result["main_issue"]},
        {"Feature": "Secondary Issue", "Output": result["secondary_issue"]},
        {"Feature": "Priority", "Output": result["priority"]},
        {"Feature": "Language", "Output": result["language"]},
        {"Feature": "Reply source", "Output": result["response_source"]},
    ]
    st.table(pd.DataFrame(rows))


def implemented_vs_proposed() -> None:
    with st.expander("Scope of this build", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Working in this version**
                - Four sections in one Streamlit app
                - CSV loaded into SQLite
                - Hugging Face sentiment model with offline fallback
                - Sentiment, confidence and complaint detection
                - Priority assignment
                - Template replies with an optional LLM hook
                - Dashboard and complaint charts
                """
            )
        with col2:
            st.markdown(
                """
                **Planned next**
                - Review-platform connectors and scheduled sync
                - Fine-tuned multilingual model
                - Retrieval from client policy documents
                - Approval queue before replies are published
                - Posting replies back to source platforms
                - Monitoring, A/B tests and retraining
                """
            )


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
conn = init_db()
seed_sample_data(conn)
analyzer = get_analyzer()
generator = get_generator("empathetic")
analyzed_count = analyze_pending(conn, analyzer, generator)

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="sidebar-mark">RA</div>
      <div>
        <div class="sidebar-title">Revio AI</div>
        <div class="sidebar-sub">Cynbit Technologies</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Review triage workspace")
page = st.sidebar.radio(
    "Workspace",
    ["Dashboard", "Review Analyzer", "AI Response Generator", "Review Insights"],
)
st.sidebar.divider()
st.sidebar.markdown(f"**Model:** `{analyzer.active_model}`")
st.sidebar.markdown("**Replies:** templates, optional LLM hook")
if analyzed_count:
    st.sidebar.caption(f"{analyzed_count} stored reviews processed.")
if st.sidebar.button("Process pending reviews"):
    count = analyze_pending(conn, analyzer, generator)
    st.sidebar.caption(f"Processed {count} reviews." if count else "No reviews pending.")

st.sidebar.divider()
implemented_vs_proposed()

df = fetch_reviews(conn)
analyzed = df[df["sentiment"].notna()].copy()


# ---------------------------------------------------------------------------
# 1. Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    page_header(
        "Review desk",
        "Review dashboard",
        "Daily view of the review queue: what came in, how it was classified and what needs attention first.",
    )

    if analyzed.empty:
        st.warning("No analyzed reviews yet. Add one in the Review Analyzer.")
    else:
        total = len(analyzed)
        counts = analyzed["sentiment"].value_counts()
        pos_pct = round(100 * counts.get("Positive", 0) / total, 1)
        neg_pct = round(100 * counts.get("Negative", 0) / total, 1)
        neu_pct = round(100 * counts.get("Neutral", 0) / total, 1)
        high_count = int((analyzed["priority"] == "High").sum())

        c1, c2, c3, c4, c5 = st.columns(5)
        metric_card(c1, "Total reviews", str(total), "Number of analyzed reviews in SQLite")
        metric_card(c2, "Positive reviews", f"{pos_pct}%", "Share of reviews classified as Positive")
        metric_card(c3, "Negative reviews", f"{neg_pct}%", "Share of reviews classified as Negative")
        metric_card(c4, "Neutral reviews", f"{neu_pct}%", "Share of reviews classified as Neutral")
        metric_card(c5, "High priority", str(high_count), "Reviews routed for urgent human attention")
        st.markdown("")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        if not analyzed.empty:
            sentiment_counts = analyzed["sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Reviews"]
            fig = px.pie(
                sentiment_counts,
                values="Reviews",
                names="Sentiment",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLOURS,
                hole=0.55,
                title="Sentiment split",
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, width="stretch")

    with col2:
        if not analyzed.empty:
            priority_counts = analyzed["priority"].value_counts().reindex(["High", "Medium", "Low"]).dropna().reset_index()
            priority_counts.columns = ["Priority", "Reviews"]
            fig = px.bar(
                priority_counts,
                x="Priority",
                y="Reviews",
                color="Priority",
                color_discrete_map=PRIORITY_COLOURS,
                title="Queue by priority",
                text="Reviews",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        if not analyzed.empty:
            rating_counts = analyzed["rating"].value_counts().sort_index().reset_index()
            rating_counts.columns = ["Rating", "Reviews"]
            fig = px.bar(
                rating_counts,
                x="Rating",
                y="Reviews",
                color="Reviews",
                color_continuous_scale="Teal",
                title="Ratings",
            )
            st.plotly_chart(fig, width="stretch")

    with col4:
        if not analyzed.empty:
            issue_counts = analyzed["main_issue"].value_counts().head(6).reset_index()
            issue_counts.columns = ["Issue", "Reviews"]
            fig = px.bar(
                issue_counts,
                x="Reviews",
                y="Issue",
                orientation="h",
                color="Reviews",
                color_continuous_scale="Blues",
                title="Top complaint themes",
            )
            st.plotly_chart(fig, width="stretch")

    if not analyzed.empty:
        st.subheader("Latest reviews")
        st.dataframe(
            analyzed[["review_id", "review_text", "rating", "source", "sentiment", "main_issue", "priority"]]
            .head(12),
            width="stretch",
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# 2. Review Analyzer
# ---------------------------------------------------------------------------
elif page == "Review Analyzer":
    page_header(
        "Assessment",
        "Review Analyzer",
        "Paste one review to check its sentiment and identify the complaint behind it.",
    )

    with st.form("analyzer_form", clear_on_submit=False):
        text = st.text_area(
            "Customer review",
            value="The delivery was very late, and customer support did not respond to my calls. Very disappointing experience.",
            height=130,
        )
        col1, col2 = st.columns([1, 2])
        with col1:
            rating = st.selectbox("Star rating", [1, 2, 3, 4, 5], index=0)
        with col2:
            source = st.selectbox("Source", ["Google", "Instagram", "Facebook", "Website", "Zomato", "LinkedIn", "Internal feedback"])
        submitted = st.form_submit_button("Analyze review", type="primary")

    if submitted:
        if not text.strip():
            st.error("Please enter a review.")
        else:
            result = analyze_review(text, rating, get_analyzer(), generator)
            result["review_id"] = None
            result["raw_text"] = text
            result["rating"] = rating
            result["source"] = source
            st.session_state.last_analysis = result
            st.caption("Review added to the current assessment.")

    result = st.session_state.get("last_analysis")
    if result:
        st.markdown("### Review assessment")
        render_result_table(result)

        col1, col2 = st.columns([1, 1])
        with col1:
            confidence = result["confidence"]
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": "", "valueformat": ".0%"},
                title={"text": "Confidence"},
                gauge={
                    "axis": {"range": [0, 1]},
                    "bar": {"color": SENTIMENT_COLOURS.get(result["sentiment"], "#6c8cff")},
                    "steps": [
                        {"range": [0, 0.4], "color": "#f2ded8"},
                        {"range": [0.4, 0.7], "color": "#f4ead4"},
                        {"range": [0.7, 1], "color": "#d9ece5"},
                    ],
                },
            ))
            fig.update_layout(height=260)
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("**Issues in this review**")
            issues = [result["main_issue"]] + ([result["secondary_issue"]] if result["secondary_issue"] != "None" else [])
            st.markdown(" ".join(f"`{issue}`" for issue in issues))
            st.markdown(f"**Class scores:** {result['scores']}")
            st.caption(f"Model used: {result['model']}")

        if st.button("Save to review table", type="primary"):
            saved = insert_review(conn, result["raw_text"], result["rating"], result["source"], get_analyzer(), generator)
            st.session_state.last_analysis = saved
            st.success(f"Saved as `{saved['review_id']}`.")


# ---------------------------------------------------------------------------
# 3. AI Response Generator
# ---------------------------------------------------------------------------
elif page == "AI Response Generator":
    page_header(
        "Drafting",
        "AI Response Generator",
        "Draft a reply from the review assessment, then edit it before saving or exporting.",
    )

    tone = st.selectbox("Reply tone", ["empathetic", "professional", "concise"], index=0)
    generator = get_generator(tone)

    source_mode = st.radio(
        "Draft from",
        ["Current review", "Paste a new review"],
        horizontal=True,
    )

    if source_mode == "Paste a new review":
        text = st.text_area(
            "Customer review",
            value="The delivery was very late, and customer support did not respond to my calls. Very disappointing experience.",
            height=110,
        )
        rating = st.selectbox("Star rating", [1, 2, 3, 4, 5], index=0, key="resp_rating")
        working = analyze_review(text, rating, get_analyzer(), generator) if text.strip() else None
        if working:
            working["raw_text"] = text
            working["rating"] = rating
            working["review_id"] = None
    else:
        working = st.session_state.get("last_analysis")

    if not working:
        st.info("Analyze a review first, or paste a new one above.")
    else:
        st.markdown(f"**Review:** {working.get('raw_text', '')}")
        st.markdown(
            f"**Sentiment:** {working['sentiment']} · "
            f"**Main issue:** {working['main_issue']} · "
            f"**Priority:** {working['priority']}"
        )

        if st.button("Generate reply", type="primary"):
            response = generator.generate(
                working.get("raw_text", ""),
                working["sentiment"],
                working["main_issue"],
                working["secondary_issue"],
                working["priority"],
                working.get("rating"),
            )
            working["suggested_reply"] = response["reply"]
            working["response_source"] = response["source"]
            st.session_state.last_analysis = working

        reply_value = working.get("suggested_reply", "")
        edited = st.text_area("Edit the suggested reply", value=reply_value, height=160)
        st.caption(f"Reply source: {working.get('response_source', 'template-based fallback')}")

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download reply (.txt)",
                data=edited,
                file_name="revio_reply.txt",
                mime="text/plain",
                disabled=not edited.strip(),
            )
        with col2:
            if st.button("Save edited reply"):
                review_id = working.get("review_id")
                if review_id:
                    save_edited_reply(conn, review_id, edited)
                    st.success("Edited reply saved.")
                else:
                    st.warning("Save the review from the Review Analyzer first.")


# ---------------------------------------------------------------------------
# 4. Review Insights
# ---------------------------------------------------------------------------
else:
    page_header(
        "Patterns",
        "Review Insights",
        "Look at repeated complaints and how sentiment moves across sources and dates.",
    )

    if analyzed.empty:
        st.warning("No analyzed reviews yet.")
    else:
        with st.expander("Filters", expanded=True):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                sentiments = st.multiselect(
                    "Sentiment",
                    options=sorted(analyzed["sentiment"].dropna().unique()),
                    default=sorted(analyzed["sentiment"].dropna().unique()),
                )
            with fcol2:
                sources = st.multiselect(
                    "Source",
                    options=sorted(analyzed["source"].dropna().unique()),
                    default=sorted(analyzed["source"].dropna().unique()),
                )
        filtered = analyzed[
            analyzed["sentiment"].isin(sentiments) & analyzed["source"].isin(sources)
        ].copy()

        if filtered.empty:
            st.info("No reviews match these filters.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                issue_counts = filtered["main_issue"].value_counts().reset_index()
                issue_counts.columns = ["Issue", "Reviews"]
                fig = px.bar(
                    issue_counts,
                    x="Reviews", y="Issue", orientation="h",
                    color="Reviews", color_continuous_scale="Teal",
                    title="Top complaint themes",
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                corpus = " ".join(filtered["review_text"].astype(str))
                keywords = get_keywords(corpus, top_n=12)
                keyword_df = pd.DataFrame(keywords, columns=["Keyword", "Mentions"])
                fig = px.bar(
                    keyword_df,
                    x="Mentions", y="Keyword", orientation="h",
                    color="Mentions", color_continuous_scale="Blues",
                    title="Common terms",
                )
                st.plotly_chart(fig, width="stretch")

            trend = filtered.copy()
            trend["date"] = pd.to_datetime(trend["created_at"], errors="coerce")
            trend["sentiment_value"] = trend["sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
            daily = (
                trend.dropna(subset=["date"])
                .groupby("date", as_index=False)
                .agg(avg_sentiment=("sentiment_value", "mean"), reviews=("review_id", "count"))
                .sort_values("date")
            )
            if not daily.empty:
                daily["rolling"] = daily["avg_sentiment"].rolling(3, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=daily["date"], y=daily["rolling"], mode="lines+markers", name="3-day average", line=dict(color="#1d6b58", width=3)))
                fig.add_hline(y=0, line_dash="dash", line_color="#8b9bb4")
                fig.update_layout(title="Sentiment over time", yaxis_title="Average score (-1 to +1)", xaxis_title="Date")
                st.plotly_chart(fig, width="stretch")

            st.subheader("Reviews in view")
            st.dataframe(
                filtered[[
                    "review_id", "created_at", "source", "rating", "sentiment",
                    "confidence", "main_issue", "secondary_issue", "priority",
                ]],
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download current view (.csv)",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="revio_review_insights.csv",
                mime="text/csv",
            )

st.markdown("---")
st.caption("Revio AI | Review triage workspace | Cynbit Technologies")
