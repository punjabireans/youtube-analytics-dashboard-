import os
import re
import json
import time
import logging
import datetime as dt
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import plotly.express as px
import plotly.graph_objects as go
print("✅ STEP 1: streamlit_app.py is running!")
# =======================
# Load API Key Securely
# =======================
try:
    YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
except KeyError:
    st.error(" API Key not found. Please set `YOUTUBE_API_KEY` in secrets.")
    st.info("Go to your app settings → Secrets → Add your key.")
    st.stop()
# Configuration
DATA_DIR = os.path.join(os.getcwd(), "yt_trends_data")
HISTORY_DIR = os.path.join(DATA_DIR, "history")
CATEGORY_CACHE = os.path.join(DATA_DIR, "category_cache.json")
os.makedirs(HISTORY_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
]

HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# Predefined health-related keywords for suggestions
HEALTH_KEYWORDS = [
    "health", "wellness", "fitness", "nutrition", "diet", "exercise",
    "sexual health", "orgasm", "prostate", "bladder", "sex", "penis", 
    "testosterone", "hormones", "mental health", "meditation"
]

# =======================
# Utility Functions
# =======================
def human_int(n: int) -> str:
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)

def parse_metric_number(s: str) -> int:
    if not isinstance(s, str):
        return 0
    s = s.strip().replace(",", "").replace(" ", "")
    match = re.match(r"^(\d+(?:\.\d+)?)([KMB])?$", s, re.IGNORECASE)
    if match:
        num = float(match.group(1))
        suffix = match.group(2).upper() if match.group(2) else ""
        multipliers = {"K": 1000, "M": 1000000, "B": 1000000000}
        return int(num * multipliers.get(suffix, 1))
    digits = re.findall(r"\d+", s)
    return int("".join(digits)) if digits else 0

def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def ts_str(ts: dt.datetime) -> str:
    return ts.isoformat()

# Function to filter videos based on keywords
def filter_videos_by_keywords(df: pd.DataFrame, keywords: List[str]) -> pd.DataFrame:
    """Filter videos based on keywords in title, description, or tags"""
    if not keywords:
        return df
    
    # Convert to lowercase for case-insensitive matching
    pattern = "|".join([re.escape(keyword.lower()) for keyword in keywords])
    
    # Check if any of the keywords are in title, description or tags
    mask = (
        df["title"].str.lower().str.contains(pattern, na=False) | 
        df["description"].str.lower().str.contains(pattern, na=False) |
        df["tags_str"].str.lower().str.contains(pattern, na=False)
    )
    
    return df[mask].reset_index(drop=True)

# ==========
# YouTube API
@dataclass
class VideoItem:
    video_id: str
    title: str
    url: str
    views: int
    views_api: int
    likes: int
    category_id: str
    category: str
    tags: List[str]
    description: str
    published_at: Optional[dt.datetime]

class YouTubeAPI:
    def __init__(self, api_key: str):
        try:
            self.youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API: {e}")
            st.error(" Failed to connect to YouTube API. Check your API key.")
            st.stop()

    def fetch_trending(self, region_code: str, max_results: int = 50) -> List[dict]:
        for _ in range(MAX_RETRIES):
            try:
                response = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    chart="mostPopular",
                    regionCode=region_code,
                    maxResults=min(max_results, 50),
                ).execute()
                return response.get("items", [])
            except Exception as e:
                logger.warning(f"API call failed: {e}")
                time.sleep(RETRY_DELAY)
        return []

    def fetch_categories(self, region_code: str) -> Dict[str, str]:
        try:
            response = self.youtube.videoCategories().list(
                part="snippet", regionCode=region_code
            ).execute()
            return {item["id"]: item["snippet"]["title"] for item in response.get("items", [])}
        except Exception as e:
            logger.error(f"Failed to fetch categories: {e}")
            return {}

def scrape_view_count(video_id: str, timeout: int = 10) -> int:
    return 0  # Fast: no delay


def load_category_cache() -> Dict[str, Dict[str, str]]:
    if os.path.exists(CATEGORY_CACHE):
        try:
            with open(CATEGORY_CACHE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return {}
    return {}

def save_category_cache(cache: Dict[str, Dict[str, str]]):
    try:
        with open(CATEGORY_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

# history 
def history_path(region_code: str) -> str:
    return os.path.join(HISTORY_DIR, f"history_{region_code.upper()}.csv")

def append_history(region_code: str, df: pd.DataFrame):
    path = history_path(region_code)
    df.to_csv(path, mode="a", index=False, header=not os.path.exists(path))

def load_history(region_code: str) -> pd.DataFrame:
    path = history_path(region_code)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "video_id", "views", "likes"])
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
        df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)
        return df.dropna(subset=["timestamp"])
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return pd.DataFrame(columns=["timestamp", "video_id", "views", "likes"])

def compute_24h_delta(current_df: pd.DataFrame, hist_df: pd.DataFrame, now: dt.datetime) -> pd.DataFrame:
    if hist_df.empty:
        current_df["views_delta_24h"] = 0
        current_df["likes_delta_24h"] = 0
        current_df["delta_mode"] = "since_start"
        return current_df

    cutoff = now - dt.timedelta(hours=24)
    baselines = {}
    for vid, group in hist_df.groupby("video_id"):
        before = group[group["timestamp"] <= cutoff]
        base_row = before.iloc[-1] if not before.empty else group.iloc[0]
        baselines[vid] = base_row

    vdelta, ldelta, modes = [], [], []
    for _, row in current_df.iterrows():
        vid = row["video_id"]
        base = baselines.get(vid)
        if base is None:
            vdelta.append(0)
            ldelta.append(0)
            modes.append("since_start")
        else:
            vdelta.append(max(0, row["views"] - base["views"]))
            ldelta.append(max(0, row["likes"] - base["likes"]))
            mode = "exact_24h" if base["timestamp"] <= cutoff else "since_start"
            modes.append(mode)

    current_df["views_delta_24h"] = vdelta
    current_df["likes_delta_24h"] = ldelta
    current_df["delta_mode"] = modes
    return current_df

# Analysis Function=
def extract_hashtags(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [tag.lower() for tag in re.findall(r"#[A-Za-z0-9_]+", text)]

def analyze_hashtags(df: pd.DataFrame) -> pd.Series:
    all_tags = [tag for desc in df["description"] for tag in extract_hashtags(desc)]
    return pd.Series(all_tags).value_counts().head(10) if all_tags else pd.Series(dtype=int)

def analyze_category_distribution(df: pd.DataFrame) -> pd.Series:
    return df["category"].value_counts().head(10)

def analyze_recent_uploads(df: pd.DataFrame, hours: int = 24) -> pd.Series:
    cutoff = utcnow() - dt.timedelta(hours=hours)
    recent = df[df["published_at"].notnull() & (df["published_at"] >= cutoff)]
    return recent["category"].value_counts().head(5)

# Plotting Function
def plot_category_pie(cat_counts: pd.Series):
    if cat_counts.empty:
        fig = go.Figure()
        fig.update_layout(title="No category data")
        return fig
    return px.pie(values=cat_counts.values, names=cat_counts.index, title="Top 10 Trending Categories")

def plot_likes_views_bar(top_df: pd.DataFrame):
    if top_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No performance data")
        return fig
    use_delta = top_df["views_delta_24h"].sum() > 0
    y_views = top_df["views_delta_24h"] if use_delta else top_df["views"]
    y_likes = top_df["likes_delta_24h"] if use_delta else top_df["likes"]
    title = "Top 5 Videos by Views & Likes (last 24h)" if use_delta else "Top 5 Videos by Views & Likes (total)"
    x = [t if len(t) <= 32 else t[:29] + "..." for t in top_df["title"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y_views, name="Views"))
    fig.add_trace(go.Bar(x=x, y=y_likes, name="Likes"))
    fig.update_layout(barmode="group", title_text=title, xaxis_title="Video", yaxis_title="Count")
    return fig

def plot_top_likes_bar(top8_df: pd.DataFrame):
    if top8_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No likes data")
        return fig
    use_delta = top8_df["likes_delta_24h"].sum() > 0
    y_likes = top8_df["likes_delta_24h"] if use_delta else top8_df["likes"]
    title = "Top 8 Videos by Likes (last 24h)" if use_delta else "Top 8 Videos by Likes (total)"
    x = [t if len(t) <= 32 else t[:29] + "..." for t in top8_df["title"]]
    fig = px.bar(x=x, y=y_likes, title=title, labels={"x": "Video", "y": "Likes"})
    return fig

def plot_recent_categories_bar(recent_cat_counts: pd.Series):
    if recent_cat_counts.empty:
        fig = go.Figure()
        fig.update_layout(title="No recent uploads in trending")
        return fig
    return px.bar(
        x=recent_cat_counts.index,
        y=recent_cat_counts.values,
        title="Top 5 Categories by Recent Uploads (last 24h)",
        labels={"x": "Category", "y": "# Trending Videos Published Recently"}
    )


def main():
    st.set_page_config(page_title="YouTube Trending Dashboard", layout="wide")

    st.title("YouTube Trending Analytics Dashboard")
    st.markdown("""
    Real-time insights into what's trending on YouTube.
    
    ### Features
    - Top trending videos by country
    - Filter by keywords or categories
    - View and like counts
    - 24-hour growth tracking
    - Category & hashtag analysis
    
    > No auto-refresh — click **'Fetch Latest Data'** to update.
    """)

    with st.sidebar:
        st.header(" Settings")
        COUNTRIES = {
            "United States": "US",
            "India": "IN",
            "United Kingdom": "GB",
            "Canada": "CA",
            "Australia": "AU",
            "Germany": "DE",
            "France": "FR",
            "Brazil": "BR",
            "Japan": "JP",
            "South Korea": "KR",
        }
        country_name = st.selectbox(" Country", options=list(COUNTRIES.keys()), index=0)  
# Default: United States
        region_code = COUNTRIES[country_name]
        
        st.header("Filter Options")
        
        # Keyword filtering
        filter_method = st.radio("Filter Method", ["Predefined Keywords", "Custom Keywords"])
        
        if filter_method == "Predefined Keywords":
            selected_keywords = st.multiselect(
                "Select Health Topics", 
                options=HEALTH_KEYWORDS,
                default=[]
            )
        else:
            keyword_input = st.text_input("Enter Custom Keywords (comma separated)")
            selected_keywords = [k.strip() for k in keyword_input.split(",")] if keyword_input else []
        
        # Category filtering
        enable_category_filter = st.checkbox("Filter by Category")
        selected_category = None
        if enable_category_filter:
            # We'll populate this after fetching data
            if "categories" in st.session_state:
                selected_category = st.selectbox(
                    "Select Category",
                    options=["All"] + list(st.session_state.categories.values())
                )
        
        max_monitor = st.slider("🎥 Max Videos to Monitor", 5, 50, 30)
        st.info("Click the button below to load data.")

    #  Main Action: Manual Refresh
    if st.button("Fetch Latest Data") or "fetched" not in st.session_state:
        st.session_state.fetched = True
        with st.spinner("Fetching trending videos..."):
            try:
                yt = YouTubeAPI(YOUTUBE_API_KEY)

                # Load or fetch categories
                cat_cache = load_category_cache()
                if region_code not in cat_cache:
                    with st.spinner(f"Loading categories for {country_name}..."):
                        cat_cache[region_code] = yt.fetch_categories(region_code)
                        save_category_cache(cat_cache)
                cat_map = cat_cache[region_code]
                
                # Store categories for the UI
                st.session_state.categories = cat_map

                # Fetch trending videos
                items = yt.fetch_trending(region_code, max_monitor)
                if not items:
                    st.error("Failed to fetch trending videos. Check region or API key.")
                    return

                # Process videos
                records = []
                now = utcnow()
                for item in items:
                    snip = item.get("snippet", {})
                    stats = item.get("statistics", {})
                    try:
                        published_dt = pd.to_datetime(snip.get("publishedAt"), utc=True)
                    except Exception:
                        published_dt = None

                    vid = item["id"]
                    views_api = int(stats.get("viewCount", 0))
                    likes_api = int(stats.get("likeCount", 0))
                    
                    tags = snip.get("tags", [])
                    tags_str = " ".join(tags).lower() if tags else ""

                    records.append({
                        "video_id": vid,
                        "title": snip.get("title", "(no title)"),
                        "url": f"https://www.youtube.com/watch?v={vid}",
                        "views": views_api,
                        "views_api": views_api,
                        "likes": likes_api,
                        "category_id": snip.get("categoryId", "Unknown"),
                        "category": cat_map.get(snip.get("categoryId", ""), "Unknown"),
                        "tags": tags,
                        "tags_str": tags_str,
                        "description": snip.get("description", ""),
                        "published_at": published_dt,
                    })

                df = pd.DataFrame(records)
                df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

                # Save to history
                hist_df = df[["video_id", "views", "likes"]].copy()
                hist_df["timestamp"] = ts_str(now)
                append_history(region_code, hist_df[["timestamp", "video_id", "views", "likes"]])

                # Compute deltas
                full_hist = load_history(region_code)
                df = compute_24h_delta(df, full_hist, now)
                
                # Apply category filtering if selected
                if enable_category_filter and selected_category and selected_category != "All":
                    df = df[df["category"] == selected_category].reset_index(drop=True)
                
                # Apply keyword filtering if any keywords selected
                if selected_keywords:
                    filtered_df = filter_videos_by_keywords(df, selected_keywords)
                    
                    # Show warning if no videos match the keywords
                    if filtered_df.empty and not df.empty:
                        st.warning(f"No trending videos match your keyword filters. Showing all trending videos instead.")
                    elif not filtered_df.empty:
                        df = filtered_df
                
                # Store the original unfiltered data
                st.session_state.all_data = df.copy()

                # Analytics
                top5 = df.nlargest(5, "views")
                top5_delta = df.nlargest(5, ["views_delta_24h", "likes_delta_24h"])
                top8_likes = df.nlargest(8, "likes_delta_24h")
                cat_counts = analyze_category_distribution(df)
                recent_cat_counts = analyze_recent_uploads(df)
                hashtag_counts = analyze_hashtags(df)

                # Store in session state to avoid re-fetching
                st.session_state.data = {
                    "top5": top5,
                    "top5_delta": top5_delta,
                    "top8_likes": top8_likes,
                    "cat_counts": cat_counts,
                    "recent_cat_counts": recent_cat_counts,
                    "hashtag_counts": hashtag_counts,
                    "updated_at": now,
                    "keywords_used": selected_keywords,
                    "category_used": selected_category if enable_category_filter else None
                }

            except Exception as e:
                st.error(f" Error: {str(e)}")
                logger.error(f"Error in main: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # 📊 Display Data (only if fetched)
    if "data" in st.session_state:
        data = st.session_state.data
        updated_at = data["updated_at"]
        
        # Show filtering information
        if data.get("keywords_used"):
            st.info(f"🔍 Filtered by keywords: {', '.join(data['keywords_used'])}")
        
        if data.get("category_used"):
            st.info(f"📂 Filtered by category: {data['category_used']}")
            
        # Show how many videos match the filters
        if "all_data" in st.session_state:
            total_count = len(st.session_state.all_data)
            filtered_count = len(data["top5"]) if not data["top5"].empty else 0
            if filtered_count < total_count:
                st.write(f"Showing {filtered_count} videos matching your filters out of {total_count} trending videos.")

        # Top 5 Videos
        st.subheader("Top 5 Trending Videos (by views)")
        if data["top5"].empty:
            st.warning("No videos match your filters.")
        else:
            for i, row in data["top5"].iterrows():
                st.markdown(f"**{i + 1}.** [{row['title']}]({row['url']}) — "
                            f"👁️ {human_int(row['views'])} views · "
                            f" {human_int(row['likes'])} likes · "
                            f"Category: {row['category']}")

        # Charts (with unique keys)
        timestamp = int(time.time())
        
        if not data["top5"].empty:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_category_pie(data["cat_counts"]), key=f"pie_{timestamp}", use_container_width=True)
            with col2:
                st.plotly_chart(plot_likes_views_bar(data["top5_delta"]), key=f"bar1_{timestamp}", use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(plot_top_likes_bar(data["top8_likes"]), key=f"bar2_{timestamp}", use_container_width=True)
            with col4:
                st.plotly_chart(plot_recent_categories_bar(data["recent_cat_counts"]), key=f"bar3_{timestamp}", use_container_width=True)

            # Hashtags
            st.subheader("# Top 10 Hashtags in Descriptions")
            if not data["hashtag_counts"].empty:
                for tag, cnt in data["hashtag_counts"].items():
                    st.markdown(f"- **{tag}** — {cnt} mentions")
            else:
                st.markdown("*No hashtags found.*")
        else:
            st.warning("No data to display. Try different filter criteria.")

        st.success(f" Last updated: {updated_at.strftime('%H:%M:%S UTC')} | Region: {country_name}")

    else:
        st.info("Click 'Fetch Latest Data' to start.")

if __name__ == "__main__":
    main()
