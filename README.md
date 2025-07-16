# Reddit User Persona Generator

## Overview
This project generates detailed user personas by analyzing Reddit users' posts and comments. Using advanced Natural Language Processing (NLP) and pre-trained models from Hugging Face, it extracts semantic insights such as top interests, named entities, sentiment, writing style, and frequent subreddits. Each trait is supported by citations (i.e., URLs) to real Reddit posts or comments, making the persona transparent and verifiable.
This tool is useful for researchers, marketers, social media analysts, and behavioral scientists who wish to analyze public 
Reddit profiles for qualitative or quantitative studies.

## Installation
#To run this project locally or on Google Colab, you need to install the following dependencies:
pip install praw
pip install spacy
pip install transformers
pip install sentence-transformers
pip install keybert
python -m spacy download en_core_web_sm

## Reddit API Setup
Visit https://www.reddit.com/prefs/apps
Click "Create App" and choose the "script" type.
Fill in required fields and note the generated:
a)client_id
b)client_secret
c)user_agent
These credentials will be used in the script to authenticate access to Reddit's API.

## Features Extracted
Top Interests (Semantic Keywords): Extracts meaningful and non-generic interests using KeyBERT + MiniLM. Handles synonyms and singular/plural variations.
Named Entities: Uses spaCy NER to extract important mentions like locations, organizations, or individuals.
Frequent Subreddits: Lists subreddits where the user posts or comments the most, along with examples.
Writing Style: Based on average word count per post/comment; labeled as either Concise or Detailed.
Sentiment: Classified using Twitter-RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) as Positive, Neutral, or Negative.
Evidence Citations: Every characteristic includes URLs to Reddit posts/comments as reference points.

## How to Run
#Once dependencies and API credentials are ready, copy and run this code in your Python environment or Google Colab:
import praw
from transformers import pipeline
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
import spacy
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="reddit-persona-bot by u/YOUR_USERNAME"
)


nlp = spacy.load("en_core_web_sm")

#Get Reddit username
username = input("Enter Reddit username: ")
user = reddit.redditor(username)

#Scrape posts and comments
posts, comments = [], []
try:
    for submission in user.submissions.new(limit=50):
        posts.append({
            "text": f"{submission.title} {submission.selftext}",
            "url": f"https://reddit.com{submission.permalink}",
            "subreddit": str(submission.subreddit)
        })
except Exception as e:
    print("Could not fetch submissions:", e)

try:
    for comment in user.comments.new(limit=50):
        comments.append({
            "text": comment.body,
            "url": f"https://reddit.com{comment.permalink}",
            "subreddit": str(comment.subreddit)
        })
except Exception as e:
    print("Could not fetch comments:", e)

all_texts = posts + comments

if not all_texts:
    print(f"❌ No posts or comments found for user '{username}'. Cannot generate a persona.")
else:
    summary_text = " ".join([entry["text"] for entry in all_texts])

    kw_model = KeyBERT(model=SentenceTransformer("all-MiniLM-L6-v2"))
    keywords = kw_model.extract_keywords(summary_text, top_n=20, stop_words='english')
    generic_terms = {'india', 'car', 'cars', 'people', 'thing', 'delhi', 'uttar', 'pradesh', 'lko'}
    top_keywords = [kw[0] for kw in keywords if kw[0].lower() not in generic_terms]
    keyword_citations = {kw[0]: [] for kw in keywords if kw[0].lower() not in generic_terms}
    for kw in keyword_citations:
        for entry in all_texts:
            if kw in entry["text"].lower():
                keyword_citations[kw].append(entry["url"])
                break

    sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    raw_label = sentiment_pipeline(summary_text[:512])[0]["label"]
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    sentiment = label_map.get(raw_label, "Unknown")
    sentiment_citation = next((entry["url"] for entry in all_texts if entry["text"][:512] in summary_text), 'N/A')

    doc = nlp(summary_text)
    named_entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON"]]
    unique_entities = list(set(named_entities))

    avg_len = sum(len(item["text"].split()) for item in all_texts) / len(all_texts)
    style = "Concise" if avg_len < 15 else "Detailed"
    style_citation = all_texts[0]["url"] if all_texts else 'N/A'

    top_subreddits = Counter([item["subreddit"] for item in all_texts]).most_common(3)
    subreddit_citations = {sub: next((entry["url"] for entry in all_texts if entry["subreddit"] == sub), 'N/A') for sub, _ in top_subreddits}

    persona_text = f"\nUser Persona for: u/{username}\n\n🔹 Top Interests:\n"
    for kw in top_keywords:
        url = keyword_citations[kw][0] if keyword_citations[kw] else "N/A"
        persona_text += f"- {kw} (e.g., {url})\n"

    persona_text += f"\n🔹 Named Entities Mentioned: {', '.join(unique_entities[:5]) if unique_entities else 'N/A'}\n\n🔹 Frequent Subreddits:\n"
    for sub, _ in top_subreddits:
        citation = subreddit_citations[sub]
        persona_text += f"- r/{sub} (e.g., {citation})\n"

    persona_text += f"\n🔹 Writing Style: {style} (e.g., {style_citation})"
    persona_text += f"\n🔹 Sentiment: {sentiment} (e.g., {sentiment_citation})\n\n🔹 Sample Evidence:\n"
    for entry in posts[:2] + comments[:2]:
        snippet = entry["text"].replace("\n", " ").strip()
        if len(snippet) > 100:
            snippet = snippet[:97] + "..."
        persona_text += f"• \"{snippet}\" — {entry['url']}\n"

    # Save to file
    with open(f"{username}_persona.txt", "w", encoding="utf-8") as f:
        f.write(persona_text)

    print("✅ Persona file saved. Check your working directory.")

## Output Format
#The final output looks like this:
User Persona for: u/exampleuser
🔹 Top Interests:
- vision pro (e.g., https://reddit.com/example1)
- xcode (e.g., https://reddit.com/example2)
🔹 Named Entities Mentioned: Apple, New York, Reddit
🔹 Frequent Subreddits:
- r/technology (e.g., https://reddit.com/example3)
- r/apple (e.g., https://reddit.com/example4)
🔹 Writing Style: Detailed (e.g., https://reddit.com/example5)
🔹 Sentiment: Positive (e.g., https://reddit.com/example6)
🔹 Sample Evidence:
• "Just got my Vision Pro and loving it!" — https://reddit.com/example7
• "Anyone else using Xcode on AVP?" — https://reddit.com/example8

## Sample Reddit Users
You can try public usernames like:
a)mcmrarm
b)TheGreatZarquon

    





