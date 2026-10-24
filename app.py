
import os
import re
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    try:
        API_KEY = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        API_KEY = None

MODEL = "openai/gpt-oss-120b"

st.set_page_config(
    page_title="SEO Content Assistant",
    page_icon="✍️",
    layout="wide"
)


# ---------------- API ----------------

def call_groq(system_prompt, user_prompt, max_tokens=4000):
    if not API_KEY:
        raise ValueError(
        "GROQ_API_KEY is missing. Configure it in your .env file "
        "or Streamlit Secrets."
    )

    client = Groq(api_key=API_KEY)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=max_tokens
    )

    result = response.choices[0].message.content

    if not result or not result.strip():
        raise ValueError("The AI returned an empty response.")

    return result.strip()


# ---------------- TOPIC CORRECTION ----------------

def correct_topic_spelling(topic):
    return call_groq(
        """
Correct spelling and grammar in a short topic.
Preserve the original meaning and proper nouns.
Return only the corrected topic, without explanations.
Do not unnecessarily rewrite the topic.
""",
        f"Correct this topic: {topic}",
        max_tokens=100
    ).strip().strip('"').strip("'")


# ---------------- GENERATE CONTENT ----------------

def generate_seo_content(topic):
    system_prompt = """
You are a professional SEO content writer.

Write clear, useful, accurate, well-structured content in simple English.

Do not invent statistics, research, citations, or facts.
Avoid keyword stuffing and repetitive sentences.
Use short paragraphs and natural language.
Use Markdown headings: #, ##, and ###.
Never display literal labels such as H1:, H2:, or H3:.
Keep SEO metadata separate from the article.
Include exactly five numbered FAQs.
Do not guarantee Google rankings.

Return this format:

SEO DETAILS

SEO Title: [30-60 character title]
Primary Keyword: [main keyword]
Secondary Keywords: [comma-separated keywords]
Meta Description: [120-160 character description]

ARTICLE

# [Main article heading]

[Introduction]

## [Relevant section]

[Explanation]

## [Relevant section]

[Explanation]

## [Relevant section]

[Explanation]

## [Challenges or important considerations]

[Explanation where relevant]

## Conclusion

[Conclusion]

## Frequently Asked Questions

### 1. [Question]
[Answer]

### 2. [Question]
[Answer]

### 3. [Question]
[Answer]

### 4. [Question]
[Answer]

### 5. [Question]
[Answer]

Adapt section names to the topic. Keep the article useful and
substantial. Do not force irrelevant sections.
"""

    return call_groq(
        system_prompt,
        f"Create a complete SEO content package about: {topic}",
        max_tokens=4000
    )


# ---------------- CONTENT PARSING ----------------

def extract_section(content, label):
    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, content)
    return match.group(1).strip() if match else ""


def get_article(content):
    match = re.search(
        r"(?im)^\s*ARTICLE\s*$([\s\S]*)",
        content
    )
    return match.group(1).strip() if match else content.strip()


def get_title(content):
    title = extract_section(content, "SEO Title")
    if title:
        return title

    match = re.search(r"(?m)^#\s+(.+)$", get_article(content))
    return match.group(1).strip() if match else ""


# ---------------- SEO SMARTSCORE ----------------

def calculate_seo_smartscore(content, topic):
    score = 0
    checks = []

    title = get_title(content)
    meta = extract_section(content, "Meta Description")
    article = get_article(content)

    # Check 1: Main topic in title
    topic_words = set(re.findall(r"\w+", topic.lower()))
    title_words = set(re.findall(r"\w+", title.lower()))

    topic_in_title = (
        bool(topic_words) and topic_words.issubset(title_words)
    )

    if topic_in_title:
        score += 20
        checks.append(("Topic in SEO title", 20, "Passed"))
    else:
        checks.append((
            "Topic in SEO title", 0,
            "Main topic is missing from the title"
        ))

    # Check 2: Title length
    if 30 <= len(title) <= 60:
        score += 20
        checks.append((
            "Title length (30–60 characters)", 20, "Passed"
        ))
    else:
        checks.append((
            "Title length (30–60 characters)", 0,
            f"Current length: {len(title)} characters"
        ))

    # Check 3: Meta description length
    if 120 <= len(meta) <= 160:
        score += 20
        checks.append((
            "Meta description length (120–160 characters)",
            20, "Passed"
        ))
    else:
        checks.append((
            "Meta description length (120–160 characters)",
            0, f"Current length: {len(meta)} characters"
        ))

    # Check 4: Article structure
    h2_count = len(re.findall(r"(?m)^##\s+\S", article))

    if h2_count >= 4:
        score += 20
        checks.append((
            "Article structure (4+ sections)", 20, "Passed"
        ))
    else:
        checks.append((
            "Article structure (4+ sections)", 0,
            f"Only {h2_count} sections detected"
        ))

    # Check 5: Five numbered FAQs
    faq_match = re.search(
        r"(?im)^##\s+Frequently Asked Questions\s*$([\s\S]*)",
        article
    )

    faq_count = 0
    if faq_match:
        faq_count = len(re.findall(
            r"(?m)^###\s+\d+[.)]?\s+",
            faq_match.group(1)
        ))

    if faq_count >= 5:
        score += 20
        checks.append(("Five FAQs", 20, "Passed"))
    else:
        checks.append((
            "Five FAQs", 0,
            f"Only {faq_count} numbered FAQs detected"
        ))

    return score, checks


# ---------------- TARGETED IMPROVEMENT ----------------

def improve_specific_issue(content, topic, issue):
    instructions = {
        "Topic in SEO title": (
            f"Rewrite only the SEO Title to include the topic "
            f"'{topic}' naturally. Keep it 30-60 characters."
        ),
        "Title length (30–60 characters)": (
            f"Rewrite only the SEO Title to be 30-60 characters "
            f"and include '{topic}' naturally."
        ),
        "Meta description length (120–160 characters)": (
            "Rewrite only the Meta Description to be 120-160 "
            "characters, including spaces. Keep it clear and relevant."
        ),
        "Article structure (4+ sections)": (
            "Change only the ARTICLE section. Ensure it has at least "
            "four meaningful second-level Markdown headings (##)."
        ),
        "Five FAQs": (
            "Change only the FAQ section. Ensure it contains exactly "
            "five numbered questions with useful answers."
        )
    }

    instruction = instructions.get(issue)
    if not instruction:
        raise ValueError("No improvement instruction found.")

    return call_groq(
        f"""
You are making one targeted SEO correction.

{instruction}

Return the complete content in the original format.
Preserve all other content as closely as possible.
Keep SEO DETAILS, SEO Title:, Primary Keyword:,
Secondary Keywords:, Meta Description:, and ARTICLE.
Use Markdown headings. Never use literal H1:, H2:, or H3: labels.
Do not add an explanation outside the content.
""",
        f"Topic: {topic}\n\nCurrent content:\n{content}",
        max_tokens=4000
    )


# ---------------- FULL ARTICLE AUTO-FIX ----------------

def improve_seo_content(content, topic):
    return call_groq(
        """
You are an SEO editor.

Improve the complete article for clarity, grammar, organization,
readability, and SEO.

Keep the original topic and useful information.
Keep the SEO title between 30 and 60 characters if possible.
Keep the meta description between 120 and 160 characters if possible.
Include at least four meaningful second-level headings.
Include exactly five numbered FAQs.
Preserve the original response format and all SEO metadata labels.
Use natural Markdown headings, never literal H1:, H2:, or H3: labels.
Do not invent statistics or guarantee search rankings.

Return the complete revised content only.
""",
        f"Topic: {topic}\n\nContent to improve:\n{content}",
        max_tokens=4000
    )


# ---------------- SESSION STATE ----------------

defaults = {
    "seo_result": "",
    "seo_original": "",
    "seo_topic": "",
    "seo_improved": False
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- USER INTERFACE ----------------

st.title("✍️ AI-Powered SEO Content Assistant")

st.write(
    "Generate SEO-friendly articles, evaluate them using SEO SmartScore, "
    "and improve individual issues with AI."
)

st.caption(
    "SEO SmartScore is a custom rule-based checklist, not an official "
    "Google score or a search-ranking prediction."
)

topic = st.text_input(
    "Enter your topic",
    placeholder="e.g., Use of Technology in Medical Science"
)

if st.button("Generate SEO Content", type="primary"):
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        try:
            with st.spinner("Checking topic spelling..."):
                corrected_topic = correct_topic_spelling(topic.strip())

            if corrected_topic.casefold() != topic.strip().casefold():
                st.info(f"Corrected topic: **{corrected_topic}**")

            with st.spinner("Generating your article..."):
                result = generate_seo_content(corrected_topic)

            st.session_state["seo_result"] = result
            st.session_state["seo_original"] = result
            st.session_state["seo_topic"] = corrected_topic
            st.session_state["seo_improved"] = False

            st.success("Content generated successfully!")

        except Exception as error:
            st.error(
                "Content generation failed. Check your API key, "
                "internet connection, and try again."
            )
            st.caption(f"Technical details: {error}")


# ---------------- RESULTS ----------------

result = st.session_state["seo_result"]
saved_topic = st.session_state["seo_topic"]

if result and saved_topic:
    st.divider()
    st.header("Generated SEO Content")

    st.subheader("SEO Details")

    st.markdown(f"**SEO Title:** {get_title(result) or 'Not detected'}")
    st.markdown(
        f"**Primary Keyword:** "
        f"{extract_section(result, 'Primary Keyword') or 'Not detected'}"
    )
    st.markdown(
        f"**Secondary Keywords:** "
        f"{extract_section(result, 'Secondary Keywords') or 'Not detected'}"
    )
    st.markdown(
        f"**Meta Description:** "
        f"{extract_section(result, 'Meta Description') or 'Not detected'}"
    )

    st.subheader("Full Article")
    st.markdown(get_article(result))

    score, checks = calculate_seo_smartscore(result, saved_topic)

    st.divider()
    st.header("SEO SmartScore")

    col1, col2 = st.columns(2)
    col1.metric("SEO SmartScore", f"{score}/100")
    col2.metric(
        "Checks Passed",
        f"{sum(1 for _, points, _ in checks if points > 0)}/5"
    )

    st.progress(score / 100)

    st.subheader("Checks and Individual Improvements")

    for name, points, status in checks:
        if points > 0:
            st.success(f"{name}: +{points}/20")
        else:
            st.warning(f"{name}: {status}")

            if st.button(
                f"Improve: {name}",
                key=f"target_{name}"
            ):
                try:
                    with st.spinner(f"Improving {name}..."):
                        updated = improve_specific_issue(
                            st.session_state["seo_result"],
                            saved_topic,
                            name
                        )

                    st.session_state["seo_result"] = updated
                    st.session_state["seo_improved"] = True
                    st.rerun()

                except Exception as error:
                    st.error(f"Improvement failed: {error}")

    st.subheader("Improvement Suggestions")

    suggestions = {
        "Topic in SEO title":
            "Include the main topic naturally in the SEO title.",
        "Title length (30–60 characters)":
            "Adjust the title to 30–60 characters.",
        "Meta description length (120–160 characters)":
            "Adjust the meta description to 120–160 characters.",
        "Article structure (4+ sections)":
            "Add meaningful sections to the article.",
        "Five FAQs":
            "Add five numbered FAQs with answers."
    }

    failed_checks = [
        name for name, points, _ in checks if points == 0
    ]

    if failed_checks:
        for name in failed_checks:
            st.write(f"- {suggestions[name]}")
    else:
        st.write(
            "All five checks passed. Review the content for factual "
            "accuracy and readability before publishing."
        )

    st.divider()
    st.header("Full Article Auto-Fix")

    if st.button("Improve Entire Article"):
        try:
            with st.spinner("Improving the entire article..."):
                updated = improve_seo_content(result, saved_topic)

            st.session_state["seo_result"] = updated
            st.session_state["seo_improved"] = True
            st.rerun()

        except Exception as error:
            st.error(f"Full article improvement failed: {error}")

    if st.session_state["seo_improved"]:
        original_score, _ = calculate_seo_smartscore(
            st.session_state["seo_original"], saved_topic
        )
        current_score, _ = calculate_seo_smartscore(
            result, saved_topic
        )

        st.subheader("Score Comparison")

        col1, col2, col3 = st.columns(3)
        col1.metric("Original Score", f"{original_score}/100")
        col2.metric("Current Score", f"{current_score}/100")
        col3.metric(
            "Score Change",
            f"{current_score - original_score:+d}"
        )

    st.download_button(
        "Download Final SEO Content",
        data=result,
        file_name="seo_content.txt",
        mime="text/plain"
    )
