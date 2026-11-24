"""Browser UI for the stress text classifier."""

import json
from pathlib import Path

import streamlit as st

from backend import AMBIGUOUS_HIGH, AMBIGUOUS_LOW, StressDetector

METRICS_FILE = Path(__file__).resolve().parent / "model" / "metrics.json"

EXAMPLES = {
    "Stressed example": (
        "I have three exams this week and I haven't started studying. I can't sleep, "
        "my chest feels tight, and every time I try to focus I just panic. I feel like "
        "I'm falling behind and I don't know how I'm going to cope."
    ),
    "Calm example": (
        "We spent the weekend at the lake with some friends. The weather was great, we "
        "grilled dinner and played cards until late. Thank you to everyone who came, we "
        "would love to host you all again soon."
    ),
    "Question example": "How do I study better for exams?",
}


@st.cache_resource
def load_detector() -> StressDetector:
    return StressDetector()


def load_metrics() -> dict:
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    return {}


def set_example(text: str) -> None:
    st.session_state["user_text"] = text


st.set_page_config(page_title="Stress Check", page_icon="🧠", layout="centered")
st.title("Stress Check")
st.write("This tool looks for stress-related patterns in text. It is not an advice chatbot or a medical diagnosis.")
st.caption("Your text is processed locally by the saved project model and is not intentionally stored by this app.")

try:
    detector = load_detector()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

metrics = load_metrics()

st.write("Quick examples:")
columns = st.columns(len(EXAMPLES))
for column, (name, example) in zip(columns, EXAMPLES.items()):
    column.button(name, on_click=set_example, args=(example,), use_container_width=True)

text = st.text_area(
    "What is on your mind?",
    key="user_text",
    height=180,
    placeholder="For example: I have been feeling overwhelmed with deadlines lately...",
)

if st.button("Check for stress signals", type="primary"):
    try:
        prediction = detector.predict(text)
    except (TypeError, ValueError) as error:
        st.info(str(error))
    else:
        st.subheader("Result")
        if prediction.result == "stress-related patterns detected":
            st.warning("Some stress-related patterns were detected.")
        elif prediction.result == "no clear stress signal":
            st.success("No clear stress signal was detected in this text.")
        else:
            st.info("This input is unclear for the stress detector.")

        st.write(prediction.input_note)
        st.metric("Stress-pattern score", f"{prediction.stress_probability:.0%}")
        st.progress(prediction.stress_probability)
        st.caption(
            f"Scores between {AMBIGUOUS_LOW:.0%} and {AMBIGUOUS_HIGH:.0%} are reported as unclear. "
            "This score is a model estimate, not a personal or medical measurement."
        )

        if prediction.needs_more_context:
            st.info("More context may help. A question, a very short message, or a borderline score cannot be judged reliably by this model.")
        elif prediction.result == "stress-related patterns detected":
            st.write("If this reflects how you are actually feeling, consider taking a short pause or talking with someone you trust.")

st.divider()
if metrics:
    with st.expander("About the model"):
        st.write(f"Model: {metrics['selected_model'].replace('_', ' ')}")
        st.write(f"Trained on {metrics['train_rows']:,} examples, tested on {metrics['test_rows']:,} unseen examples.")
        st.write(f"Test accuracy: {metrics['accuracy']:.1%}")
    st.caption(
        f"The model was trained on {metrics['rows_used']:,} labelled English Reddit posts. "
        "It can misunderstand sarcasm, slang, study questions, and text outside its training distribution."
    )
else:
    st.caption("Run train_model.py to create model/metrics.json.")
