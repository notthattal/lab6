import os
from dotenv import load_dotenv
import streamlit as st
import llm
import storage

load_dotenv()

st.set_page_config(page_title="RLHF Data Collector", layout="wide")
st.title("RLHF Preference Data Collector")
st.caption(f"Model: `{llm.LLM_MODEL}`")

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    st.error("OPENAI_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    count = storage.record_count()
    st.metric("Records saved", count)
    if count > 0:
        st.download_button(
            "Download CSV",
            data=storage.csv_bytes(),
            file_name="preference_data.csv",
            mime="text/csv",
        )

client = llm.get_client(api_key)

# --- Prompt input ---
prompt = st.text_area("Prompt", height=120, placeholder="Type a prompt and click Generate…")

col_gen, col_reset, _ = st.columns([1, 1, 6])
with col_gen:
    generate_clicked = st.button("Generate", type="primary", disabled=not prompt.strip())
with col_reset:
    if st.button("Reset"):
        for key in ("resp_a", "resp_b", "saved"):
            st.session_state.pop(key, None)

# --- Generation ---
if generate_clicked and prompt.strip():
    for key in ("resp_a", "resp_b", "saved"):
        st.session_state.pop(key, None)
    with st.spinner("Generating two responses…"):
        try:
            st.session_state["resp_a"] = llm.generate(client, prompt)
            st.session_state["resp_b"] = llm.generate(client, prompt)
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

# --- Display responses ---
if "resp_a" in st.session_state:
    resp_a: llm.LLMResponse = st.session_state["resp_a"]
    resp_b: llm.LLMResponse = st.session_state["resp_b"]

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Response A")
        st.write(resp_a.text)
        st.caption(f"Output tokens: {resp_a.output_tokens} | Latency: {resp_a.latency_s}s")
    with col_b:
        st.subheader("Response B")
        st.write(resp_b.text)
        st.caption(f"Output tokens: {resp_b.output_tokens} | Latency: {resp_b.latency_s}s")

    # --- Preference ---
    st.divider()
    st.subheader("Which response do you prefer?")

    if st.session_state.get("saved"):
        st.success("Preference saved! Enter a new prompt or click Reset to continue.")
    else:
        p_a, p_tie, p_b = st.columns(3)
        preference = None
        if p_a.button("Prefer A", use_container_width=True):
            preference = "A"
        if p_tie.button("Tie", use_container_width=True):
            preference = "tie"
        if p_b.button("Prefer B", use_container_width=True):
            preference = "B"

        if preference is not None:
            storage.save(prompt, resp_a, resp_b, preference)
            st.session_state["saved"] = True
            st.rerun()
