import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ingest import load_model, get_qdrant_client, setup_collection, ingest_pdf
from app.retriever import retrieve_pages
from app.generator import generate_answer, format_citations

st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Multi-Modal Document QA System")
st.caption("Built with ColPali + Qdrant + Groq (LLaMA 4) — DSAI 413 Assignment 1")


@st.cache_resource
def init_model():
    model, processor = load_model()
    client = get_qdrant_client()
    setup_collection(client)
    return model, processor, client


with st.spinner("Loading ColPali model... please wait"):
    model, processor, qdrant_client = init_model()

st.success(" Model loaded and ready!")

with st.sidebar:
    st.header(" Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        save_dir = os.path.abspath("data/sample_docs")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button(" Ingest Document", type="primary"):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                num_pages = ingest_pdf(save_path, model, processor, qdrant_client)
            st.success(f" Ingested {num_pages} pages!")

    st.divider()
    st.header(" Settings")
    top_k = st.slider("Pages to retrieve", min_value=1, max_value=5, value=3)

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF and click Ingest")
    st.markdown("2. Ask any question about the document")
    st.markdown("3. ColPali finds the most relevant pages")
    st.markdown("4. LLaMA 4 generates a cited answer")

st.header(" Ask a Question")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("What is the GDP growth forecast?"):
        st.session_state.query = "What is the GDP growth forecast?"
with col2:
    if st.button("Summarize the key findings"):
        st.session_state.query = "Summarize the key findings"
with col3:
    if st.button("What do the charts show?"):
        st.session_state.query = "What do the charts show?"

query = st.text_input(
    "Your question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What are the inflation figures in the table?"
)

if st.button("🔍 Get Answer", type="primary") and query:
    with st.spinner("Searching relevant pages..."):
        retrieved_pages = retrieve_pages(
            query=query,
            model=model,
            processor=processor,
            client=qdrant_client,
            top_k=top_k
        )

    if not retrieved_pages:
        st.warning("No pages found. Make sure you have ingested a document first.")
    else:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_pages)

        st.header(" Answer")
        st.markdown(answer)

        st.header(" Sources")
        st.markdown(format_citations(retrieved_pages))

        st.header(" Retrieved Pages")
        cols = st.columns(len(retrieved_pages))
        for i, (col, page) in enumerate(zip(cols, retrieved_pages)):
            with col:
                if page["image"] is not None:
                    st.image(
                        page["image"],
                        caption=f"Page {page['page_number']} — {page['doc_name']}",
                        use_container_width=True
                    )