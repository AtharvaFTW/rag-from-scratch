import streamlit as st
import requests


API_URL = "http://localhost:8000"

st.title("🐾 Animal Welfare Legal Assistant")

query = st.text_area("Ask a question about animal welfare laws in India")
if "data" not in st.session_state:
    st.session_state.data = None

if st.button("Submit"):
    res = requests.post(f"{API_URL}/query", json= {"query": query})
    st.session_state.data = res.json()

if st.session_state.data:
    data = st.session_state.data
    st.text(f"Answer: {data["response"]}")
    
    if st.button("Show References"):
        for i, ref in enumerate(data["references"]):
            with st.expander(f"Reference {i+1}"):
                st.info(ref)

    if st.button("Show Ragas"):
        scores = data["ragas_scores"]
        st.metric("Faithfulness", scores["faithfulness"])
        st.metric("Answer Relevancy", scores["answer_relevancy"])
        st.metric("Context Precision", scores["context_precision"])
        st.metric("Context Recall", scores["context_recall"])
        st.caption(f"Config: {scores['config']}")


    



