import streamlit as st
import os
from data_utils import load_data, preprocess, quick_inspect
from metrics import compute_country_metrics, compute_product_metrics
from rag import get_relevant_rows_for_question, build_context_table, ask_groq

st.set_page_config(page_title="Retail RAG Chatbot", layout="wide")
st.title("Retail RAG Chatbot — Demo (RAG + Groq)")

st.sidebar.header("About")
st.sidebar.write("The chatbot uses the dataset to answer queries. If a Groq API key isn't available, it uses a simple local fallback.")

DATA_PATH = os.path.join("data","Retail.csv")
try:
    df_raw = load_data(DATA_PATH)
    df = preprocess(df_raw)
except Exception as e:
    st.error(f"Could not load dataset at {DATA_PATH}: {e}")
    st.stop()

# Precompute global metrics once
try:
    country_metrics = compute_country_metrics(df)
except Exception:
    country_metrics = None

try:
    product_metrics = compute_product_metrics(df)
except Exception:
    product_metrics = None

st.sidebar.subheader("Example queries")
st.sidebar.write(
    "- Which country has the highest sales?\n"
    "- What is the best product by sales?\n"
    "- Best products in united kingdom\n"
    "- Compare united kingdom and france\n"
    "- Show top products in france\n"
    "- What is the total sales of assorted colour bird ornament?\n"
    "- List products with quantity more than 20 in united kingdom\n"
    "- What is the best outlet in surat? (tests fallback)"
)

col1, col2 = st.columns([3,1])
with col1:
    query = st.text_input("Ask a question", placeholder="e.g. Which country has the highest sales in Germany?")
    if st.button("Ask"):
        if not query.strip():
            st.warning("Type a question first.")
        else:
            q_lower = query.strip().lower()

            country_list = []
            if 'country' in df.columns:
                country_list = list(df['country'].dropna().astype(str).str.lower().unique()[:500])
            product_list = []
            if 'product' in df.columns:
                product_list = list(df['product'].dropna().astype(str).str.lower().unique()[:500])

            is_country_query = ("which country" in q_lower or "highest sales" in q_lower or "top country" in q_lower)
            is_product_query = ("best product" in q_lower or "top product" in q_lower or "best products" in q_lower)

            mentions_country = any(name in q_lower for name in country_list)
            mentions_product = any(name in q_lower for name in product_list)

            if is_country_query and not mentions_country:
                try:
                    country_metrics = compute_country_metrics(df)
                    if country_metrics is None or country_metrics.shape[0] == 0:
                        answer = "I don't know / not available in the dataset."
                    else:
                        top = country_metrics.iloc[0]
                        answer = f"Top country by total sales: {top['country']} with total sales {top['total_sales']:.2f}."
                except Exception:
                    answer = "I don't know / not available in the dataset."
                st.markdown("**Answer**")
                st.write(answer)
                st.markdown("**Note:** This used global precomputed aggregates (positive sales only).")
            elif is_product_query and not (mentions_product or mentions_country):
                try:
                    product_metrics = compute_product_metrics(df)
                    if product_metrics is None or product_metrics.shape[0] == 0:
                        answer = "I don't know / not available in the dataset."
                    else:
                        top = product_metrics.iloc[0]
                        answer = f"Top product by total sales: {top['product']} with total sales {top['total_sales']:.2f}."
                except Exception:
                    answer = "I don't know / not available in the dataset."
                st.markdown("**Answer**")
                st.write(answer)
                st.markdown("**Note:** This used global product aggregates (positive sales only).")
            else:
                relevant = get_relevant_rows_for_question(df, query, top_n=20)
                context = build_context_table(relevant, cols=['product','country','quantity','unit_price','total_sale_value'], max_rows=12)
                sys_prompt = (
                    "You are a helpful assistant. ONLY use information provided in the Context. "
                    "If info is not present, reply: \"I don't know / not available in the dataset.\" "
                    "State which metric you used. Keep answer concise."
                )
                answer = ask_groq(sys_prompt, context, query)
                st.markdown("**Answer**")
                st.write(answer)
                st.markdown("**Context passed to the model**")
                st.code(context)

with col2:
    st.markdown("### Top countries (computed)")
    try:
        cm = compute_country_metrics(df)
        st.dataframe(cm.head(10))
    except Exception as e:
        st.write("Could not compute country metrics:", e)
    st.markdown("### Top products (computed)")
    try:
        pm = compute_product_metrics(df)
        st.dataframe(pm.head(10))
    except Exception as e:
        st.write("Could not compute product metrics:", e)
