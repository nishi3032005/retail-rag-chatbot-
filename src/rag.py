import os
import pandas as pd
import requests
import re

# 1. MULTI-MATCH RETRIEVER (supports comparisons)
def get_relevant_rows_for_question(df: pd.DataFrame, question: str, top_n=15):
    
    """ Detect ALL referenced countries/products in a query.
    Returns combined rows for every match (for comparison queries).
    Fallback: top-N highest sale rows. """
   
    q = question.lower()
    collected = []

    search_cols = ['country', 'product', 'invoice', 'customer_id']

    for col in search_cols:
        if col not in df.columns:
            continue

        uniques = df[col].dropna().astype(str).str.lower().unique()

        for v in uniques:
            if len(v) < 2:
                continue

            # Use full-word matching to avoid substring issues
            pattern = r'\b' + re.escape(v) + r'\b'

            try:
                if re.search(pattern, q):
                    matched_rows = df[df[col].astype(str).str.lower() == v].head(top_n)
                    collected.append(matched_rows)
            except Exception:
                continue

    # If multiple entities found gives combine
    if len(collected) > 1:
        combined = pd.concat(collected).sort_values("total_sale_value", ascending=False)
        return combined.head(max_rows := max(top_n, len(combined)))

    # If exactly one match found
    if len(collected) == 1:
        return collected[0]

    # Nothing matched then fallback: show top-selling rows
    if "total_sale_value" in df.columns:
        return df.sort_values("total_sale_value", ascending=False).head(top_n)

    return df.head(top_n)


# 2. SAFE CONTEXT BUILDER (no commas/newlines)

def build_context_table(df_rows: pd.DataFrame, cols=None, max_rows=10):
   
    """Builds a clean multi-line context table using '|'.
    Removes newlines, extra pipes, and dangerous characters inside cells."""
    
    if cols is None:
        cols = [c for c in ['product', 'country', 'quantity', 'unit_price', 'total_sale_value']
                if c in df_rows.columns]

    dfc = df_rows[cols].head(max_rows).copy().fillna("")

    clean_rows = []

    for _, row in dfc.iterrows():
        cleaned = []
        for c in cols:
            val = str(row[c])
            val = val.replace("\n", " ").replace("\r", " ")
            val = val.replace("|", " ")  # avoid breaking delimiter
            cleaned.append(val.strip())
        clean_rows.append(cleaned)

    header = " | ".join(cols)
    lines = [header]

    for row in clean_rows:
        lines.append(" | ".join(row))

    return "\n".join(lines)



# 3. GROQ API WRAPPER

def ask_groq(system_prompt: str, context: str, user_question: str, model="llama3-70b-8192"):
   
    """Calls Groq LLM. If no API key is found, uses fallback mode."""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return local_fallback_answer(context, user_question)

    url = "https://api.groq.ai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": user_question}
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]
    except Exception:
        return local_fallback_answer(context, user_question)


# 4. DETERMINISTIC FALLBACK (no hallucinations)

def local_fallback_answer(context: str, question: str):
   
    """ Uses ONLY the provided context to answer.
    Safe, deterministic, zero hallucinations."""

    lines = [ln for ln in context.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "I don't know / not available in the dataset."

    header = [h.strip().lower() for h in lines[0].split("|")]
    rows = []

    for r in lines[1:]:
        parts = [p.strip() for p in r.split("|")]
        if len(parts) < len(header):
            parts += [""] * (len(header) - len(parts))
        rows.append(parts)

    try:
        df = pd.DataFrame(rows, columns=header)
    except Exception:
        return "I don't know / not available in the dataset."

    # Numeric cleanup
    for col in ['total_sale_value', 'quantity', 'unit_price']:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace("[^0-9.-]", "", regex=True),
                errors="coerce"
            ).fillna(0)

    if "total_sale_value" not in df.columns:
        return "I don't know / not available in the dataset."

    positive = df[df["total_sale_value"] > 0]
    if positive.empty:
        return "No positive sales found in the shown context. (Fallback)"

    # COUNTRY COMPARISON
    if "country" in df.columns:
        agg = positive.groupby("country", as_index=False)["total_sale_value"].sum()
        top = agg.sort_values("total_sale_value", ascending=False).iloc[0]
        return f"Top country by total sales: {top['country']} with {top['total_sale_value']:.2f}. (Fallback)"

    # PRODUCT COMPARISON
    if "product" in df.columns:
        agg = positive.groupby("product", as_index=False)["total_sale_value"].sum()
        top = agg.sort_values("total_sale_value", ascending=False).iloc[0]
        return f"Top product by total sales: {top['product']} with {top['total_sale_value']:.2f}. (Fallback)"

    return "I don't know / not available in the dataset."
