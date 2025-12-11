## Retail RAG Chatbot (Groq + Streamlit)

## Project Overview
This project is an AI-powered chatbot built using Retrieval-Augmented Generation (RAG) and Groq Llama models.  
The chatbot analyzes a retail transactions dataset and answers natural language questions such as:

- Which outlet/country has the highest sales?
- What is the best product based on sales?
- Compare the performance of two outlets.
- Show top products for a specific country.

The system retrieves relevant rows from the dataset, builds a clean context table, and then uses the Groq LLM to generate answers.  
When no Groq API key is provided, a deterministic fallback mode is used to ensure safe, non-hallucinated responses.


##  Dataset Link & Description
Dataset Source:  
https://www.datayb.com/datasets/dataset-details/datayb_dataset_details_p333awduhf2dv5t/

This dataset contains retail transaction records including:
- Product name  
- Country / Outlet  
- Quantity sold  
- Unit price  
- Invoice and customer identifiers  

A new derived field is added during preprocessing:
  total_sale_value = quantity × unit_price

This allows ranking products and outlets based on sales.

##  Definitions
Best Outlet:
“The country/outlet with the highest total sales, computed by summing the 'total_sale_value' of all its transactions.”

###  **Best Product**
The *best product* is defined as:
 “The product with the highest total sales, based on aggregated 'total_sale_value' across all transactions.”

These metrics are calculated dynamically from the dataset.

## Environment Setup

# 1. Create a virtual environment
python -m venv venv

 # 2. Activate it
Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt


## How to Run the App
Run the Streamlit interface:

streamlit run src/app_streamlit.py

The app will open in your browser at:

http://localhost:8501

## Example Queries (Recommended)

These queries demonstrate all core features of the chatbot:

# 🔹 Sales / Outlet Queries

Which country has the highest sales?

Show total sales by country.

# 🔹 Product Ranking

What is the best product by sales?

Best products in united kingdom

# 🔹 Comparison Queries

Compare united kingdom and france

Compare germany and france

# 🔹 Product-level Lookup

What is the total sales of assorted colour bird ornament?

# 🔹 Quantity / Filter Queries

List products with quantity more than 20 in united kingdom

# 🔹 Fallback Demonstration

What is the best outlet in surat?
(Dataset does not contain Surat → model shows fallback behavior.)