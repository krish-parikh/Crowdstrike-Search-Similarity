# Paste Similarity Search with Pinecone Built-in Embeddings

A semantic similarity search system for paste data using Pinecone for performing similarity search.

## Overview

### Step 1: Data Cleaning

The original paste dataset had 172615 records, with information relating to the paste such as the title, id, content, source, language, and more. Due to the large size of the dataset I wanted to reduce the size of the data to make it more manageable and reduce the computational cost of the search. In order to achieve this I only upserted the title and id data into Pinecone for search similarity. I initially tried to include the content of the paste, but found that this had a signficant computational cost, so as a workaround, once the user performed the search, it would use the id to return the content and additional information relating to the paste.

To further reduce the size of the data, I filtered out any pastes that were not marked as "en" in the language field, and removed "Untitled" pastes. This reduced the number of records to 67836. Having inspected examples of titles, I noticed that although they were tagged as english, some were gibberish for example a title such as "ESHSIzAnWgPKaR". To overcome this I used langdetect to check if the title was english and removed any non-english titles, although this was not exact and resulted in some titles that although didn't make grammatical sense but were still english being removed like "HOW TO BECOME A DARK KNIGHT A WEEK", it was a worthwhile trade off as it further reduced the records to just 26056.

The records for the "english" titles can be found in the file "english_records.json" and the records for the "non-english" titles can be found in the file "non_english_records.json".

Following this, I then cleaned the title data to remove any urls, file extensions, leading and trailing punctuation, collapse multiple spaces and lowercaseing the title in preparation for embedding.

### Step 2: Embedding/Upserting

I used Pinecone's built-in multilingual-e5-large model for state-of-the-art embeddings. Pinecone has a limit when it comes to upserting data in one go, so I had to batch the data into smaller chunks of 90 records at a time, resulting in 290 batches being upserted. The index had the following configuration:

```json
{
  "dimension": 1024,
  "index_fullness": 0.0,
  "metric": "cosine",
  "namespaces": { "paste_data": { "vector_count": 25156 } },
  "total_vector_count": 25156,
  "vector_type": "dense"
}
```

### Step 3: Search

The search function is a simple function that takes a query and returns the top k results based on the title similarity, with the id, score and text of the paste.

```python

def get_info(query, top_k = 10):
    # Search the dense index and rerank results
    results = dense_index.search(
        namespace="paste_data",
        query={
            "top_k": top_k,
            "inputs": {
                'text': query
            }
        }
    )

    # Print the results
    for hit in results['result']['hits']:
            print(f"id: {hit['_id']:<5} | score: {round(hit['_score'], 2):<5} | text: {hit['fields']['chunk_text']:<50}")

query = "bank account hacks"
```

Output:

```
id: 674f3e28-d4b0-4130-9c17-8740e89e99dc | score: 0.45  | text: western union transfer hack bank transfer hack paypal credit card hack transfer
id: c56a39c1-46a4-42a3-a0f0-01240dca0b83 | score: 0.45  | text: western union transfer hack bank transfer hack paypal credit card hack transfer
id: 45c9da7b-1ba4-4209-b8de-7705d7fdf8a3 | score: 0.44  | text: western union transfer hack bank transfer hack paypal credit card hack
id: 7f06c490-3b4c-4970-aff5-7d5e15f1d544 | score: 0.44  | text: western union transfer hack bank transfer hack paypal credit card hack
id: 6f923f4e-00b8-46b3-9df3-d5ff731cc93a | score: 0.44  | text: western union transfer hack bank transfer hack paypal credit card hack
```

The user can then use the id to get the content of the paste and additional information relating to the paste by directly searching the paste_extract.json file. For example, by inspecting the id "674f3e28-d4b0-4130-9c17-8740e89e99dc" we can see the following:

```json
{
  "iocs": {
    "emails": ["jonathanreed506@gmail.com"],
    "phone_numbers": ["+13202701152"],
    "emails_localpart": ["jonathanreed506"],
    "domains": ["gmail.com"],
    "available_types": [
      "domains",
      "emails",
      "emails_localpart",
      "phone_numbers"
    ]
  },
  "created_at": "2022-01-17T14:00:25-06:00",
  "language": "en",
  "title": "Western union transfer hack bank transfer hack PayPal credit card hack transfer",
  "content": "I have an authenticate western union,bank transfer,money gram,credit card,PayPal transfer bug software which is capable of running multiple transactions credentials via MTCN,bank transfer and sender's info less than an hour cashing out with zero theft and no traces of future charge back fee\r\nContact me\r\nName : Jonathan Reed\r\n\u00a0WhatsApp number : +13202701152\r\nE-mail address: Jonathanreed506@gmail.com\r\nPhone text : +13202701152",
  "id": "674f3e28-d4b0-4130-9c17-8740e89e99dc",
  "source_name": "pastebin.com",
  "user_name": null
}
```
