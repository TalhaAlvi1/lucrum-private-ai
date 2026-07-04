import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
import re

def process_document_query(query, collection):
    """Process a query specifically about document content"""
    try:
        # Get document metadata first
        meta_results = collection.query(
            query_texts=["document information"],
            n_results=1,
            where={"type": "metadata"}
        )
        
        # Search for content with the actual query
        content_results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
        
        if not content_results["documents"] or len(content_results["documents"][0]) == 0:
            return "I don't find any relevant information about that in the uploaded documents."
            
        # Filter and format results
        relevant_texts = []
        sources = set()

        # Normalize results: chroma may return nested lists or flat lists depending on client
        docs_raw = content_results.get("documents", [])
        metas_raw = content_results.get("metadatas", [])
        dists_raw = content_results.get("distances", [])

        # Unwrap one level if needed
        docs = docs_raw[0] if docs_raw and isinstance(docs_raw[0], list) else docs_raw
        metas = metas_raw[0] if metas_raw and isinstance(metas_raw[0], list) else metas_raw
        dists = dists_raw[0] if dists_raw and isinstance(dists_raw[0], list) else dists_raw

        # Ensure all are iterable lists of the same length
        if not isinstance(docs, list):
            docs = [docs]
        if not isinstance(metas, list):
            metas = [metas]
        if not isinstance(dists, list):
            dists = [dists]

        for doc, meta, dist in zip(docs, metas, dists):
            try:
                dist_val = float(dist)
            except Exception:
                # If distance can't be parsed, accept it conservatively
                dist_val = 1.0

            if dist_val < 0.8:  # Only use relatively close matches
                relevant_texts.append(doc)
                if isinstance(meta, dict) and "source" in meta:
                    sources.add(meta["source"])
                    
        if not relevant_texts:
            return "While I found the documents, I couldn't find specific information answering your question."
            
        # Create a context-aware response
        response = f"Based on the content from {', '.join(sources)}:\n\n"
        response += "\n\n".join(relevant_texts)
        
        return response
        
    except Exception as e:
        st.error(f"Error processing document query: {str(e)}")
        return "I encountered an error while trying to search the documents. Please try again."