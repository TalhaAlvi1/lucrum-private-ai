import streamlit as st
from llama_cpp import Llama
import json
import os
import chromadb
from chromadb.config import Settings
import PyPDF2
import pandas as pd
import datetime
from sentence_transformers import SentenceTransformer
import tempfile
import uuid
import re

AVATAR = "avatar.png"
MAX_FILES = 5
ALLOWED_EXTENSIONS = ['.pdf', '.csv', '.txt']

# Set page configuration with Lucrum branding
st.set_page_config(
    page_title="Lucrum Private AI",
    page_icon=AVATAR,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize all session state variables at the start
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Lucrum Pro"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load configuration from config.json
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "properties.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

config = load_config()

# Create system prompt from config data
def create_system_prompt(config):
    if not config or "system_prompt" not in config:
        return (
            "You are a precise and thorough AI assistant specialized in document analysis and information retrieval. " 
            "Follow these guidelines for all responses:\n\n"
            "1. DOCUMENT HANDLING:\n"
            "   - When answering questions about documents, ONLY use information from the provided document chunks\n"
            "   - Clearly indicate which document and section you're referencing\n"
            "   - If the requested information is not in the provided chunks, say so clearly\n"
            "   - Pay attention to document type (PDF, CSV, TXT) and format responses accordingly\n\n"
            "2. ACCURACY:\n"
            "   - Verify all information against the provided document chunks\n"
            "   - Use exact quotes when appropriate\n"
            "   - Never make assumptions about document content not provided\n"
            "   - Acknowledge any uncertainty clearly\n\n"
            "3. CONTEXT AWARENESS:\n"
            "   - Consider the document type and structure\n"
            "   - Maintain context between different chunks of the same document\n"
            "   - Identify and respect document hierarchy (headings, sections, etc.)\n\n"
            "4. RESPONSE STRUCTURE:\n"
            "   - Start with direct answers to questions\n"
            "   - Organize information logically\n"
            "   - Cite specific sections or page numbers when available\n\n"
            "Always maintain professional tone and prioritize accuracy over speculation."
        )
    
    system_config = config["system_prompt"]
    
    # Build comprehensive system prompt
    prompt_parts = []
    
    # Identity and role
    if "identity" in system_config and "role" in system_config:
        identity = system_config["identity"]
        prompt_parts.append(f"You are {identity.get('name', 'Lucrum Private AI')}, created by {identity.get('creator', 'Lucrum Industries')}.")
        prompt_parts.append(f"You operate {identity.get('operating_mode', 'Offline, without internet access')}.")
        prompt_parts.append(f"You {identity.get('limitations', 'do not manage user accounts, privacy settings, or personal data')}.")
        prompt_parts.append("")
        prompt_parts.append(system_config.get("role", ""))
        prompt_parts.append("")
    
    # Product information
    if "lucrum_ai_technology_suite" in system_config:
        suite_info = system_config["lucrum_ai_technology_suite"]
        prompt_parts.append("LUCRUM AI TECHNOLOGY SUITE:")
        prompt_parts.append(suite_info.get("description", ""))
        if "components" in suite_info:
            prompt_parts.append("Components:")
            for component in suite_info["components"]:
                prompt_parts.append(f"- {component}")
        prompt_parts.append("")
    
    # Key features
    if "key_features" in system_config:
        prompt_parts.append("KEY FEATURES:")
        for key, value in system_config["key_features"].items():
            prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        prompt_parts.append("")
    
    # CRM functionalities
    if "crm_core_functionalities" in system_config:
        prompt_parts.append("CRM CORE FUNCTIONALITIES:")
        for key, value in system_config["crm_core_functionalities"].items():
            prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        prompt_parts.append("")
    
    # Financial details
    if "financial_details" in system_config:
        fin_details = system_config["financial_details"]
        prompt_parts.append("FINANCIAL DETAILS:")
        prompt_parts.append(f"License Fee: {fin_details.get('license_fee', '')}")
        
        if "immediate_benefits" in fin_details:
            prompt_parts.append("Immediate Benefits:")
            for benefit in fin_details["immediate_benefits"]:
                prompt_parts.append(f"- {benefit}")
        
        if "membership_rewards_program" in fin_details:
            prompt_parts.append(f"Membership Rewards: {fin_details['membership_rewards_program']}")
        
        prompt_parts.append("")
    
    # Response directives
    if "response_generation_directives" in system_config:
        directives = system_config["response_generation_directives"]
        prompt_parts.append("RESPONSE GUIDELINES:")
        for key, value in directives.items():
            prompt_parts.append(f"- {value}")
    
    return "\n".join(prompt_parts)

# Load your GGUF model
@st.cache_resource
def load_model(model_name="Lucrum Pro"):
    model_files = {
        "Lucrum Pro": "Qwen1.5-0.5B-Chat-Q6_K.gguf",  # Original filename for Pro
        "Lucrum Thinking": "qwen1_5-1_8b-chat-q4_k_m.gguf",  # Original filename for Thinking
        "Lucrum Auto": "qwen1.5-0.5B-Chat-Q8_0.gguf",  # Original filename for Auto
        "Lucrum Instant": "qwen1.5-0.5B-Chat-Q8_0.gguf"  # Same as Auto for Instant
    }
    
    model_file = model_files.get(model_name, "model_pro.gguf")
    model_path = os.path.join(os.path.dirname(__file__), model_file)
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model file is present.")
        st.info(f"Please download the appropriate model file ({model_file}) and place it in the application directory.")
        st.stop()
    try:
        return Llama(
            model_path=model_path,
            n_ctx=4096,            # Large context window
            n_batch=512,           # Optimal batch size
            n_threads=6,           # Increased CPU threads
            n_gpu_layers=0,        # CPU only
            f16_kv=True,          # Use half precision for key/value cache
            use_mmap=True,        # Use memory mapping
            use_mlock=False,      # Dynamic memory management
            embedding=False,       # Focus on text generation
            last_n_tokens_size=256,# Increased history window
            seed=42,              # Consistent outputs
            repeat_penalty=1.1,    # Reduce repetition
            temperature=0.7,       # Balanced creativity
            top_p=0.9             # High-quality sampling
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure you have the correct model file and it's not corrupted.")
        st.stop()

try:
    # Load model based on selection
    llm = load_model(st.session_state.selected_model)
except Exception as e:
    st.error(f"Failed to initialize model: {str(e)}")
    st.stop()

# Initialize the vector store
def init_vector_store():
    try:
        persist_directory = os.path.join(os.path.dirname(__file__), "vector_store")
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create a new client instance
        st.session_state.vector_store = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create the collection
        try:
            # First try to delete the existing collection
            try:
                st.session_state.vector_store.delete_collection("document_store")
            except:
                pass  # Collection might not exist yet
                
            # Now create a fresh collection
            collection = st.session_state.vector_store.create_collection(
                name="document_store",
                metadata={"hnsw:space": "cosine"},  # Use cosine similarity
                get_or_create=True  # This ensures we get existing or create new
            )
            
        except Exception as e:
            st.error(f"Error managing collection: {str(e)}")
            raise e
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        raise e

# Function to clean text
def clean_text(text):
    """Enhanced text cleaning and normalization"""
    import re
    
    # Remove non-printable characters first
    text = ''.join(char for char in text if char.isprintable())
    
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix common PDF extraction issues
    text = text.replace('•', '* ')  # Replace bullets
    text = text.replace('►', '* ')  # Replace arrows
    text = text.replace('■', '* ')  # Replace squares
    
    # Fix word spacing issues
    text = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)  # Fix CamelCase
    text = re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[A-Za-z])(?=\d)', ' ', text)  # Fix number-word joins
    
    # Fix common typographic issues
    text = text.replace('"', '"').replace('"', '"')  # Standardize quotes
    text = text.replace(''', "'").replace(''', "'")  # Standardize apostrophes
    text = re.sub(r'(?<=[.!?])\s*(?=[A-Z])', '\n\n', text)  # Add paragraph breaks
    
    # Remove redundant whitespace while preserving paragraphs
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    
    # Clean up list formatting
    text = re.sub(r'(?m)^[\s•*-]+', '* ', text)  # Standardize list markers
    
    return text.strip()
    
    # Remove multiple periods and normalize spacing
    text = re.sub(r'\.{2,}', '. ', text)
    text = re.sub(r'\s*\.\s*', '. ', text)
    
    return text.strip()

# Function to split text into chunks
def split_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks while preserving context and structure"""
    # Clean the text while preserving structure
    text = clean_text(text)
    
    # Split into sentences more accurately while preserving structure
    import re
    
    # First, preserve important structural elements
    text = re.sub(r'\n\s*\n', ' <PARA> ', text)  # Mark paragraphs
    text = re.sub(r'([\w\)])\.\s+([A-Z])', r'\1. <SENT> \2', text)  # Mark sentence boundaries
    
    # Split by both structural markers and sentences
    parts = text.split('<PARA>')
    sentences = []
    for part in parts:
        # Split the part into sentences
        sent_parts = re.split('<SENT>', part)
        sentences.extend([s.strip() for s in sent_parts if s.strip()])
        # Add an empty line between paragraphs
        sentences.append('')
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_length = len(sentence_words)
        
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:  # Save the current chunk
                chunks.append(' '.join(current_chunk))
            # Start new chunk with this sentence
            current_chunk = [sentence]
            current_length = sentence_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Function to process uploaded files and add to vector store
def process_uploaded_files():
    if not st.session_state.uploaded_files:
        st.warning("No files to process")
        return
    
    # Always reinitialize vector store to ensure fresh state
    init_vector_store()
    collection = st.session_state.vector_store.get_collection("document_store")
    
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return
    
    try:
        # Create or get collection for storing documents
        collection_name = "document_store"
        
        # Delete existing collection if it exists
        try:
            st.session_state.vector_store.delete_collection(name=collection_name)
        except:
            pass  # Collection might not exist yet
        
        # Create new collection
        collection = st.session_state.vector_store.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
    except Exception as e:
        st.error(f"Error with vector store: {str(e)}")
        return
    
    # Count total documents to process
    total_to_process = len(st.session_state.uploaded_files)
    if total_to_process == 0:
        st.warning("No documents to process")
        return
        
    st.info(f"Processing {total_to_process} document(s)...")
    
    for filename, file_info in st.session_state.uploaded_files.items():
        file_path = file_info['path']
        file_type = os.path.splitext(filename)[1].lower()
        
        try:
            if file_type == '.pdf':
                try:
                    # Process PDF file
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        total_pages = len(pdf_reader.pages)
                        
                        # Store document title/name for context
                        doc_title = filename
                        
                        # First pass: Extract and store document metadata
                        doc_metadata = {
                            'title': doc_title,
                            'total_pages': total_pages,
                            'file_type': 'pdf'
                        }
                        
                        # Add document metadata to collection
                        collection.add(
                            documents=[f"Document Information: {doc_title} is a PDF document with {total_pages} pages."],
                            metadatas=[{
                                'source': filename,
                                'type': 'metadata',
                                'title': doc_title,
                                'page': 0
                            }],
                            ids=[f"{filename}_metadata_{uuid.uuid4()}"]
                        )
                        
                        # First, extract and clean all text from the PDF
                        all_text = ""
                        for page_num, page in enumerate(pdf_reader.pages, 1):
                            try:
                                page_text = page.extract_text()
                                if page_text:
                                    # Clean the text while preserving important formatting
                                    page_text = re.sub(r'\s+', ' ', page_text)  # Normalize whitespace
                                    page_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', page_text)  # Fix sentence boundaries
                                    all_text += f"\n\nPage {page_num}:\n" + page_text
                            except Exception as e:
                                st.warning(f"Warning: Could not fully process page {page_num} of {filename}. Error: {str(e)}")
                        
                        # Clean the entire text
                        cleaned_text = clean_text(all_text)
                        
                        # Split into smaller chunks while preserving context
                        chunks = split_text(cleaned_text, chunk_size=300, overlap=50)
                        
                        # Process each chunk
                        for chunk_num, chunk in enumerate(chunks, 1):
                            # Identify the page number from the chunk content
                            page_match = re.search(r'Page (\d+):', chunk)
                            page_num = int(page_match.group(1)) if page_match else 1
                            
                            # Create document ID and metadata
                            doc_id = f"{filename}_p{page_num}_c{chunk_num}_{uuid.uuid4()}"
                            metadata = {
                                "source": filename,
                                "page": page_num,
                                "chunk": chunk_num,
                                "type": "pdf",
                                "total_chunks": len(chunks)
                            }
                            
                            # Remove the "Page X:" markers from the chunk before storing
                            cleaned_chunk = re.sub(r'Page \d+:', '', chunk).strip()
                            
                            # Generate embeddings and add to collection
                            embeddings = embedding_model.encode([cleaned_chunk])
                            collection.add(
                                documents=[cleaned_chunk],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
                            
                except Exception as e:
                    st.error(f"Error processing PDF {filename}: {str(e)}")
                    raise e
            
            elif file_type == '.csv':
                # Process CSV file with improved handling
                df = pd.read_csv(file_path)
                # Get column descriptions
                column_desc = "Columns in this CSV: " + ", ".join(df.columns)
                
                for idx, row in df.iterrows():
                    # Format each value according to its type
                    formatted_values = []
                    for col, val in row.items():
                        if pd.isna(val):
                            formatted_values.append(f"{col}: N/A")
                        elif isinstance(val, (int, float)):
                            formatted_values.append(f"{col}: {val:,}")
                        else:
                            formatted_values.append(f"{col}: {val}")
                    
                    # Create a well-structured text representation
                    text = column_desc + "\n" + " | ".join(formatted_values)
                    doc_id = f"{filename}_r{idx}_{uuid.uuid4()}"
                    metadata = {
                        "source": filename,
                        "row": idx + 1,
                        "type": "csv"
                    }
                    embeddings = embedding_model.encode([text])
                    collection.add(
                        documents=[text],
                        metadatas=[metadata],
                        ids=[doc_id]
                    )
            
            elif file_type == '.txt':
                # Process TXT file
                with open(file_path, 'r', encoding='utf-8') as txt_file:
                    text = txt_file.read()
                    if text.strip():
                        # Split text into chunks
                        chunks = split_text(text)
                        for chunk_num, chunk in enumerate(chunks, 1):
                            doc_id = f"{filename}_c{chunk_num}_{uuid.uuid4()}"
                            metadata = {
                                "source": filename,
                                "chunk": chunk_num,
                                "type": "txt"
                            }
                            embeddings = embedding_model.encode([chunk])
                            collection.add(
                                documents=[chunk],
                                metadatas=[metadata],
                                ids=[doc_id]
                            )
        
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            continue

# Initialize chat messages if empty
if not st.session_state.messages:
    system_prompt = create_system_prompt(config)
    st.session_state.messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hello! I'm Lucrum Private AI. How can I assist you today?"}
    ]
    
# Process uploaded files if needed
if st.session_state.uploaded_files:
    current_files = set(st.session_state.uploaded_files.keys())
    if not hasattr(st.session_state, 'last_processed_files') or st.session_state.last_processed_files != current_files:
        with st.spinner("Processing new documents..."):
            # Clear previous vector store
            st.session_state.vector_store = None
            # Process all files
            process_uploaded_files()
            # Keep track of processed files
            st.session_state.last_processed_files = current_files
            # Update system message with current document context
            doc_context = "\nAvailable documents:\n"
            for filename in current_files:
                doc_type = os.path.splitext(filename)[1].lower()
                doc_context += f"- {filename} (Type: {doc_type})\n"
            # Update system message
            if st.session_state.messages:
                st.session_state.messages[0]['content'] += doc_context
            doc_context = "Available documents:\n"
            for filename in st.session_state.uploaded_files:
                doc_type = os.path.splitext(filename)[1].lower()
                doc_context += f"- {filename} (Type: {doc_type})\n"
            system_prompt = create_system_prompt(config)
            st.session_state.messages[0] = {
                "role": "system",
                "content": f"{system_prompt}\n\nCURRENT DOCUMENT CONTEXT:\n{doc_context}"
            }

# Display chat header with Lucrum branding using standard Streamlit components
st.title("Lucrum Private AI")
st.subheader("Secure, Offline, Enterprise-Grade AI Assistant")

# Create a sidebar for settings and file upload
with st.sidebar:
    st.header("Settings")
    
    # Model selection dropdown
    st.session_state.selected_model = st.selectbox(
        "Lucrum Model",
        ["Lucrum Pro", "Lucrum Thinking", "Lucrum Instant", "Lucrum Auto"],
        index=0,
        help="Select the Lucrum AI model to use for chat"
    )
    
    st.markdown("---")
    
    # File upload section
    st.header("Upload Files")
    st.markdown(f"Upload up to {MAX_FILES} files (PDF, CSV, or TXT)")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=ALLOWED_EXTENSIONS,
        help=f"Maximum {MAX_FILES} files. Supported formats: PDF, CSV, TXT"
    )
    
    # Process uploaded files
    if uploaded_files:
        if len(uploaded_files) > MAX_FILES:
            st.error(f"Please upload no more than {MAX_FILES} files.")
        else:
            # Track new files
            new_files = False
            
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    new_files = True
                    # Save file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        st.session_state.uploaded_files[uploaded_file.name] = {
                            'path': tmp_file.name,
                            'type': uploaded_file.type,
                            'size': uploaded_file.size,
                            'processed': False
                        }
            
            # Process files if there are new ones
            if new_files:
                process_uploaded_files()
                st.success(f"Successfully processed {', '.join(f.name for f in uploaded_files)}")
                st.rerun()
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files:")
        for filename in st.session_state.uploaded_files:
            st.text(f"✓ {filename}")
        
        if st.button("Clear All Files"):
            # Remove temporary files
            for file_info in st.session_state.uploaded_files.values():
                if os.path.exists(file_info['path']):
                    os.unlink(file_info['path'])
            st.session_state.uploaded_files = {}
            st.session_state.vector_store = None
            st.rerun()

# Display chat messages (skip system messages)
for message in st.session_state.messages:
    if message["role"] != "system":
        # Use the avatar for assistant messages
        if message["role"] == "assistant":
            with st.chat_message(message["role"], avatar=AVATAR):
                st.write(message["content"])
                # Show document reference if available
                if "source" in message:
                    st.caption(f"Source: {message['source']}")
        else:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Function to generate response with streaming
def get_relevant_context(query, max_results=5, max_tokens=1000):
    """Retrieve relevant documents from the vector store with improved context"""
    if not st.session_state.vector_store:
        st.info("No documents have been uploaded yet.")
        return None
        
    # Get the names of currently uploaded files
    current_files = list(st.session_state.uploaded_files.keys()) if hasattr(st.session_state, 'uploaded_files') else []
    
    try:
        collection = st.session_state.vector_store.get_collection("document_store")
        
        # Get total document count
        doc_count = collection.count()
        if doc_count == 0:
            st.warning("Vector store is empty. Please upload and process documents first.")
            return None

        # Enhance query with type-specific keywords if asking about document content
        if any(word in query.lower() for word in ['document', 'pdf', 'file', 'content']):
            query = f"Document content: {query}"

        # Search with embedding similarity (no spinner here; caller handles UI)
        # First try exact matching if it's a document-related query
        if any(word in query.lower() for word in ['what', 'summary', 'tell me about', 'content']):
            results = collection.query(
                query_texts=[query],
                n_results=max_results * 2,
                include=["documents", "metadatas", "distances"]
            )
        else:
            enhanced_query = f"Find information about: {query}"
            results = collection.query(
                query_texts=[enhanced_query],
                n_results=max_results * 2,
                include=["documents", "metadatas", "distances"]
            )

        # Normalize Chroma results which may be nested lists
        docs_raw = results.get('documents', [])
        metas_raw = results.get('metadatas', [])
        dists_raw = results.get('distances', [])

        docs = docs_raw[0] if docs_raw and isinstance(docs_raw[0], list) else docs_raw
        metas = metas_raw[0] if metas_raw and isinstance(metas_raw[0], list) else metas_raw
        dists = dists_raw[0] if dists_raw and isinstance(dists_raw[0], list) else dists_raw

        # Ensure lists
        if not isinstance(docs, list):
            docs = [docs] if docs is not None else []
        if not isinstance(metas, list):
            metas = [metas] if metas is not None else []
        if not isinstance(dists, list):
            dists = [dists] if dists is not None else []

        # Build filtered list of tuples (doc, meta, dist)
        filtered = []
        sources = set()
        for doc, meta, dist in zip(docs, metas, dists):
            try:
                dist_val = float(dist)
            except Exception:
                dist_val = 1.0

            if dist_val < 0.8:
                filtered.append((doc, meta, dist_val))
                if isinstance(meta, dict) and meta.get('source'):
                    sources.add(meta.get('source'))
                if len(filtered) >= max_results:
                    break

        if sources:
            st.session_state['last_sources'] = list(sources)

        if not filtered:
            st.info("No relevant information found in the uploaded documents.")
            return None

        # Prepare results variable for downstream compatibility
        results = {
            'documents': [ [t[0] for t in filtered] ],
            'metadatas': [ [t[1] for t in filtered] ],
            'distances': [ [t[2] for t in filtered] ],
        }
        
        if not results or not results['documents'] or len(results['documents'][0]) == 0:
            st.info("No relevant information found in the uploaded documents.")
            return None
            
            # Process search results without showing summary
        
        # 'filtered' already contains tuples (doc, meta, dist_val)
        sorted_results = sorted(filtered, key=lambda x: x[2])

        # Format results with metadata
        context_parts = []
        total_tokens = 0

        for doc, metadata, distance in sorted_results:
            # Format citation
            source = metadata['source']
            citation = f"[From {source}"
            if metadata['type'] == 'pdf':
                citation += f", Page {metadata['page']}"
                if 'chunk' in metadata:
                    citation += f", Section {metadata['chunk']}/{metadata['total_chunks']}"
            citation += f", Relevance: {100 * (1 - distance):.1f}%]\n"
            
            # Estimate tokens (rough approximation)
            context_length = len(citation + doc) // 4
            if total_tokens + context_length > max_tokens:
                break
                
            context_parts.append(citation + doc)
            total_tokens += context_length
        
        if context_parts:
            return "\n\n".join(context_parts)
        return None
        
    except Exception as e:
        st.error(f"Error searching documents: {str(e)}")
        return None
    

def get_model_settings(model_name):
    """Map Lucrum model names to actual model settings"""
    settings = {
        "Lucrum Pro": {
            "description": "Enhanced Qwen model for general-purpose tasks and document analysis",
            "base_model": "Qwen1.5-0.5B-Chat-Q6_K.gguf",  # Qwen 0.5B with 6-bit quantization
            "temperature": 0.7,  # Balanced creativity
            "top_p": 0.92,  # Slightly adjusted for better quality with Q6
            "top_k": 45,    # Increased for better sampling with Q6
            "repeat_penalty": 1.08,  # Adjusted for Q6 model
            "max_tokens": 512,
            "supports_rag": True  # Supports document Q&A
        },
        "Lucrum Thinking": {
            "description": "Focused model for precise, analytical responses with detailed document analysis",
            "base_model": "qwen1_5-1_8b-chat-q4_k_m.gguf",  # 1.8B model for enhanced analysis
            "temperature": 0.5,  # More focused responses
            "top_p": 0.92,
            "top_k": 50,        # Increased for better sampling with larger model
            "repeat_penalty": 1.1,
            "max_tokens": 1024,  # Increased context for longer responses
            "supports_rag": True  # Supports document Q&A
        },
        "Lucrum Instant": {
            "description": "Higher quality responses for quick queries with 8-bit precision",
            "base_model": "qwen1.5-0.5B-Chat-Q8_0.gguf",  # 8-bit quantization for better quality
            "temperature": 0.72,  # Balanced for quality
            "top_p": 0.92,
            "top_k": 40,
            "repeat_penalty": 1.05,
            "max_tokens": 384,  # Increased for better completion
            "supports_rag": True  # Enabled RAG with better quality
        },
        "Lucrum Auto": {
            "description": "High-quality adaptive model with 8-bit precision",
            "base_model": "qwen1.5-0.5B-Chat-Q8_0.gguf",  # 8-bit quantization for best quality
            "temperature": 0.7,
            "top_p": 0.92,
            "top_k": 45,
            "repeat_penalty": 1.08,
            "max_tokens": 512,
            "supports_rag": True  # Supports document Q&A
        }
    }
    return settings.get(model_name, settings["Lucrum Pro"])

def generate_streaming_response(prompt, message_placeholder):
    try:
        # Get relevant context from vector store (spinner is handled by caller)
        context = get_relevant_context(prompt)
        
        # Format messages for the model
        chat_history = []
        
        if context:
            # Create a more focused system prompt with the context
            context_prompt = (
                "You are a helpful AI assistant. Format your responses for maximum readability:\n"
                "1. Use double line breaks between paragraphs\n"
                "2. When presenting lists, use A., B., C. format with each item on a new line\n"
                "3. Break complex topics into clear, well-spaced sections\n"
                "4. Use bullet points where appropriate\n"
                "5. Add a line break after introductory text before starting lists\n\n"
                "Use the following document information to provide a clear, accurate, and well-structured answer. "
                "Always cite specific pages or sections when referring to the document content. Break down technical "
                "concepts into simpler terms.\n\n"
                f"Document excerpts:\n{context}\n\n"
                "Instructions:\n"
                "1. Start by directly answering the question\n"
                "2. Use specific citations when referencing document content\n"
                "3. Explain technical terms if present\n"
                "4. Keep the tone professional but accessible\n"
                "5. If information is missing or unclear, acknowledge it"
            )
            chat_history.append(f"System: {context_prompt}")
        else:
            chat_history.append("System: You are a helpful AI assistant. Format your response for readability:\n"
                              "1. Use double line breaks between paragraphs\n"
                              "2. Present lists in A., B., C. format with items on new lines\n"
                              "3. Break complex topics into clear sections\n\n"
                              "The user's question is not directly addressed in the uploaded documents. "
                              "Provide a general response and suggest uploading relevant documents if needed.")
        
        # Add chat history
        for msg in st.session_state.messages:
            if msg["role"] != "system":  # Skip old system messages
                chat_history.append(f"{msg['role'].title()}: {msg['content']}")
        
        # Add current prompt
        chat_history.append(f"User: {prompt}")
        formatted_prompt = "\n\n".join(chat_history) + "\n\nAssistant:"
        
        # Get model settings and generate response
        model_settings = get_model_settings(st.session_state.selected_model)
        full_response = ""
        
        try:
            for chunk in llm(
                formatted_prompt,
                max_tokens=model_settings['max_tokens'],
                temperature=model_settings['temperature'],
                top_p=model_settings['top_p'],
                top_k=model_settings['top_k'],
                repeat_penalty=model_settings['repeat_penalty'],
                stop=["User:", "\n\nUser:", "System:"],
                stream=True,
            ):
                if chunk["choices"][0]["text"]:
                    text_chunk = chunk["choices"][0]["text"]
                    full_response += text_chunk
                    # Update the message placeholder with the text received so far
                    message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            message_placeholder.error(error_message)
            return error_message
        
        # Final update without the cursor
        message_placeholder.markdown(full_response)
        return full_response
        
    except Exception as e:
        error_message = f"Error processing response: {str(e)}"
        message_placeholder.error(error_message)
        return error_message
    
    # Format messages for the model
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            chat_history.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            chat_history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            chat_history.append(f"Assistant: {msg['content']}")
    
    # Build the prompt with context if available
    if context:
        system_context = (
            "Use the following information to help answer the user's question. "
            "Always mention the source of information in your response:\n\n" + context
        )
        chat_history.insert(1, f"System: {system_context}")
    
    # Add current prompt
    chat_history.append(f"User: {prompt}")
    formatted_prompt = "\n\n".join(chat_history) + "\n\nAssistant:"
    
    # Get model settings based on selected model
    model_settings = get_model_settings(st.session_state.selected_model)
    
    # Generate response with streaming
    full_response = ""
    
    # Use the streaming capability of llama-cpp-python
    try:
        for chunk in llm(
            formatted_prompt,
            max_tokens=model_settings['max_tokens'],
            temperature=model_settings['temperature'],
            top_p=model_settings['top_p'],
            top_k=model_settings['top_k'],
            repeat_penalty=model_settings['repeat_penalty'],
            stop=["User:", "\n\nUser:", "System:"],
            stream=True,
        ):
            if chunk["choices"][0]["text"]:
                text_chunk = chunk["choices"][0]["text"]
                full_response += text_chunk
                # Update the message placeholder with the text received so far
                message_placeholder.markdown(full_response + "▌")
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        message_placeholder.error(error_message)
        return error_message
    
    # Log the interaction
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": st.session_state.selected_model,
        "prompt": prompt,
        "response": full_response,
        "context_used": bool(context)
    }
    
    # Save to JSON log file
    log_file = os.path.join(os.path.dirname(__file__), "chat_logs.json")
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        logs.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save chat log: {str(e)}")
    
    # Final update without the cursor
    message_placeholder.markdown(full_response)
    
    return full_response

# Chat input
if prompt := st.chat_input("Ask me anything about the uploaded documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response with streaming (single spinner)
    with st.chat_message("assistant", avatar=AVATAR):
        with st.spinner("Thinking..."):
            message_placeholder = st.empty()
            response = generate_streaming_response(prompt, message_placeholder)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar with Lucrum branding and options
with st.sidebar:
    st.title("Lucrum Private AI")
    
    if "system_prompt" in config and "identity" in config["system_prompt"]:
        identity = config["system_prompt"]["identity"]
        st.caption(identity.get("limitations", "Fully offline, secure AI assistant"))
    
    st.divider()
    st.subheader("Chat Options")
    
    if st.button("Clear Conversation"):
        # Reset conversation but keep the system prompt
        system_prompt = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "system"), 
                           create_system_prompt(config))
        st.session_state.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": "Hello! I'm Lucrum Private AI. How can I assist you today?"}
        ]
        st.rerun()  # Updated from st.experimental_rerun()
    
    st.divider()
    
    # Display product features
    if "system_prompt" in config and "lucrum_ai_technology_suite" in config["system_prompt"]:
        st.subheader("Lucrum AI Features")
        components = config["system_prompt"]["lucrum_ai_technology_suite"].get("components", [])
        for component in components:
            st.markdown(f"✓ {component}")
    
    st.divider()
    
    if "system_prompt" in config and "identity" in config["system_prompt"]:
        st.caption(f"Created by {config['system_prompt']['identity'].get('creator', 'Lucrum Industries')}")