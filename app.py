import os
import warnings
import logging
import json
import threading
import pickle
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from difflib import SequenceMatcher
import re
from gtts import gTTS
import uuid
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

try:
    from docx import Document
    import PyPDF2
    import pdfplumber
    import fitz
except ImportError:
    print("Please install: pip install python-docx PyPDF2 pdfplumber PyMuPDF")

try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    print("Please install: pip install langchain langchain-openai langchain-community faiss-cpu")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOCUMENTS_FOLDER'] = os.getenv('DOCUMENTS_FOLDER', '/opt/render/project/src/documents')
app.config['MAX_DOCUMENTS'] = 10
app.config['VIDEOS_FOLDER'] = 'static/videos'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SESSION_CACHE_FOLDER'] = 'session_cache'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOCUMENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEOS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'audio'), exist_ok=True)
os.makedirs(app.config['SESSION_CACHE_FOLDER'], exist_ok=True)

VECTORSTORE_CACHE = os.path.join(app.config['DOCUMENTS_FOLDER'], '.vectorstore_cache.pkl')
CACHE_HASH_FILE = os.path.join(app.config['DOCUMENTS_FOLDER'], '.cache_hash.txt')

# Global shared state (across all workers via file system)
GLOBAL_STATE_FILE = os.path.join(app.config['SESSION_CACHE_FOLDER'], 'global_state.json')
global_state_lock = threading.Lock()

# In-memory sessions for current worker
sessions = {}

GREETING_PATTERNS = [
    'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
    'good evening', 'howdy', 'hiya', 'whats up', "what's up", 'sup'
]

GREETING_RESPONSES = [
    "Hello! I'm your HR Assistant. How can I help you today?",
    "Hi there! I'm here to help with any HR-related questions you have.",
    "Greetings! What HR information can I assist you with today?",
    "Hello! Feel free to ask me anything about HR policies and procedures."
]

FAREWELL_PATTERNS = ['bye', 'goodbye', 'exit', 'quit', 'end', 'see you', 'farewell']

def load_global_state():
    """Load global state from file"""
    try:
        if os.path.exists(GLOBAL_STATE_FILE):
            with open(GLOBAL_STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading global state: {e}")
    return {
        'documents_loaded': False,
        'loading_in_progress': False,
        'loading_error': None,
        'preloaded_files': [],
        'load_started_at': None
    }

def save_global_state(state):
    """Save global state to file"""
    try:
        with global_state_lock:
            with open(GLOBAL_STATE_FILE, 'w') as f:
                json.dump(state, f)
    except Exception as e:
        print(f"Error saving global state: {e}")

def get_global_state():
    """Get current global state"""
    return load_global_state()

def update_global_state(**kwargs):
    """Update specific fields in global state"""
    with global_state_lock:
        state = load_global_state()
        state.update(kwargs)
        save_global_state(state)

def get_session(session_id):
    """Get or create session"""
    if session_id not in sessions:
        sessions[session_id] = {
            'chat_history': [],
            'uploaded_files': [],
            'feedback_history': []
        }
    return sessions[session_id]

def is_greeting(message):
    msg_lower = message.lower().strip()
    return any(pattern in msg_lower for pattern in GREETING_PATTERNS)

def is_farewell(message):
    msg_lower = message.lower().strip()
    return any(pattern in msg_lower for pattern in FAREWELL_PATTERNS)

def get_directory_hash(directory):
    hash_obj = hashlib.md5()
    try:
        file_count = 0
        for root, dirs, files in os.walk(directory):
            for filename in sorted(files):
                if not filename.startswith('.') and filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                    filepath = os.path.join(root, filename)
                    hash_obj.update(filename.encode())
                    try:
                        hash_obj.update(str(os.path.getmtime(filepath)).encode())
                        file_count += 1
                    except:
                        pass
        print(f"Directory hash calculated for {file_count} files")
    except Exception as e:
        print(f"Error calculating directory hash: {e}")
    return hash_obj.hexdigest()

def extract_pdf_text(pdf_file):
    text = ""
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)
        import sys
        from contextlib import redirect_stderr
        from io import StringIO
        f = StringIO()
        with redirect_stderr(f):
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for page_num in range(min(doc.page_count, 50)):
                try:
                    page = doc.load_page(page_num)
                    text += page.get_text() + "\n"
                except:
                    continue
            doc.close()
        if text.strip():
            return text
    except:
        pass
    try:
        pdf_file.seek(0)
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages[:50]:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except:
                    continue
    except:
        pass
    return text if text.strip() else "Could not extract text from PDF"

def extract_docx_text(docx_file):
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except:
        return "Error reading DOCX file"

def extract_txt_text(txt_file):
    try:
        return txt_file.read().decode('utf-8')
    except:
        try:
            txt_file.seek(0)
            return txt_file.read().decode('latin-1')
        except:
            return "Error reading TXT file"

def process_file(file_path_or_obj, filename):
    file_ext = filename.lower().split('.')[-1]
    
    if isinstance(file_path_or_obj, str):
        with open(file_path_or_obj, 'rb') as f:
            if file_ext == 'pdf':
                return extract_pdf_text(f)
            elif file_ext in ['docx', 'doc']:
                return extract_docx_text(f)
            elif file_ext == 'txt':
                return extract_txt_text(f)
    else:
        if file_ext == 'pdf':
            return extract_pdf_text(file_path_or_obj)
        elif file_ext in ['docx', 'doc']:
            return extract_docx_text(file_path_or_obj)
        elif file_ext == 'txt':
            return extract_txt_text(file_path_or_obj)
    return ""

def load_documents_from_directory(directory, max_files=10):
    all_text = ""
    processed_files = []
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return all_text, processed_files
    
    print(f"Scanning directory: {directory}")
    
    all_files_with_size = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                filepath = os.path.join(root, filename)
                try:
                    size = os.path.getsize(filepath)
                    all_files_with_size.append((root, filename, size))
                except Exception as e:
                    print(f"Could not get size for {filename}: {e}")
    
    if len(all_files_with_size) == 0:
        print("No documents found in directory")
        return all_text, processed_files
    
    print(f"Found {len(all_files_with_size)} documents")
    
    all_files_with_size.sort(key=lambda x: x[2])
    
    if len(all_files_with_size) > max_files:
        print(f"Limiting to {max_files} documents")
        all_files_with_size = all_files_with_size[:max_files]
    
    for idx, (root, filename, size) in enumerate(all_files_with_size, 1):
        file_path = os.path.join(root, filename)
        try:
            if size > 3 * 1024 * 1024:
                print(f"Skipping large file: {filename}")
                continue
            
            print(f"Processing [{idx}/{len(all_files_with_size)}]: {filename}")
            text = process_file(file_path, filename)
            
            if text and len(text) > 50:
                if len(text) > 30000:
                    text = text[:30000] + "...[truncated]"
                
                all_text += f"\n\n--- {filename} ---\n{text}"
                processed_files.append(filename)
                print(f"Processed: {filename}")
            else:
                print(f"No text from: {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"Successfully processed {len(processed_files)} files")
    return all_text, processed_files

def load_or_create_vectorstore(docs_folder):
    print("Checking for cached vectorstore...")
    current_hash = get_directory_hash(docs_folder)
    
    if os.path.exists(VECTORSTORE_CACHE) and os.path.exists(CACHE_HASH_FILE):
        try:
            with open(CACHE_HASH_FILE, 'r') as f:
                cached_hash = f.read().strip()
            
            if cached_hash == current_hash:
                print("Loading from cache...")
                with open(VECTORSTORE_CACHE, 'rb') as f:
                    vectorstore = pickle.load(f)
                print("Loaded from cache!")
                return vectorstore, True
        except Exception as e:
            print(f"Cache load error: {e}")
    
    print("Creating new vectorstore...")
    all_text, processed_files = load_documents_from_directory(docs_folder, max_files=10)
    
    if not all_text or len(processed_files) == 0:
        print("No documents to process")
        return None, False
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=150,
        length_function=len
    )
    texts = text_splitter.split_text(all_text)
    print(f"Created {len(texts)} chunks")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts, embeddings)
    
    try:
        with open(VECTORSTORE_CACHE, 'wb') as f:
            pickle.dump(vectorstore, f)
        with open(CACHE_HASH_FILE, 'w') as f:
            f.write(current_hash)
        print("Cached successfully")
    except Exception as e:
        print(f"Cache save error: {e}")
    
    return vectorstore, False

# Global vectorstore (shared across all requests in this worker)
global_vectorstore = None
global_chain = None
vectorstore_lock = threading.Lock()

def load_docs_background(docs_folder):
    """Background document loading with global state"""
    global global_vectorstore, global_chain
    
    print("=" * 60)
    print("STARTING BACKGROUND LOADING")
    print("=" * 60)
    
    try:
        # Set loading flag
        update_global_state(
            loading_in_progress=True,
            loading_error=None,
            load_started_at=datetime.now().isoformat()
        )
        
        vectorstore, was_cached = load_or_create_vectorstore(docs_folder)
        
        if vectorstore:
            processed_files = []
            for root, dirs, files in os.walk(docs_folder):
                for filename in files:
                    if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                        processed_files.append(filename)
            
            processed_files = processed_files[:10]
            
            with vectorstore_lock:
                global_vectorstore = vectorstore
                global_chain = create_chain(vectorstore)
            
            # Update global state
            update_global_state(
                documents_loaded=True,
                loading_in_progress=False,
                loading_error=None,
                preloaded_files=processed_files
            )
            
            cache_status = "FROM CACHE" if was_cached else "NEWLY CREATED"
            print("=" * 60)
            print(f"SUCCESS! Loaded {len(processed_files)} documents {cache_status}")
            print("=" * 60)
        else:
            update_global_state(
                documents_loaded=False,
                loading_in_progress=False,
                loading_error="No documents found"
            )
            print("FAILED: No documents found")
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        update_global_state(
            documents_loaded=False,
            loading_in_progress=False,
            loading_error=error_msg
        )
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())

def generate_tts(text):
    try:
        clean_text = re.sub(r'<[^>]+>', '', text)
        clean_text = re.sub(r'\*\*', '', clean_text)
        
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join('static', 'audio', audio_filename)
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        tts.save(audio_path)
        
        return f'/static/audio/{audio_filename}'
    except Exception as e:
        print(f"TTS error: {e}")
        return None

def similarity_score(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_related_video(query, threshold=0.5):
    videos_folder = app.config['VIDEOS_FOLDER']
    
    if not os.path.exists(videos_folder):
        return None
    
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    query_words = set(query_clean.split())
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
    query_words = query_words - stop_words
    
    if len(query_words) == 0:
        return None
    
    best_match = None
    best_score = 0
    
    for filename in os.listdir(videos_folder):
        if filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mov', '.avi')):
            name_without_ext = os.path.splitext(filename)[0]
            name_clean = re.sub(r'[^\w\s]', ' ', name_without_ext.lower())
            name_words = set(name_clean.split())
            name_words = name_words - stop_words
            
            if len(name_words) == 0:
                continue
            
            common_words = query_words.intersection(name_words)
            if len(common_words) < 2:
                continue
            
            word_overlap = len(common_words) / max(len(query_words), 1)
            string_sim = similarity_score(query_clean, name_clean)
            combined_score = (word_overlap * 0.7) + (string_sim * 0.3)
            
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                best_match = filename
    
    return best_match

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    prompt = ChatPromptTemplate.from_template("""
You are an HR Assistant for Invenio Business Solutions. Answer questions ONLY using the provided context from HR policy documents.

Context: {context}
Question: {input}

IMPORTANT RULES:
1. Answer ONLY based on the information in the Context above
2. If the context doesn't contain the answer, say: "I don't have that specific information in the HR policy documents. Please contact HR directly or ask a different question."
3. Do NOT make assumptions or provide general knowledge
4. Do NOT add extra information not present in the context
5. Quote specific policy details when available
6. Be concise and direct

Answer:
""")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    try:
        session_id = request.json.get('session_id', 'default')
        get_session(session_id)  # Create session
        
        docs_folder = app.config['DOCUMENTS_FOLDER']
        state = get_global_state()
        
        # Check if already loaded
        if state['documents_loaded']:
            files = state.get('preloaded_files', [])
            return jsonify({
                'success': True,
                'files': files[:10],
                'message': f'{len(files)} documents loaded and ready',
                'status': 'ready'
            })
        
        # Check if there was an error
        if state.get('loading_error'):
            return jsonify({
                'success': True,
                'files': [],
                'message': f'Error: {state["loading_error"]}',
                'status': 'error'
            })
        
        # Check if loading is in progress
        if state.get('loading_in_progress'):
            # Check if loading has been stuck for too long
            if state.get('load_started_at'):
                try:
                    start_time = datetime.fromisoformat(state['load_started_at'])
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > 300:  # 5 minutes
                        print("Loading stuck, resetting...")
                        update_global_state(loading_in_progress=False)
                    else:
                        return jsonify({
                            'success': True,
                            'files': [],
                            'message': 'Loading documents...',
                            'status': 'loading'
                        })
                except:
                    pass
        
        # Start loading (only one worker will actually start it)
        update_global_state(
            loading_in_progress=True,
            load_started_at=datetime.now().isoformat()
        )
        
        thread = threading.Thread(
            target=load_docs_background,
            args=(docs_folder,),
            daemon=True
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'files': [],
            'message': 'Loading documents...',
            'status': 'loading'
        })
        
    except Exception as e:
        import traceback
        print(f"ERROR in init_session: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/check_status', methods=['GET'])
def check_status():
    try:
        state = get_global_state()
        
        if state.get('documents_loaded'):
            files = state.get('preloaded_files', [])
            return jsonify({
                'ready': True,
                'files_count': len(files),
                'message': f'Ready - {len(files)} documents loaded',
                'files': files[:10]
            })
        elif state.get('loading_error'):
            return jsonify({
                'ready': False,
                'loading': False,
                'error': True,
                'message': f'Error: {state["loading_error"]}'
            })
        elif state.get('loading_in_progress'):
            return jsonify({
                'ready': False,
                'loading': True,
                'message': 'Loading documents...'
            })
        else:
            return jsonify({
                'ready': False,
                'loading': False,
                'message': 'Not initialized'
            })
    except Exception as e:
        print(f"Status check error: {e}")
        return jsonify({
            'ready': False,
            'loading': False,
            'error': True,
            'message': str(e)
        })

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        session_id = request.form.get('session_id', 'default')
        session = get_session(session_id)
        
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if not files or len(files) == 0:
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        all_text = ""
        processed_files = []
        
        for file in files:
            if file.filename:
                try:
                    filename = secure_filename(file.filename)
                    text = process_file(file, filename)
                    if text and text.strip():
                        all_text += f"\n\n--- {filename} ---\n{text}"
                        processed_files.append(filename)
                        
                        file.seek(0)
                        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                except Exception as e:
                    print(f"Error with {filename}: {e}")
        
        if not all_text:
            return jsonify({'success': False, 'error': 'No text extracted'}), 400
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=150,
            length_function=len
        )
        texts = text_splitter.split_text(all_text)
        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        global global_vectorstore, global_chain
        
        with vectorstore_lock:
            if global_vectorstore:
                global_vectorstore.add_texts(texts)
            else:
                global_vectorstore = FAISS.from_texts(texts, embeddings)
            
            global_chain = create_chain(global_vectorstore)
        
        # Update global state
        state = get_global_state()
        current_files = state.get('preloaded_files', [])
        current_files.extend(processed_files)
        update_global_state(
            documents_loaded=True,
            preloaded_files=current_files
        )
        
        session['uploaded_files'].extend(processed_files)
        
        return jsonify({
            'success': True,
            'files': processed_files
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        session_id = request.json.get('session_id', 'default')
        message = request.json.get('message', '')
        is_voice_input = request.json.get('is_voice_input', False)
        
        session = get_session(session_id)
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        session['chat_history'].append({
            'message': message,
            'is_user': True,
            'timestamp': datetime.now().isoformat()
        })
        
        # Handle greetings
        if is_greeting(message):
            import random
            response = random.choice(GREETING_RESPONSES)
            
            session['chat_history'].append({
                'message': response,
                'is_user': False,
                'timestamp': datetime.now().isoformat()
            })
            
            audio_url = generate_tts(response)
            
            return jsonify({
                'response': response,
                'audio_url': audio_url,
                'auto_play': is_voice_input
            })
        
        # Handle farewells
        if is_farewell(message):
            response = "Thank you for using HR Assistant! Have a great day!"
            session['chat_history'].append({
                'message': response,
                'is_user': False,
                'timestamp': datetime.now().isoformat()
            })
            
            audio_url = generate_tts(response)
            
            return jsonify({
                'response': response,
                'audio_url': audio_url,
                'show_feedback': True
            })
        
        # Check if ready
        state = get_global_state()
        if not state.get('documents_loaded'):
            if state.get('loading_in_progress'):
                error_msg = 'Knowledge base is still loading. Please wait a moment and try again.'
                return jsonify({
                    'error': error_msg,
                    'loading': True
                }), 503
            else:
                error_msg = 'No knowledge base loaded. Please refresh the page.'
                return jsonify({
                    'error': error_msg,
                    'loading': False
                }), 503
        
        global global_chain
        if not global_chain:
            return jsonify({
                'error': 'System not ready. Please refresh.'
            }), 500
        
        # Process with chain
        with vectorstore_lock:
            result = global_chain.invoke({'input': message})
            response = result['answer']
        
        session['chat_history'].append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        
        # Always generate TTS
        audio_url = generate_tts(response)
        
        response_data = {
            'response': response,
            'audio_url': audio_url,
            'auto_play': is_voice_input
        }
        
        # Check for video
        related_video = find_related_video(message)
        if related_video:
            response_data['video'] = f'/static/videos/{related_video}'
            response_data['video_name'] = os.path.splitext(related_video)[0].replace('_', ' ').title()
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    session_id = request.json.get('session_id', 'default')
    rating = request.json.get('rating')
    comment = request.json.get('comment', '')
    session = get_session(session_id)
    
    feedback_data = {
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    session['feedback_history'].append(feedback_data)
    
    return jsonify({
        'success': True,
        'message': 'Thank you for your feedback!'
    })

@app.route('/export/json', methods=['POST'])
def export_json():
    session_id = request.json.get('session_id', 'default')
    session = get_session(session_id)
    
    chat_data = {
        "export_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": len(session['chat_history']),
            "format": "HR Assistant JSON Export"
        },
        "chat_history": session['chat_history'],
        "feedback": session['feedback_history']
    }
    
    return jsonify(chat_data)

@app.route('/export/feedback', methods=['POST'])
def export_feedback():
    session_id = request.json.get('session_id', 'default')
    session = get_session(session_id)
    
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Timestamp', 'Rating', 'Comment'])
    
    for feedback in session['feedback_history']:
        writer.writerow([
            feedback['timestamp'],
            feedback['rating'],
            feedback.get('comment', '')
        ])
    
    csv_content = output.getvalue()
    
    return jsonify({
        'success': True,
        'csv_data': csv_content,
        'filename': f'feedback_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    })

@app.route('/clear', methods=['POST'])
def clear_session():
    session_id = request.json.get('session_id', 'default')
    if session_id in sessions:
        sessions[session_id] = {
            'chat_history': [],
            'uploaded_files': [],
            'feedback_history': []
        }
    return jsonify({'success': True})

@app.route('/feedback/stats', methods=['GET'])
def feedback_stats():
    session_id = request.args.get('session_id', 'default')
    session = get_session(session_id)
    
    feedback_history = session['feedback_history']
    if not feedback_history:
        return jsonify({'count': 0, 'average': 0})
    
    avg_rating = sum(f['rating'] for f in feedback_history) / len(feedback_history)
    
    return jsonify({
        'count': len(feedback_history),
        'average': round(avg_rating, 1)
    })

# Debug endpoint
@app.route('/test/docs')
def test_docs():
    folder = app.config['DOCUMENTS_FOLDER']
    files = []
    
    if os.path.exists(folder):
        for root, dirs, filenames in os.walk(folder):
            for filename in filenames:
                if filename.endswith(('.pdf', '.docx', '.txt')):
                    files.append(filename)
    
    state = get_global_state()
    
    return jsonify({
        'folder': folder,
        'exists': os.path.exists(folder),
        'file_count': len(files),
        'files': files,
        'global_state': state
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
