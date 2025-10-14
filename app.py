import os
import warnings
import logging
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from difflib import SequenceMatcher
import re
from gtts import gTTS
import uuid

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
app.config['DOCUMENTS_FOLDER'] = 'documents'
app.config['VIDEOS_FOLDER'] = 'static/videos'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['SESSION_FOLDER'] = 'sessions'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate that the key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please configure it in Render dashboard.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOCUMENTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEOS_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'audio'), exist_ok=True)

sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            'vectorstore': None,
            'conversation_chain': None,
            'chat_history': [],
            'uploaded_files': [],
            'feedback_history': [],
            'preloaded_files': []
        }
    return sessions[session_id]

def extract_pdf_text(pdf_file):
    """Extract text from PDF file"""
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
            for page_num in range(doc.page_count):
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
            for page in pdf.pages:
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
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except:
        return "Error reading DOCX file"

def extract_txt_text(txt_file):
    """Extract text from TXT file"""
    try:
        return txt_file.read().decode('utf-8')
    except:
        try:
            txt_file.seek(0)
            return txt_file.read().decode('latin-1')
        except:
            return "Error reading TXT file"

def process_file(file_path_or_obj, filename):
    """Process a single file and extract text"""
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

def load_documents_from_directory(directory):
    """
    Load all documents from a directory - FIXED VERSION
    """
    all_text = ""
    processed_files = []
    MAX_CHARS = 2000000  # 2 million chars limit
    
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return all_text, processed_files
    
    files = os.listdir(directory)
    print(f"Found {len(files)} files in {directory}")
    
    for filename in files:
        if len(all_text) >= MAX_CHARS:
            print(f"Reached max chars limit ({MAX_CHARS}), stopping at {filename}")
            break
            
        file_path = os.path.join(directory, filename)
        
        # Skip directories
        if not os.path.isfile(file_path):
            continue
        
        try:
            text = ""
            
            # FIXED: Use correct function names
            if filename.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    text = extract_pdf_text(f)  # ✅ CORRECT
            elif filename.lower().endswith(('.docx', '.doc')):
                with open(file_path, 'rb') as f:
                    text = extract_docx_text(f)  # ✅ CORRECT
            elif filename.lower().endswith('.txt'):
                with open(file_path, 'rb') as f:
                    text = extract_txt_text(f)
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue
            
            if not text or len(text.strip()) == 0:
                print(f"⚠️ No text extracted from {filename}")
                continue
            
            # Truncate if adding this would exceed limit
            remaining_space = MAX_CHARS - len(all_text)
            if len(text) > remaining_space:
                text = text[:remaining_space]
                print(f"⚠️ Truncated {filename} to fit within limit")
            
            all_text += f"\n\n--- {filename} ---\n\n{text}"
            processed_files.append(filename)
            print(f"✓ Processed {filename}: {len(text)} chars")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ TOTAL: {len(all_text)} chars from {len(processed_files)} files")
    print(f"{'='*60}\n")
    return all_text, processed_files


def similarity_score(str1, str2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_related_video(query, threshold=0.3):
    """Find related video based on query"""
    videos_folder = app.config['VIDEOS_FOLDER']
    
    if not os.path.exists(videos_folder):
        return None
    
    query_clean = re.sub(r'[^\w\s]', '', query.lower())
    query_words = set(query_clean.split())
    
    best_match = None
    best_score = 0
    
    for filename in os.listdir(videos_folder):
        if filename.lower().endswith(('.mp4', '.webm', '.ogg', '.mov', '.avi')):
            name_without_ext = os.path.splitext(filename)[0]
            name_clean = re.sub(r'[^\w\s]', ' ', name_without_ext.lower())
            name_words = set(name_clean.split())
            
            common_words = query_words.intersection(name_words)
            word_overlap = len(common_words) / max(len(query_words), 1)
            
            string_sim = similarity_score(query_clean, name_clean)
            
            combined_score = (word_overlap * 0.6) + (string_sim * 0.4)
            
            if combined_score > best_score and combined_score >= threshold:
                best_score = combined_score
                best_match = filename
    
    return best_match

def create_chain(vectorstore):
    """Create conversation chain with retriever"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful HR Assistant. Answer the question based on the following context from HR documents.

Context: {context}

Question: {input}

Provide a clear, detailed, and helpful answer. If the context contains relevant information,
use it to provide specific details. If you're explaining a process or concept, be thorough
but concise. If the information is not in the context, say so politely.

Answer:""")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

@app.route('/')
def index():
    """Render home page"""
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize session and load documents from knowledge base"""
    try:
        session_id = request.json.get('session_id', 'default')
        session = get_session(session_id)
        
        print(f"\n{'='*60}")
        print(f"INITIALIZING SESSION: {session_id}")
        print(f"{'='*60}\n")
        
        # Load documents from the documents folder
        all_text, processed_files = load_documents_from_directory(app.config['DOCUMENTS_FOLDER'])
        
        if all_text and len(all_text.strip()) > 0:
            try:
                print("Creating text chunks...")
                # Split text into chunks
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                texts = text_splitter.split_text(all_text)
                
                # Ensure we have texts to embed
                if not texts:
                    return jsonify({
                        'success': False,
                        'error': 'No text chunks created from documents'
                    }), 500
                
                print(f"✓ Created {len(texts)} chunks")
                print("Creating FAISS vectorstore...")
                
                # Initialize embeddings
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                
                # Create FAISS vectorstore
                vectorstore = FAISS.from_texts(texts, embeddings)
                print("✓ Vectorstore created")
                
                # Create conversation chain
                print("Creating conversation chain...")
                conversation_chain = create_chain(vectorstore)
                print("✓ Conversation chain ready")
                
                # Store in session
                session['vectorstore'] = vectorstore
                session['conversation_chain'] = conversation_chain
                session['preloaded_files'] = processed_files
                
                print(f"\n{'='*60}")
                print(f"✅ SUCCESS: Loaded {len(processed_files)} documents")
                print(f"{'='*60}\n")
                
                return jsonify({
                    'success': True,
                    'files': processed_files,
                    'message': f'Successfully loaded {len(processed_files)} documents from knowledge base'
                })
                
            except Exception as e:
                print(f"❌ Error creating vectorstore: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': f'Error creating knowledge base: {str(e)}'
                }), 500
        else:
            print("⚠️ No documents found or no text extracted")
            # No documents found - return success but with empty knowledge base
            return jsonify({
                'success': True,
                'files': [],
                'message': 'No documents found in knowledge base. You can upload documents to get started.'
            })
    
    except Exception as e:
        print(f"❌ Init session error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error initializing session'
        }), 500

@app.route('/upload', methods=['POST'])
def upload_files():
    """Upload and process user files"""
    session_id = request.form.get('session_id', 'default')
    session = get_session(session_id)
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    all_text = ""
    processed_files = []
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            text = process_file(file, filename)
            if text:
                all_text += f"\n\n--- {filename} ---\n{text}"
                processed_files.append(filename)
                
                file.seek(0)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    if all_text:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(all_text)
        embeddings = OpenAIEmbeddings()
        
        if session['vectorstore']:
            session['vectorstore'].add_texts(texts)
        else:
            vectorstore = FAISS.from_texts(texts, embeddings)
            session['vectorstore'] = vectorstore
        
        session['conversation_chain'] = create_chain(session['vectorstore'])
        session['uploaded_files'].extend(processed_files)
        
        return jsonify({
            'success': True,
            'files': processed_files,
            'message': f'Successfully processed {len(processed_files)} files'
        })
    
    return jsonify({'error': 'No text could be extracted from files'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    session_id = request.json.get('session_id', 'default')
    message = request.json.get('message', '')
    is_voice_input = request.json.get('is_voice_input', False)
    
    session = get_session(session_id)
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Ensure chat_history exists
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    session['chat_history'].append({
        'message': message,
        'is_user': True,
        'is_voice': is_voice_input,
        'timestamp': datetime.now().isoformat()
    })
    
    # Handle goodbye messages
    if message.lower().strip() in ['bye', 'goodbye', 'exit', 'quit', 'end']:
        response = "Thank you for using HR Assistant! Have a great day!"
        session['chat_history'].append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        
        audio_url = None
        if is_voice_input:
            try:
                audio_filename = f"response_{uuid.uuid4().hex}.mp3"
                audio_path = os.path.join('static', 'audio', audio_filename)
                tts = gTTS(text=response, lang='en', slow=False)
                tts.save(audio_path)
                audio_url = f'/static/audio/{audio_filename}'
            except Exception as e:
                print(f"TTS error: {e}")
        
        return jsonify({
            'response': response,
            'session_ended': True,
            'should_speak': is_voice_input,
            'audio_url': audio_url
        })
    
    # Find related video
    related_video = find_related_video(message)
    
    # Process with conversation chain
    if session['conversation_chain']:
        try:
            result = session['conversation_chain'].invoke({'input': message})
            response = result['answer']
            
            session['chat_history'].append({
                'message': response,
                'is_user': False,
                'timestamp': datetime.now().isoformat()
            })
            
            response_data = {
                'response': response,
                'should_speak': is_voice_input
            }
            
            # Generate audio for voice input
            if is_voice_input:
                try:
                    audio_filename = f"response_{uuid.uuid4().hex}.mp3"
                    audio_path = os.path.join('static', 'audio', audio_filename)
                    tts = gTTS(text=response, lang='en', slow=False)
                    tts.save(audio_path)
                    response_data['audio_url'] = f'/static/audio/{audio_filename}'
                except Exception as e:
                    print(f"TTS error: {e}")
            
            if related_video:
                response_data['video'] = f'/static/videos/{related_video}'
                response_data['video_name'] = os.path.splitext(related_video)[0].replace('_', ' ').title()
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Chat error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Please wait for knowledge base to load or upload documents first'}), 400

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    text = request.json.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        audio_filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_path = os.path.join('static', 'audio', audio_filename)
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save(audio_path)
        
        return jsonify({
            'success': True,
            'audio_url': f'/static/audio/{audio_filename}'
        })
    except Exception as e:
        return jsonify({'error': f'TTS error: {str(e)}'}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback"""
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
    """Export chat history as JSON"""
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
    """Export feedback as CSV"""
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
    """Clear session data"""
    session_id = request.json.get('session_id', 'default')
    if session_id in sessions:
        sessions[session_id] = {
            'vectorstore': None,
            'conversation_chain': None,
            'chat_history': [],
            'uploaded_files': [],
            'feedback_history': [],
            'preloaded_files': []
        }
    return jsonify({'success': True})

@app.route('/feedback/stats', methods=['GET'])
def feedback_stats():
    """Get feedback statistics"""
    try:
        session_id = request.args.get('session_id', 'default')
        session = get_session(session_id)
        
        feedback_history = session.get('feedback_history', [])
        
        stats = {
            'total': len(feedback_history),
            'positive': sum(1 for f in feedback_history if f.get('rating', 0) > 3),
            'negative': sum(1 for f in feedback_history if f.get('rating', 0) <= 3),
            'average_rating': sum(f.get('rating', 0) for f in feedback_history) / len(feedback_history) if feedback_history else 0
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        print(f"Feedback stats error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'stats': {'total': 0, 'positive': 0, 'negative': 0, 'average_rating': 0}
        })

@app.route('/get_loaded_files', methods=['GET'])
def get_loaded_files():
    """Get list of loaded files"""
    session_id = request.args.get('session_id', 'default')
    session = get_session(session_id)
    
    return jsonify({
        'preloaded': session.get('preloaded_files', []),
        'uploaded': session.get('uploaded_files', [])
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
