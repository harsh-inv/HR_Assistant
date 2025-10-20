import os
import warnings
import logging
import json
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
import pickle
import html as html_module
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import re

# Video processing imports
try:
    import moviepy.editor as mp
    import speech_recognition as sr
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    print("Video processing not available. Install: pip install moviepy SpeechRecognition")
    VIDEO_PROCESSING_AVAILABLE = False

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

# PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    import html as html_module
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    print("ReportLab not available for PDF export")
    PDF_EXPORT_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Persistent disk configuration
PERSISTENT_DISK = os.environ.get('PERSISTENT_DISK_PATH', '/opt/render/project/src')
DOCUMENTS_FOLDER = os.path.join(PERSISTENT_DISK, 'documents')  # Preloaded documents
app.config['UPLOAD_FOLDER'] = os.path.join(PERSISTENT_DISK, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate that the key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please configure it in Render dashboard.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'}

# Session storage (in production, use Redis or database)
sessions = {}

# ============================================================================
# VIDEO INDEX CLASS
# ============================================================================

class VideoIndex:
    """Handles video transcription and semantic search"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.video_metadata = {}
        self.vector_store = None
        
    def transcribe_video(self, video_path):
        """Extract audio and transcribe to text"""
        if not VIDEO_PROCESSING_AVAILABLE:
            return "Video transcription not available - missing dependencies"
            
        try:
            video = mp.VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_data)
            
            os.remove(audio_path)
            video.close()
            return transcript
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
        
    def add_video(self, video_path, custom_description=None):
        """Index a video with its transcript and metadata"""
        filename = os.path.basename(video_path)
        transcript = self.transcribe_video(video_path)
        
        if custom_description:
            content = f"{transcript} {custom_description}"
        else:
            content = transcript
        
        self.video_metadata[filename] = {
            'path': video_path,
            'transcript': transcript,
            'description': custom_description or '',
            'upload_date': datetime.now().isoformat()
        }
        
        if content.strip():
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(content)
            
            metadatas = [{'filename': filename, 'chunk_id': i} 
                         for i in range(len(chunks))]
            
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    chunks, 
                    self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(chunks, metadatas=metadatas)
            
            self.save_index()
        
    def search_relevant_videos(self, query, k=3):
        """Find most relevant videos for a query"""
        if self.vector_store is None:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            video_matches = {}
            for doc, score in results:
                filename = doc.metadata['filename']
                if filename not in video_matches:
                    video_matches[filename] = {
                        'metadata': self.video_metadata[filename],
                        'relevance_score': score
                    }
            
            return list(video_matches.values())
        except Exception as e:
            print(f"Video search error: {e}")
            return []
    
    def save_index(self):
        """Persist vector store and metadata"""
        try:
            if self.vector_store:
                index_path = os.path.join(PERSISTENT_DISK, "video_index")
                self.vector_store.save_local(index_path)
            
            metadata_path = os.path.join(PERSISTENT_DISK, 'video_metadata.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.video_metadata, f)
        except Exception as e:
            print(f"Error saving video index: {e}")
    
    def load_index(self):
        """Load existing index"""
        try:
            index_path = os.path.join(PERSISTENT_DISK, "video_index")
            self.vector_store = FAISS.load_local(
                index_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            metadata_path = os.path.join(PERSISTENT_DISK, 'video_metadata.pkl')
            with open(metadata_path, 'rb') as f:
                self.video_metadata = pickle.load(f)
        except:
            pass

# Initialize global video index
video_index = VideoIndex()
video_index.load_index()

# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def get_session(session_id):
    """Get or create a session"""
    if session_id not in sessions:
        sessions[session_id] = {
            'vectorstore': None,
            'conversation_chain': None,
            'chat_history': [],
            'uploaded_files': [],
            'uploaded_images': [],
            'feedback_history': [],
            'consecutive_no_count': 0,
            'last_analysis': None,
            'awaiting_followup': False,
            'last_interaction': time.time(),
            'upload_completed_time': None,
            'feedback_submitted': False
        }
    return sessions[session_id]

def is_greeting(message):
    """Detect if message is a greeting - handles punctuation like 'Hello.'"""
    greetings = [
        'hello', 'hi', 'hii', 'hey', 'howdy',
        'good morning', 'good afternoon', 'good evening', 
        'greetings', 'hai', 'hey there', 'hiya'
    ]
    
    # Remove trailing punctuation and convert to lowercase
    lower_message = re.sub(r'[.,!?;:]+$', '', message.lower().strip())
    
    return any(
        lower_message == greeting or lower_message.startswith(greeting + ' ')
        for greeting in greetings
    )


def is_goodbye(message):
    """Detect if message is a goodbye"""
    goodbyes = ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'exit', 'quit']
    lower_message = message.lower().strip()
    return any(goodbye in lower_message for goodbye in goodbyes)


def is_acknowledgment(message):
    """Detect if message is just an acknowledgment - handles punctuation like 'okay.' 'nice.'"""
    acknowledgments = [
        'ok', 'okay', 'okey', 'oke', 'k',
        'nice', 'good', 'great', 'excellent', 'awesome', 'perfect', 'cool', 'fine',
        'thanks', 'thank you', 'thankyou', 'thx', 'ty',
        'alright', 'got it', 'understood', 'i see', 'i understand',
        'yes', 'yeah', 'yep', 'yup', 'sure', 'of course'
    ]
    
    negative_responses = [
        'no', 'nope', 'nah', 'not needed', 'no need', 'no thanks',
        'not really', 'im good', "i'm good", 'all good', 'thats all',
        "that's all", 'nothing else', 'nothing more'
    ]
    acknowledgments.extend(negative_responses)
    
    # Remove ALL punctuation for comparison
    normalized = re.sub(r'[.,!?;:\'"]+', '', message.lower().strip())
    
    question_words = [
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could',
        'would', 'should', 'is', 'are', 'does', 'do', 'tell', 'show', 'explain', 'describe'
    ]
    
    is_ack = (
        normalized in acknowledgments or
        (len(normalized.split()) <= 3 and any(ack in normalized for ack in acknowledgments))
    ) and not any(qw in normalized for qw in question_words)
    
    return is_ack

def detect_query_type(message):
    """Categorize query to provide contextual help"""
    msg_lower = message.lower()
    
    if any(word in msg_lower for word in ['leave', 'pto', 'vacation', 'sick', 'time off', 'absence']):
        return 'leave_policy'
    elif any(word in msg_lower for word in ['benefit', 'insurance', 'health', '401k', 'dental', 'medical']):
        return 'benefits'
    elif any(word in msg_lower for word in ['payroll', 'salary', 'pay', 'direct deposit', 'compensation']):
        return 'payroll'
    elif any(word in msg_lower for word in ['onboarding', 'new hire', 'orientation', 'first day']):
        return 'onboarding'
    elif any(word in msg_lower for word in ['performance', 'review', 'evaluation', 'appraisal']):
        return 'performance'
    elif any(word in msg_lower for word in ['remote', 'work from home', 'wfh', 'hybrid']):
        return 'remote_work'
    elif any(word in msg_lower for word in ['training', 'course', 'learning', 'development']):
        return 'training'
    elif any(word in msg_lower for word in ['grievance', 'complaint', 'issue', 'concern']):
        return 'grievance'
    elif any(word in msg_lower for word in ['policy', 'handbook', 'guidelines', 'rules']):
        return 'policy'
    else:
        return 'general'

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def extract_pdf_text(pdf_file):
    """Extract text from PDF using multiple methods"""
    text = ""
    
    # Try PyMuPDF first (most reliable)
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
    
    # Fallback to pdfplumber
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

def process_file(uploaded_file, filename):
    """Process uploaded file based on type"""
    file_ext = filename.lower().split('.')[-1]
    
    # Handle video files
    if file_ext in ALLOWED_VIDEO_EXTENSIONS:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        uploaded_file.save(filepath)
        return f"Video file saved: {filename}"
    
    # Handle image files
    if file_ext in ALLOWED_IMAGE_EXTENSIONS:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        uploaded_file.save(filepath)
        return f"Image file saved: {filename}"
    
    # Handle document files
    if file_ext == 'pdf':
        return extract_pdf_text(uploaded_file)
    elif file_ext in ['docx', 'doc']:
        return extract_docx_text(uploaded_file)
    elif file_ext == 'txt':
        return extract_txt_text(uploaded_file)
    
    return ""

def preload_documents():
    """Load all documents from the documents folder on startup"""
    all_text = ""
    loaded_files = []
    
    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Documents folder not found: {DOCUMENTS_FOLDER}")
        return None, []
    
    for filename in os.listdir(DOCUMENTS_FOLDER):
        file_ext = filename.lower().split('.')[-1]
        if file_ext in ALLOWED_EXTENSIONS:
            filepath = os.path.join(DOCUMENTS_FOLDER, filename)
            try:
                with open(filepath, 'rb') as f:
                    text = process_file(f, filename)
                    if text and not text.startswith("Error") and not text.startswith("Video") and not text.startswith("Image"):
                        all_text += f"\n\n--- {filename} ---\n{text}"
                        loaded_files.append(filename)
                        print(f"✓ Loaded document: {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
    
    if all_text:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        texts = text_splitter.split_text(all_text)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings)
        print(f"✓ Successfully preloaded {len(loaded_files)} documents")
        return vectorstore, loaded_files
    
    print("✗ No documents found to preload")
    return None, []

def create_chain(vectorstore):
    """Create RAG chain with custom prompt"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    prompt = ChatPromptTemplate.from_template("""You are an HR Assistant for Invenio Business Solutions. Answer questions ONLY using the provided context from HR policy documents.

Context: {context}

Question: {input}

STRICT ACCURACY RULES:
1. Answer ONLY based on the information in the Context above
2. If the context doesn't contain the answer, say: "I don't have that specific information in the HR policy documents. Please contact HR directly at [hr@company.com] or ask a different question."
3. Do NOT make assumptions or provide general knowledge
4. Do NOT add extra information not present in the context
5. Quote specific policy details, form numbers, and document names when available
6. Be concise, direct, and professional

CONVERSATION HANDLING:
- For acknowledgments like "ok", "thanks", "got it", "yes", "no": Respond with ONE short sentence only
- For follow-up questions: Answer briefly based on previous context (2-3 sentences)
- For new questions: Provide full structured response

RESPONSE STRUCTURE (for procedural questions like "How do I...?"):

**Overview**: [1-2 sentence summary]

**Step-by-Step Process**:
- Step 1: [Action from context]
- Step 2: [Action from context]
- Step 3: [Action from context]
(Continue as needed)

**Required Forms & Documents**: 
- [List any forms the employee must complete/submit - include form numbers and names from context]
- [Where to obtain them if mentioned]

**Reference Documents**: 
- [List policy documents, handbook sections, or guides mentioned in context for additional reading]

**Timeline**: [If mentioned in context]

**Contact Information**: [If mentioned in context]

**Important Notes**: [Any critical requirements, deadlines, or warnings from context]

RESPONSE STRUCTURE (for policy questions like "What is...?", "Can I...?"):

**Answer**: [Direct clear answer]

**Policy Details**: [Specific rules, eligibility, conditions from context]

**Required Forms & Documents**: [If any are needed]

**Reference Documents**: [Policy docs or handbook sections mentioned]

**Exceptions**: [If any special cases mentioned]

**Next Steps**: [What employee should do]

CRITICAL INSTRUCTIONS:
- Extract EXACT names: form numbers (e.g., "HR-GR-001"), section numbers (e.g., "Section 8.3"), document titles
- Separate "Required Forms" (what they fill out) from "Reference Documents" (what they read)
- Include specific details: numbers, dates, names, emails, departments from context
- If info is missing, be specific about what's missing and suggest contacting HR

Answer:""")
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# Preload documents on startup
print("=" * 60)
print("PRELOADING DOCUMENTS FROM:", DOCUMENTS_FOLDER)
print("=" * 60)
preloaded_vectorstore, preloaded_files = preload_documents()
print("=" * 60)

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize session with preloaded documents"""
    session_id = request.json.get('session_id', 'default')
    session = get_session(session_id)
    
    # Initialize with preloaded documents if available
    if preloaded_vectorstore and not session['vectorstore']:
        session['vectorstore'] = preloaded_vectorstore
        session['conversation_chain'] = create_chain(preloaded_vectorstore)
        session['uploaded_files'] = preloaded_files.copy()
        print(f"✓ Session {session_id} initialized with {len(preloaded_files)} preloaded documents")
    
    # Calculate total counts
    total_documents = len(session['uploaded_files'])
    total_images = len(session.get('uploaded_images', []))
    
    return jsonify({
        'success': True,
        'preloaded_files': preloaded_files,
        'message': f'{len(preloaded_files)} documents ready' if preloaded_files else 'No preloaded documents',
        'feedback_submitted': session['feedback_submitted'],
        'total_documents': total_documents,
        'total_images': total_images,
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads (documents, videos, images)"""
    session_id = request.form.get('session_id', 'default')
    session = get_session(session_id)
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    all_text = ""
    processed_files = []
    processed_images = []
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            file_ext = filename.lower().split('.')[-1]
            
            # Handle image files separately
            if file_ext in ALLOWED_IMAGE_EXTENSIONS:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read and encode as base64 for preview
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')
                
                processed_images.append({
                    'filename': filename,
                    'base64': base64_data,
                    'mimetype': f'image/{file_ext}'
                })
                session['uploaded_images'].append({
                    'filename': filename,
                    'path': filepath
                })
                continue
            
            # Handle video files
            if file_ext in ALLOWED_VIDEO_EXTENSIONS:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                processed_files.append(filename)
                continue
            
            # Handle document files
            text = process_file(file, filename)
            if text and not text.startswith("Video file saved") and not text.startswith("Image file saved") and not text.startswith("Error"):
                all_text += f"\n\n--- {filename} ---\n{text}"
                processed_files.append(filename)
    
    # Process text documents into vector store
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
            # Merge with existing vectorstore
            new_vectorstore = FAISS.from_texts(texts, embeddings)
            session['vectorstore'].merge_from(new_vectorstore)
        else:
            session['vectorstore'] = FAISS.from_texts(texts, embeddings)
        
        session['conversation_chain'] = create_chain(session['vectorstore'])
        session['uploaded_files'].extend(processed_files)
    
    # Update upload completion time
    upload_time = time.time()
    session['last_interaction'] = upload_time
    session['upload_completed_time'] = upload_time
    
    # Calculate total document count
    total_documents = len(session['uploaded_files'])
    total_images = len(session['uploaded_images'])
    
    return jsonify({
        'success': True,
        'files': processed_files,
        'images': processed_images,
        'message': f'Successfully processed {len(processed_files) + len(processed_images)} file(s)',
        'upload_completed_time': upload_time,
        'total_documents': total_documents,
        'total_images': total_images,
        'document_count_message': f'{total_documents} document{"s" if total_documents != 1 else ""} loaded. You can upload more documents to expand the collection.'
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video uploads with optional descriptions"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    description = request.form.get('description', '')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        video_index.add_video(filepath, description)
        return jsonify({
            'success': True,
            'message': f'Video uploaded and indexed: {filename}'
        })
    except Exception as e:
        return jsonify({'error': f'Indexing failed: {str(e)}'}), 500

@app.route('/video/<filename>')
def serve_video(filename):
    """Serve stored videos"""
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': f'Video not found: {str(e)}'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages with RAG and video search"""
    session_id = request.json.get('session_id', 'default')
    message = request.json.get('message', '')
    is_voice_input = request.json.get('is_voice_input', False)
    session = get_session(session_id)
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Auto-initialize session with preloaded documents if not already done
    if preloaded_vectorstore and not session['vectorstore']:
        session['vectorstore'] = preloaded_vectorstore
        session['conversation_chain'] = create_chain(preloaded_vectorstore)
        session['uploaded_files'] = preloaded_files.copy()
        print(f"✓ Auto-initialized session {session_id} with {len(preloaded_files)} preloaded documents")
    
    # Update last interaction time
    session['last_interaction'] = time.time()
    
    # Add user message to history
    session['chat_history'].append({
        'message': message,
        'is_user': True,
        'timestamp': datetime.now().isoformat()
    })
    
    # Normalize message (remove punctuation for comparison)
    import re
    normalized_message = message.lower().strip()
    normalized_message = re.sub(r'[.,!?;:\'"]+', '', message.lower().strip())
    
    # ========================================================================
    # PRIORITY 1: GREETING DETECTION (Must be FIRST!)
    # ========================================================================
    if is_greeting(message):
        greeting_response = "Hello! I'm your HR Assistant. How can I help you today? You can ask about policies, benefits, leave procedures, or any HR-related questions."
        session['chat_history'].append({
            'message': greeting_response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        return jsonify({
            'response': greeting_response,
            'is_voice_input': is_voice_input,
            'relevant_videos': [],
            'query_type': 'greeting'
        })
    
    # ========================================================================
    # PRIORITY 2: GOODBYE DETECTION
    # ========================================================================
    if is_goodbye(message):
        response = "Thank you for using HR Assistant! Have a great day! 👋"
        session['chat_history'].append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        return jsonify({
            'response': response,
            'is_voice_input': is_voice_input,
            'session_ended': True,
            'trigger_feedback': True,
            'relevant_videos': []
        })
    
    # ========================================================================
    # PRIORITY 3: ACKNOWLEDGMENT DETECTION (Saves API costs!)
    # ========================================================================
    if is_acknowledgment(message) and session.get('last_analysis'):
        if 'no' in normalized_message or 'not' in normalized_message:
            brief_response = "Understood. Let me know if you need anything else."
        elif 'yes' in normalized_message:
            brief_response = "What specific aspect would you like me to elaborate on?"
        else:
            brief_response = "You're welcome! Feel free to ask if you need anything else."
        
        session['chat_history'].append({
            'message': brief_response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        return jsonify({
            'response': brief_response,
            'is_voice_input': is_voice_input,
            'relevant_videos': [],
            'is_acknowledgment': True
        })
    
    # ========================================================================
    # PRIORITY 4: NEGATIVE RESPONSE TRACKING
    # ========================================================================
    negative_responses = [
        'no', 'nope', 'nah', 'not needed', 'no need', 'no thanks',
        'not really', 'im good', "i'm good", 'all good', 'thats all',
        "that's all", 'nothing else', 'nothing more'
    ]
    is_negative = any(neg in normalized_message for neg in negative_responses)
    
    # Track consecutive "no" responses
    if is_negative:
        session['consecutive_no_count'] = session.get('consecutive_no_count', 0) + 1
    else:
        session['consecutive_no_count'] = 0
    
    # If user has said "no" 2 or more times consecutively, end the session
    if session['consecutive_no_count'] >= 2:
        bot_response = "Thank you for using HR Assistant! Have a great day!"
        session['chat_history'].append({
            'message': bot_response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        session['consecutive_no_count'] = 0
        return jsonify({
            'response': bot_response,
            'is_voice_input': is_voice_input,
            'session_ended': True,
            'trigger_feedback': True,
            'relevant_videos': []
        })
    
    # ========================================================================
    # PRIORITY 5: QUERY TYPE DETECTION
    # ========================================================================
    query_type = detect_query_type(message)
    
    # ========================================================================
    # PRIORITY 6: VIDEO SEARCH
    # ========================================================================
    relevant_videos = video_index.search_relevant_videos(message, k=2)
    
    # ========================================================================
    # PRIORITY 7: RAG RESPONSE GENERATION (LAST!)
    # ========================================================================
    if session['conversation_chain']:
        try:
            result = session['conversation_chain'].invoke({'input': message})
            response = result['answer']
            
            # Store as last analysis
            session['last_analysis'] = response
            session['awaiting_followup'] = True
            
            session['chat_history'].append({
                'message': response,
                'is_user': False,
                'timestamp': datetime.now().isoformat()
            })
            
            # Return text answer and relevant videos
            return jsonify({
                'response': response,
                'is_voice_input': is_voice_input,
                'query_type': query_type,
                'relevant_videos': [{
                    'filename': v['metadata']['path'].split('/')[-1],
                    'url': f"/video/{v['metadata']['path'].split('/')[-1]}",
                    'transcript_excerpt': v['metadata']['transcript'][:200],
                    'relevance': float(v['relevance_score']) if isinstance(v['relevance_score'], (int, float)) else 0.0,
                    'description': v['metadata']['description']
                } for v in relevant_videos]
            })
        except Exception as e:
            print(f"Chat error: {e}")
            return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    else:
        # If no documents, still search for videos
        if relevant_videos:
            return jsonify({
                'response': "I found some relevant videos for your query:",
                'is_voice_input': is_voice_input,
                'relevant_videos': [{
                    'filename': v['metadata']['path'].split('/')[-1],
                    'url': f"/video/{v['metadata']['path'].split('/')[-1]}",
                    'transcript_excerpt': v['metadata']['transcript'][:200],
                    'relevance': float(v['relevance_score']) if isinstance(v['relevance_score'], (int, float)) else 0.0,
                    'description': v['metadata']['description']
                } for v in relevant_videos]
            })
        else:
            return jsonify({
                'error': 'Please upload documents or videos first',
                'response': 'I don\'t have any documents or videos to reference. Please upload some documents or ask me to help you get started.'
            }), 400
            
@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions or not sessions[session_id]['messages']:
        return jsonify({'error': 'No chat history found'}), 404
    
    try:
        # Create PDF filename
        pdf_filename = f'chat_export_{session_id}_{int(time.time())}.pdf'
        pdf_path = os.path.join(STATIC_FOLDER, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#2c3e50',
            spaceAfter=30
        )
        
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2980b9',
            leftIndent=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        bot_style = ParagraphStyle(
            'BotMessage',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#34495e',
            leftIndent=20,
            spaceAfter=15
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Image/Audio Assistant - Chat Export", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Add messages
        for msg in sessions[session_id]['messages']:
            if msg['role'] == 'user':
                story.append(Paragraph(f"<b>You:</b> {html.escape(msg['content'])}", user_style))
            else:
                # Clean bot response for PDF
                content = msg['content'].replace('**', '')
                story.append(Paragraph(f"<b>Assistant:</b> {html.escape(content)}", bot_style))
        
        # Build PDF
        doc.build(story)
        
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_data,
            'filename': pdf_filename
        })
    
    except Exception as e:
        print(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/clear', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        # Delete uploaded files
        for file_info in sessions[session_id]['files']:
            filename = file_info['filename'] if isinstance(file_info, dict) else file_info
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")
        
        # Clear session data but keep ticket counter
        sessions[session_id]['messages'] = []
        sessions[session_id]['files'] = []
        sessions[session_id]['ticket_created'] = False
        sessions[session_id]['ticket_button_clicked'] = False
        sessions[session_id]['last_interaction'] = time.time()
        sessions[session_id]['last_analysis'] = None
        sessions[session_id]['awaiting_followup'] = False
        sessions[session_id]['consecutive_no_count'] = 0
        
        return jsonify({'success': True})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    session_id = data.get('session_id')
    rating = data.get('rating')
    comment = data.get('comment', '')
    
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False
        }
    
    feedback_entry = {
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    sessions[session_id]['feedback'].append(feedback_entry)
    sessions[session_id]['feedback_submitted'] = True
    sessions[session_id]['last_interaction'] = time.time()
    
    return jsonify({
        'success': True,
        'feedback_submitted': True
    })

@app.route('/check_idle', methods=['POST'])
def check_idle():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        current_time = time.time()
        last_interaction = sessions[session_id]['last_interaction']
        upload_completed_time = sessions[session_id].get('upload_completed_time')
        
        # Calculate idle time
        idle_time = current_time - last_interaction
        
        # If file was recently uploaded, use 10 second threshold
        # Otherwise use 7 second threshold
        if upload_completed_time and (current_time - upload_completed_time) < 15:
            # Within 15 seconds of upload, use 10 second threshold
            idle_threshold = 10
        else:
            # Normal idle threshold
            idle_threshold = 7
            # Clear the upload_completed_time after threshold period
            if upload_completed_time:
                sessions[session_id]['upload_completed_time'] = None
        
        if idle_time >= idle_threshold:
            return jsonify({
                'is_idle': True,
                'idle_time': idle_time
            })
        else:
            return jsonify({
                'is_idle': False,
                'idle_time': idle_time
            })
    else:
        return jsonify({
            'is_idle': False,
            'idle_time': 0
        })

@app.route('/export/feedback', methods=['POST'])
def export_feedback():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions and sessions[session_id]['feedback']:
        csv_data = "Timestamp,Rating,Comment\n"
        for fb in sessions[session_id]['feedback']:
            csv_data += f"{fb['timestamp']},{fb['rating']},\"{fb['comment']}\"\n"
        
        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'feedback_{session_id}.csv'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No feedback data available'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
