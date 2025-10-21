import os
import base64
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import html

# LangChain and OpenAI imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import docx

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configure folders
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# Render disk path for pre-loaded documents and videos
RENDER_DISK_PATH = '/opt/render/project/src/documents'
PRELOAD_FOLDER = os.path.join(RENDER_DISK_PATH, 'preload_documents') if os.path.exists(RENDER_DISK_PATH) else 'preload_documents'
PRELOAD_VIDEOS_FOLDER = os.path.join(RENDER_DISK_PATH, 'preload_videos') if os.path.exists(RENDER_DISK_PATH) else 'preload_videos'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(PRELOAD_FOLDER, exist_ok=True)
os.makedirs(PRELOAD_VIDEOS_FOLDER, exist_ok=True)

# OpenAI API Key - CRITICAL: Set this in your environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY or OPENAI_API_KEY == 'your-api-key-here':
    print("="*70)
    print("WARNING: OPENAI_API_KEY is not set or is using placeholder value!")
    print("Please set your OpenAI API key in one of these ways:")
    print("1. Create a .env file with: OPENAI_API_KEY=sk-your-actual-key")
    print("2. Set environment variable: export OPENAI_API_KEY=sk-your-actual-key")
    print("3. On Render: Add OPENAI_API_KEY in Environment Variables section")
    print("="*70)
else:
    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
    print(f"✓ OpenAI API Key loaded: {OPENAI_API_KEY[:8]}...")

# Session storage
sessions = {}

# Enhanced prompt template for HR Assistant
HR_ASSISTANT_TEMPLATE = """You are an HR Assistant for Invenio Business Solutions. Answer questions ONLY using the provided context from HR policy documents.

Context: {context}

Chat History: {chat_history}

Question: {question}

STRICT ACCURACY RULES:
1. Answer ONLY based on the information in the Context above
2. If the context doesn't contain the answer, say: "I don't have that specific information in the HR policy documents. Please contact HR directly or ask a different question."
3. Do NOT make assumptions or provide general knowledge
4. Do NOT add extra information not present in the context
5. Quote specific policy details, form numbers, and document names when available
6. Be concise, direct, and professional

CONVERSATION HANDLING:
- For greetings (hello, hi, hey): Respond warmly with a brief welcome
- For goodbyes (bye, goodbye): Respond courteously and professionally
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

Your Response:"""

# Video metadata storage
video_metadata = {}

def load_video_metadata():
    """Load metadata about videos in the preload_videos folder"""
    global video_metadata
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    if not os.path.exists(PRELOAD_VIDEOS_FOLDER):
        print(f"Videos folder not found: {PRELOAD_VIDEOS_FOLDER}")
        return
    
    for filename in os.listdir(PRELOAD_VIDEOS_FOLDER):
        file_path = os.path.join(PRELOAD_VIDEOS_FOLDER, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in video_extensions):
            video_name = os.path.splitext(filename)[0]
            video_metadata[filename] = {
                'name': video_name,
                'path': file_path,
                'keywords': video_name.lower().split('_')
            }
    
    print(f"Loaded {len(video_metadata)} videos from {PRELOAD_VIDEOS_FOLDER}")

def get_relevant_videos(query):
    """Find relevant videos based on query keywords"""
    query_lower = query.lower()
    relevant_videos = []
    
    for filename, metadata in video_metadata.items():
        if any(keyword in query_lower for keyword in metadata['keywords']):
            relevant_videos.append({
                'filename': filename,
                'name': metadata['name']
            })
    
    return relevant_videos

def read_docx(file_path):
    """Extract text from DOCX files"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
        return ""

def read_txt(file_path):
    """Extract text from TXT files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading TXT {file_path}: {e}")
        return ""

def get_preloaded_documents():
    """Load all documents from preload folder"""
    text = ""
    supported_extensions = ['.pdf', '.docx', '.txt']
    
    if not os.path.exists(PRELOAD_FOLDER):
        print(f"Preload folder not found: {PRELOAD_FOLDER}")
        return text
    
    for filename in os.listdir(PRELOAD_FOLDER):
        file_path = os.path.join(PRELOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.pdf':
                try:
                    pdf_reader = PdfReader(file_path)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n\n--- From {filename} ---\n\n"
                            text += page_text + "\n"
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
            
            elif ext == '.docx':
                doc_text = read_docx(file_path)
                if doc_text:
                    text += f"\n\n--- From {filename} ---\n\n"
                    text += doc_text + "\n"
            
            elif ext == '.txt':
                txt_text = read_txt(file_path)
                if txt_text:
                    text += f"\n\n--- From {filename} ---\n\n"
                    text += txt_text + "\n"
    
    print(f"Loaded preloaded documents from {PRELOAD_FOLDER}")
    return text

def get_document_text(file_paths):
    """Extract text from various document types"""
    text = ""
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            try:
                pdf_reader = PdfReader(file_path)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception as e:
                print(f"Error reading PDF {file_path}: {e}")
        
        elif ext == '.docx':
            text += read_docx(file_path) + "\n"
        
        elif ext == '.txt':
            text += read_txt(file_path) + "\n"
    
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Create FAISS vectorstore from text chunks"""
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Create conversational retrieval chain with enhanced prompt"""
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:"
    )
    
    qa_prompt = PromptTemplate(
        template=HR_ASSISTANT_TEMPLATE,
        input_variables=["context", "chat_history", "question"]
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=False,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    return conversation_chain

# ==================== FLASK ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': True,
            'upload_completed_time': None,
            'preloaded_files': []
        }
        
        preloaded_text = get_preloaded_documents()
        if preloaded_text.strip():
            try:
                text_chunks = get_text_chunks(preloaded_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation_chain = get_conversation_chain(vectorstore)
                
                sessions[session_id]['vectorstore'] = vectorstore
                sessions[session_id]['conversation_chain'] = conversation_chain
                sessions[session_id]['chat_active'] = True
                
                if os.path.exists(PRELOAD_FOLDER):
                    preloaded_files = [f for f in os.listdir(PRELOAD_FOLDER) 
                                     if os.path.isfile(os.path.join(PRELOAD_FOLDER, f))]
                    sessions[session_id]['preloaded_files'] = preloaded_files
                
                print(f"Session {session_id} initialized with preloaded documents")
            except Exception as e:
                print(f"Error loading preloaded documents: {e}")
    
    load_video_metadata()
    
    return jsonify({
        'success': True,
        'pdf_files': [f['filename'] for f in sessions[session_id]['pdf_files']],
        'preloaded_files': sessions[session_id].get('preloaded_files', []),
        'feedback_submitted': sessions[session_id]['feedback_submitted'],
        'chat_active': sessions[session_id]['chat_active']
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': True,
            'upload_completed_time': None,
            'preloaded_files': []
        }
    
    for file_info in sessions[session_id]['pdf_files']:
        filepath = os.path.join(UPLOAD_FOLDER, file_info['filename'])
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file: {e}")
    
    sessions[session_id]['pdf_files'] = []
    
    uploaded_files = []
    files = request.files.getlist('files')
    saved_paths = []
    
    for file in files:
        if file and (file.filename.lower().endswith('.pdf') or 
                    file.filename.lower().endswith('.docx') or 
                    file.filename.lower().endswith('.txt')):
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            saved_paths.append(filepath)
            
            sessions[session_id]['pdf_files'].append({
                'filename': filename,
                'original_name': file.filename
            })
            
            uploaded_files.append({
                'filename': filename,
                'original_name': file.filename
            })
    
    if saved_paths or sessions[session_id].get('preloaded_files'):
        try:
            all_text = get_preloaded_documents()
            uploaded_text = get_document_text(saved_paths)
            
            if uploaded_text.strip():
                all_text += "\n\n--- User Uploaded Documents ---\n\n" + uploaded_text
            
            if all_text.strip():
                text_chunks = get_text_chunks(all_text)
                vectorstore = get_vectorstore(text_chunks)
                conversation_chain = get_conversation_chain(vectorstore)
                
                sessions[session_id]['vectorstore'] = vectorstore
                sessions[session_id]['conversation_chain'] = conversation_chain
                sessions[session_id]['chat_active'] = True
                
                welcome_msg = 'Documents processed successfully! I have access to company documents and your uploaded files. How can I assist you today?'
                sessions[session_id]['messages'].append({
                    'role': 'assistant',
                    'content': welcome_msg,
                    'timestamp': datetime.now().isoformat(),
                    'exclude_from_export': False
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No text could be extracted from the documents'
                })
        except Exception as e:
            print(f"Document processing error: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to process documents: {str(e)}'
            })
    
    upload_time = time.time()
    sessions[session_id]['last_interaction'] = upload_time
    sessions[session_id]['upload_completed_time'] = upload_time
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'chat_active': sessions[session_id]['chat_active'],
        'upload_completed_time': upload_time,
        'welcome_message': sessions[session_id]['messages'][-1]['content'] if sessions[session_id]['messages'] else None
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    is_voice_input = data.get('is_voice_input', False)
    
    if session_id not in sessions:
        return jsonify({
            'error': 'Session not found',
            'response': 'Please wait for the system to initialize.'
        })
    
    if not sessions[session_id]['chat_active']:
        return jsonify({
            'error': 'Chat not active',
            'response': 'Please wait for documents to be processed.'
        })
    
    sessions[session_id]['last_interaction'] = time.time()
    
    try:
        sessions[session_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat(),
            'exclude_from_export': False
        })
        
        normalized_message = message.lower().strip().replace("'", "").replace(",", "").replace(".", "")
        
        negative_responses = ['no', 'nope', 'nah', 'not needed', 'no need', 'no thanks', 
                             'not really', 'im good', "i'm good", 'all good', 'thats all', 
                             "that's all", 'nothing else', 'nothing more']
        
        is_negative = any(neg in normalized_message for neg in negative_responses)
        
        if is_negative:
            sessions[session_id]['consecutive_no_count'] = sessions[session_id].get('consecutive_no_count', 0) + 1
        else:
            sessions[session_id]['consecutive_no_count'] = 0
        
        if sessions[session_id]['consecutive_no_count'] >= 2:
            bot_response = "Thank you for using the HR Assistant! Have a great day!"
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat(),
                'exclude_from_export': False
            })
            
            sessions[session_id]['consecutive_no_count'] = 0
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': True,
                'trigger_feedback': True
            })
        
        greetings = ['hello', 'hi', 'hii', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(greet in normalized_message for greet in greetings) and len(normalized_message.split()) <= 3:
            bot_response = 'Hello! I\'m your HR Assistant. I have access to company documents and can help answer your questions. How may I assist you today?'
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat(),
                'exclude_from_export': False
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': False
            })
        
        goodbyes = ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'exit', 'quit']
        if any(goodbye in normalized_message for goodbye in goodbyes):
            bot_response = 'Thank you for using the HR Assistant! If you have any more questions in the future, feel free to reach out. Have a wonderful day!'
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat(),
                'exclude_from_export': False
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': True,
                'trigger_feedback': True
            })
        
        acknowledgments = [
            'ok', 'okay', 'okey', 'oke', 'k',
            'nice', 'good', 'great', 'excellent', 'awesome', 'perfect', 'cool', 'fine',
            'thanks', 'thank you', 'thankyou', 'thx', 'ty',
            'alright', 'got it', 'understood', 'i see', 'i understand',
            'yes', 'yeah', 'yep', 'yup', 'sure', 'of course'
        ]
        
        acknowledgments.extend(negative_responses)
        
        is_acknowledgment = (
            normalized_message in acknowledgments or
            (len(normalized_message.split()) <= 3 and any(ack in normalized_message for ack in acknowledgments))
        ) and not any(question_word in normalized_message for question_word in [
            'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 
            'would', 'should', 'is', 'are', 'does', 'do', 'explain', 'tell', 'show', 'describe'
        ])
        
        if is_acknowledgment:
            if is_negative:
                bot_response = "Understood. I'm here if you need anything else."
            else:
                bot_response = "You're welcome! Feel free to ask if you have any other questions."
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat(),
                'exclude_from_export': False
            })
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'session_ended': False
            })
        
        relevant_videos = get_relevant_videos(message)
        video_context = ""
        if relevant_videos:
            video_context = "\n\n**Related Video Resources:**\n"
            for video in relevant_videos:
                video_context += f"- {video['name']} (Video available in system)\n"
        
        conversation_chain = sessions[session_id]['conversation_chain']
        
        if conversation_chain:
            response = conversation_chain({'question': message})
            bot_response = response['answer']
            
            if video_context:
                bot_response += video_context
            
            sessions[session_id]['last_analysis'] = bot_response
            sessions[session_id]['awaiting_followup'] = True
        else:
            bot_response = "I apologize, but the system is not fully initialized. Please try again or contact support."
        
        sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': datetime.now().isoformat(),
            'exclude_from_export': False
        })
        
        return jsonify({
            'success': True,
            'response': bot_response,
            'is_voice_input': is_voice_input,
            'feedback_submitted': sessions[session_id]['feedback_submitted'],
            'session_ended': False,
            'videos': relevant_videos if relevant_videos else None
        })
    
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': str(e),
            'response': 'An error occurred while processing your request. Please try again.'
        })

@app.route('/export/json', methods=['POST'])
def export_json():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        return jsonify({
            'session_id': session_id,
            'messages': sessions[session_id]['messages'],
            'pdf_files': [f['filename'] for f in sessions[session_id]['pdf_files']]
        })
    else:
        return jsonify({'error': 'Session not found'})

@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions or not sessions[session_id]['messages']:
        return jsonify({'error': 'No chat history found'}), 404
    
    try:
        exportable_messages = sessions[session_id]['messages']
        
        if not exportable_messages:
            return jsonify({'error': 'No messages found'}), 404
        
        pdf_filename = f'chat_export_{session_id}_{int(time.time())}.pdf'
        pdf_path = os.path.join(STATIC_FOLDER, pdf_filename)
        
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
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
        
        story = []
        story.append(Paragraph("HR Assistant - Chat Export", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        for msg in exportable_messages:
            if msg['role'] == 'user':
                story.append(Paragraph(f"<b>You:</b> {html.escape(msg['content'])}", user_style))
            else:
                content = msg['content'].replace('**', '')
                story.append(Paragraph(f"<b>Assistant:</b> {html.escape(content)}", bot_style))
        
        doc.build(story)
        
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
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
        for file_info in sessions[session_id]['pdf_files']:
            filepath = os.path.join(UPLOAD_FOLDER, file_info['filename'])
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error deleting file: {e}")
        
        sessions[session_id]['messages'] = []
        sessions[session_id]['pdf_files'] = []
        sessions[session_id]['last_interaction'] = time.time()
        sessions[session_id]['last_analysis'] = None
        sessions[session_id]['awaiting_followup'] = False
        sessions[session_id]['consecutive_no_count'] = 0
        sessions[session_id]['feedback_submitted'] = False
        sessions[session_id]['feedback'] = []
        
        has_preloaded = bool(sessions[session_id].get('preloaded_files'))
        
        if sessions[session_id].get('conversation_chain'):
            try:
                sessions[session_id]['conversation_chain'].memory.clear()
                print("Conversation memory cleared successfully")
            except Exception as e:
                print(f"Error clearing memory: {e}")
        
        sessions[session_id]['chat_active'] = has_preloaded or bool(sessions[session_id].get('vectorstore'))
        
        return jsonify({
            'success': True, 
            'chat_active': sessions[session_id]['chat_active'],
            'has_preloaded': has_preloaded
        })
    
    return jsonify({'success': False, 'error': 'Session not found'})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    session_id = data.get('session_id')
    rating = data.get('rating')
    comment = data.get('comment', '')
    
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'pdf_files': [],
            'vectorstore': None,
            'conversation_chain': None,
            'feedback': [],
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0,
            'chat_active': False,
            'preloaded_files': []
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
        
        idle_time = current_time - last_interaction
        
        # 10 seconds after upload, 7 seconds otherwise
        if upload_completed_time and (current_time - upload_completed_time) < 15:
            idle_threshold = 10
        else:
            idle_threshold = 7
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
