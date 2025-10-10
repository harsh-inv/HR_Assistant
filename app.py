import os
import warnings
import logging
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

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
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret-key-for-development')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate that the key exists
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please configure it in Render dashboard.")

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Session storage (in production, use Redis or database)
sessions = {}

def get_session(session_id):
    if session_id not in sessions:
        sessions[session_id] = {
            'vectorstore': None,
            'conversation_chain': None,
            'chat_history': [],
            'uploaded_files': [],
            'feedback_history': []
        }
    return sessions[session_id]

# ---------- Document Processing Functions ----------
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

def process_file(uploaded_file, filename):
    file_ext = filename.lower().split('.')[-1]
    if file_ext == 'pdf':
        return extract_pdf_text(uploaded_file)
    elif file_ext in ['docx', 'doc']:
        return extract_docx_text(uploaded_file)
    elif file_ext == 'txt':
        return extract_txt_text(uploaded_file)
    return ""

def create_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context:

        Context: {context}

        Question: {input}

        Answer:""")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
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
        session['vectorstore'] = vectorstore
        session['conversation_chain'] = create_chain(vectorstore)
        session['uploaded_files'] = processed_files
        
        return jsonify({
            'success': True,
            'files': processed_files,
            'message': f'Successfully processed {len(processed_files)} files'
        })
    
    return jsonify({'error': 'No text could be extracted from files'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    session_id = request.json.get('session_id', 'default')
    message = request.json.get('message', '')
    session = get_session(session_id)
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    # Add user message to history
    session['chat_history'].append({
        'message': message,
        'is_user': True,
        'timestamp': datetime.now().isoformat()
    })
    
    # Check for goodbye
    if message.lower().strip() in ['bye', 'goodbye', 'exit', 'quit', 'end']:
        response = "Thank you for using HR Assistant! Have a great day! 👋"
        session['chat_history'].append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        return jsonify({
            'response': response,
            'session_ended': True
        })
    
    # Get AI response
    if session['conversation_chain']:
        try:
            result = session['conversation_chain'].invoke({'input': message})
            response = result['answer']
            session['chat_history'].append({
                'message': response,
                'is_user': False,
                'timestamp': datetime.now().isoformat()
            })
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Please upload documents first'}), 400

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
    
    # Create CSV content
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Timestamp', 'Rating', 'Comment'])
    
    # Write feedback data
    for feedback in session['feedback_history']:
        writer.writerow([
            feedback['timestamp'],
            feedback['rating'],
            feedback.get('comment', '')
        ])
    
    # Create the response
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
            'vectorstore': None,
            'conversation_chain': None,
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)