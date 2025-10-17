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
app.config['DOCUMENTS_FOLDER'] = os.getenv('DOCUMENTS_FOLDER', '/opt/render/project/src/documents')
app.config['MAX_DOCUMENTS'] = 100  # Maximum number of documents to load
app.config['VIDEOS_FOLDER'] = 'static/videos'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

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

def load_documents_from_directory(directory):
    all_text = ""
    processed_files = []
    
    if not os.path.exists(directory):
        return all_text, processed_files
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.pdf', '.docx', '.doc', '.txt')):
                file_path = os.path.join(root, filename)
                try:
                    text = process_file(file_path, filename)
                    if text:
                        all_text += f"\n\n--- {filename} ---\n{text}"
                        processed_files.append(filename)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    return all_text, processed_files

def similarity_score(str1, str2):
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def find_related_video(query, threshold=0.3):
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
    retriever = vectorstore.as_retriever()
    
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
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # Changed to 0.0 for less creativity
    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    session_id = request.json.get('session_id', 'default')
    session = get_session(session_id)
    
    # Debug: Check documents folder
    print(f"Documents folder path: {os.path.abspath(app.config['DOCUMENTS_FOLDER'])}")
    print(f"Documents folder exists: {os.path.exists(app.config['DOCUMENTS_FOLDER'])}")
    if os.path.exists(app.config['DOCUMENTS_FOLDER']):
        files_in_folder = os.listdir(app.config['DOCUMENTS_FOLDER'])
        print(f"Files in documents folder: {len(files_in_folder)} files found")
        if len(files_in_folder) > app.config['MAX_DOCUMENTS']:
            print(f"WARNING: Found {len(files_in_folder)} files, limiting to {app.config['MAX_DOCUMENTS']} files")
    
    all_text, processed_files = load_documents_from_directory(app.config['DOCUMENTS_FOLDER'])
    
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
        session['preloaded_files'] = processed_files
        
        return jsonify({
            'success': True,
            'files': processed_files,
            'message': f'Loaded {len(processed_files)} documents from knowledge base'
        })
    
    return jsonify({
        'success': True,
        'files': [],
        'message': 'No documents found in knowledge base. You can upload documents to get started.'
    })

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
    session_id = request.json.get('session_id', 'default')
    message = request.json.get('message', '')
    is_voice_input = request.json.get('is_voice_input', False)
    
    session = get_session(session_id)
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400
    
    session['chat_history'].append({
        'message': message,
        'is_user': True,
        'is_voice': is_voice_input,
        'timestamp': datetime.now().isoformat()
    })
    
    if message.lower().strip() in ['bye', 'goodbye', 'exit', 'quit', 'end']:
        response = "Thank you for using HR Assistant! Have a great day!"
        session['chat_history'].append({
            'message': response,
            'is_user': False,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate audio for voice input
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
    
    related_video = find_related_video(message)
    
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
            
            # Generate audio automatically if voice input
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
            return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Please upload documents first or wait for knowledge base to load'}), 400

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    text = request.json.get('text', '')
    session_id = request.json.get('session_id', 'default')
    
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

@app.route('/get_loaded_files', methods=['GET'])
def get_loaded_files():
    session_id = request.args.get('session_id', 'default')
    session = get_session(session_id)
    
    return jsonify({
        'preloaded': session.get('preloaded_files', []),
        'uploaded': session.get('uploaded_files', [])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

