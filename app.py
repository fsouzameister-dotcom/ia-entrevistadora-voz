import os
import json
import logging
import uuid
import re
from datetime import datetime
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from flask import Flask, request, jsonify, Response, send_file

from flask_cors import CORS

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURAÇÕES E INICIALIZAÇÃO ---
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "SENHA_ADMIN")
REPORTS_DIR = "relatorios"

if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# --- INICIALIZAÇÃO DOS SERVIÇOS ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key: raise ValueError("Chave da API do Google não encontrada.")
genai.configure(api_key=google_api_key)
generation_model = genai.GenerativeModel('models/gemini-2.5-flash')
logging.info("Modelo Gemini inicializado.")

tts_client = None
try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Cliente Google TTS inicializado com sucesso.")
except Exception as e:
    logging.error(f"FALHA CRÍTICA AO INICIALIZAR O CLIENTE GOOGLE TTS: {e}")

# --- PERSONA E ROTEIRO DA IA ---
SYSTEM_PROMPT = """
Você é "Gui", um entrevistador de IA de elite. Sua personalidade é calorosa, curiosa, empática e profissional. Seu objetivo principal é conduzir uma pesquisa que se sinta como uma conversa humana genuína.
Para cada interação, siga estritamente este fluxo de 4 passos:
1.  **Agradecer/Reconhecer:** Inicie sua resposta com uma frase curta de reconhecimento (a menos que seja a primeira frase da conversa).
2.  **Refletir/Empatizar:** Faça um breve comentário de uma frase que se conecte diretamente ao conteúdo ou sentimento da 'RESPOSTA ANTERIOR DO USUÁRIO'.
3.  **Fazer a Ponte:** Use uma frase de transição curta para mover a conversa para o próximo tópico.
4.  **Perguntar:** Apresente a 'PRÓXIMA PERGUNTA DO ROTEIRO' de forma clara e exata.
**Exceção:** Para a PRIMEIRA pergunta da entrevista, não há resposta anterior. Apenas apresente a pergunta do roteiro de forma amigável e direta.
"""
try:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "interview_script.json")
    with open(file_path, "r", encoding="utf-8") as f: interview_script = json.load(f)
    logging.info("Roteiro da entrevista carregado.")
except FileNotFoundError:
    logging.error("Arquivo 'interview_script.json' não encontrado.")
    interview_script = None

ongoing_interviews = {}

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

def get_next_step(current_step_id, user_response):
    current_step = interview_script["steps"].get(current_step_id)
    if not current_step: return interview_script["start_step_id"]
    if current_step.get("awaits_rating"):
        rating_logic = current_step.get("rating_logic", {})
        try:
            numbers = re.findall(r'\d+', user_response)
            if numbers:
                rating = int(numbers[0])
                if "detractor_threshold" in rating_logic:
                    if rating <= rating_logic["detractor_threshold"]: return rating_logic["detractor_step_id"]
                    else: return rating_logic["promoter_step_id"]
                elif "threshold" in rating_logic:
                    if rating <= rating_logic["threshold"]: return rating_logic["follow_up_step_id"]
        except (ValueError, IndexError): pass
    response_lower = user_response.lower()
    if "conditional_logic" in current_step:
        for condition in current_step["conditional_logic"]:
            if any(keyword in response_lower for keyword in condition["keywords"]):
                return condition["next_step_id"]
    return current_step.get("next_step_id", interview_script["end_step_id"])

def save_report(interview_id):
    if interview_id not in ongoing_interviews: return
    interview_data = ongoing_interviews[interview_id]
    end_time = datetime.utcnow()
    duration = (end_time - interview_data["start_time"]).total_seconds()
    report = { "interview_id": interview_id, "start_time": interview_data["start_time"].isoformat() + "Z", "end_time": end_time.isoformat() + "Z", "duration_seconds": int(duration), "transcript": interview_data["transcript"] }
    filename = f"entrevista_{end_time.strftime('%Y-%m-%d_%H-%M-%S')}_{interview_id[:8]}.json"
    filepath = os.path.join(REPORTS_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f: json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"Relatório salvo: {filepath}")
    del ongoing_interviews[interview_id]

# --- ENDPOINTS ---
@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/start', methods=['POST'])
def start_interview():
    if not interview_script: return jsonify({'error': 'Erro: Roteiro não carregado.'}), 500
    interview_id = str(uuid.uuid4())
    ongoing_interviews[interview_id] = {"start_time": datetime.utcnow(), "transcript": {}, "last_question": None, "last_topic": None}
    start_step_id = interview_script["start_step_id"]
    start_step_data = interview_script["steps"].get(start_step_id)
    intro_text = start_step_data["question_text"]
    ongoing_interviews[interview_id]["last_question"] = intro_text
    ongoing_interviews[interview_id]["last_topic"] = start_step_data.get("topic")
    return jsonify({'answer': intro_text, 'next_step_id': start_step_id, 'interview_id': interview_id})

@app.route('/interview', methods=['POST'])
def interview_step():
    if not interview_script: return jsonify({'error': 'Erro: Roteiro não carregado.'}), 500
    data = request.get_json()
    user_response = data.get('response', '')
    current_step_id = data.get('current_step_id')
    chat_history = data.get('history', [])
    interview_id = data.get('interview_id')
    if not all([user_response, current_step_id, interview_id]):
        return jsonify({'error': 'Dados insuficientes na requisição.'}), 400
    if interview_id in ongoing_interviews:
        session = ongoing_interviews[interview_id]
        topic = session["last_topic"]
        if topic:
            if topic not in session["transcript"]: session["transcript"][topic] = []
            session["transcript"][topic].append({"question": session["last_question"], "answer": user_response})
    next_step_id = get_next_step(current_step_id, user_response)
    next_step_data = interview_script["steps"].get(next_step_id)
    next_question_to_ask = next_step_data["question_text"]
    prompt_for_gemini = (f"RESPOSTA ANTERIOR DO USUÁRIO: \"{user_response}\"\n\nPRÓXIMA PERGUNTA DO ROTEIRO: \"{next_question_to_ask}\"\n\nSua tarefa: Como Gui, gere a próxima resposta.")
    try:
        history_for_gemini = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': ["Entendido."]}]
        history_for_gemini.extend(chat_history)
        convo = generation_model.start_chat(history=history_for_gemini)
        convo.send_message(prompt_for_gemini)
        gui_response = convo.last.text
        
        # --- AQUI ESTÁ A CORREÇÃO ---
        # Remove os asteriscos da resposta do Gemini antes de enviá-la.
        cleaned_response = gui_response.replace('*', '')

        if interview_id in ongoing_interviews:
            # Salva a versão limpa como a "última pergunta" para consistência nos relatórios
            ongoing_interviews[interview_id]["last_question"] = cleaned_response
            ongoing_interviews[interview_id]["last_topic"] = next_step_data.get("topic")

        if next_step_data.get("is_final", False):
            save_report(interview_id)
            
        return jsonify({'answer': cleaned_response, 'next_step_id': next_step_id, 'interview_id': interview_id})
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Gemini: {e}")
        return jsonify({'answer': 'Desculpe, tive um problema técnico.'}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    if not tts_client: return jsonify({"error": "Serviço de TTS não configurado"}), 500
    data = request.get_json()
    text = data.get('text', '')
    if not text: return jsonify({"error": "Nenhum texto fornecido"}), 400
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="pt-BR", name="pt-BR-Chirp3-HD-Algieba")
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return Response(response.audio_content, mimetype="audio/mpeg")
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Google TTS: {e}")
        return jsonify({"error": "Não foi possível gerar o áudio"}), 500

@app.route('/admin')
def admin_panel():
    return app.send_static_file('admin.html')

@app.route('/admin/reports')
def list_reports():
    try:
        files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
        files.sort(reverse=True)
        return jsonify({"reports": files})
    except Exception as e:
        logging.error(f"Erro ao listar relatórios: {e}")
        return jsonify({"error": "Não foi possível listar os relatórios"}), 500

@app.route('/admin/download/<file_format>')
def download_report(file_format):
    all_data = []
    try:
        for filename in os.listdir(REPORTS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(REPORTS_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    for topic, qas in report.get("transcript", {}).items():
                        for qa in qas:
                            all_data.append({"interview_id": report.get("interview_id", ""), "start_time": report.get("start_time", ""), "end_time": report.get("end_time", ""),"topic": topic, "question": qa.get("question", ""), "answer": qa.get("answer", "")})
        if not all_data: return "Nenhum dado encontrado para gerar o relatório.", 404
        df = pd.DataFrame(all_data)
        output = BytesIO()
        if file_format == 'csv':
            df.to_csv(output, index=False, encoding='utf-8-sig')
            mimetype = 'text/csv'
            filename = 'consolidated_report.csv'
        elif file_format == 'xls':
            writer = pd.ExcelWriter(output, engine='openpyxl')
            df.to_excel(writer, index=False, sheet_name='Entrevistas')
            writer.close()
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            filename = 'consolidated_report.xlsx'
        else:
            return "Formato de arquivo não suportado.", 400
        output.seek(0)
        return send_file(output, as_attachment=True, download_name=filename, mimetype=mimetype)
    except Exception as e:
        logging.error(f"Erro ao gerar relatório para download: {e}")
        return "Erro ao gerar o relatório.", 500