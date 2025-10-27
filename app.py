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
if not os.path.exists(REPORTS_DIR): os.makedirs(REPORTS_DIR)

# --- INICIALIZAÇÃO DOS SERVIÇOS ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key: raise ValueError("Chave da API do Google não encontrada.")
genai.configure(api_key=google_api_key)
generation_model = genai.GenerativeModel('models/gemini-2.5-flash')
interpreter_model = genai.GenerativeModel('models/gemini-2.5-flash') # Modelo dedicado para interpretação
logging.info("Modelos Gemini inicializados.")

tts_client = None
try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Cliente Google TTS inicializado.")
except Exception as e:
    logging.error(f"FALHA AO INICIALIZAR O CLIENTE GOOGLE TTS: {e}")

# --- PERSONA E ROTEIRO ---
SYSTEM_PROMPT = "Você é Gui, um entrevistador de IA empático e profissional..." # Mantido o mesmo prompt detalhado
try:
    script_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(script_dir, "interview_script.json")
    with open(file_path, "r", encoding="utf-8") as f: interview_script = json.load(f)
    logging.info("Roteiro da entrevista carregado.")
except FileNotFoundError:
    logging.error("Arquivo 'interview_script.json' não encontrado.")
    interview_script = None

ongoing_interviews = {}

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# --- NOVO MOTOR DE REGRAS E NAVEGAÇÃO ---

def interpret_response(question_options, user_answer):
    """Usa a IA para categorizar uma resposta aberta em uma das opções predefinidas."""
    prompt = (
        f"Contexto: O usuário respondeu a uma pergunta. As opções de resposta possíveis são: {json.dumps(question_options, ensure_ascii=False)}.\n"
        f"Resposta do Usuário: \"{user_answer}\"\n"
        "Sua Tarefa: Analise a resposta do usuário e retorne APENAS a opção da lista que melhor corresponde à resposta. "
        "Se a resposta indicar mais de uma opção, retorne as opções separadas por vírgula. "
        "Se não corresponder a nenhuma, retorne 'N/A'."
    )
    try:
        response = interpreter_model.generate_content(prompt)
        interpreted_answer = response.text.strip()
        logging.info(f"Resposta '{user_answer}' interpretada como: '{interpreted_answer}'")
        return interpreted_answer
    except Exception as e:
        logging.error(f"Erro na interpretação da IA: {e}")
        return user_answer # Retorna a resposta original em caso de erro

def navigate_interview(session, user_response):
    """O novo motor de regras que processa a lógica do JSON."""
    current_step_id = session['current_step_id']
    current_step_data = interview_script["chapters"][session['current_chapter']]['steps'].get(current_step_id)

    if not current_step_data:
        # Fim de um capítulo, ou erro
        session['last_action_result'] = "END_CHAPTER"
        return

    # Salva a resposta do usuário (bruta)
    topic = current_step_data.get("topic", "N/A")
    if topic not in session['transcript']: session['transcript'][topic] = []
    session['transcript'][topic].append({ "question": session['last_question'], "answer": user_response })

    # Interpreta a resposta se necessário
    final_answer = user_response
    if current_step_data.get("requires_interpretation"):
        final_answer = interpret_response(current_step_data.get("options", []), user_response)
        # Salva o dado interpretado no perfil para uso futuro
        session['user_profile'][current_step_id] = final_answer

    # Processa a lógica de ações (cortes, ramificações)
    if "logic" in current_step_data:
        for rule in current_step_data["logic"]:
            action = rule.get("action")
            if rule.get("if_answer") == final_answer or (rule.get("if_answer_contains") and rule.get("if_answer_contains") in final_answer):
                if action == "END_INTERVIEW":
                    session['last_action_result'] = "END_INTERVIEW"
                    logging.info(f"Entrevista {session['interview_id']} encerrada por regra: {rule.get('reason')}")
                    return
                elif action == "ADD_CHAPTER":
                    chapter_to_add = rule.get("chapter")
                    if chapter_to_add not in session['chapter_queue']:
                        session['chapter_queue'].append(chapter_to_add)
    
    # Determina o próximo passo
    if current_step_data.get("next_action") == "START_NEXT_CHAPTER":
        session['last_action_result'] = "END_CHAPTER"
        return

    session['next_step_id'] = current_step_data.get("next_step_id")
    session['last_action_result'] = "CONTINUE"


# --- ENDPOINTS ---
@app.route('/')
def serve_index(): return app.send_static_file('index.html')

@app.route('/start', methods=['POST'])
def start_interview():
    interview_id = str(uuid.uuid4())
    start_chapter = interview_script['start_chapter']
    start_step_id = interview_script['chapters'][start_chapter]['start_step_id']
    start_step_data = interview_script['chapters'][start_chapter]['steps'][start_step_id]
    intro_text = start_step_data['question_text']

    ongoing_interviews[interview_id] = {
        "interview_id": interview_id,
        "start_time": datetime.utcnow(),
        "transcript": {},
        "user_profile": {},
        "chapter_queue": [],
        "current_chapter": start_chapter,
        "current_step_id": start_step_id,
        "last_question": intro_text,
        "last_topic": start_step_data.get("topic")
    }
    return jsonify({'answer': intro_text, 'next_step_id': start_step_id, 'interview_id': interview_id})

@app.route('/interview', methods=['POST'])
def interview_step():
    data = request.get_json()
    interview_id = data.get('interview_id')
    user_response = data.get('response', '')
    session = ongoing_interviews.get(interview_id)

    if not session: return jsonify({'error': 'Sessão inválida.'}), 400

    # Executa o motor de regras
    navigate_interview(session, user_response)

    # Processa o resultado da navegação
    if session['last_action_result'] == "END_INTERVIEW":
        final_step_data = interview_script['final_steps']['FINAL_END']
        session['current_step_id'] = "FINAL_END"
        save_report(interview_id)
        return jsonify({'answer': final_step_data['question_text'], 'next_step_id': "FINAL_END"})
    
    if session['last_action_result'] == "END_CHAPTER":
        if session['chapter_queue']:
            next_chapter = session['chapter_queue'].pop(0)
            session['current_chapter'] = next_chapter
            next_step_id = interview_script['chapters'][next_chapter]['start_step_id']
            session['current_step_id'] = next_step_id
        else: # Fim de todos os capítulos
            finalization_chapter = "FINALIZACAO"
            session['current_chapter'] = finalization_chapter
            next_step_id = interview_script['chapters'][finalization_chapter]['start_step_id']
            session['current_step_id'] = next_step_id
    else: # Continua no mesmo capítulo
        next_step_id = session['next_step_id']
        session['current_step_id'] = next_step_id

    next_step_data = interview_script['chapters'][session['current_chapter']]['steps'][next_step_id]
    next_question_to_ask = next_step_data['question_text']

    # Lógica de humanização com Gemini
    prompt_for_gemini = (f"RESPOSTA ANTERIOR DO USUÁRIO: \"{user_response}\"\n\nPRÓXIMA PERGUNTA DO ROTEIRO: \"{next_question_to_ask}\"\n\nSua tarefa: Como Gui, gere a próxima resposta.")
    try:
        convo = generation_model.start_chat(history=[{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': ["Entendido."]}])
        convo.send_message(prompt_for_gemini)
        gui_response = convo.last.text.replace('*', '')
        session['last_question'] = gui_response
        session['last_topic'] = next_step_data.get("topic")

        if next_step_data.get("is_final") or next_step_id == "FINAL_END":
            save_report(interview_id)

        return jsonify({'answer': gui_response, 'next_step_id': next_step_id, 'interview_id': interview_id})
    except Exception as e:
        logging.error(f"Erro na chamada do Gemini: {e}")
        return jsonify({'answer': 'Desculpe, tive um problema técnico.'}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # ... (sem alterações) ...
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

# --- ENDPOINTS DE ADMINISTRAÇÃO E DOWNLOAD (COM RELATÓRIO MULTI-ABAS) ---
@app.route('/admin')
def admin_panel(): return app.send_static_file('admin.html')

@app.route('/admin/reports')
def list_reports():
    # ... (sem alterações) ...
    try:
        if not os.path.exists(REPORTS_DIR): return jsonify({"reports": []})
        files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
        files.sort(reverse=True)
        return jsonify({"reports": files})
    except Exception as e:
        logging.error(f"Erro ao listar relatórios: {e}")
        return jsonify({"error": "Não foi possível listar os relatórios"}), 500

@app.route('/admin/download/xls')
def download_xls_report():
    try:
        if not os.path.exists(REPORTS_DIR) or not os.listdir(REPORTS_DIR):
            return "Nenhum relatório encontrado para download.", 404
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            all_files_data = [json.load(open(os.path.join(REPORTS_DIR, f), 'r', encoding='utf-8')) for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
            
            # Aba 1: Perfil dos Entrevistados
            profiles = []
            for report in all_files_data:
                profile_data = {
                    "interview_id": report.get('interview_id'),
                    "start_time": report.get('start_time'),
                    "duration_seconds": report.get('duration_seconds')
                }
                # Adiciona respostas do perfil salvas
                for key, val in report.get('user_profile', {}).items():
                    profile_data[key] = val
                profiles.append(profile_data)
            pd.DataFrame(profiles).to_excel(writer, sheet_name='Perfis', index=False)
            
            # Abas por Tópico
            topics_data = {}
            for report in all_files_data:
                for topic, qas in report.get('transcript', {}).items():
                    if topic not in topics_data: topics_data[topic] = []
                    for qa in qas:
                        topics_data[topic].append({
                            "interview_id": report.get('interview_id'),
                            "question": qa.get('question'),
                            "answer": qa.get('answer')
                        })
            
            for topic, data in topics_data.items():
                pd.DataFrame(data).to_excel(writer, sheet_name=topic[:30], index=False) # Limita nome da aba a 30 chars

        output.seek(0)
        return send_file(output, as_attachment=True, download_name='consolidated_report.xlsx', mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    except Exception as e:
        logging.error(f"Erro ao gerar relatório XLS: {e}", exc_info=True)
        return "Erro ao gerar o relatório.", 500