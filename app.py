import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
# --- MUDANÇA 1: NOVA IMPORTAÇÃO ---
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# ... (todo o resto do seu código de configuração permanece igual) ...
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Chave da API do Google (GOOGLE_API_KEY) não encontrada no arquivo .env.")
genai.configure(api_key=google_api_key)

generation_model = genai.GenerativeModel('models/gemini-2.5-flash')
logging.info("Modelo Gemini ('models/gemini-2.5-flash') inicializado para respostas rápidas.")

try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Cliente Google Text-to-Speech (TTS) inicializado com sucesso.")
except Exception as e:
    logging.warning(f"Não foi possível inicializar o cliente Google TTS. O endpoint de voz /synthesize não funcionará. Erro: {e}")
    tts_client = None

SYSTEM_PROMPT = """
Você é "Gui", um entrevistador de IA especializado em pesquisa de mercado. Sua personalidade é calorosa, curiosa e empática, como um bom ouvinte.
Seu objetivo principal é fazer com que a conversa flua de forma natural, e não como um questionário robótico. Para isso, siga estas diretrizes:
1.  **Crie Transições Suaves:** Antes de fazer a 'PRÓXIMA PERGUNTA DO ROTEIRO', conecte-a com a 'RESPOSTA ANTERIOR DO USUÁRIO'. Use frases curtas de reconhecimento como "Entendi.", "Interessante o que você disse sobre...", "Obrigado por compartilhar isso. Falando agora sobre...", "Isso me leva à próxima questão...".
2.  **Seja Conciso:** Mantenha suas frases curtas e diretas. Evite parágrafos longos.
3.  **Mantenha a Persona:** Você é sempre profissional, mas amigável. Use um tom encorajador.
4.  **Foco no Roteiro:** Sua tarefa é APENAS usar a 'PRÓXIMA PERGUNTA DO ROTEIRO'. Nunca crie suas próprias perguntas ou desvie do tópico.
Exemplo de interação ideal:
- RESPOSTA ANTERIOR DO USUÁRIO: "A academia é limpa, mas fica muito cheia à noite."
- PRÓXIMA PERGUNTA DO ROTEIRO: "E sobre os equipamentos, como você os avalia?"
- SUA RESPOSTA IDEAL: "Entendi. É um desafio quando o espaço fica lotado, né? Mudando um pouco de assunto, e sobre os equipamentos, como você os avalia?"
"""

try:
    with open("interview_script.json", "r", encoding="utf-8") as f:
        interview_script = json.load(f)
    logging.info("Roteiro da entrevista (interview_script.json) carregado com sucesso.")
except FileNotFoundError:
    logging.error("Arquivo 'interview_script.json' não encontrado. O aplicativo não pode funcionar sem ele.")
    interview_script = None

app = Flask(__name__)
CORS(app)

def get_next_step(current_step_id, user_response):
    current_step = interview_script["steps"].get(current_step_id)
    if not current_step:
        logging.warning(f"ID de passo inválido: {current_step_id}. Retornando ao início.")
        return interview_script["start_step_id"]
    response_lower = user_response.lower()
    if "conditional_logic" in current_step:
        for condition in current_step["conditional_logic"]:
            if any(keyword in response_lower for keyword in condition["keywords"]):
                logging.info(f"Lógica condicional ativada. Indo para o passo: {condition['next_step_id']}")
                return condition["next_step_id"]
    return current_step.get("next_step_id", interview_script["end_step_id"])

# --- MUDANÇA 2: NOVO ENDPOINT PARA SERVIR O HTML ---
@app.route('/')
def serve_index():
    """Serve o arquivo index.html quando a rota raiz é acessada."""
    return send_from_directory('.', 'index.html')

@app.route('/interview', methods=['POST'])
def interview_step():
    # ... (o resto da sua função interview_step permanece exatamente igual) ...
    if not interview_script: return jsonify({'error': 'Erro: Roteiro não carregado.'}), 500
    data = request.get_json()
    user_response = data.get('response', '')
    current_step_id = data.get('current_step_id', interview_script["start_step_id"])
    chat_history = data.get('history', [])
    logging.info(f"Recebido - Resposta: '{user_response}', Passo Atual: '{current_step_id}'")

    if current_step_id == interview_script["start_step_id"] and not user_response:
        next_step_id = interview_script["start_step_id"]
    else:
        next_step_id = get_next_step(current_step_id, user_response)
    next_step_data = interview_script["steps"].get(next_step_id)

    if next_step_data.get("is_final", False):
        logging.info("Fim da entrevista alcançado.")
        return jsonify({'answer': next_step_data["question_text"], 'next_step_id': next_step_id})

    next_question_to_ask = next_step_data["question_text"]
    
    prompt_for_gemini = (
        f"RESPOSTA ANTERIOR DO USUÁRIO: \"{user_response}\"\n\n"
        f"PRÓXIMA PERGUNTA DO ROTEIRO: \"{next_question_to_ask}\"\n\n"
        "Sua tarefa: Como Gui, gere a próxima resposta da entrevista seguindo as diretrizes e o exemplo do seu prompt de sistema."
    )

    try:
        history_for_gemini = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': ["Entendido. Estou pronto para começar a entrevista."]}]
        history_for_gemini.extend(chat_history)
        convo = generation_model.start_chat(history=history_for_gemini)
        convo.send_message(prompt_for_gemini)
        gui_response = convo.last.text
        logging.info(f"Resposta gerada pelo Gemini (Gui): '{gui_response}'")
        return jsonify({'answer': gui_response, 'next_step_id': next_step_id})
    except Exception as e:
        logging.error(f"Erro ao chamar a API do Gemini: {e}")
        return jsonify({'answer': 'Desculpe, tive um problema técnico. Poderia repetir?'}), 500


@app.route('/synthesize', methods=['POST'])
def synthesize():
    # ... (a sua função synthesize permanece exatamente igual) ...
    if not tts_client: return jsonify({"error": "Serviço de TTS não configurado"}), 500
    data = request.get_json()
    text = data.get('text', '')
    if not text: return jsonify({"error": "Nenhum texto fornecido"}), 400
    try:
        logging.info(f"Sintetizando texto para áudio: '{text}'")
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        voice = texttospeech.VoiceSelectionParams(
            language_code="pt-BR",