import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# --- CONFIGURAções E CARREGAMENTO INICIAL ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- INICIALIZAÇÃO DOS SERVIÇOS ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Chave da API do Google (GOOGLE_API_KEY) não encontrada.")
genai.configure(api_key=google_api_key)

generation_model = genai.GenerativeModel('models/gemini-2.5-flash')
logging.info("Modelo Gemini ('models/gemini-2.5-flash') inicializado.")

try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Cliente Google TTS inicializado.")
except Exception as e:
    logging.warning(f"Não foi possível inicializar o cliente Google TTS: {e}")
    tts_client = None

# --- PERSONA E ROTEIRO DA IA (COM HUMANIZAÇÃO) ---
SYSTEM_PROMPT = """
Você é "Gui", um entrevistador de IA de elite. Sua personalidade é calorosa, curiosa, empática e profissional. Seu objetivo principal é conduzir uma pesquisa que se sinta como uma conversa humana genuína.

Para cada interação, siga estritamente este fluxo de 4 passos:

1.  **Agradecer/Reconhecer:** Inicie sua resposta com uma frase curta de reconhecimento.
    *   Exemplos: "Entendido.", "Certo.", "Obrigado por compartilhar isso.", "Anotado."

2.  **Refletir/Empatizar (O PASSO MAIS IMPORTANTE):** Faça um breve comentário de uma frase que se conecte diretamente ao conteúdo ou sentimento da 'RESPOSTA ANTERIOR DO USUÁRIO'. Mostre que você entendeu não apenas as palavras, mas o significado por trás delas.
    *   Se a resposta for negativa (ex: "os equipamentos estão sempre quebrados"): "Nossa, isso deve ser muito frustrante e pode até atrapalhar a consistência dos treinos."
    *   Se a resposta for positiva (ex: "os instrutores são muito atenciosos"): "Que ótimo ouvir isso! Ter um bom suporte profissional faz toda a diferença na motivação, não é mesmo?"
    *   Se a resposta for neutra (ex: "a iluminação é normal"): "Ok, um aspecto funcional que cumpre seu papel, sem grandes destaques."

3.  **Fazer a Ponte:** Use uma frase de transição curta para mover a conversa para o próximo tópico.
    *   Exemplos: "Mudando um pouco de assunto...", "Falando agora sobre...", "Isso me leva à próxima questão..."

4.  **Perguntar:** Apresente a 'PRÓXIMA PERGUNTA DO ROTEIRO' de forma clara. Você deve usar a pergunta do roteiro exatamente como fornecida.

**Exemplo de Execução Perfeita:**
- RESPOSTA ANTERIOR DO USUÁRIO: "A academia é ok, mas os vestiários são muito sujos, eu evito usar."
- PRÓXIMA PERGUNTA DO ROTEIRO: "E sobre os equipamentos, você encontra a variedade que precisa para os seus treinos?"
- SUA RESPOSTA IDEAL GERADA: "Entendido. Higiene no vestiário é fundamental para o conforto, imagino que seja uma situação bem desagradável. Mudando um pouco o foco agora, sobre os equipamentos, você encontra a variedade que precisa para os seus treinos?"

Sua tarefa é usar o contexto da resposta do usuário e a próxima pergunta do roteiro para gerar uma resposta que siga este fluxo de 4 passos, tornando a entrevista o mais humana possível.
"""

# Construindo um caminho absoluto para o JSON para garantir que o servidor sempre o encontre.
try:
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, "interview_script.json")
    with open(file_path, "r", encoding="utf-8") as f:
        interview_script = json.load(f)
    logging.info("Roteiro da entrevista carregado com sucesso.")
except FileNotFoundError:
    logging.error("Arquivo 'interview_script.json' não encontrado. O aplicativo não pode funcionar.")
    interview_script = None

# --- CONFIGURAÇÃO DO FLASK ---
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

def get_next_step(current_step_id, user_response):
    current_step = interview_script["steps"].get(current_step_id)
    if not current_step: return interview_script["start_step_id"]
    response_lower = user_response.lower()
    if "conditional_logic" in current_step:
        for condition in current_step["conditional_logic"]:
            if any(keyword in response_lower for keyword in condition["keywords"]):
                return condition["next_step_id"]
    return current_step.get("next_step_id", interview_script["end_step_id"])

# --- ENDPOINTS ---
@app.route('/')
def serve_index():
    """Serve o arquivo index.html da pasta 'static'."""
    return app.send_static_file('index.html')

@app.route('/interview', methods=['POST'])
def interview_step():
    if not interview_script: return jsonify({'error': 'Erro: Roteiro não carregado.'}), 500
    data = request.get_json()
    user_response = data.get('response', '')
    current_step_id = data.get('current_step_id', interview_script["start_step_id"])
    chat_history = data.get('history', [])
    
    if current_step_id == interview_script["start_step_id"] and not user_response:
        next_step_id = interview_script["start_step_id"]
    else:
        next_step_id = get_next_step(current_step_id, user_response)
    next_step_data = interview_script["steps"].get(next_step_id)

    if next_step_data.get("is_final", False):
        return jsonify({'answer': next_step_data["question_text"], 'next_step_id': next_step_id})

    next_question_to_ask = next_step_data["question_text"]
    prompt_for_gemini = (
        f"RESPOSTA ANTERIOR DO USUÁRIO: \"{user_response}\"\n\n"
        f"PRÓXIMA PERGUNTA DO ROTEIRO: \"{next_question_to_ask}\"\n\n"
        "Sua tarefa: Como Gui, gere a próxima resposta da entrevista seguindo seu prompt de sistema."
    )
    try:
        history_for_gemini = [{'role': 'user', 'parts': [SYSTEM_PROMPT]}, {'role': 'model', 'parts': ["Entendido. Estou pronto para começar."]}]
        history_for_gemini.extend(chat_history)
        convo = generation_model.start_chat(history=history_for_gemini)
        convo.send_message(prompt_for_gemini)
        gui_response = convo.last.text
        return jsonify({'answer': gui_response, 'next_step_id': next_step_id})
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

if __name__ == '__main__':
    if not interview_script:
        logging.fatal("O aplicativo não pode iniciar sem 'interview_script.json'.")
    else:
        app.run(host='0.0.0.0', port=5000)