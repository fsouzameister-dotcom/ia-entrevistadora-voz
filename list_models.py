import os
from dotenv import load_dotenv
import google.generativeai as genai

# Carrega a chave da API do seu arquivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Chave da API do Google não encontrada no arquivo .env")
else:
    try:
        genai.configure(api_key=api_key)
        
        print("===============================================")
        print("   Modelos de IA Disponíveis para Sua Chave    ")
        print("===============================================")
        
        for model in genai.list_models():
            # Verifica se o modelo suporta a geração de conteúdo (a nossa tarefa)
            if 'generateContent' in model.supported_generation_methods:
                print(f"-> Nome do Modelo: {model.name}")
                print(f"   Descrição: {model.description}\n")

        print("===============================================")
        print("COPIE o 'Nome do Modelo' que você quer usar e cole no seu app.py.")

    except Exception as e:
        print(f"\nOcorreu um erro ao tentar listar os modelos: {e}")