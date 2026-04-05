#!/usr/bin/env python3
"""
Test rapide d'Ollama avec Mistral
"""
from langchain_ollama import ChatOllama

def test_ollama():
    try:
        llm = ChatOllama(
            model="mistral",
            base_url="http://localhost:11500",
            temperature=0.7
        )

        response = llm.invoke("Bonjour, peux-tu me dire en une phrase ce que tu es ?")
        print("✅ Ollama fonctionne !")
        print(f"Réponse : {response.content}")
        return True
    except Exception as e:
        print(f"❌ Erreur Ollama : {e}")
        return False

if __name__ == "__main__":
    test_ollama()