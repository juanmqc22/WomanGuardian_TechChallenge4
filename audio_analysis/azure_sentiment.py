from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os


AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")  # ex: "https://seu-recurso.cognitiveservices.azure.com/"


# Palavras-chave de risco clínico para saúde da mulher
RISK_KEYWORDS = {
    "violencia": ["bater", "machucar", "medo", "ameaça", "fugir", "socorro", "agredida", "violência"],
    "depressao_pos_parto": ["não consigo cuidar", "não me sinto mãe", "choro muito", "não quero sair", "sem energia"],
    "ansiedade": ["nervosa", "angustiada", "sufocando", "coração acelerado", "não consigo dormir"],
    "dor": ["dor", "sangramento", "cólica", "tontura", "desmaio"],
}


def _get_client():
    if not AZURE_LANGUAGE_KEY or not AZURE_LANGUAGE_ENDPOINT:
        return None
    credential = AzureKeyCredential(AZURE_LANGUAGE_KEY)
    return TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, credential=credential)


def analyze_sentiment(transcript: str) -> dict:
    """
    Analisa o sentimento do texto transcrito via Azure Text Analytics.
    Também detecta palavras-chave de risco clínico.
    """

    if not transcript or transcript.strip() == "":
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "riskKeywordsFound": [],
            "textRisk": "INDETERMINADO"
        }

    client = _get_client()

    if client is None:
        return {"error": "Credenciais Azure Language não configuradas"}

    try:
        print(f"[Azure Sentiment] Analisando texto: {transcript[:80]}...")

        response = client.analyze_sentiment(
            documents=[{"id": "1", "language": "pt", "text": transcript}]
        )

        result = response[0]

        sentiment = result.sentiment  # "positive", "neutral", "negative", "mixed"
        scores = result.confidence_scores

        confidence_map = {
            "positive": scores.positive,
            "neutral": scores.neutral,
            "negative": scores.negative,
            "mixed": max(scores.positive, scores.negative)
        }

        confidence = confidence_map.get(sentiment, 0.0)

        # -----------------------------
        # DETECÇÃO DE PALAVRAS DE RISCO
        # -----------------------------
        transcript_lower = transcript.lower()
        risk_keywords_found = []

        for category, keywords in RISK_KEYWORDS.items():
            for kw in keywords:
                if kw in transcript_lower:
                    risk_keywords_found.append({
                        "category": category,
                        "keyword": kw
                    })

        # -----------------------------
        # NÍVEL DE RISCO TEXTUAL
        # -----------------------------
        has_violence = any(r["category"] == "violencia" for r in risk_keywords_found)
        has_depression = any(r["category"] == "depressao_pos_parto" for r in risk_keywords_found)

        if has_violence or (sentiment == "negative" and confidence > 0.85):
            text_risk = "URGENTE"
        elif has_depression or sentiment == "negative":
            text_risk = "MONITORAR"
        elif len(risk_keywords_found) > 0:
            text_risk = "MONITORAR"
        else:
            text_risk = "ROTINA"

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "positiveScore": float(scores.positive),
            "negativeScore": float(scores.negative),
            "neutralScore": float(scores.neutral),
            "riskKeywordsFound": risk_keywords_found,
            "textRisk": text_risk
        }

    except Exception as e:
        return {
            "sentiment": "unknown",
            "confidence": 0.0,
            "riskKeywordsFound": [],
            "textRisk": "INDETERMINADO",
            "error": str(e)
        }


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    test_text = "Estou com muito medo, ele me ameaçou ontem e não consigo dormir."
    result = analyze_sentiment(test_text)
    print(result)