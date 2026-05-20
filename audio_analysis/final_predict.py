import sys
import os

BASE_DIR = os.path.dirname(__file__)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from audio_analysis.predict import predict_emotion
from risk_engine.risk_model import classify_risk
from audio_analysis.azure_transcribe import transcribe_audio
from audio_analysis.azure_sentiment import analyze_sentiment
from video_analysis.detect_behavior import analyze_video


def full_analysis(audio_path: str, video_path: str = None) -> dict:

    # -----------------------------
    # 1. EMOÇÃO (modelo local)
    # -----------------------------
    emotion_result = predict_emotion(audio_path)

    if emotion_result is None or "error" in emotion_result:
        return {"error": "Falha ao processar áudio"}

    emotion = emotion_result["emotion"]
    confidence = emotion_result["confidence"]

    # -----------------------------
    # 2. TRANSCRIÇÃO AZURE SPEECH
    # -----------------------------
    transcription_result = transcribe_audio(audio_path)
    transcript = transcription_result.get("transcript", "")

    # -----------------------------
    # 3. SENTIMENTO AZURE
    # -----------------------------
    sentiment_result = analyze_sentiment(transcript)
    text_risk = sentiment_result.get("textRisk", "INDETERMINADO")

    # -----------------------------
    # 4. ANÁLISE DE VÍDEO (YOLOv8)
    # -----------------------------
    video_result = {}
    video_risk = "LOW"

    if video_path:
        video_result = analyze_video(video_path)
        video_risk = video_result.get("visualRisk", "LOW")

    # -----------------------------
    # 5. RISK ENGINE (fusão completa)
    # -----------------------------
    risk_result = classify_risk(emotion, video_risk, confidence)

    final_risk = _merge_risks(
        risk_result["riskLevel"],
        text_risk,
        video_result.get("visualRisk", "LOW")
    )

    # Consolida razões de todos os canais
    reasons = risk_result.get("reasons", [])

    if sentiment_result.get("sentiment") == "negative":
        reasons.append(f"Sentimento negativo no texto (confiança: {sentiment_result.get('confidence', 0):.0%})")

    for kw in sentiment_result.get("riskKeywordsFound", []):
        reasons.append(f"Palavra de risco: '{kw['keyword']}' ({kw['category']})")

    for r in video_result.get("reasons", []):
        reasons.append(f"[Vídeo] {r}")

    # -----------------------------
    # 6. RETORNO FINAL
    # -----------------------------
    return {
        "emotion": emotion,
        "confidence": confidence,
        "transcript": transcript,
        "transcriptionSuccess": transcription_result.get("success", False),
        "sentiment": sentiment_result.get("sentiment", "unknown"),
        "sentimentConfidence": sentiment_result.get("confidence", 0.0),
        "riskKeywordsFound": sentiment_result.get("riskKeywordsFound", []),
        "videoRisk": video_risk,
        "videoDetails": video_result,
        "riskLevel": final_risk,
        "score": risk_result.get("score", 0),
        "humanReviewRequired": final_risk in ["URGENTE", "MONITORAR"],
        "reasons": reasons
    }


def _merge_risks(audio_risk: str, text_risk: str, video_risk_raw: str) -> str:
    priority = {"URGENTE": 3, "MONITORAR": 2, "ROTINA": 1, "INDETERMINADO": 0}

    video_map = {"HIGH": "URGENTE", "MEDIUM": "MONITORAR", "LOW": "ROTINA"}
    video_risk = video_map.get(video_risk_raw, "ROTINA")

    return max(audio_risk, text_risk, video_risk, key=lambda r: priority.get(r, 0))