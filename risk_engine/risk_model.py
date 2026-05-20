def classify_risk(emotion, videoRisk, confidence):

    # -----------------------------
    # EMOTION RISK MAP
    # -----------------------------
    emotion_risk_map = {
        "neutral": 5,
        "happy": 0,
        "surprised": 20,
        "sad": 40,
        "disgust": 50,
        "fearful": 70,
        "angry": 60
    }

    # -----------------------------
    # VIDEO RISK MAP
    # -----------------------------
    video_risk_map = {
        "LOW": 10,
        "MEDIUM": 40,
        "HIGH": 70
    }

    base_score = emotion_risk_map.get(emotion, 20)
    video_score = video_risk_map.get(videoRisk, 20)

    uncertainty_penalty = (1 - confidence) * 30

    final_score = base_score + video_score + uncertainty_penalty


    # -----------------------------
    # EXPLANATION ENGINE
    # -----------------------------
    reasons = []

    if emotion in ["sad", "fearful", "angry", "disgust"]:
        reasons.append(f"Emoção detectada: {emotion}")

    if videoRisk in ["MEDIUM", "HIGH"]:
        reasons.append(f"Comportamento visual de risco: {videoRisk}")

    if confidence < 0.5:
        reasons.append("Baixa confiança do modelo de áudio")

    if uncertainty_penalty > 15:
        reasons.append("Alta incerteza na análise emocional")


    # -----------------------------
    # DECISÃO FINAL
    # -----------------------------
    if final_score >= 120:
        return {
            "riskLevel": "URGENTE",
            "score": float(final_score),
            "humanReviewRequired": True,
            "reasons": reasons
        }

    elif final_score >= 70:
        return {
            "riskLevel": "MONITORAR",
            "score": float(final_score),
            "humanReviewRequired": True,
            "reasons": reasons
        }

    else:
        return {
            "riskLevel": "ROTINA",
            "score": float(final_score),
            "humanReviewRequired": False,
            "reasons": reasons
        }