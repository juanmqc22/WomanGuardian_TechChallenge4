import streamlit as st
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from audio_analysis.final_predict import full_analysis


# -------- CONFIG --------
st.set_page_config(
    page_title="Woman Care AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Woman Care AI - Risk Analysis System")
st.caption("Análise multimodal: Áudio + Vídeo + Azure Speech + Sentiment")

st.divider()

# -------- UPLOADS --------
audio_file = st.file_uploader("🎙️ Áudio (.wav)", type=["wav"])
video_file = st.file_uploader("🎥 Vídeo (.mp4) — opcional", type=["mp4"])

if audio_file is not None:

    audio_path = f"temp_{audio_file.name}"
    with open(audio_path, "wb") as f:
        f.write(audio_file.read())

    st.audio(audio_file)

    video_path = None
    if video_file is not None:
        video_path = f"temp_{video_file.name}"
        with open(video_path, "wb") as f:
            f.write(video_file.read())
        st.video(video_file)

    with st.spinner("🔍 Analisando com Azure + YOLOv8 + Modelo Local..."):
        result = full_analysis(audio_path, video_path)

    # Limpeza
    try:
        os.remove(audio_path)
        if video_path:
            os.remove(video_path)
    except:
        pass

    if "error" in result:
        st.error(result["error"])
        st.stop()

    # -------- ALERTA PRINCIPAL --------
    risk = result.get("riskLevel", "ROTINA")

    if risk == "URGENTE":
        st.error("🚨 CASO URGENTE — Revisão humana obrigatória")
    elif risk == "MONITORAR":
        st.warning("⚠️ Caso de atenção — Monitoramento recomendado")
    else:
        st.success("✔ Caso de rotina")

    st.divider()

    # -------- TRANSCRIÇÃO --------
    st.subheader("🎙️ Transcrição (Azure Speech-to-Text)")
    transcript = result.get("transcript", "")
    if transcript:
        st.info(f'"{transcript}"')
    else:
        st.caption("Nenhuma fala detectada.")

    st.divider()

    # -------- SENTIMENTO --------
    st.subheader("💬 Sentimento (Azure Language)")

    col1, col2 = st.columns(2)
    sentiment_label = {
        "positive": "😊 Positivo", "neutral": "😐 Neutro",
        "negative": "😢 Negativo", "mixed": "😕 Misto", "unknown": "❓ Indeterminado"
    }
    with col1:
        st.metric("Sentimento", sentiment_label.get(result.get("sentiment"), "—"))
    with col2:
        st.metric("Confiança", f"{result.get('sentimentConfidence', 0):.0%}")

    keywords = result.get("riskKeywordsFound", [])
    if keywords:
        st.markdown("**🔍 Palavras de risco:**")
        for kw in keywords:
            st.markdown(f"- `{kw['keyword']}` → _{kw['category']}_")

    st.divider()

    # -------- EMOÇÃO --------
    st.subheader("🎵 Emoção no Áudio (Modelo Local)")

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Emoção", result.get("emotion", "N/A").capitalize())
    with col4:
        st.metric("Confiança", f"{result.get('confidence', 0):.0%}")

    st.divider()

    # -------- VÍDEO (YOLOv8) --------
    if video_file is not None:
        st.subheader("🎥 Análise de Vídeo (YOLOv8)")

        video_details = result.get("videoDetails", {})
        col5, col6, col7 = st.columns(3)

        with col5:
            st.metric("Risco Visual", result.get("videoRisk", "—"))
        with col6:
            st.metric("Frames Analisados", video_details.get("framesAnalyzed", "—"))
        with col7:
            st.metric("Pessoas Detectadas", video_details.get("avgPersonsDetected", "—"))

        video_reasons = video_details.get("reasons", [])
        if video_reasons:
            st.markdown("**⚠️ Alertas visuais:**")
            for r in video_reasons:
                st.markdown(f"- {r}")

        st.divider()

    # -------- SCORE EXPLICADO --------
    st.subheader("📊 Risco Final (Fusão Multimodal)")

    score = result.get("score", 0)
    risk_level = result.get("riskLevel", "ROTINA")

    # Barra visual do score
    progress_value = min(score / 200, 1.0)  # Normaliza para 0-1
    st.progress(progress_value, text=f"Score: {score:.1f}/200")

    # Explicação do score
    st.markdown("**Como o score é calculado:**")
    st.markdown("""
    - **Emoção**: Adiciona pontos baseado na emoção detectada (neutral=5, happy=0, sad=40, angry=60, fearful=70, etc)
    - **Vídeo**: Adiciona pontos baseado no risco visual (LOW=10, MEDIUM=40, HIGH=70)
    - **Incerteza**: Penalidade se o modelo tiver baixa confiança (até 30 pontos)
    - **Total**: Soma das três componentes (máximo teórico: ~200 pontos)
    
    **Classificação Final:**
    - **URGENTE**: Score ≥ 120 (requer ação imediata)
    - **MONITORAR**: Score 70-119 (vigilância recomendada)
    - **ROTINA**: Score < 70 (sem alerta)
    """)

    col_score_1, col_score_2 = st.columns(2)
    with col_score_1:
        st.metric("Nível de Risco", risk_level, delta=f"{score:.1f} pts")
    with col_score_2:
        # Cor dinamicamente
        if risk_level == "URGENTE":
            st.metric("Status", "🚨 Crítico")
        elif risk_level == "MONITORAR":
            st.metric("Status", "⚠️ Alerta")
        else:
            st.metric("Status", "✅ Normal")

    st.divider()

    # -------- JUSTIFICATIVAS --------
    st.subheader("🧠 Justificativa Detalhada")

    reasons = result.get("reasons", [])
    if reasons:
        st.markdown("**Fatores que influenciaram a decisão:**")
        for i, reason in enumerate(reasons, 1):
            st.markdown(f"**{i}.** {reason}")
    else:
        st.success("Nenhum fator de risco detectado.")

    st.divider()

    # -------- RODAPÉ -------- 
    st.caption("⚠️ Esta ferramenta é um auxílio diagnóstico. Decisões clínicas finais devem ser tomadas por profissionais de saúde qualificados.")