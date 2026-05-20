# 🩺 Woman Care AI - WomanGuardian

**Análise multimodal de dados médicos especializados em saúde da mulher com IA integrada**

Sistema de monitoramento preventivo que processa áudio, vídeo e texto de consultas médicas especializadas em saúde feminina para detectar sinais precoces de risco clínico em tempo real.

---

## 📋 Requisitos do Tech Challenge — Atendimento Completo

### ✅ 1. Análise de Vídeo Especializada
- **Modelo**: YOLOv8 (detecção em tempo real de pessoas e padrões corporais)
- **Indicadores analisados**:
  - Presença/ausência de pessoas em cena
  - Variação de movimento corporal (postura retraída vs. agitada)
  - Múltiplas pessoas (potencial confronto)
  - Imobilidade anormal (possível queda ou colapso)
- **Output**: Classificação de risco visual (LOW, MEDIUM, HIGH)

### ✅ 2. Análise de Áudio Especializada
- **Modelo**: MLP Neural Network treinado em dataset RAVDESS
- **Features extraídas**: MFCC, Chroma, Mel Spectrogram, Spectral Contrast, Zero Crossing Rate, RMS Energy, Spectral Centroid
- **Emoções detectadas**: Neutral, Happy, Sad, Angry, Fearful, Disgust, Surprised
- **Acurácia**: ~75%+ no dataset de validação
- **Indicadores clínicos**: Ansiedade gestacional, fadiga hormonal, depressão pós-parto

### ✅ 3. Análise de Texto (Transcrição + Sentimento)
- **Serviço**: Azure Speech-to-Text + Azure Language (Text Analytics)
- **Detecção de sinais de risco**:
  - Violência doméstica (linguagem de ameaça, medo)
  - Depressão pós-parto (fadiga, perda de interesse)
  - Ansiedade (agitação, insônia)
  - Sinais físicos anormais (dor, sangramento, tontura)
- **Output**: Sentimento + palavras-chave de risco + classificação (URGENTE/MONITORAR/ROTINA)

### ✅ 4. Integração Azure Cognitive Services
```
Áudio (.wav)
    ↓
[Transcrição Azure Speech-to-Text]
    ↓
[Análise Sentimento Azure Language]
    ↓
[Detecção Palavras-chave de Risco]
```

### ✅ 5. Fusão Multimodal com Engine de Risco
- **Scoring combinado**:
  - Emoção: 0-70 pontos (emoções negativas = mais risco)
  - Vídeo: 10-70 pontos (comportamento visual anômalo)
  - Incerteza: até 30 pontos (baixa confiança do modelo)
- **Classificação final**: URGENTE (≥120) | MONITORAR (70-119) | ROTINA (<70)

### ✅ 6. Dashboard Interativo
- Streamlit com upload de áudio + vídeo
- Visualização em tempo real de todas as análises
- Explicabilidade completa (justificativas de cada decisão)
- Indicadores visuais (barras de progresso, cores, emojis contextuais)

### ✅ 7. Documentação Técnica
Repositório Git completo com:
- Código-fonte estruturado
- Modelos treinados e serializados
- Documentação inline
- Instruções de setup
- Exemplos de resultado

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Streamlit)               │
│                  Upload: Áudio + Vídeo (Opcional)           │
└────────────────┬────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
    ┌───▼────┐        ┌──▼──────┐
    │ ÁUDIO  │        │ VÍDEO   │
    └───┬────┘        └──┬──────┘
        │                │
    ┌───▼──────────┐  ┌─▼────────────┐
    │ Emoção Local │  │ YOLOv8 Real  │
    │ (MLP)        │  │ (Pessoa Pose)│
    └───┬──────────┘  └─┬────────────┘
        │               │
    ┌───▼──────────┐    │
    │ Azure Speech │    │
    │ (Transcrição)│    │
    └───┬──────────┘    │
        │               │
    ┌───▼──────────┐    │
    │ Azure Lang   │    │
    │ (Sentimento) │    │
    └───┬──────────┘    │
        │               │
        └───┬───────────┘
            │
        ┌───▼──────────┐
        │ Risk Engine  │
        │ (Fusão)      │
        └───┬──────────┘
            │
        ┌───▼──────────┐
        │ Dashboard    │
        │ Resultado    │
        └──────────────┘
```

---

## 📊 Pipeline de Análise

### 1️⃣ Entrada
- Arquivo de áudio (WAV) — obrigatório
- Arquivo de vídeo (MP4) — opcional

### 2️⃣ Processamento Áudio
```python
# Feature Extraction (137 features)
MFCC (80) + Chroma (12) + Mel (32) + Contrast (7) + ZCR (2) + RMS (2) + Centroid (2)

# Classificação
MLP Neural Network → Emoção + Confiança

# Transcrição + Sentimento
Azure Speech → Texto
Azure Language → Sentimento + Palavras-chave
```

### 3️⃣ Processamento Vídeo
```python
# Detecção
YOLOv8 Nano → Pessoas detectadas por frame

# Análise de Padrão
- Presença humana (%)
- Movimento médio
- Agitação/Repouso

# Classificação
Regras heurísticas → LOW | MEDIUM | HIGH
```

### 4️⃣ Fusão e Decisão
```python
Score Final = Emoção_Score + Vídeo_Score + Incerteza_Penalidade

if Score ≥ 120: URGENTE
elif Score ≥ 70: MONITORAR
else: ROTINA
```

### 5️⃣ Saída
- Nível de risco (URGENTE / MONITORAR / ROTINA)
- Score numérico (0-200)
- Justificativas detalhadas
- Recomendação de ação

---

## 🚀 Instalação

### Pré-requisitos
- Python 3.8+
- Pip
- Conta Azure (para Speech + Language Services)

### Setup Rápido

```bash
# 1. Clone o repositório
git clone https://github.com/juanmqc22/TechChallenge4_WomanGuardian.git
cd TechChallenge4_WomanGuardian

# 2. Crie ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instale dependências
pip install -r requirements.txt

# 4. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas chaves Azure
```

### Chaves Azure (obtém gratuitamente no F0)
1. Acesse [portal.azure.com](https://portal.azure.com)
2. Crie **Speech** → copie Key + Region
3. Crie **Language Service** → copie Key + Endpoint
4. Adicione ao `.env`

```bash
AZURE_SPEECH_KEY=sua_chave
AZURE_SPEECH_REGION=eastus
AZURE_LANGUAGE_KEY=sua_chave
AZURE_LANGUAGE_ENDPOINT=https://seu-recurso.cognitiveservices.azure.com/
```

---

## 🎯 Uso

### Treinar Modelo de Áudio (primeira vez)
```bash
cd audio_analysis
python train_model.py
# Gera: emotion_model.pkl + scaler.pkl
```

### Executar Dashboard
```bash
streamlit run dashboard/app.py
```

Acesse: `http://localhost:8501`

### Teste Rápido
```python
from audio_analysis.final_predict import full_analysis

result = full_analysis("seu_audio.wav", "seu_video.mp4")
print(result)
```

---

## 📈 Resultados & Métricas

### Modelo de Emoção (Audio)
- **Dataset**: RAVDESS (1248 amostras, 24 atores)
- **Arquitetura**: MLP (256-128-64 neurônios)
- **Acurácia**: ~75%
- **Features**: 137 características acústicas
- **Tempo de inferência**: ~200ms por arquivo

### Análise de Vídeo (YOLOv8)
- **Modelo**: YOLOv8 Nano (pré-treinado COCO)
- **Detecções**: Pessoas, postura, movimento
- **FPS**: ~30 (real-time capable)
- **Tempo análise**: ~500ms por vídeo (amostragem)

### Azure Services
- **Transcrição**: ~5s por minuto de áudio
- **Sentimento**: ~1s por transcrição
- **Confiabilidade**: Português Brasil nativo
- **Custo**: Plano F0 (gratuito com limites)

---

## 📁 Estrutura do Projeto

```
woman_care_ai/
├── audio_analysis/
│   ├── feature_extraction.py       # Extração de 137 features
│   ├── train_model.py              # Treino do MLP
│   ├── predict.py                  # Inferência local
│   ├── azure_transcribe.py         # Azure Speech-to-Text
│   ├── azure_sentiment.py          # Azure Language + Keywords
│   ├── final_predict.py            # Orquestração
│   ├── emotion_model.pkl           # Modelo treinado
│   ├── scaler.pkl                  # Normalizador
│   └── dataset/                    # RAVDESS (1440 áudios)
│
├── video_analysis/
│   └── detect_behavior.py          # YOLOv8 + análise de padrão
│
├── risk_engine/
│   ├── risk_model.py               # Scoring + Engine de risco
│   └── fusion_engine.py            # Fusão áudio + vídeo
│
├── dashboard/
│   └── app.py                      # Streamlit UI
│
├── .env                            # Variáveis de ambiente
├── .env.example                    # Template
└── README.md                       # Este arquivo
```

---

## 🎓 Conceitos Implementados

### Processamento de Áudio
- ✅ MFCC (Mel-Frequency Cepstral Coefficients)
- ✅ Espectrograma (Mel e convencional)
- ✅ Chroma Features (conteúdo harmônico)
- ✅ Zero Crossing Rate (transições no sinal)
- ✅ Spectral Contrast (textura da voz)

### Machine Learning
- ✅ MLP Neural Network (classificação multiclasse)
- ✅ StandardScaler (normalização)
- ✅ Feature Engineering (137 features customizadas)
- ✅ Early Stopping (regularização)

### Computer Vision
- ✅ YOLOv8 (detecção em tempo real)
- ✅ Análise de movimento (delta de posição entre frames)
- ✅ Detecção de anomalias comportamentais

### Cloud Services
- ✅ Azure Cognitive Services (Speech + Language)
- ✅ Processamento multimodal
- ✅ Detecção de palavras-chave contextual

### UX/Explicabilidade
- ✅ Dashboard interativo (Streamlit)
- ✅ Visualizações de score (barra de progresso)
- ✅ Justificativas detalhadas de decisão
- ✅ Status visual (cores + emojis)

---

## 🔍 Exemplos de Detecção

### Caso 1: Depressão Pós-Parto (URGENTE)
```
Áudio: Emoção "sad" (confiança 92%)
Texto: "não consigo cuidar do bebê", "sem energia", "choro constantemente"
Vídeo: Movimento muito reduzido, postura retraída
Score: 145 → URGENTE
Ações: Alerta para obstetra + psicólogo
```

### Caso 2: Violência Doméstica (URGENTE)
```
Áudio: Emoção "fearful" (confiança 87%)
Texto: "ele me ameaçou", "tenho medo dele", "não durmo"
Vídeo: Detecção de múltiplas pessoas + agitação
Score: 165 → URGENTE
Ações: Alerta para equipe de proteção
```

### Caso 3: Monitoramento Gestacional (MONITORAR)
```
Áudio: Emoção "surprised" (confiança 78%)
Texto: Sentimento neutro, "tontura", "pressão alta"
Vídeo: Presença normal, movimento reduzido (esperado)
Score: 85 → MONITORAR
Ações: Monitoramento contínuo de sinais vitais
```

### Caso 4: Consulta de Rotina (ROTINA)
```
Áudio: Emoção "neutral" (confiança 89%)
Texto: Sentimento positivo, sem palavras de risco
Vídeo: Presença humana normal, movimento esperado
Score: 28 → ROTINA
Ações: Nenhuma ação especial
```

---

## 🔐 Privacidade & Segurança

- ✅ Arquivos temporários deletados após processamento
- ✅ Sem armazenamento de áudio/vídeo no servidor
- ✅ Azure services com criptografia end-to-end
- ✅ Senhas nunca commitadas (`.env` + `.gitignore`)
---


- **Autores**: Juan Quezada (juanmqc22)
- **Instituição**: FIAP - Pós-Graduação em IA para Devs

---

## ✅ Checklist de Conformidade com Tech Challenge

- [x] Análise de Vídeo especializada (YOLOv8)
- [x] Análise de Áudio especializada (MLP + Features ricas)
- [x] Processamento multimodal (fusão)
- [x] Integração com Azure Cognitive Services
- [x] Engine de risco com scoring
- [x] Dashboard interativo
- [x] Relatório técnico (README)
- [x] Código-fonte no Git
- [x] Vídeo de demonstração (15 min) 

