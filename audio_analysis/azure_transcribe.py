import azure.cognitiveservices.speech as speechsdk
import os


AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")


def transcribe_audio(file_path: str) -> dict:
    """
    Transcreve áudio COMPLETO usando reconhecimento contínuo.
    Sem limite de duração.
    """

    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        return {"error": "Credenciais Azure Speech não configuradas"}

    try:
        speech_config = speechsdk.SpeechConfig(
            subscription=AZURE_SPEECH_KEY,
            region=AZURE_SPEECH_REGION
        )

        speech_config.speech_recognition_language = "pt-BR"

        audio_config = speechsdk.audio.AudioConfig(filename=file_path)

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        print(f"[Azure Speech] Transcrevendo arquivo completo: {file_path}")

        all_text = []
        done = False

        def on_recognized(evt):
            """Callback quando frases são reconhecidas"""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                print(f"[Azure] Frase: {evt.result.text}")
                all_text.append(evt.result.text)

        def on_session_stopped(evt):
            """Callback quando sessão termina"""
            nonlocal done
            print("[Azure] Transcrição completa")
            done = True

        def on_canceled(evt):
            """Callback para erros"""
            cancellation = evt.result.cancellation_details
            print(f"[Azure ERROR] {cancellation.reason}: {cancellation.error_details}")
            nonlocal done
            done = True

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_session_stopped)
        recognizer.canceled.connect(on_canceled)

        # Inicia reconhecimento contínuo
        recognizer.start_continuous_recognition()

        while not done:
            import time
            time.sleep(0.1)

        recognizer.stop_continuous_recognition()

        transcript = " ".join(all_text).strip()

        if transcript:
            return {
                "transcript": transcript,
                "success": True
            }
        else:
            return {
                "transcript": "",
                "success": False,
                "warning": "Nenhuma fala detectada no áudio"
            }

    except Exception as e:
        return {
            "transcript": "",
            "success": False,
            "error": f"Exceção ao transcrever: {str(e)}"
        }


if __name__ == "__main__":
    test_file = "sample.wav"
    result = transcribe_audio(test_file)
    print(result)