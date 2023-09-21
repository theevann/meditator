from streamlit import secrets
from google.oauth2 import service_account
from google.cloud import texttospeech

# Create API client.
credentials = service_account.Credentials.from_service_account_info(secrets["gcp_service_account"])

def get_voice_list(language_code):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    voices = client.list_voices().voices
    voices = [voice for voice in voices if voice.language_codes[0].startswith(language_code)]
    return voices


def synthesize_ssml(ssml, model="en-US-Standard-C", locale="en-US"):
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    input_text = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code=locale, name=model)
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=0.85)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)
    return response.audio_content


if __name__ == "__main__":
    response = synthesize_ssml("<speak>Hello there.</speak>")
    with open("output.mp3", "wb") as out:
        out.write(response)