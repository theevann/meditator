import wave
import io
import re
import base64

import numpy as np
import streamlit as st
from streamlit.components.v1 import html

from voice_synthesis import synthesize_ssml, get_voice_list
from text_generation import generate_text_v1


TESTING = st.secrets["TESTING"]


st.set_page_config(
    page_title="Meditator",
    page_icon=":sun:",
    layout="centered",
    initial_sidebar_state="expanded" if TESTING else "collapsed",
    menu_items={
        'Report a bug': 'https://github.com/theevann/meditator/issues',
        'About': "https://github.com/theevann/meditator/"
    }
)

st.markdown(
    f"""
     <style>
     body .stApp {{
         // background: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
         background: url(data:image/png;base64,{base64.b64encode(open("images/background.png", "rb").read()).decode()});
         background-repeat: no-repeat;
         background-position: right 50% bottom 95%;
         background-size: cover;
         background-attachment: scroll;
     }}
     .stApp > div > section:first-child {{
         background-color: rgba(240, 242, 246, 0.7);
     }}
     .main > .block-container {{
            background-color: rgba(255, 255, 255, 0.65);
            border-radius: 10px;
            margin-top: 3.5em;
            padding-top: 1em;
            padding-bottom: 1em;
     }}
     header {{
            // background-color: rgba(176, 191, 214, 1)!important;
            background-color: rgba(255, 255, 255, 0.5)!important;
     }}
     </style>
     """,
    unsafe_allow_html=True,
)


languages = {
    "en-US": "English (US)",
    "en-GB": "English (GB)",
    "fr-FR": "French",
    "es-ES": "Spanish",
    "it-IT": "Italian",
    "de-DE": "German",
    "ru-RU": "Russian",
}

best_models = {
    "MALE": {
        "en-US": "en-US-Neural2-D",
        "en-GB": "en-GB-Neural2-D",
        "fr-FR": "fr-FR-Neural2-D",
        "ru-RU": "ru-RU-Wavenet-D",
    },
    "FEMALE": {
        "en-US": "en-US-Neural2-C",
        "en-GB": "en-GB-Neural2-C",
        "fr-FR": "fr-FR-Neural2-C",
        "ru-RU": "ru-RU-Wavenet-C",
    }
}

musics = {
    "No Music": [None, 0],
    "Deep Sound": ["musics/music_1.wav", 0.5],
    "Water Sound": ["musics/music_2.wav", 0.25],
}

safari_warning_html = """
<div id="safari-warning" style="display: none;">
<div style="background-color: #ffe3121a; color: #926c05; padding: 10px; border-radius: 5px; font-style: italic; font-family: 'Source Sans Pro'">
    <b>Warning:</b> In Safari, the audio player is sometime throwing an error, please use another browser or download the meditation.
</div>
</div>
<script>
var isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
if (isSafari) {
document.getElementById("safari-warning").style.display = "block";
}
</script>
"""


@st.cache_data
def get_voices(locale, gender):
    voices = get_voice_list(locale)
    if gender:
        gender_index = 1 + (gender == "FEMALE")
        voices = [voice for voice in voices if voice.ssml_gender == gender_index]
    return voices


def superpose_music(meditation, music_path, mix_ratio=0.5):
    wav_file1 = wave.open(io.BytesIO(meditation), 'rb')
    wav_file2 = wave.open(music_path, 'rb')

    params_1 = wav_file1.getparams()
    params_2 = wav_file2.getparams()

    frames1 = wav_file1.readframes(params_1.nframes)
    frames2 = wav_file2.readframes(params_2.nframes)

    frames1 = np.frombuffer(frames1, dtype=np.int16)
    frames2 = np.frombuffer(frames2, dtype=np.int16)[:len(frames1)]

    frames1 = frames1.astype(np.int32)
    frames2 = frames2.astype(np.int32)

    frames = mix_ratio * frames1 + (1-mix_ratio) * frames2
    frames = frames.astype(np.int16).tobytes()

    output_audio_io = io.BytesIO()
    output_audio_header = wave.open(output_audio_io, 'wb')
    output_audio_header.setparams(params_1)
    output_audio_header.writeframes(frames)
    output_audio_header.close()

    wav_file1.close()
    wav_file2.close()
    
    return output_audio_io.getvalue()


def concatenate_audio(audio1, audio2, pause_duration=0):
    wav_file1 = wave.open(io.BytesIO(audio1), 'rb')
    wav_file2 = wave.open(io.BytesIO(audio2), 'rb')

    params = wav_file1.getparams()

    pause_frames = int(params.framerate * pause_duration)
    silent_segment = np.zeros((pause_frames, params.sampwidth), dtype=np.uint8)
    silent_audio_byte_object = silent_segment.tobytes()

    output_audio_io = io.BytesIO()
    with wave.open(output_audio_io, 'wb') as output_audio_header:
        output_audio_header.setparams(params)
        output_audio_header.writeframes(wav_file1.readframes(params.nframes))
        output_audio_header.writeframes(silent_audio_byte_object)
        output_audio_header.writeframes(wav_file2.readframes(params.nframes))
    
    return output_audio_io.getvalue()


def generate_audio(meditation, music, sentence_break_time, speaking_rate):
    full_audio = None
    with st.status("Generating voice...", expanded=True) as status:
        meditation = meditation.replace(". <",".<").replace(". ",f". <break time=\"{sentence_break_time}s\" /> ")
        try:
            pattern = r'\[PAUSE=(\d+)\]'
            split_text = re.split(pattern, meditation)
            times = [int(x) for x in split_text[1::2]]
            chunks = split_text[::2]
            
            ph = st.empty()
            ph.checkbox(f"Generating voice... 1 / {len(chunks)}", False, disabled=True, )
            full_audio = synthesize_ssml("<speak>" + chunks[0] + "</speak>", model=voice_model.name, locale=locale, speaking_rate=speaking_rate)
            ph.checkbox(f"Generating voice... 1 / {len(chunks)}", True, disabled=True)
            for i, (chunk, time) in enumerate(zip(chunks[1:], times)):
                ph = st.empty()
                ph.checkbox(f"Generating voice... {i+2} / {len(chunks)}", False, disabled=True)
                audio = synthesize_ssml("<speak>" + chunk + "</speak>", model=voice_model.name, locale=locale, speaking_rate=speaking_rate)
                full_audio = concatenate_audio(full_audio, audio, pause_duration=time)
                ph.checkbox(f"Generating voice... {i+2} / {len(chunks)}", True, disabled=True)

            if music[0]:
                ph = st.empty()
                ph.checkbox(f"Adding Music...", False, disabled=True)
                full_audio = superpose_music(full_audio, music[0], music[1])
                ph.checkbox(f"Adding Music...", True, disabled=True)

            status.update(label="Voice generation complete!", state="complete", expanded=False)
        except Exception as e:
            st.write(e)
            status.update(label="Voice generation failed, retry.", state="error", expanded=True)
            if TESTING:
                raise e
    return full_audio


### Start of app

st.title("Meditator")




with st.form("my_form"):
    col1, col2, col3 = st.columns([1, 1, 1])
    gender = col3.radio("Select voice gender:", ["MALE", "FEMALE"])
    locale = col1.selectbox("Select language:", languages.keys(), format_func=lambda language: f"{languages[language]}")
    music = col2.selectbox("Select music:", list(musics.keys()), format_func=lambda music: f"{music}", index=1)
    text = st.text_input("Optional meditation theme:", max_chars=100, placeholder="What do you want this meditation to be about ?", help="You can write for instance: Relaxation, Sleep, Peace, Joy, Sounds, Inner Child, etc.")
    clicked = st.form_submit_button("Generate !", type="primary", use_container_width=True)

with st.sidebar:
    #st.sidebar.expander("Advanced options", expanded=True):
    st.title("Advanced options")
    st.header("Text generation")
    openai_model = st.radio("Select OpenAI model:", ["gpt-3.5-turbo", "gpt-4"])
    time = st.slider("Wanted duration (in minutes):", 1, 15, 5, 1)
    max_tokens = st.slider("Max tokens:", 10, 4000, 100, 10) if TESTING else 2000
    st.markdown("---")
    st.header("Voice generation")
    voice_models = get_voices(locale, gender)
    index = next((i for i, model in enumerate(voice_models) if model.name == best_models[gender].get(locale, "")), 0)
    voice_model = st.selectbox("Select voice model:", voice_models, format_func=lambda voice: f"{voice.name}", index=index)
    sentence_break_time = st.slider("Sentence break time (in seconds):", 1., 5., 1.5, 0.5)
    speaking_rate = st.slider("Speaking rate:", 0.5, 1.0, 0.75, 0.01)

if clicked:
    # st.session_state.meditation = "Welcome to this guided meditation. Find a comfortable position, whether it's sitting on a cushion, chair, or lying down. Close your eyes and take a deep breath in, [PAUSE=12] savoring the sensation of the air [PAUSE=12] filling your lungs. And now, gently exhale, releasing any tension or stress that you may be holding onto."
    st.session_state.meditation = generate_text_v1(text, time=time, max_tokens=max_tokens, model=openai_model, language=languages[locale])
    st.session_state.voice = generate_audio(st.session_state.meditation, musics[music], sentence_break_time, speaking_rate)


if st.session_state.get("voice", False):
    st.audio(st.session_state.voice, format="audio/wav")
    st.download_button("Download meditation", data=st.session_state.voice, file_name="meditation.wav", mime="audio/wav", use_container_width=True)
    html(safari_warning_html, height=70)