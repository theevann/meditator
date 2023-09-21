import wave
import io

import numpy as np
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from voice_synthesis import synthesize_ssml, get_voice_list

TESTING = False



st.set_page_config(
    page_title="Meditator",
    page_icon=":sun:",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This the meditator app."
    }
)

languages = {
    "en-US": "English",
    "fr-FR": "French"
}

best_models = {
    "MALE": {
        "en-US": "en-US-Neural2-D",
        "fr-FR": "fr-FR-Neural2-D",
    },
    "FEMALE": {
        "en-US": "en-US-Neural2-C",
        "fr-FR": "fr-FR-Neural2-C",
    }
}

system_prompt = """You are a meditation generator. You generate a meditation based on the user's input.
- The meditation should be about {time} minutes long.
- The meditation will be read by a voice assistant, so make sure to pause for a few seconds between sentences using <break time="Xs"/>. You can only enter a time in seconds and lower than 5.
- You can also use the tag "emphasis" this way: <emphasis level="strong">This is an important announcement</emphasis> with level being one of: none, reduced, moderate, strong.
- During the meditation leave a 1 min pause for users to enjoy the state of relaxation using the tag <longpause /> as is without specifying any duration (you can duplicate it if you want a longer pause). DO NOT add the tag after bringing the user back to reality !
"""

human_prompt = "Generate a meditation using the following prompt:\n\n{user_input}\n\nMake sure to add multiple break and a long pause. Write the meditation in {language}."


@st.cache_data
def get_voices(locale, gender):
    voices = get_voice_list(locale)
    if gender:
        gender_index = 1 + (gender == "FEMALE")
        voices = [voice for voice in voices if voice.ssml_gender == gender_index]
    return voices

def generate_response(input_text, time, max_tokens):
    llm = ChatOpenAI(temperature=1, openai_api_key=st.secrets["OPENAI_API_KEY"], model=openai_model, max_tokens=max_tokens)
    
    input_messages = [
        SystemMessage(content=system_prompt.format(time=time)),
        HumanMessage(content=human_prompt.format(user_input=input_text, language=languages[locale]))
    ]

    meditation = ""
    with st.status("Generating text...", expanded=True) as status:
        placeholder = st.empty()
        for response in llm.stream(input_messages):
            meditation += response.content
            # meditation = meditation.replace(". <",".<").replace(". ",'. <break time=\"1s\" /> ')
            placeholder.markdown(meditation + "▌")
        placeholder.markdown(meditation)
        status.update(label="Meditation generation complete!", state="complete", expanded=False)

    return meditation

def concatenate_audio(audio1, audio2, pause_duration=0):
    wav_file1 = wave.open(io.BytesIO(audio1), 'rb')
    wav_file2 = wave.open(io.BytesIO(audio2), 'rb')

    params = wav_file1.getparams()

    pause_frames = int(params.framerate * pause_duration)
    silent_segment = np.zeros((pause_frames, params.sampwidth), dtype=np.uint8)
    silent_audio_byte_object = silent_segment.tobytes()

    output_audio_io = io.BytesIO()
    output_audio_header = wave.open(output_audio_io, 'wb')
    output_audio_header.setparams(params)
    output_audio_header.writeframes(wav_file1.readframes(params.nframes))
    output_audio_header.writeframes(silent_audio_byte_object)
    output_audio_header.writeframes(wav_file2.readframes(params.nframes))
    output_audio_header.close()
    
    return output_audio_io.getvalue()

def generate_voice(meditation):
    full_audio = None
    with st.status("Generating voice...", expanded=True) as status:
        meditation = meditation.replace(". <",".<").replace(". ",'. <break time=\"1s\" /> ')
        try:
            chunks = meditation.split("<longpause />")
            full_audio = synthesize_ssml("<speak>" + chunks[0] + "</speak>", model=voice_model.name, locale=locale)
            for chunk in chunks[1:]:
                audio = synthesize_ssml("<speak>" + chunk + "</speak>", model=voice_model.name, locale=locale)
                full_audio = concatenate_audio(full_audio, audio, pause_duration=60)
            status.update(label="Voice generation complete!", state="complete", expanded=False)
        except Exception as e:
            st.write(e)
            status.update(label="Voice generation failed, retry.", state="error", expanded=True)
    return full_audio


### Start of app

st.title("Meditator")

openai_model = st.sidebar.radio("Select OpenAI model:", ["gpt-3.5-turbo", "gpt-4"])
locale = st.sidebar.selectbox("Select language:", languages.keys(), format_func=lambda language: f"{languages[language]}")
gender = st.sidebar.radio("Select voice gender:", ["MALE", "FEMALE"])

voice_models = get_voices(locale, gender)
index = next((i for i, model in enumerate(voice_models) if model.name == best_models[gender][locale]), 0)

if TESTING:
    voice_model = st.sidebar.selectbox("Select voice model:", voice_models, format_func=lambda voice: f"{voice.name}", index=index)
    max_tokens = st.sidebar.slider("Max tokens:", 10, 4000, 1000, 10)
else:
    voice_model = voice_models[index]
    max_tokens = 2000

time = st.sidebar.slider("Wanted duration (in minutes):", 1, 15, 5, 1)

with st.form("my_form"):
    text = st.text_input("Optional meditation theme:", max_chars=100, placeholder="What do you want this meditation to be about ?")
    clicked = st.form_submit_button("Generate !")

    if clicked:
        st.session_state.meditation = generate_response(text, time=time, max_tokens=max_tokens)
        st.session_state.voice = generate_voice(st.session_state.meditation)

if st.session_state.get("voice", False):
    st.audio(st.session_state.voice)
    st.download_button("Download meditation", data=st.session_state.voice, file_name="meditation.wav", mime="audio/wav")