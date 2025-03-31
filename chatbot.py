import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os

# Load the pre-trained emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Emotion-based responses
responses = {
    "joy": "I'm glad to hear that! 😊",
    "sadness": "I'm here for you. Things will get better. 💙",
    "anger": "I understand. Take a deep breath. Let's talk about it. 😌",
    "enthusiasm": "I love your energy! Let's go! 🚀",
    "neutral": "That's interesting. I’d love to hear more! 🤖",
    "fear": "I'm here if you want to talk. Stay strong! 💪",
    "surprise": "Wow, that’s surprising! Tell me more! 😲"
}

def detect_emotion(text):
    """Function to detect emotion from text."""
    try:
        result = emotion_classifier(text)
        if result and isinstance(result, list) and "label" in result[0]:  
            label = result[0]["label"]
        else:
            label = "neutral"  

        return responses.get(label, "That’s an interesting point! Could you elaborate? 🤖")

    except Exception as e:
        return "Error in emotion detection. Try again!"

def speak_response(text):
    """Generate speech from text using gTTS and play in Streamlit."""
    try:
        tts = gTTS(text=text, lang="en")
        audio_file = "response.mp3"
        tts.save(audio_file)  # Save audio file

        # Streamlit audio player
        st.audio(audio_file, format="audio/mp3")

    except Exception as e:
        st.error("Error in generating speech.")

def get_voice_input():
    """Function to get voice input and convert to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Could not request results. Please check your internet connection."

# Streamlit UI
st.title("🤖 Voice-Enabled Emotion-Aware Chatbot")
st.write("Type or speak a message, and I'll respond based on your emotions!")

# Initialize session state for user input
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

user_input = st.text_input("Type here:", key="text_input", value=st.session_state.user_input)

if st.button("🎤 Speak"):
    voice_input = get_voice_input()
    if voice_input:
        st.session_state.user_input = voice_input
        st.text(f"You said: {voice_input}")

if user_input:
    st.session_state.user_input = user_input

if st.session_state.user_input.strip():
    bot_response = detect_emotion(st.session_state.user_input)
    st.text_area("Chatbot:", value=bot_response, height=100, max_chars=None)

    # Generate and play speech response
    speak_response(bot_response)
