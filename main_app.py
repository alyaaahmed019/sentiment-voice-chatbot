import streamlit as st
from modules.data_preprocessor import DataPreprocessor
from modules.embeddings import Embedder
from modules.emotion_detection import EmotionDetector
from modules.text_to_speech import speak_text

def main():
    # Sidebar settings
    st.sidebar.title("⚙️ Settings")

    # Load dataset and preprocess (only once, cache with session_state)
    if "df" not in st.session_state:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        df = preprocessor.preprocess()

        st.session_state.df = df
        st.session_state.embedder = Embedder(df["text"].tolist())
        st.session_state.detector = EmotionDetector()

    embedder = st.session_state.embedder
    detector = st.session_state.detector

    # App UI
    st.title("🎤 EmpathyBot: Sentiment-Driven Voice Chat")

    user_input = st.text_area("📝 Type your message here:")

    if st.button("Analyze"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            retrieved = embedder.retrieve(user_input, k=3)
            emotion = detector.classify(user_input, retrieved)

            st.success(f"Detected Emotion: **{emotion.upper()}**")

            st.write("📚 Retrieved examples:")
            for i, (doc, dist) in enumerate(retrieved, 1):
                st.write(f"{i}. {doc} (distance={dist:.4f})")

            st.audio(speak_text(f"The detected emotion is {emotion}"), format="audio/mp3")

# This makes sure Streamlit runs main()
if __name__ == "__main__":
    main()
