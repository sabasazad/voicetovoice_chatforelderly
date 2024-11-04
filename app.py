# Set up Groq API client
os.environ['GROQ_API_KEY'] = "" 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load the Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Initialize an emotion detection model
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Function to transcribe the audio input
def transcribe(audio_path):
    transcription = whisper_model.transcribe(audio_path)
    return transcription['text']

# Function to detect emotion from transcribed text
def detect_emotion(text):
    emotion_results = emotion_analyzer(text)
    # Filter for the highest-scoring emotion
    emotion = max(emotion_results, key=lambda x: x['score'])
    return emotion['label']

# Function to interact with Groq's LLM and get a response
def interact_with_llm(text, emotion):
    # Include the detected emotion in the prompt for more context-aware responses
    prompt = f"User emotion: {emotion}. Respond to the user's message: {text}"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to convert the text response into audio using gTTS
def text_to_speech(text):
    tts = gTTS(text)
    output_audio_path = "response.mp3"
    tts.save(output_audio_path)
    return output_audio_path

# Complete pipeline function: audio input -> transcription -> emotion detection -> LLM response -> audio output
def chatbot_pipeline(audio):
    # Step 1: Transcribe audio
    transcription = transcribe(audio)
    print(f"Transcription: {transcription}")

    # Step 2: Detect emotion
    emotion = detect_emotion(transcription)
    print(f"Detected Emotion: {emotion}")

    # Step 3: Generate response using LLM with emotion context
    response = interact_with_llm(transcription, emotion)
    print(f"LLM Response: {response}")

    # Step 4: Convert response to speech
    audio_output = text_to_speech(response)
    
    # Return transcription, detected emotion, and response audio
    return transcription, emotion, audio_output

# Gradio interface for real-time interaction
iface = gr.Interface(
    fn=chatbot_pipeline,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="Transcription"),
        gr.Textbox(label="Detected Emotion"),
        gr.Audio(label="Chatbot Response (Audio)")
    ]
)

# Launch the Gradio app
iface.launch()
