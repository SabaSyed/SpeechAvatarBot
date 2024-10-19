# Avatar Chatbot Project

## Description

This project creates an interactive voice-based chatbot with a visual avatar. The bot listens to user input, generates responses using the LLaMA model, and speaks responses out loud using eSpeak. The avatar's visual behavior switches between idle and speaking modes depending on the bot's activity.

## Features

- **Voice Input**: Users can speak into the microphone to interact with the bot.
- **Response Generation**: LLaMA model generates conversational and concise responses.
- **Voice Output**: Responses are spoken aloud using eSpeak.
- **Visual Avatar**: A video of an avatar is displayed, switching between idle and speaking modes during interactions.
- **Vosk Model**: Speech-to-text functionality is powered by the Vosk model, providing real-time voice input recognition.

## Requirements

### Software Dependencies:

- Python 3.8+
- eSpeak
- Vosk
- Pyaudio
- PyGame
- PyAv
- Ollama (LLaMA)

### Install Required Python Libraries:

```bash
pip install -r requirements.txt
```

### Install eSpeak:

- **Linux**:
  ```bash
  sudo apt-get install espeak
  ```

- **Windows**:
  Download and install eSpeak from the [official site](http://espeak.sourceforge.net/).

### Install Ollama and LLaMA Model

1. **Download Ollama**:
   - Visit [Ollama's download page](https://ollama.com/downloads) to download the appropriate version for your operating system.

2. **Install Ollama**:
   - **macOS**:
     ```bash
     brew install ollama/tap/ollama
     ```
   - **Windows/Linux**:
     Follow the installation guide after downloading the binary from the [Ollama download page](https://ollama.com/downloads).

3. **Download the LLaMA 3.2 model**:
   After installing Ollama, download the LLaMA 3.2 model by running:
   ```bash
   ollama pull llama3.2
   ```

4. **Start Ollama**:
   To use Ollama in the project, you need to start the Ollama server:
   ```bash
   ollama serve
   ```

## How to Run

1. **Clone the project**:
   ```bash
   git clone <https://github.com/SabaSyed/SpeechAvatarBot>
   cd <project-directory>
   ```

2. **Ensure all dependencies are installed** (see requirements above).

3. **Run the `app.py` script**:
   ```bash
   python app.py
   ```

4. **Interaction**:
   - The bot will listen for user input.
   - It will generate a response using LLaMA and speak it using eSpeak.
   - The avatar's video will switch between idle and speaking videos based on activity.

## Project Structure

```
├── idle.mp4                # Idle video for avatar
├── speaking.mp4            # Speaking video for avatar
├── app.py                  # Main script for chatbot functionality
├── requirements.txt        # File containing python libraries to be installed
└── README.md               # Project documentation
```

## Known Issues

- Ensure that the audio device is correctly configured for Vosk to capture input and for eSpeak to produce audio.
- The performance may vary depending on system resources, especially with LLaMA model response generation.
- Make sure that the Ollama server is running before starting the bot.

## Acknowledgments

- **Vosk** for speech-to-text recognition.
- **Ollama's LLaMA** for response generation.
- **eSpeak** for quick and efficient text-to-speech conversion.
