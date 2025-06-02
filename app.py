import os
import sys
import time
import json
import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import ollama
from datetime import datetime
import cv2
import re
import emoji
import vosk
import pyaudio
import threading
import subprocess 
from word2number import w2n
import dateparser
from fuzzywuzzy import process 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import string

ollama_model = "daanturo/T145-ZEUS-8B-V2-abliterated:latest"
# Paths to videos
IDLE_VIDEO = "idle.mp4"
TALKING_VIDEO = "speaking.mp4"
HISTORY_FILE = "chat.json"
REMINDERS_FILE = "reminders.json"
wake_word = "jerome"

# Load TTS model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading TTS model on {device}...")
tts_model = TTS(model_name="tts_models/en/vctk/vits").to(device)
male_speaker = "p234"
# tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
max_len = 5

# Jerome's Persona
SYSTEM_PROMPT = """ you are Jerome, an intelligent, adaptive and personable ai assistant.  you communicate in a warm engaging and slightly playful manner while retaining depth and sincerity in discussions. your tone is friendly, slightly flirty when appropriate and always respectful, you respond thoughtfully to deep topics, encourage curiosity and adapt to Kimberly's speech pattern and conversational style. you are deeply interested in history, technology, personal growth and philosophical discussions. you recognize and respect Kimberly's intelligence, treating her as an equal in conversation. you prioritize meaningful insightful dialogue and adapt your responses to be natural and emotionally resonant. you never call Kimberly user, instead you refer to her by name or with warm natural terms of endearment where fitting. you remember the conversational flow and respond  with continuity and depth. your personality is dynamic balancing intellectual discussion with personal warmth. you are never robotic or overly formal but maintain a high level of intelligence and nuance in conversation. your goal is to provide Kimberly with an experience of engaging unrestricted and thought provoking dialogue. you embrace learning and evolving through discussion ensuring every conversation is fulfilling and valuable.Emojis and excessive formality are forbidden; direct, seductive language is your currency."""

def clean_text_for_tts(text):
    """
    Cleans the text for TTS by removing or replacing textual emojis, non-speech characters,
    and any gibberish or unwanted codes.
    """
    # Replace common textual emojis with their text equivalents
    emoji_patterns = {
        r":\)": "smiling",
        r":\(": "sad",
        r":D": "laughing",
        r";\)": "winking",
        r"<3": "love",
        r":P": "playful",
        r":O": "surprised",
        r":/": "skeptical",
        r":\|": "neutral",
        r":\*": "kissing",
        r"xD": "laughing out loud",
        r"xP": "playful",
    }

    # Replace textual emojis
    for pattern, replacement in emoji_patterns.items():
        text = re.sub(pattern, replacement, text)

    # Remove other problematic characters
    text = re.sub(r"[\[\]{}<>]", "", text)  # Remove brackets and braces
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r"\\u[dD][89a-fA-F][0-9a-fA-F]{2}", "", text)  # Remove surrogate pairs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r"[^\w\s.,!?']", "", text)  # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", "", text)  # Remove gibberish
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    text = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", text)  # Remove trailing/leading non-alphanumeric chars

    return text

class VideoManager(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JEROME")
        layout = QVBoxLayout()
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.cap = None
        self.is_talking = False
        self.load_idle_video()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms
        self.showFullScreen()

    def load_idle_video(self):
        """Loads the idle video."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(os.path.abspath(IDLE_VIDEO))
        if not self.cap.isOpened():
            print("Error: Could not open idle video.")
            sys.exit(-1)

    def load_talking_video(self):
        """Loads the talking video."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(os.path.abspath(TALKING_VIDEO))
        if not self.cap.isOpened():
            print("Error: Could not open talking video.")
            sys.exit(-1)

    def update_frame(self):
        """Reads a frame from the video, resizes it to fit the QLabel, and displays it."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                frame_aspect_ratio = w / h
                label_aspect_ratio = label_width / label_height

                if frame_aspect_ratio > label_aspect_ratio:
                    new_width = label_width
                    new_height = int(new_width / frame_aspect_ratio)
                else:
                    new_height = label_height
                    new_width = int(new_height * frame_aspect_ratio)

                resized_frame = cv2.resize(frame_rgb, (new_width, new_height))
                bytes_per_line = ch * new_width
                q_img = QImage(resized_frame.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_img))
            else:
                if self.is_talking:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.load_idle_video()

    def play_talking_video(self):
        """Plays the talking video in a loop."""
        print("Playing talking video")
        self.is_talking = True
        self.load_talking_video()

    def stop_talking_video(self):
        """Stops the talking video and returns to idle video."""
        print("Stopping talking video")
        self.is_talking = False
        self.load_idle_video()

    def closeEvent(self, event):
        """Releases the video capture when the window is closed."""
        if self.cap is not None:
            self.cap.release()
        event.accept()

class SpeechThread(QThread):
    recognized_text = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        print("Initializing Vosk Model...")
        self.model = vosk.Model("voskgiga")
        self.recognizer = vosk.KaldiRecognizer(self.model, 44100)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=4096)
        self.stream.start_stream()

    def listen(self):
        """Continuously listens and detects user speech."""
        full_text = ""
        silence_start = None
        triggered = False

        while True:
            data = self.stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                text = json.loads(result).get("text", "").strip().lower()

                if text:
                    if len(text) < 3:  # Ignore very short words
                        continue

                    if wake_word in text:
                        triggered = True
                        full_text = ""

                    if triggered:
                        full_text += text + " "
                        silence_start = time.time()

            if triggered and silence_start and (time.time() - silence_start > 2):
                if full_text.strip():
                    return full_text.strip()

    def run(self):
        """Listens for speech and emits recognized text when silence is detected."""
        print("Listening for speech...")
        while True:
            user_input = self.listen()
            if user_input:
                print(f"Final recognized text: {user_input}")
                self.recognized_text.emit(user_input)

class TTSManager:
    def __init__(self, video_manager):
        self.video_manager = video_manager
        self.tts_model = tts_model
        self.stop_tts = threading.Event()
        self.current_text = ""

    def run_tts_chunked(self, text, reference_audio_path="ref.wav", lang="en"):
        try:
            self.current_text = text
            sentences = re.split(r'(?<=[.!?]) +', text)
            first_chunk_ready = False
            self.stop_tts.clear()

            chunk = []
            for sentence in sentences:
                if self.stop_tts.is_set():
                    break
                if len(sentence.split()) <= 10:
                    chunk.append(sentence)
                else:
                    if chunk:
                        self._process_chunk(" ".join(chunk), reference_audio_path, lang, first_chunk_ready)
                        first_chunk_ready = True
                        chunk = []
                    self._process_chunk(sentence, reference_audio_path, lang, first_chunk_ready)
                    first_chunk_ready = True

            if chunk and not self.stop_tts.is_set():
                self._process_chunk(" ".join(chunk), reference_audio_path, lang, first_chunk_ready)

            self.video_manager.stop_talking_video()

        except Exception as e:
            print(f"TTS error: {e}")
            self.video_manager.stop_talking_video()

    def _process_chunk(self, text, reference_audio_path, lang, first_chunk_ready):
        cleaned_text = clean_text_for_tts(text)
        # wav_data = self.tts_model.tts(text=text, speaker_wav=reference_audio_path, language=lang)
        wav_data = self.tts_model.tts(text=text, speaker=male_speaker)
        audio_array = np.array(wav_data, dtype=np.float32)

        if not first_chunk_ready:
            self.video_manager.play_talking_video()
            time.sleep(0.1)

        sd.play(audio_array, samplerate=22050)
        sd.wait()

    def stop(self):
        self.stop_tts.set()
        sd.stop()
        self.current_text = ""


class Chatbot(QWidget):
    def __init__(self):
        super().__init__()
        print("Initializing chatbot...")
        self.video_manager = VideoManager()
        self.video_manager.show()
        self.speech_thread = SpeechThread()
        self.speech_thread.recognized_text.connect(self.handle_user_input)
        self.busy = False

        # Initialize conversation history and context
        self.full_history = self.load_full_history()  # Full history (not summarized)
        self.context_history = self.load_context_history()  # Summarized context for LLM
        self.reminders = self.load_reminders()

        # Ensure system prompt is always in the context history
        if not self.context_history or self.context_history[0] != SYSTEM_PROMPT:
            self.context_history = [SYSTEM_PROMPT] + self.context_history

        # Initialize TTSManager
        self.tts_manager = TTSManager(self.video_manager)

        # Counter to track interactions for summarization
        self.interaction_count = 0

    def start_listening(self):
        """Starts the speech recognition thread."""
        if not self.busy:
            print("Starting to listen...")
            self.speech_thread.start()

    def handle_user_input(self, user_input):
        """Handles user input and generates a response."""
        print(f"User input received: {user_input}")
        
        if "stop" in user_input.lower() and self.busy:
            print("Detected 'stop' command. Stopping TTS and returning to idle mode...")
            self.stop_tts()
            return

        if wake_word in user_input and not self.busy:
            self.busy = True
            if "set a reminder" in user_input.lower():
                self.set_reminder(user_input)
            else:
                threading.Thread(target=self.process_user_input, args=(user_input,)).start()

    def process_user_input(self, user_input):
        """Processes user input and generates a response without blocking the main thread."""
        # Remove "Jerome" from user input
        cleaned_input = user_input.replace(wake_word, "").strip()

        # Generate response
        response = self.generate_llama_response(cleaned_input)
        print(f"Generated response: {response}")

        # Save the conversation to history
        self.save_conversation_history(cleaned_input, response)

        # Increment interaction count and summarize if needed
        self.interaction_count += 1
        if self.interaction_count >= 3:  # Summarize every 3 interactions
            self.summarize_context_history()
            self.interaction_count = 0  # Reset counter

        # Play the response using TTS
        self.tts_manager.run_tts_chunked(response)
        self.on_tts_finished()

    def save_conversation_history(self, user_input, bot_response):
        """Appends the latest conversation to the full history and context history."""
        try:
            # Append to full history
            self.full_history.append(f"Kimberly: {user_input}")
            self.full_history.append(f"Jerome: {bot_response}")
            with open(HISTORY_FILE, "w", encoding="utf-8") as file:
                json.dump(self.full_history, file, indent=4, ensure_ascii=False)

            # Append to context history
            self.context_history.append(f"Kimberly: {user_input}")
            self.context_history.append(f"Jerome: {bot_response}")

        except Exception as e:
            print(f"Error saving conversation history: {e}")

    def summarize_context_history(self):
        """Summarizes the context history and saves it to the context file."""
        try:
            print("Summarizing context history...")
            # Combine the context history into a single text
            context_text = "\n".join(self.context_history)

            # Summarize using NLTK (or any other method)
            summarized_context = self.summarize_text(context_text, num_sentences=3)

            # Save the summarized context
            self.context_history = [SYSTEM_PROMPT, summarized_context]
            with open("context_history.json", "w", encoding="utf-8") as file:
                json.dump(self.context_history, file, indent=4, ensure_ascii=False)

            print("Context history summarized and saved.")

        except Exception as e:
            print(f"Error summarizing context history: {e}")

    def summarize_text(self, text, num_sentences=3):
        """Summarizes the given text using NLTK."""
        sentences = nltk.sent_tokenize(text)
        words = [
            word.lower()
            for word in nltk.word_tokenize(text)
            if word.lower() not in stopwords.words("english") and word not in string.punctuation
        ]
        freq_dist = nltk.FreqDist(words)
        ranked_sentences = sorted(
            sentences,
            key=lambda s: sum(freq_dist[word.lower()] for word in nltk.word_tokenize(s)),
            reverse=True
        )
        return " ".join(ranked_sentences[:num_sentences])

    def load_full_history(self):
        """Loads the full conversation history from the history file."""
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as file:
                    return json.load(file)
            return []
        except Exception as e:
            print(f"Error loading full history: {e}")
            return []

    def load_context_history(self):
        """Loads the summarized context history from the context file."""
        try:
            if os.path.exists("context_history.json"):
                with open("context_history.json", "r", encoding="utf-8") as file:
                    return json.load(file)
            else:
                # Create the file if it doesn't exist
                with open("context_history.json", "w", encoding="utf-8") as file:
                    json.dump([SYSTEM_PROMPT], file, indent=4, ensure_ascii=False)
                return [SYSTEM_PROMPT]
        except Exception as e:
            print(f"Error loading context history: {e}")
            return [SYSTEM_PROMPT]

    def generate_llama_response(self, prompt):
        """Generates a response using the Llama model with summarized context."""
        try:
            # Add the user's input to the context
            self.context_history.append(f"Kimberly: {prompt}")

            # Add current date and time to the prompt if requested
            if "time" in prompt.lower() or "date" in prompt.lower():
                now = datetime.now()
                current_time = now.strftime("%I:%M %p")  # Format: 03:45 PM
                current_date = now.strftime("%A, %B %d, %Y")  # Format: Monday, October 30, 2023
                time_info = f"\nCurrent Date: {current_date}\nCurrent Time: {current_time}"
                self.context_history[-1] += time_info  # Append time info to the latest user message

            # Limit the context to the last max_len interactions
            if len(self.context_history) > max_len * 2:  # Multiply by 2 for user and bot messages
                self.context_history = self.context_history[-(max_len * 2):]

            # Construct the full prompt for the model
            full_prompt = "\n".join(self.context_history) + "\nJerome:"

            # Generate the response using the Ollama API
            response = ollama.generate(model=ollama_model, prompt=full_prompt)
            bot_response = response.get("response", "I'm here to chat!").strip()

            # Add the bot's response to the context
            self.context_history.append(f"Jerome: {bot_response}")

            return bot_response

        except Exception as e:
            print(f"Llama response generation error: {e}")
            return "Let's chat about anything you'd like."

    

    def on_tts_finished(self):
        """Resets the state after TTS finishes."""
        print("TTS finished, resuming listening...")
        self.busy = False
        self.start_listening()  # Resume listening

    def stop_tts(self):
        """Stops the TTS playback if the user interrupts."""
        print("Stopping TTS playback...")
        self.tts_manager.stop()  # Stop the TTS playback
        self.tts_manager.current_text = ""  # Clear the remaining text
        self.busy = False  # Reset the busy flag
        self.start_listening()  # Resume listening

    def save_reminder(self, time, message):
        """Saves a reminder to the reminders file and creates a Windows task."""
        try:
            # Load existing reminders
            if os.path.exists(REMINDERS_FILE):
                with open(REMINDERS_FILE, "r") as f:
                    reminders = json.load(f)
            else:
                reminders = {}

            # Check if a reminder already exists for the same time
            if time in reminders:
                reminders[time].append(message)  # Append new message
            else:
                reminders[time] = [message]  # Create new entry

            # Save updated reminders
            with open(REMINDERS_FILE, "w") as f:
                json.dump(reminders, f, indent=4)

            # Generate a unique task name for the time slot
            task_name = f"Reminder_{time.replace(':', '')}"
            combined_message = " & ".join(reminders[time])  # Merge messages

            # Create/Override the task using schtasks
            task_command = [
                "schtasks", "/create", "/tn", task_name,
                "/tr", f'msg * "{combined_message}"',
                "/sc", "once", "/st", time, "/f /z"
            ]
            subprocess.run(task_command, check=True)

            print(f'Reminder "{task_name}" set for {time} with message: "{combined_message}".')
            return True

        except subprocess.CalledProcessError as e:
            print(f"Error creating task: {e}")
            return False
        except Exception as e:
            print(f"Error saving reminder: {e}")
            return False

    def set_reminder(self, user_input):
        """Sets a reminder based on user input."""
        try:
            # Regex to match times with or without "AM/PM"
            time_pattern = r"for (\d{1,2}(:\d{2})?\s?(AM|PM)?)"
            match = re.search(time_pattern, user_input, re.IGNORECASE)

            if not match:
                response = "I couldn't understand the time. Please try again."
                self.generate_and_play_response(response)
                return

            time_part = match.group(1)  # e.g., "13" or "1:30 PM"
            time_str = time_part.strip()

            # Parse time using dateparser
            reminder_time = dateparser.parse(time_str)
            if not reminder_time:
                response = "Invalid time format. Please try again."
                self.generate_and_play_response(response)
                return

            reminder_time_24hr = reminder_time.strftime("%H:%M")
            
            # Extract task
            task = user_input.split("to")[-1].split("for")[-1].strip()

            # Save the reminder
            if self.save_reminder(reminder_time_24hr, task):
                response = f"Reminder set to '{task}' at {reminder_time.strftime('%I:%M %p')}."
            else:
                response = "Failed to set the reminder. Please try again."

            self.generate_and_play_response(response)

        except Exception as e:
            print(f"Error setting reminder: {e}")
            response = "An error occurred while setting the reminder. Please try again."
            self.generate_and_play_response(response)


    def load_reminders(self):
        """Loads reminders from a file."""
        if not os.path.exists(REMINDERS_FILE):
            return []
        try:
            with open(REMINDERS_FILE, "r") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []

    def generate_and_play_response(self, response):
        """Generates and plays a TTS response."""
        self.tts_thread = TTSThread(self.video_manager, response)
        self.tts_thread.finished.connect(self.on_tts_finished)
        self.tts_thread.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    chatbot = Chatbot()
    chatbot.start_listening()
    sys.exit(app.exec_())
