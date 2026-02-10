import os
from dotenv import load_dotenv
import openai
import sounddevice as sd
import numpy as np
import simpleaudio as sa
import whisper
from openwakeword.model import Model
from openwakeword.utils import download_models
import threading
import queue
import collections

# Optional imports
try:
    from elevenlabs import generate, stream
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except ImportError:
    WEBRTCVAD_AVAILABLE = False

# ---- 1. Load environment variables securely ----
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# ---- 2. Paths ----
SOUNDS_DIR = "sounds/"
os.makedirs(SOUNDS_DIR, exist_ok=True)
STARTUP_SOUND_PATH = os.path.join(SOUNDS_DIR, "jarvis_startup.wav")
BEEP_SOUND_PATH = os.path.join(SOUNDS_DIR, "beep.wav")
TEMP_RECORDING_PATH = os.path.join(SOUNDS_DIR, "temp_prompt.wav")


def play_sound(path: str):
    if not os.path.exists(path):
        print(f"Audio file not found at: {path}")
        return
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")


class JarvisAssistant:
    def __init__(self):
        self.first_interaction = True
        self.conversation_memory = [
            {"role": "system", "content": "You are Jarvis, a calm, helpful, and concise British voice assistant."}
        ]

        self.samplerate = 16000
        self.channels = 1
        self.chunk_size = 1280
        self.vad_chunk_size = 480

        print("Loading models, please wait...")
        self.oww_model = Model(inference_framework='onnx')
        self.whisper_model = whisper.load_model("base")
        if WEBRTCVAD_AVAILABLE:
            self.vad = webrtcvad.Vad(3)
        print("Models loaded successfully.")

        self.audio_queue = queue.Queue()
        self.stop_listening_flag = threading.Event()

    def _audio_callback_raw(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self.audio_queue.put(bytes(indata))

    def listen_for_wake_word(self):
        print(f"\n🎧 Listening... (Say 'Hey Jarvis' to activate)")

        with sd.RawInputStream(samplerate=self.samplerate,
                               blocksize=self.chunk_size,
                               device=None,
                               dtype='int16',
                               channels=self.channels,
                               callback=self._audio_callback_raw):

            while not self.stop_listening_flag.is_set():
                try:
                    chunk = self.audio_queue.get()
                    prediction = self.oww_model.predict(np.frombuffer(chunk, dtype=np.int16))

                    if prediction.get('hey_jarvis', 0) > 0.5:
                        self.handle_interaction()
                        print(f"\n🎧 Listening... (Say 'Hey Jarvis' to activate)")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"An error occurred in the listening loop: {e}")

        print("\nWake word listener stopped.")

    def handle_interaction(self):
        print("\n🚀 Wake word detected!")

        if self.first_interaction:
            play_sound(STARTUP_SOUND_PATH)
            self.first_interaction = False
        else:
            play_sound(BEEP_SOUND_PATH)

        user_audio = self.record_with_vad()
        if user_audio is None:
            print("No speech detected.")
            return

        print("🎙️ Transcribing...")
        user_text = self.transcribe_audio(user_audio)
        if not user_text:
            print("Could not transcribe audio. Please try again.")
            return

        print(f"👤 You: {user_text}")

        if "stop listening" in user_text.lower():
            print("🛑 'Stop listening' detected. Shutting down.")
            self.speak_response("Alright, shutting down.")
            self.stop_listening_flag.set()
            return

        print("🧠 Thinking...")
        gpt_response_text = self.get_gpt_response(user_text)
        if gpt_response_text:
            self.speak_response(gpt_response_text)
        else:
            print("Sorry, I couldn't get a response.")

    def record_with_vad(self):
        if not WEBRTCVAD_AVAILABLE:
            print("VAD not available. Recording for a fixed 5 seconds.")
            recording = sd.rec(int(5 * self.samplerate), samplerate=self.samplerate, channels=self.channels, dtype='float32')
            sd.wait()
            return recording.flatten()

        print("🔴 Voice activity detected, recording...")
        ring_buffer = collections.deque(maxlen=15)
        triggered = False
        voice_frames = []

        with sd.RawInputStream(samplerate=self.samplerate,
                               blocksize=self.vad_chunk_size,
                               device=None,
                               dtype='int16',
                               channels=self.channels) as stream:

            while True:
                frame, _ = stream.read(self.vad_chunk_size)
                is_speech = self.vad.is_speech(frame, self.samplerate)

                if not triggered:
                    ring_buffer.append((frame, is_speech))
                    num_voiced = len([f for f, speech in ring_buffer if speech])
                    if num_voiced > 0.8 * ring_buffer.maxlen:
                        triggered = True
                        print("🎤 Started recording your prompt.")
                        voice_frames.extend([f for f, s in ring_buffer])
                        ring_buffer.clear()
                else:
                    voice_frames.append(frame)
                    ring_buffer.append((frame, is_speech))
                    num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                    if num_unvoiced > 0.9 * ring_buffer.maxlen:
                        print("✅ Silence detected, recording finished.")
                        break

        if not voice_frames:
            return None

        recording_bytes = b''.join(voice_frames)
        return np.frombuffer(recording_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    def transcribe_audio(self, audio_data: np.ndarray) -> str:
        try:
            result = self.whisper_model.transcribe(audio_data, language="en")
            return result.get('text', '')
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return ""

    def get_gpt_response(self, user_input: str) -> str:
        self.conversation_memory.append({"role": "user", "content": user_input})

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=self.conversation_memory
            )
            assistant_response = response.choices[0].message.content
            self.conversation_memory.append({"role": "assistant", "content": assistant_response})
            return assistant_response.strip()
        except Exception as e:
            print(f"An error occurred while contacting OpenAI: {e}")
            self.conversation_memory.pop()
            return ""

    def speak_response(self, text: str):
        print(f"🤖 Jarvis: {text}")
        if not ELEVENLABS_AVAILABLE:
            return

        try:
            audio_stream = generate(
                api_key=ELEVENLABS_API_KEY,
                text=text,
                voice=ELEVENLABS_VOICE_ID,
                model="eleven_multilingual_v2",
                stream=True
            )
            stream(audio_stream)
        except Exception as e:
            print(f"An error occurred with ElevenLabs TTS: {e}")

    def run(self):
        try:
            self.listen_for_wake_word()
        except Exception as e:
            print(f"A critical error occurred: {e}")
        finally:
            print("Jarvis has shut down.")


if __name__ == "__main__":
    print("--- Jarvis AI Voice Assistant (Upgraded) ---")

    if not WEBRTCVAD_AVAILABLE:
        print("WARNING: 'webrtcvad-wheels' not found. Recording will be a fixed 5-second duration.")
        print("         For dynamic recording, run: pip install webrtcvad-wheels")

    print("Checking for wake word models...")
    download_models()
    print("Model check complete.")

    try:
        jarvis = JarvisAssistant()
        jarvis.run()
    except Exception as e:
        print(f"Failed to initialize Jarvis: {e}")
