#!/usr/bin/env python3

import subprocess
import requests
import sounddevice as sd
import numpy as np

from scipy.io.wavfile import write
from faster_whisper import WhisperModel


# ---------- CONFIG ----------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "translategemma:4b"

WHISPER_MODEL = "small"

SAMPLE_RATE = 16000
RECORD_SECONDS = 5

PIPER_MODEL = "en_US-lessac-medium"

EXIT_WORDS = {
    "exit",
    "quit",
    "shutdown",
    "stop",
}


# ---------- INIT ----------

print("Loading Whisper...")
whisper = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8",
)

print("Ready.")


# ---------- AUDIO ----------

def record_microphone(
    filename="input.wav",
):

    print("\nListening...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    sd.wait()

    audio = np.squeeze(audio)

    write(
        filename,
        SAMPLE_RATE,
        audio,
    )

    return filename



def speak(text):

    print("\nAssistant:")
    print(text)

    subprocess.run(
        [
            "piper",
            "--model",
            PIPER_MODEL,
            "--output_file",
            "reply.wav",
        ],
        input=text.encode("utf-8"),
    )

    subprocess.run(
        [
            "aplay",
            "reply.wav",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ---------- AI ----------

def transcribe(filename):

    segments, _ = whisper.transcribe(
        filename
    )

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()



def ask_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    response.raise_for_status()

    return response.json()["response"].strip()



# ---------- MAIN ----------

def main():

    while True:

        try:

            audio = record_microphone()

            text = transcribe(audio)

            if not text:
                continue

            print("\nYou:")
            print(text)

            if text.lower() in EXIT_WORDS:
                speak("Goodbye.")
                break


            answer = ask_ollama(
                text
            )

            speak(answer)


        except KeyboardInterrupt:
            print("\nStopping.")
            break


        except Exception as e:
            print(
                "Error:",
                e,
            )


if __name__ == "__main__":
    main()
