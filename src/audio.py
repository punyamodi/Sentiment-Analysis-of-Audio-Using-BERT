import os
from pathlib import Path
from typing import List, Optional

import whisper


class AudioTranscriber:
    SUPPORTED_FORMATS = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}

    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{path.suffix}'. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )

        options = {}
        if language:
            options["language"] = language

        result = self.model.transcribe(str(path), **options)
        return result["text"].strip()

    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
    ) -> List[str]:
        return [self.transcribe(path, language=language) for path in audio_paths]

    def transcribe_with_timestamps(self, audio_path: str) -> dict:
        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        result = self.model.transcribe(str(path), verbose=False)
        return {
            "text": result["text"].strip(),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "language": result.get("language", "unknown"),
        }
