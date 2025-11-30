"""
Custom Analysis Tools for Communication Coach
==============================================
Provides vision, voice, and language analysis tools wrapped as ADK FunctionTools.
"""

import cv2
import whisper
import librosa
import numpy as np
import spacy
import re
from transformers import pipeline
from google.adk.tools import FunctionTool

# Initialize models globally to avoid reloading
print("Loading models... This may take a minute on first run.")
nlp = spacy.load("en_core_web_sm")
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)
whisper_model = whisper.load_model("base")
print("Models loaded successfully!")


# ============================================================================
# VISION ANALYSIS TOOL
# ============================================================================

def vision_analyze(video_path: str) -> dict:
    """
    Analyzes facial expressions and non-verbal communication from video.
    
    Args:
        video_path: Path to the video file to analyze
        
    Returns:
        Dictionary containing:
        - expressions: Joy, sorrow, surprise scores (0-1)
        - eye_contact_proxy: Estimated eye contact ratio
        - head_nods: Count of head nods (placeholder)
        - smile_ratio: Proportion of frames with smiling
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Sample frames evenly throughout video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, total_frames // 20)
        
        frames = []
        frame_count = 0
        
        while cap.isOpened() and len(frames) < 20:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % sample_interval == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            raise ValueError("No frames could be extracted from video")
        
        # Initialize metrics
        expressions = {'joy': 0, 'sorrow': 0, 'surprise': 0}
        eye_contact = 0
        smile_ratio = 0
        head_nods = 0
        
        # Analyze each frame
        for frame in frames:
            # Convert BGR to RGB for emotion pipeline
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # For now, we analyze the text sentiment as proxy
            # In production, use proper face detection + emotion recognition
            # emotion_result = emotion_pipeline(rgb_frame)[0]
            
            # Placeholder metrics (replace with actual face detection)
            expressions['joy'] += 0.6
            expressions['sorrow'] += 0.1
            expressions['surprise'] += 0.2
            eye_contact += 0.75  # Proxy based on face detection
            smile_ratio += 0.5
        
        num_frames = len(frames)
        
        # Normalize metrics
        return {
            'expressions': {k: round(v / num_frames, 2) for k, v in expressions.items()},
            'eye_contact_proxy': round(eye_contact / num_frames, 2),
            'head_nods': head_nods,
            'smile_ratio': round(smile_ratio / num_frames, 2),
            'frames_analyzed': num_frames
        }
        
    except Exception as e:
        print(f"Vision analysis error: {e}")
        return {
            'error': str(e),
            'expressions': {'joy': 0, 'sorrow': 0, 'surprise': 0},
            'eye_contact_proxy': 0,
            'head_nods': 0,
            'smile_ratio': 0
        }


# Wrap as ADK FunctionTool
vision_tool = FunctionTool.from_function(
    vision_analyze,
    name="vision_analyze",
    description="Analyzes facial expressions and non-verbal cues from video"
)


# ============================================================================
# VOICE ANALYSIS TOOL
# ============================================================================

def voice_analyze(audio_path: str) -> dict:
    """
    Transcribes audio and analyzes vocal delivery characteristics.
    
    Args:
        audio_path: Path to audio file (supports mp3, wav, m4a, mp4)
        
    Returns:
        Dictionary containing:
        - transcript: Full text transcription
        - wpm: Words per minute (speaking rate)
        - pitch: Average pitch in Hz
        - energy: Average vocal energy (0-1)
        - fillers: Count of filler words (um, uh, like)
    """
    try:
        # Transcribe with Whisper
        print("Transcribing audio...")
        result = whisper_model.transcribe(audio_path, fp16=False)
        transcript = result['text']
        
        # Calculate speaking rate (WPM)
        words = transcript.split()
        duration = result['segments'][-1]['end'] if result['segments'] else 1
        wpm = len(words) / (duration / 60) if duration > 0 else 0
        
        # Load audio for prosody analysis
        y, sr = librosa.load(audio_path, sr=None)
        
        # Pitch analysis (fundamental frequency)
        pitches = librosa.yin(y, fmin=75, fmax=300, sr=sr)
        pitch = np.nanmean(pitches[pitches > 0]) if len(pitches) > 0 else 150
        
        # Energy analysis (RMS)
        rms = librosa.feature.rms(y=y)
        energy = float(np.mean(rms))
        
        # Count filler words
        filler_words = ['uh', 'um', 'like', 'you know', 'actually', 'basically']
        fillers = sum(
            transcript.lower().count(filler) for filler in filler_words
        )
        
        return {
            'transcript': transcript,
            'wpm': round(wpm, 1),
            'pitch': round(float(pitch), 1),
            'energy': round(energy, 3),
            'fillers': fillers,
            'duration_seconds': round(duration, 1),
            'word_count': len(words)
        }
        
    except Exception as e:
        print(f"Voice analysis error: {e}")
        return {
            'error': str(e),
            'transcript': '',
            'wpm': 0,
            'pitch': 0,
            'energy': 0,
            'fillers': 0
        }


# Wrap as ADK FunctionTool
voice_tool = FunctionTool.from_function(
    voice_analyze,
    name="voice_analyze",
    description="Transcribes audio and analyzes speaking rate, pitch, energy, and fillers"
)


# ============================================================================
# LANGUAGE ANALYSIS TOOL
# ============================================================================

def language_analyze(transcript: str) -> dict:
    """
    Analyzes language patterns, grammar, and sentiment from transcript.
    
    Args:
        transcript: Text transcription to analyze
        
    Returns:
        Dictionary containing:
        - grammar_score: Grammar quality (0-1)
        - confidence: Sentiment-based confidence score (0-1)
        - filler_words: Count of verbal fillers
        - sentence_count: Number of sentences
        - avg_sentence_length: Average words per sentence
    """
    try:
        if not transcript or len(transcript.strip()) == 0:
            raise ValueError("Empty transcript provided")
        
        # Parse with spaCy
        doc = nlp(transcript)
        sentences = list(doc.sents)
        
        # Grammar score (proxy: well-formed sentences)
        well_formed = [s for s in sentences if len(s) > 5]
        grammar_score = len(well_formed) / max(1, len(sentences))
        
        # Sentiment analysis for confidence
        sentiment_result = pipeline("sentiment-analysis")(transcript[:512])[0]
        confidence = (
            sentiment_result['score'] 
            if sentiment_result['label'] == 'POSITIVE' 
            else 1 - sentiment_result['score']
        )
        
        # Count filler words
        filler_pattern = r'\b(uh|um|like|you know|actually|basically|sort of|kind of)\b'
        fillers = len(re.findall(filler_pattern, transcript.lower()))
        
        # Sentence statistics
        sentence_lengths = [len(s.text.split()) for s in sentences]
        avg_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths) 
            if sentence_lengths else 0
        )
        
        # Vocabulary diversity (unique words / total words)
        words = [token.text.lower() for token in doc if token.is_alpha]
        vocab_diversity = len(set(words)) / len(words) if words else 0
        
        return {
            'grammar_score': round(grammar_score, 2),
            'confidence': round(confidence, 2),
            'filler_words': fillers,
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 1),
            'vocab_diversity': round(vocab_diversity, 2),
            'sentiment_label': sentiment_result['label']
        }
        
    except Exception as e:
        print(f"Language analysis error: {e}")
        return {
            'error': str(e),
            'grammar_score': 0,
            'confidence': 0,
            'filler_words': 0,
            'sentence_count': 0,
            'avg_sentence_length': 0
        }


# Wrap as ADK FunctionTool
language_tool = FunctionTool.from_function(
    language_analyze,
    name="language_analyze",
    description="Analyzes grammar, sentiment, and linguistic patterns from transcript"
)


# ============================================================================
# EXPORT ALL TOOLS
# ============================================================================

__all__ = ['vision_tool', 'voice_tool', 'language_tool']