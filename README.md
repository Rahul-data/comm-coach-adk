# Communication Coach ADK

An AI-powered multi-agent communication coaching system built with Google's Agent Development Kit (ADK). This system analyzes video interviews across three dimensions—vision, voice, and language—to provide actionable feedback and personalized recommendations.

## 🎯 Features

- **Multi-Modal Analysis**: Comprehensive evaluation of facial expressions, vocal delivery, and language patterns
- **Multi-Agent Architecture**: Sequential analysis pipeline with parallel coaching agents
- **Intelligent Tool Integration**: Custom vision, voice, and language analysis tools
- **Memory & Progress Tracking**: Session-based memory for tracking improvement over time
- **Smart Recommendations**: Google Search integration for personalized practice exercises
- **Observability**: Built-in logging and tracing for debugging and monitoring
- **Evaluation Framework**: Automated feedback quality assessment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Video Input                      │
└─────────────────┬───────────────────────────────────────┘
                  │
         ┌────────▼────────┐
         │ Sequential      │
         │ Analysis        │
         │ Pipeline        │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼────┐  ┌────▼─────┐  ┌───▼──────┐
│ Vision │  │  Voice   │  │ Language │
│ Agent  │  │  Agent   │  │  Agent   │
└───┬────┘  └────┬─────┘  └───┬──────┘
    │             │             │
    └─────────────┼─────────────┘
                  │
         ┌────────▼────────┐
         │   Parallel      │
         │   Coach Agent   │
         └────────┬────────┘
                  │
         ┌────────┼────────┐
         │                 │
    ┌────▼─────┐    ┌─────▼──────┐
    │Aggregator│    │Recommender │
    │  Agent   │    │   Agent    │
    └────┬─────┘    └─────┬──────┘
         │                 │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Actionable      │
         │ Feedback +      │
         │ Exercises       │
         └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- Google Gemini API Key ([Get one here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM (for ML models)
- FFmpeg (for audio processing)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/comm-coach-adk.git
cd comm-coach-adk
```

2. **Install dependencies**
```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

3. **Set up API Key**
```bash
# Option 1: Environment variable
export GEMINI_API_KEY='your-api-key-here'

# Option 2: Create .env file
echo "GEMINI_API_KEY=your-api-key-here" > .env
```

## 💻 Usage

### Basic Usage

```python
from main import run_coaching_session

# Analyze a video interview
results = run_coaching_session(
    video_path="path/to/interview.mp4",
    user_id="user123",
    session_id="session1"
)

print(results)
```

### Expected Output

```json
{
  "vision_analysis": {
    "expressions": {"joy": 0.65, "sorrow": 0.1, "surprise": 0.25},
    "eye_contact_proxy": 0.75,
    "smile_ratio": 0.5
  },
  "voice_analysis": {
    "wpm": 145,
    "pitch": 180.5,
    "energy": 0.42,
    "fillers": 8
  },
  "language_analysis": {
    "grammar_score": 0.82,
    "confidence": 0.78,
    "filler_words": 8
  },
  "feedback": [
    "Eye contact is good at 75% - maintain this consistency",
    "Speaking pace (145 WPM) is ideal for interviews",
    "Reduce filler words (8 instances) - try pause instead"
  ],
  "recommendations": [
    "Practice: Mirror technique for maintaining eye contact",
    "Exercise: 'Power pause' to replace um/uh fillers",
    "Drill: Record & review 2-minute answers"
  ],
  "progress": "Improvement from prior: WPM 130 → 145",
  "eval": {"relevance_score": 0.89, "actionability": 0.92}
}
```

## 📁 Project Structure

```
comm-coach-adk/
├── README.md              # This file
├── main.py                # Main orchestration and session runner
├── agents.py              # All agent definitions
├── tools.py               # Custom analysis tools
├── requirements.txt       # Python dependencies
├── diagrams/
│   └── architecture.png   # System architecture diagram
└── .gitignore            # Git ignore rules
```

## 🔧 Components

### Tools (`tools.py`)
- **Vision Tool**: Analyzes facial expressions, eye contact, and non-verbal cues using emotion detection
- **Voice Tool**: Transcribes audio and analyzes pace, pitch, energy, and filler words
- **Language Tool**: Evaluates grammar, sentiment, and confidence from transcripts

### Agents (`agents.py`)
- **Vision Agent**: Processes video for non-verbal analysis
- **Voice Agent**: Handles audio transcription and prosody analysis
- **Language Agent**: Analyzes linguistic patterns and structure
- **Analysis Pipeline**: Sequential agent coordinating all analysis
- **Aggregator Agent**: Synthesizes multi-modal results into prioritized feedback
- **Recommender Agent**: Searches for tailored practice exercises
- **Coach Agent**: Parallel agent combining feedback and recommendations

### Main Orchestration (`main.py`)
- Session management with memory
- Runner configuration with tracing
- Progress tracking across sessions
- Evaluation framework

## 🎓 Advanced Features

### Memory & Progress Tracking
```python
# Retrieve historical data
prior_sessions = memory_bank.retrieve(session_id, "prior_metrics")

# Track improvement
progress = compare_metrics(current_metrics, prior_sessions)
```

### Context Compaction
Automatically handles long transcripts by compacting context when token limits are approached.

### Observability
```python
import logging
logging.basicConfig(level=logging.INFO)

# View agent execution traces
runner = InMemoryRunner(session_service=session_service, trace=True)
```

## 🧪 Testing

Run with sample videos:
```bash
python main.py --video sample.mp4 --user test_user --session test1
```

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables or `.env` files
- Add `.env` to `.gitignore`

## 📊 Performance

- Vision analysis: ~2-3s per video (20 frames)
- Voice transcription: ~5-10s per minute of audio
- Language analysis: <1s for typical transcripts
- Total processing: ~15-30s for 2-3 minute interview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

MIT License - feel free to use this project for learning and development.

## 🙏 Acknowledgments

- Google ADK team for the agent framework
- OpenAI Whisper for transcription
- Hugging Face for emotion detection models
- spaCy for NLP capabilities

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [rahulpandey2345@gmail.com]

## 🗺️ Roadmap

- [ ] Add real-time video analysis
- [ ] Implement advanced MediaPipe integration for gesture detection
- [ ] Multi-language support
- [ ] Web UI with Streamlit
- [ ] Cloud deployment to Vertex AI
- [ ] Mobile app integration
- [ ] Collaborative coaching sessions