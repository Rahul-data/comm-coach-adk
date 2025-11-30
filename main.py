"""
Communication Coach - Main Orchestration
=========================================
Main script for running multi-agent communication coaching sessions.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Configure Google Generative AI
import google.generativeai as genai

# ADK imports
from google.adk.runners import InMemoryRunner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import MemoryBank
from google.adk.evaluation import AgentEvaluator

# Import our agents
from agents import analysis_pipeline, coach
from tools import vision_analyze, voice_analyze, language_analyze

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

def configure_api():
    """Configure Google Generative AI API"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set")
        print("\n❌ Error: GEMINI_API_KEY not found!")
        print("Get your key at: https://makersuite.google.com/app/apikey")
        print("Then set it: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    logger.info("✓ API configured successfully")


# ============================================================================
# INITIALIZE SESSION & MEMORY
# ============================================================================

# Session service for managing agent conversations
session_service = InMemorySessionService()

# Memory bank for long-term progress tracking
memory_bank = MemoryBank()

# Runner with tracing enabled for observability
runner = InMemoryRunner(
    session_service=session_service,
    trace=True  # Enable execution tracing
)

# Evaluator for assessing feedback quality
evaluator = AgentEvaluator(
    metrics=['relevance_score', 'actionability']
)


# ============================================================================
# CORE COACHING SESSION FUNCTION
# ============================================================================

def run_coaching_session(
    video_path: str,
    user_id: str = "user1",
    session_id: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a complete coaching session on a video interview.
    
    Args:
        video_path: Path to video file (mp4, avi, mov, etc.)
        user_id: Unique identifier for the user
        session_id: Unique session identifier (auto-generated if None)
        verbose: Print detailed progress logs
        
    Returns:
        Dictionary containing:
        - vision_analysis: Facial expression and non-verbal metrics
        - voice_analysis: Transcription and vocal delivery metrics
        - language_analysis: Grammar and sentiment analysis
        - feedback: Prioritized coaching feedback points
        - recommendations: Personalized exercise suggestions
        - progress: Comparison with previous sessions
        - eval: Quality scores for the feedback
    """
    
    # Generate session ID if not provided
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting coaching session: {session_id} for user: {user_id}")
    logger.info(f"Analyzing video: {video_path}")
    
    # Validate video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        # ====================================================================
        # STEP 1: CREATE SESSION
        # ====================================================================
        if verbose:
            print("\n🎬 Creating analysis session...")
        
        session = session_service.create_session(user_id=user_id)
        logger.info(f"Session created: {session.id}")
        
        # ====================================================================
        # STEP 2: SEQUENTIAL ANALYSIS PIPELINE
        # ====================================================================
        if verbose:
            print("\n🔍 Running multi-modal analysis...")
            print("  ├─ Vision: Analyzing facial expressions...")
        
        # Prepare input for analysis pipeline
        analysis_input = {
            "video_path": video_path,
            "audio_path": video_path  # Most video formats contain audio
        }
        
        # Run sequential analysis (vision → voice → language)
        logger.info("Running analysis pipeline")
        analysis_results = runner.run(
            analysis_pipeline,
            session,
            analysis_input
        )
        
        if verbose:
            print("  ├─ Voice: Transcribing and analyzing audio...")
            print("  └─ Language: Analyzing linguistic patterns...")
            print("  ✓ Analysis complete!")
        
        logger.info(f"Analysis results: {len(str(analysis_results))} chars")
        
        # ====================================================================
        # STEP 3: PARALLEL COACHING (Feedback + Recommendations)
        # ====================================================================
        if verbose:
            print("\n🎯 Generating personalized coaching...")
            print("  ├─ Aggregating feedback...")
            print("  └─ Searching for practice exercises...")
        
        # Run parallel coaching agents
        coach_input = analysis_results
        coach_results = runner.run(coach, session, coach_input)
        
        if verbose:
            print("  ✓ Coaching complete!")
        
        logger.info("Coaching results generated")
        
        # ====================================================================
        # STEP 4: STORE IN MEMORY FOR PROGRESS TRACKING
        # ====================================================================
        current_metrics = {
            'vision': analysis_results.get('vision_analysis', {}),
            'voice': analysis_results.get('voice_analysis', {}),
            'language': analysis_results.get('language_analysis', {}),
            'timestamp': datetime.now().isoformat()
        }
        
        memory_bank.store(
            session_id,
            {
                "metrics": current_metrics,
                "feedback": coach_results
            }
        )
        logger.info("Results stored in memory bank")
        
        # ====================================================================
        # STEP 5: EVALUATE FEEDBACK QUALITY
        # ====================================================================
        if verbose:
            print("\n📊 Evaluating feedback quality...")
        
        # Evaluate coaching quality (using basic metrics)
        eval_score = evaluator.evaluate(
            coach_results,
            ground_truth={
                "expected": "actionable feedback with specific metrics"
            }
        )
        
        logger.info(f"Evaluation score: {eval_score}")
        
        # ====================================================================
        # STEP 6: CALCULATE PROGRESS VS PREVIOUS SESSIONS
        # ====================================================================
        progress_note = "First session - baseline established"
        
        try:
            # Retrieve previous session data
            prior_data = memory_bank.retrieve(user_id, "prior_metrics")
            
            if prior_data:
                prior_voice = prior_data.get('voice', {})
                current_voice = current_metrics.get('voice', {})
                
                prior_wpm = prior_voice.get('wpm', 0)
                current_wpm = current_voice.get('wpm', 0)
                
                if prior_wpm > 0:
                    wpm_change = current_wpm - prior_wpm
                    progress_note = (
                        f"Speaking pace: {prior_wpm:.0f} → {current_wpm:.0f} WPM "
                        f"({'↑' if wpm_change > 0 else '↓'} {abs(wpm_change):.0f})"
                    )
                    
                    if verbose:
                        print(f"\n📈 Progress: {progress_note}")
        
        except Exception as e:
            logger.warning(f"Could not retrieve prior session: {e}")
        
        # ====================================================================
        # STEP 7: COMPILE FINAL RESULTS
        # ====================================================================
        final_results = {
            'session_id': session_id,
            'user_id': user_id,
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            
            # Analysis results
            'vision_analysis': analysis_results.get('vision_analysis', {}),
            'voice_analysis': analysis_results.get('voice_analysis', {}),
            'language_analysis': analysis_results.get('language_analysis', {}),
            
            # Coaching outputs
            'feedback': coach_results.get('feedback', []),
            'recommendations': coach_results.get('recommendations', []),
            'strengths': coach_results.get('strengths', []),
            'priorities': coach_results.get('priorities', []),
            
            # Meta information
            'progress': progress_note,
            'eval': eval_score
        }
        
        logger.info(f"Session {session_id} completed successfully")
        
        if verbose:
            print("\n" + "="*60)
            print("✅ COACHING SESSION COMPLETE")
            print("="*60)
        
        return final_results
    
    except Exception as e:
        logger.error(f"Session failed: {e}", exc_info=True)
        raise


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command line usage"""
    
    parser = argparse.ArgumentParser(
        description='AI Communication Coach - Analyze interview videos'
    )
    parser.add_argument(
        '--video',
        required=True,
        help='Path to video file'
    )
    parser.add_argument(
        '--user',
        default='user1',
        help='User ID (default: user1)'
    )
    parser.add_argument(
        '--session',
        default=None,
        help='Session ID (auto-generated if not provided)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    # Configure API
    configure_api()
    
    # Run coaching session
    try:
        results = run_coaching_session(
            video_path=args.video,
            user_id=args.user,
            session_id=args.session,
            verbose=not args.quiet
        )
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print("\n📊 Key Metrics:")
        voice = results.get('voice_analysis', {})
        language = results.get('language_analysis', {})
        vision = results.get('vision_analysis', {})
        
        print(f"  • Speaking pace: {voice.get('wpm', 0):.0f} WPM")
        print(f"  • Filler words: {voice.get('fillers', 0)}")
        print(f"  • Confidence score: {language.get('confidence', 0):.2f}")
        print(f"  • Eye contact: {vision.get('eye_contact_proxy', 0):.0%}")
        
        print("\n💡 Top Feedback:")
        for i, fb in enumerate(results.get('feedback', [])[:3], 1):
            print(f"  {i}. {fb}")
        
        print("\n🎯 Recommended Exercises:")
        for i, rec in enumerate(results.get('recommendations', [])[:3], 1):
            print(f"  {i}. {rec}")
        
        print("\n📈 Progress:", results.get('progress', 'N/A'))
        
        print("\n✅ Full results saved in session:", results['session_id'])
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()