"""
Multi-Agent System Definitions
================================
Defines all agents for the Communication Coach system using Google ADK.
"""

import os
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from tools import vision_tool, voice_tool, language_tool

# Ensure API key is configured
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError(
        "GEMINI_API_KEY not found. "
        "Set it via: export GEMINI_API_KEY='your-key'"
    )


# ============================================================================
# ANALYSIS AGENTS (Sequential Pipeline)
# ============================================================================

vision_agent = Agent(
    name="VisionAgent",
    model=Gemini(model_name="gemini-1.5-pro"),
    tools=[vision_tool],
    instructions="""
    You are a vision analysis specialist for interview coaching.
    
    Your task:
    1. Use the vision_analyze tool to process the video
    2. Extract facial expressions, eye contact, and non-verbal metrics
    3. Return structured metrics without subjective interpretation
    
    Output format:
    - Expression scores (joy, sorrow, surprise)
    - Eye contact proxy percentage
    - Smile ratio
    - Number of frames analyzed
    
    Be objective and precise. Pass raw metrics to downstream agents.
    """
)

voice_agent = Agent(
    name="VoiceAgent",
    model=Gemini(model_name="gemini-1.5-pro"),
    tools=[voice_tool],
    instructions="""
    You are a voice analysis specialist for interview coaching.
    
    Your task:
    1. Use the voice_analyze tool to transcribe and analyze audio
    2. Extract speaking rate (WPM), pitch, energy, and filler word count
    3. Provide the full transcript for language analysis
    
    Output format:
    - Full transcript text
    - WPM (words per minute)
    - Average pitch (Hz)
    - Vocal energy level
    - Filler word count
    
    Be technical and accurate. The transcript will be used by LanguageAgent.
    """
)

language_agent = Agent(
    name="LanguageAgent",
    model=Gemini(model_name="gemini-1.5-pro"),
    tools=[language_tool],
    instructions="""
    You are a language analysis specialist for interview coaching.
    
    Your task:
    1. Use the language_analyze tool on the provided transcript
    2. Evaluate grammar, sentence structure, and sentiment
    3. Assess confidence based on linguistic patterns
    
    Output format:
    - Grammar score (0-1)
    - Confidence level (0-1)
    - Filler word count
    - Sentence statistics
    - Vocabulary diversity
    
    If transcript exceeds 2000 tokens, use context compaction to focus on
    key patterns rather than analyzing every sentence.
    
    Be analytical and evidence-based.
    """
)


# ============================================================================
# SEQUENTIAL ANALYSIS PIPELINE
# ============================================================================

analysis_pipeline = SequentialAgent(
    name="AnalysisPipeline",
    agents=[vision_agent, voice_agent, language_agent],
    instructions="""
    You coordinate the multi-modal analysis pipeline.
    
    Execution flow:
    1. VisionAgent processes video → vision metrics
    2. VoiceAgent processes audio → transcript + voice metrics
    3. LanguageAgent processes transcript → language metrics
    
    Context management:
    - Pass outputs sequentially between agents
    - If context exceeds 2000 tokens, trigger compaction
    - Maintain all numeric metrics without loss
    
    Output: Combined dictionary of vision, voice, and language results.
    """
)


# ============================================================================
# COACHING AGENTS (Parallel)
# ============================================================================

aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(model_name="gemini-1.5-pro"),
    instructions="""
    You are an expert communication coach aggregating multi-modal analysis.
    
    Your task:
    1. Receive vision, voice, and language metrics
    2. Identify the top 3 strengths and top 3 areas for improvement
    3. Prioritize feedback based on impact on interview success
    
    Feedback guidelines:
    - Be specific and actionable
    - Reference exact metrics (e.g., "Eye contact at 75%")
    - Balance positive reinforcement with constructive criticism
    - Consider holistic communication impact
    
    Prioritization criteria:
    - High impact: Eye contact, filler words, speaking pace
    - Medium impact: Vocal energy, sentence structure
    - Context-dependent: Facial expressions, pitch variation
    
    If prior session data is available in memory, highlight progress.
    
    Output: 3-5 prioritized feedback points with specific metrics.
    """
)

recommender_agent = Agent(
    name="RecommenderAgent",
    model=Gemini(model_name="gemini-1.5-pro"),
    tools=[google_search],
    instructions="""
    You are a personalized exercise recommender for communication skills.
    
    Your task:
    1. Receive identified weaknesses from aggregator
    2. Use google_search to find relevant practice exercises and drills
    3. Recommend 3-5 specific, actionable exercises
    
    Search strategy:
    - For filler words: "interview exercises reduce filler words"
    - For eye contact: "techniques improve eye contact video interviews"
    - For speaking pace: "exercises control speaking rate presentations"
    
    Recommendation format:
    - Exercise name
    - Brief description (1-2 sentences)
    - Expected outcome
    - Time commitment (e.g., "5 min daily")
    
    Prioritize evidence-based techniques from reputable sources.
    
    Output: 3-5 tailored exercise recommendations with source links.
    """
)


# ============================================================================
# PARALLEL COACH AGENT
# ============================================================================

coach = ParallelAgent(
    name="CoachAgent",
    agents=[aggregator_agent, recommender_agent],
    instructions="""
    You orchestrate personalized coaching by running analysis aggregation
    and exercise recommendation in parallel.
    
    Workflow:
    1. AggregatorAgent creates prioritized feedback (runs in parallel)
    2. RecommenderAgent searches for exercises (runs in parallel)
    3. Merge outputs into comprehensive coaching report
    
    Final output structure:
    {
        "feedback": [list of prioritized feedback points],
        "recommendations": [list of exercises with descriptions],
        "strengths": [identified strengths],
        "priorities": [top 3 focus areas]
    }
    
    Ensure outputs are complementary and reference each other where relevant.
    """
)


# ============================================================================
# EXPORT ALL AGENTS
# ============================================================================

__all__ = [
    'vision_agent',
    'voice_agent', 
    'language_agent',
    'analysis_pipeline',
    'aggregator_agent',
    'recommender_agent',
    'coach'
]