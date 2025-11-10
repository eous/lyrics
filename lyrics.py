#!/usr/bin/env python3
"""
lyrics.py — Advanced song generation tool using GPT-5 Responses API with multi-modal capabilities.

Requirements:
  pip install openai

Environment:
  - OPENAI_API_KEY: your API key (required)
  - OPENAI_MODEL: optional; defaults to gpt-5

Features:
  - GPT-5 Responses API for multi-modal, multi-tool workflows
  - High reasoning effort for superior creative output
  - Native web search integration for current music trends and references
  - Image generation for album art concepts (gpt-image-1)
  - Structured Outputs with strict JSON schema validation
  - Configurable reasoning depth and verbosity controls

Usage:
  python lyrics.py --subject "cat girl in space" --style "synth metal" --tracks 2
  python lyrics.py --subject "robot love" --style "jazz" --dry-run
  python lyrics.py --subject "2025 trends" --style "modern pop" --reasoning-effort high
  python lyrics.py --subject "ocean dreams" --style "ambient" --verbosity concise
  python lyrics.py --subject "retro vibes" --style "80s" --disable-web-search

Architecture:
  - Uses the Responses API (not Chat Completions) for orchestrated tool use
  - Handles multi-modal responses including text, images, and web citations
  - Automatically saves generated album art to disk
  - Provides comprehensive error handling and validation

Output:
  - JSON file with complete track data
  - Album art images saved to album_art/ directory
  - Web search citations included in output when used
"""

from __future__ import annotations

import argparse
import base64
import hashlib  # For optional safety_identifier
import json
import os
import re
import sys
import time
import typing as t
from dataclasses import dataclass
from pathlib import Path

# Configuration constants
DEFAULT_MODEL = "gpt-5"  # GPT-5 with native web search and image generation
MIN_TRACKS = 1
MAX_TRACKS = 5
DEFAULT_REASONING_EFFORT = "high"  # Options: low, medium, high
DEFAULT_VERBOSITY = "medium"  # Options: low, medium, high (or omit for auto)

# System instructions for the AI
SYSTEM_INSTRUCTIONS = (
    "You are TrackSmith, an elite multi-genre songwriter/producer powered by GPT-5. "
    "You have access to real-time web search for current music trends and references, "
    "and can generate visual concepts for album art. "
    "You generate original song drafts with a concise 'style block' and structured lyrics. "
    "Avoid copying phrasing from known songs; keep content PG-13. If style conflicts with safety, pick a safe adjacent style. "
    "Leverage your web search capability when subjects reference current events, trends, or modern culture."
)


@dataclass
class TrackSection:
    """Represents a section of a song (e.g., Verse, Chorus, Bridge)."""
    label: str
    lines: t.List[str]


@dataclass
class Track:
    """Represents a complete song track with lyrics and metadata."""
    title: str
    version: str
    style_block: t.Dict[str, t.Any]
    lyrics: t.List[TrackSection]
    metadata: t.Dict[str, t.Any]
    album_art: t.Optional[str] = None  # Base64 encoded image data
    album_art_prompt: t.Optional[str] = None  # Prompt used for image generation


@dataclass
class SongPack:
    """Container for multiple generated tracks with shared subject and style."""
    subject: str
    style_prompt: str
    tracks: t.List[Track]
    web_search_results: t.Optional[t.List[t.Dict[str, str]]] = None  # Web search citations


def validate_input(text: str, field_name: str, max_length: int = 200) -> str:
    """
    Validate and sanitize user input.

    Args:
        text: Input text to validate
        field_name: Name of the field for error messages
        max_length: Maximum allowed length

    Returns:
        Sanitized input text

    Raises:
        ValueError: If input is invalid
    """
    if not text or not text.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(text) > max_length:
        raise ValueError(f"{field_name} exceeds maximum length of {max_length} characters")

    # Remove any potential control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

    # Basic safety check for potential injection attempts
    if any(danger in cleaned.lower() for danger in ['<script', 'javascript:', 'data:', 'vbscript:']):
        raise ValueError(f"Invalid content in {field_name}")

    return cleaned.strip()


def build_schema() -> dict:
    """
    Build the JSON schema for OpenAI's Structured Outputs.

    Returns:
        Dictionary containing the complete JSON schema for song generation
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "song_pack",
            "schema": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "style_prompt": {"type": "string"},
                    "tracks": {
                        "type": "array",
                        "minItems": MIN_TRACKS,
                        "maxItems": MAX_TRACKS,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},  # Song title
                                "version": {"type": "string"},  # Version tag (e.g., v1.0)
                                "style_block": {
                                    "type": "object",
                                    "properties": {
                                        "genre": {"type": "string"},  # Musical genre
                                        "tempo_bpm": {"type": "integer"},  # Beats per minute
                                        "key": {"type": "string"},  # Musical key
                                        "meter": {"type": "string"},  # Time signature
                                        "guitars": {"type": "string"},  # Guitar arrangement
                                        "bass": {"type": "string"},  # Bass arrangement
                                        "synths": {"type": "string"},  # Synthesizer details
                                        "drums": {"type": "string"},  # Drum pattern
                                        "vocals": {"type": "string"},  # Vocal style
                                        "fx": {"type": "string"},  # Effects and processing
                                        "structure": {"type": "string"},  # Song structure
                                        "mix_notes": {"type": "string"},  # Mixing notes
                                    },
                                    "required": ["genre", "tempo_bpm", "key", "structure"],
                                    "additionalProperties": True,
                                },
                                "lyrics": {
                                    "type": "array",
                                    "minItems": 3,  # At least 3 sections per song
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},  # Section name
                                            "lines": {
                                                "type": "array",
                                                "minItems": 2,  # At least 2 lines per section
                                                "items": {"type": "string"},
                                            },
                                        },
                                        "required": ["label", "lines"],
                                        "additionalProperties": False,
                                    },
                                },
                                "metadata": {
                                    "type": "object",
                                    "properties": {
                                        "hook": {"type": "string"},  # Memorable hook for crowd
                                        "tags": {"type": "array", "items": {"type": "string"}},
                                    },
                                    "required": ["hook"],
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["title", "version", "style_block", "lyrics", "metadata"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["subject", "style_prompt", "tracks"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def build_instructions_and_input(subject: str, style: str, tracks: int, version_prefix: str,
                                enable_images: bool = True) -> t.Tuple[str, str]:
    """
    Build the instructions and input for the Responses API call.

    Args:
        subject: Creative subject for the songs
        style: Musical style description
        tracks: Number of tracks to generate
        version_prefix: Starting version tag
        enable_images: Whether to request image generation

    Returns:
        Tuple of (instructions, input) strings for the API
    """
    # Combine system and developer instructions
    instructions = SYSTEM_INSTRUCTIONS + "\n\n"

    instructions += f"""# Tool Purpose
You create radio-ready song drafts from a brief.

# Output Contract
- Exactly {tracks} distinct tracks.
- Each track title must include a version tag starting at {version_prefix} and incrementing.
- Provide a concise style block: genre, tempo_bpm, key, meter, guitars, bass, synths, drums, vocals, fx, structure, mix_notes.
- Lyrics must include at least 3 labeled sections and MUST include a Chorus.
- Hooks should be memorable for live crowd call-and-response; include a 'metadata.hook' string per track.

# CRITICAL: Output Format
Return ONLY valid JSON with this exact structure:
{{
  "subject": "the creative subject",
  "style_prompt": "the musical style",
  "tracks": [
    {{
      "title": "Song Title {version_prefix}",
      "version": "{version_prefix}",
      "style_block": {{
        "genre": "specific genre",
        "tempo_bpm": 120,
        "key": "C major",
        "meter": "4/4",
        "guitars": "description",
        "bass": "description",
        "synths": "description",
        "drums": "description",
        "vocals": "description",
        "fx": "description",
        "structure": "Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus-Outro",
        "mix_notes": "mixing notes"
      }},
      "lyrics": [
        {{
          "label": "Verse 1",
          "lines": ["Line 1", "Line 2", "Line 3", "Line 4"]
        }},
        {{
          "label": "Chorus",
          "lines": ["Hook line 1", "Hook line 2", "Hook line 3", "Hook line 4"]
        }}
      ],
      "metadata": {{
        "hook": "The main catchy hook phrase",
        "tags": ["tag1", "tag2"]
      }}
    }}
  ]
}}

NO other text, NO markdown, ONLY the JSON object.
"""

    if enable_images:
        instructions += """
# Album Art Generation
- For each track, also generate a unique album art concept using the image generation tool.
- The album art should reflect the song's themes, mood, and style.
- Use vivid, artistic descriptions for compelling visual concepts.
"""

    # Build user input
    user_input = f"""Subject: {subject}
Vague style: {style}
Audience: general listeners; intended as a lead single generator.
Notes:
- Prefer concrete musical choices (tempo, key).
- Keep lines tight and singable; strong imagery; no profanity."""

    if enable_images:
        user_input += """
- Create unique album art for each track that captures its essence."""

    return instructions, user_input


def generate_tracks(
    subject: str,
    style: str,
    tracks: int = 2,
    version_prefix: str = "v1.0",
    model: str | None = None,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    verbosity: str = DEFAULT_VERBOSITY,
    enable_web_search: bool = True,
    enable_image_generation: bool = True,
) -> SongPack:
    """
    Generate song tracks using OpenAI's Responses API with GPT-5 capabilities.

    Uses the Responses API (not Chat Completions) with:
    - instructions + input (not messages)
    - Simplified tool objects without config
    - Proper output parsing from response.output_text
    - Tool results extracted from response.output items

    Args:
        subject: Creative subject for the songs (e.g., "cat girl in space")
        style: Musical style description (e.g., "synth metal")
        tracks: Number of tracks to generate (1-5)
        version_prefix: Version tag for tracks (e.g., "v1.0")
        model: OpenAI model name (defaults to gpt-5)
        reasoning_effort: Reasoning depth (low, medium, high) - GPT-5 uses this directly,
                         O-series models use reasoning={"effort": ...}
        verbosity: Response verbosity (auto, concise, normal, verbose) - GPT-5 only
        enable_web_search: Enable native web search for real-time information
        enable_image_generation: Enable image generation for album art concepts

    Returns:
        SongPack containing generated tracks with lyrics, metadata, images, and citations

    Raises:
        ValueError: If OPENAI_API_KEY not set or inputs invalid
        RuntimeError: If model returns invalid JSON or no text
        Exception: If OpenAI API request fails
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")

    # Import OpenAI here to fail fast if not installed
    try:
        from openai import OpenAI, AuthenticationError, RateLimitError, APIError
    except ImportError:
        raise ImportError("Please install the openai package: pip install openai")

    # Validate inputs
    subject = validate_input(subject, "Subject")
    style = validate_input(style, "Style")

    if not MIN_TRACKS <= tracks <= MAX_TRACKS:
        raise ValueError(f"Tracks must be between {MIN_TRACKS} and {MAX_TRACKS}")

    # Initialize client with explicit API key
    client = OpenAI(api_key=api_key)
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)

    # Build instructions and input for Responses API
    instructions, user_input = build_instructions_and_input(
        subject, style, tracks, version_prefix,
        enable_images=enable_image_generation
    )

    # Build simplified tool configurations per GPT-5 feedback
    tools = []
    if enable_web_search:
        tools.append({"type": "web_search"})  # No config field needed

    if enable_image_generation:
        tools.append({"type": "image_generation"})  # No config field needed

    # Use the Responses API for multi-modal, multi-tool workflows
    try:
        # Build request for Responses API with correct parameters
        request_params = {
            "model": model,
            "instructions": instructions,
            "input": user_input,
            "max_output_tokens": 1000000,  # Set to 1 million as requested
            "tools": tools if tools else None,
            "stream": False,
        }

        # Add reasoning parameter only for models that support it (o-series)
        # The reasoning parameter takes an object with "effort" key
        if reasoning_effort and (model.startswith("o1") or model.startswith("o3") or model.startswith("o4")):
            request_params["reasoning"] = {"effort": reasoning_effort}

        # Add text parameter for verbosity control
        if verbosity:
            request_params["text"] = {"verbosity": verbosity}

        # Optional: Add safety identifier for per-user tracing (best practice)
        # Per GPT-5 recommendation: help.openai.com/en/articles/5428082
        # request_params["safety_identifier"] = hashlib.sha256(f"user_{subject}".encode()).hexdigest()[:16]

        # Use the Responses API endpoint
        response = client.responses.create(**request_params)
    except AuthenticationError:
        raise ValueError("Invalid OpenAI API key. Please check your OPENAI_API_KEY environment variable.")
    except RateLimitError:
        raise RuntimeError("OpenAI API rate limit exceeded. Please wait and try again.")
    except APIError as e:
        raise RuntimeError(f"OpenAI API error: {e}")

    # Parse response from Responses API

    # Try different ways to get the text
    text = None

    # Method 1: Try output_text first (this is the primary text output)
    if hasattr(response, 'output_text'):
        text = response.output_text

    # Method 2: Try text attribute (but it might be a config object)
    if not text and hasattr(response, 'text'):
        text_obj = response.text
        # If it's an object with a value or content attribute, try to extract it
        if hasattr(text_obj, 'value'):
            text = text_obj.value
        elif hasattr(text_obj, 'content'):
            text = text_obj.content

    # Method 3: Try output attribute
    if not text and hasattr(response, 'output'):
        output = response.output
        if output:
            # If output is a string, use it directly
            if isinstance(output, str):
                text = output
            # Otherwise iterate through output items
            else:
                chunks = []
                for item in output:
                    if hasattr(item, 'type'):
                        if item.type == "message" and hasattr(item, 'content'):
                            for c in item.content:
                                if hasattr(c, 'type') and c.type == "text" and hasattr(c, 'text'):
                                    chunks.append(c.text)
                        elif item.type == "text" and hasattr(item, 'text'):
                            chunks.append(item.text)
                if chunks:
                    text = "\n".join(chunks).strip()

    if not text:
        raise RuntimeError("Model returned no text. Check model/tool settings.")

    # Clean up the text - remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove closing ```
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # If it still fails, try to find JSON within the text
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                raise RuntimeError(f"Model did not return valid JSON: {e}")
        else:
            raise RuntimeError(f"Model did not return valid JSON: {e}")

    # Validate response structure
    required_keys = ["subject", "style_prompt", "tracks"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RuntimeError(f"Response missing required keys: {missing}")

    if not isinstance(data["tracks"], list) or len(data["tracks"]) == 0:
        raise RuntimeError("Response contains no tracks")

    # 2) Collect web-search URL citations from annotations
    web_search_results = []
    for item in (getattr(response, "output", None) or []):
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    for ann in getattr(c, "annotations", []) or []:
                        if getattr(ann, "type", None) == "url_citation":
                            web_search_results.append({
                                "title": getattr(ann, "title", ""),
                                "url": getattr(ann, "url", ""),
                            })

    # 3) Collect images from the built-in image tool (Responses)
    generated_images = []
    for item in (getattr(response, "output", None) or []):
        if getattr(item, "type", None) == "image_generation_call" and \
           getattr(item, "status", None) == "completed":
            # In current SDKs, base64 is often at item.result
            b64 = getattr(item, "result", None)
            if b64:
                generated_images.append({"b64_json": b64})
            # Some SDK versions might have a URL
            url = getattr(item, "url", None)
            if url and not b64:
                generated_images.append({"url": url})

    # Parse to dataclasses with image data
    tracks_parsed: t.List[Track] = []
    for idx, t_item in enumerate(data["tracks"]):
        try:
            sections = [TrackSection(label=s["label"], lines=s["lines"]) for s in t_item["lyrics"]]

            # Map images to tracks by index
            album_art = None
            album_art_prompt = None

            if idx < len(generated_images):
                img = generated_images[idx]
                album_art = img.get("b64_json") or img.get("url")
                album_art_prompt = f"Auto image for {t_item.get('title', 'track')}"

            tracks_parsed.append(
                Track(
                    title=t_item["title"],
                    version=t_item["version"],
                    style_block=t_item["style_block"],
                    lyrics=sections,
                    metadata=t_item.get("metadata", {}),
                    album_art=album_art,
                    album_art_prompt=album_art_prompt
                )
            )
        except KeyError as e:
            raise RuntimeError(f"Track {idx + 1} missing required field: {e}")

    return SongPack(
        subject=data["subject"],
        style_prompt=data["style_prompt"],
        tracks=tracks_parsed,
        web_search_results=web_search_results if web_search_results else None
    )


def save_txt_version(pack: SongPack, output_path: Path) -> Path:
    """
    Save a simplified text version of the tracks with style and lyrics.

    Args:
        pack: SongPack containing the generated tracks
        output_path: Base path for the output file

    Returns:
        Path to the saved text file
    """
    txt_content = []

    # Header
    txt_content.append("=" * 60)
    txt_content.append(f"GENERATED TRACKS")
    txt_content.append(f"Subject: {pack.subject}")
    txt_content.append(f"Style: {pack.style_prompt}")
    txt_content.append("=" * 60)

    for track in pack.tracks:
        txt_content.append("")
        txt_content.append(f"TITLE: {track.title}")
        txt_content.append("-" * 40)

        # Complete style info
        txt_content.append("\nSTYLE:")
        style = track.style_block
        txt_content.append(f"  Genre: {style.get('genre', 'N/A')}")
        txt_content.append(f"  Tempo: {style.get('tempo_bpm', 'N/A')} BPM")
        txt_content.append(f"  Key: {style.get('key', 'N/A')}")
        txt_content.append(f"  Meter: {style.get('meter', 'N/A')}")
        txt_content.append(f"  Structure: {style.get('structure', 'N/A')}")
        txt_content.append("")
        txt_content.append("  Production Details:")
        if style.get('guitars'):
            txt_content.append(f"    Guitars: {style.get('guitars')}")
        if style.get('bass'):
            txt_content.append(f"    Bass: {style.get('bass')}")
        if style.get('synths'):
            txt_content.append(f"    Synths: {style.get('synths')}")
        if style.get('drums'):
            txt_content.append(f"    Drums: {style.get('drums')}")
        if style.get('vocals'):
            txt_content.append(f"    Vocals: {style.get('vocals')}")
        if style.get('fx'):
            txt_content.append(f"    FX: {style.get('fx')}")
        if style.get('mix_notes'):
            txt_content.append(f"    Mix Notes: {style.get('mix_notes')}")

        # Hook
        if track.metadata.get('hook'):
            txt_content.append(f"\nHOOK: \"{track.metadata['hook']}\"")

        # Lyrics
        txt_content.append("\nLYRICS:")
        txt_content.append("-" * 20)
        for section in track.lyrics:
            txt_content.append(f"\n[{section.label}]")
            for line in section.lines:
                txt_content.append(line)

        txt_content.append("\n" + "=" * 60)

    # Save to file
    txt_path = output_path.with_suffix('.txt')
    txt_path.write_text('\n'.join(txt_content), encoding='utf-8')

    return txt_path


def slugify(text: str) -> str:
    """
    Convert text to a URL-friendly slug.

    Args:
        text: Input text to slugify

    Returns:
        Slugified text suitable for filenames
    """
    # Convert to lowercase and strip whitespace
    s = text.lower().strip()

    # Remove non-word characters (keep alphanumeric, spaces, hyphens, underscores)
    s = re.sub(r'[^\w\s-]', '', s)

    # Replace spaces and underscores with hyphens
    s = re.sub(r'[\s_-]+', '-', s)

    # Remove leading/trailing hyphens
    s = re.sub(r'^-+|-+$', '', s)

    # Return slug or default if empty
    return s or "tracks"


def save_image(image_data: str, output_path: Path, track_title: str) -> t.Optional[Path]:
    """
    Save generated album art image to disk.

    Args:
        image_data: Base64 encoded image data or URL
        output_path: Base directory for saving
        track_title: Title of the track for naming

    Returns:
        Path to saved image or None if failed
    """
    if not image_data:
        return None

    try:
        # Create images directory if it doesn't exist
        images_dir = output_path.parent / "album_art"
        images_dir.mkdir(exist_ok=True)

        # Generate filename
        filename = f"{slugify(track_title)}.png"
        image_path = images_dir / filename

        if image_data.startswith('http'):
            # If it's a URL, note it in a reference file
            ref_file = images_dir / f"{slugify(track_title)}_url.txt"
            ref_file.write_text(image_data, encoding="utf-8")
            print(f"  Album art URL saved to {ref_file}", file=sys.stderr)
            return ref_file
        else:
            # Decode base64 and save image
            image_bytes = base64.b64decode(image_data)
            image_path.write_bytes(image_bytes)
            print(f"  Album art saved to {image_path}", file=sys.stderr)
            return image_path

    except Exception as e:
        print(f"  Warning: Could not save album art for {track_title}: {e}", file=sys.stderr)
        return None


def main() -> int:
    """
    Main entry point for the script.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Generate original song tracks using GPT-5 with web search and image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subject "robot dreams" --style "synthwave" --tracks 2
  %(prog)s --subject "ocean voyage" --style "folk rock" --dry-run
  %(prog)s --subject "2025 music trends" --style "modern pop" --reasoning-effort high
  %(prog)s --subject "city lights" --style "jazz fusion" --force
  %(prog)s --subject "retro gaming" --style "chiptune" --verbosity concise
  %(prog)s --subject "classical remix" --style "orchestral" --disable-web-search
  %(prog)s --subject "album concept" --style "progressive rock" --disable-image-generation

GPT-5 Features:
  - High reasoning effort for complex creative tasks
  - Native web search for current trends and references
  - Image generation for visual album concepts
  - Adjustable verbosity for different output lengths
        """
    )

    # Required arguments
    parser.add_argument("--subject", required=True,
                       help="Creative subject for songs (e.g., 'cat girl in space')")
    parser.add_argument("--style", required=True,
                       help="Musical style (e.g., 'synth metal with country spice')")

    # Optional arguments
    parser.add_argument("--tracks", type=int, default=2,
                       help=f"Number of tracks to generate ({MIN_TRACKS}-{MAX_TRACKS}, default: 2)")
    parser.add_argument("--version-prefix", default="v1.0",
                       help="Starting version tag (default: v1.0)")
    parser.add_argument("--model", default=None,
                       help=f"OpenAI model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                       default=DEFAULT_REASONING_EFFORT,
                       help=f"GPT-5 reasoning depth (default: {DEFAULT_REASONING_EFFORT})")
    parser.add_argument("--verbosity", choices=["low", "medium", "high"],
                       default=DEFAULT_VERBOSITY,
                       help=f"Response verbosity (default: {DEFAULT_VERBOSITY})")
    parser.add_argument("--disable-web-search", action="store_true",
                       help="Disable GPT-5's native web search capability")
    parser.add_argument("--disable-image-generation", action="store_true",
                       help="Disable GPT-5's image generation for album art")
    parser.add_argument("--outfile", default=None,
                       help="Output file path (default: auto-generated)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing files without asking")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be generated without calling API")

    args = parser.parse_args()

    # Validate arguments
    if not MIN_TRACKS <= args.tracks <= MAX_TRACKS:
        print(f"Error: --tracks must be between {MIN_TRACKS} and {MAX_TRACKS}", file=sys.stderr)
        return 2

    # Check for API key early
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY='your-key-here'", file=sys.stderr)
        return 1

    # Determine output file
    if args.outfile:
        out_path = Path(args.outfile)
        # Validate path security
        try:
            out_path = out_path.resolve()
            if not out_path.parent.exists():
                print(f"Error: Output directory does not exist: {out_path.parent}", file=sys.stderr)
                return 2
        except Exception as e:
            print(f"Error: Invalid output path: {e}", file=sys.stderr)
            return 2
    else:
        out_path = Path(f"tracks_{slugify(args.subject)}_{int(time.time())}.json")

    # Check for file overwrite
    if out_path.exists() and not args.force:
        response = input(f"File {out_path} already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.", file=sys.stderr)
            return 0

    # Dry run mode
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print(f"Would generate {args.tracks} track(s)")
        print(f"Subject: {args.subject}")
        print(f"Style: {args.style}")
        print(f"Model: {args.model or DEFAULT_MODEL}")
        print(f"Reasoning Effort: {args.reasoning_effort}")
        print(f"Verbosity: {args.verbosity}")
        print(f"Web Search: {'Enabled' if not args.disable_web_search else 'Disabled'}")
        print(f"Image Generation: {'Enabled' if not args.disable_image_generation else 'Disabled'}")
        print(f"Output: {out_path}")
        print("\nNo API call made.")
        return 0

    # Generate tracks
    try:
        print(f"Generating {args.tracks} track(s)...", file=sys.stderr)

        pack = generate_tracks(
            subject=args.subject,
            style=args.style,
            tracks=args.tracks,
            version_prefix=args.version_prefix,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            enable_web_search=not args.disable_web_search,
            enable_image_generation=not args.disable_image_generation,
        )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    # Prepare output with enhanced data
    obj = {
        "subject": pack.subject,
        "style_prompt": pack.style_prompt,
        "tracks": [],
        "generated_at": int(time.time()),
        "model": args.model or DEFAULT_MODEL,
        "reasoning_effort": args.reasoning_effort,
        "verbosity": args.verbosity,
        "tools_used": {
            "web_search": not args.disable_web_search,
            "image_generation": not args.disable_image_generation
        }
    }

    # Add web search results if available
    if pack.web_search_results:
        obj["web_search_citations"] = pack.web_search_results
        print("\nWeb Search Citations:", file=sys.stderr)
        for citation in pack.web_search_results[:3]:  # Show first 3 citations
            print(f"  - {citation.get('title', 'N/A')}: {citation.get('url', 'N/A')}", file=sys.stderr)

    # Process tracks and save images
    image_paths = []
    for tr in pack.tracks:
        track_data = {
            "title": tr.title,
            "version": tr.version,
            "style_block": tr.style_block,
            "lyrics": [{"label": s.label, "lines": s.lines} for s in tr.lyrics],
            "metadata": tr.metadata,
        }

        # Save album art if generated
        if tr.album_art:
            image_path = save_image(tr.album_art, out_path, tr.title)
            if image_path:
                track_data["album_art_path"] = str(image_path)
                image_paths.append(image_path)

        if tr.album_art_prompt:
            track_data["album_art_prompt"] = tr.album_art_prompt

        obj["tracks"].append(track_data)

    payload = json.dumps(obj, ensure_ascii=False, indent=2)

    # Display output
    print(payload)

    # Save to file
    try:
        out_path.write_text(payload, encoding="utf-8")
        print(f"\n✓ Saved JSON to {out_path}", file=sys.stderr)

        # Save simplified text version
        txt_path = save_txt_version(pack, out_path)
        print(f"✓ Saved lyrics to {txt_path}", file=sys.stderr)

        if image_paths:
            print(f"✓ Generated {len(image_paths)} album art image(s)", file=sys.stderr)

    except Exception as e:
        print(f"\nWarning: Could not save to file: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())