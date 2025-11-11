# lyrics.py - AI-Powered Song Generation Tool

## Overview

`lyrics.py` is a single-file Python utility that generates complete, radio-ready song tracks using OpenAI's GPT-5 model via the Responses API. It creates professional-quality songs with detailed production specifications, structured lyrics, and optional album art.

## Features

- **GPT-5 Integration** - Leverages the GPT-5 model with high reasoning effort for superior creative output
- **Multi-Modal Output** - Generates lyrics, production specs, and album art in a single API call
- **Responses API** - Uses OpenAI's Responses API for orchestrated multi-tool workflows
- **Web Search Integration** - Native web search with high context for current trends and references
- **Image Generation** - Automatic album art generation using gpt-image-1
- **Dual Output Formats** - Saves both JSON (complete data) and TXT (readable lyrics)
- **Professional Production Details** - Industry-standard specifications including BPM, key, arrangement, and mix notes

## Installation

### Requirements

```bash
pip install openai
```

### Environment Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

Optional: Override the default model:

```bash
export OPENAI_MODEL='gpt-5'  # This is the default
```

## Usage

### Basic Usage

```bash
python lyrics.py --subject "cat girl in space" --style "synth metal" --tracks 2
```

### Common Examples

```bash
# Generate a single track with custom settings
python lyrics.py --subject "late night drives" --style "modern R&B" --tracks 1

# High-energy dance track
python lyrics.py --subject "summer party" --style "dance-pop with funk" --tracks 1 --verbosity high

# Disable web search for offline generation
python lyrics.py --subject "classic love song" --style "acoustic ballad" --disable-web-search

# Dry run to preview settings without API call
python lyrics.py --subject "test" --style "rock" --dry-run

# Custom output location
python lyrics.py --subject "ocean dreams" --style "ambient" --outfile my-song.json --force
```

### Command-Line Arguments

#### Required Arguments

- `--subject` - Creative subject for the songs (e.g., "cat girl in space")
- `--style` - Musical style description (e.g., "synth metal with country spice")

#### Optional Arguments

- `--tracks N` - Number of tracks to generate (1-5, default: 2)
- `--version-prefix VERSION` - Starting version tag (default: v1.0)
- `--model MODEL` - OpenAI model to use (default: gpt-5)
- `--reasoning-effort LEVEL` - Reasoning depth: low, medium, high (default: high)
- `--verbosity LEVEL` - Response verbosity: low, medium, high (default: medium)
- `--disable-web-search` - Disable GPT-5's native web search capability
- `--disable-image-generation` - Disable album art generation
- `--outfile PATH` - Output file path (default: auto-generated)
- `--force` - Overwrite existing files without asking
- `--dry-run` - Show what would be generated without calling API

## Design & Architecture

### API Structure

The tool uses OpenAI's **Responses API** for multi-modal workflows:

```python
client.responses.create(
    model="gpt-5",
    instructions="...",  # System-level instructions
    input="...",         # User input
    tools=[
        {"type": "web_search"},
        {"type": "image_generation"}
    ],
    reasoning={"effort": "high"},
    text={"verbosity": "medium"},
    max_output_tokens=1000000
)
```

**Key Differences from Chat Completions:**
- Uses `instructions` + `input` (not `messages`)
- Tools are simplified objects without config fields
- Response parsed from `response.output_text`
- Tool results extracted from `response.output` items
- No `temperature` or `response_format` parameters

### Response Parsing

The tool implements robust parsing with multiple fallback strategies:

1. **Primary**: Extract from `response.output_text`
2. **Fallback**: Stitch together from `response.output` message items
3. **Markdown Cleanup**: Removes code blocks if present
4. **Regex Extraction**: Falls back to JSON pattern matching if needed

### Tool Integration

#### Web Search

When enabled, the tool:
- Adds `{"type": "web_search"}` to the tools array
- Extracts citations from `response.output` annotations
- Displays first 3 citations in console output
- Saves all citations in JSON output

#### Image Generation

When enabled, the tool:
- Adds `{"type": "image_generation"}` to the tools array
- Extracts images from `response.output` items with type `image_generation_call`
- Saves base64-encoded images to `album_art/` directory
- Associates images with tracks by index

### Data Models

```python
@dataclass
class TrackSection:
    label: str           # e.g., "Verse 1", "Chorus"
    lines: List[str]     # Lyric lines

@dataclass
class Track:
    title: str
    version: str
    style_block: Dict[str, Any]       # Production details
    lyrics: List[TrackSection]
    metadata: Dict[str, Any]
    album_art: Optional[str]          # Base64 or URL
    album_art_prompt: Optional[str]

@dataclass
class SongPack:
    subject: str
    style_prompt: str
    tracks: List[Track]
    web_search_results: Optional[List[Dict[str, str]]]
```

### JSON Schema

The tool provides GPT-5 with a strict JSON schema defining:

- **Track Structure**: title, version, style_block, lyrics, metadata
- **Style Block**: genre, tempo_bpm, key, meter, guitars, bass, synths, drums, vocals, fx, structure, mix_notes
- **Lyrics**: Array of sections with label and lines
- **Metadata**: hook (required), tags (optional)

The schema enforces structure but allows flexibility with `additionalProperties: true` on style_block and metadata.

## Output Formats

### JSON Output (`tracks_*.json`)

Complete data structure with:
- Subject and style prompt
- Full track array with all production details
- Generated timestamp and model info
- Reasoning effort and verbosity settings
- Tools used flags
- Web search citations (if used)
- Album art paths (if generated)

Example structure:
```json
{
  "subject": "...",
  "style_prompt": "...",
  "tracks": [{
    "title": "Song Title v1.0",
    "version": "v1.0",
    "style_block": { /* production details */ },
    "lyrics": [ /* sections */ ],
    "metadata": { /* hook, tags */ },
    "album_art_path": "album_art/song-title-v10.png"
  }],
  "web_search_citations": [ /* if used */ ],
  "generated_at": 1762767405,
  "model": "gpt-5",
  "reasoning_effort": "high",
  "verbosity": "medium"
}
```

### Text Output (`tracks_*.txt`)

Readable, formatted version with:

```
============================================================
GENERATED TRACKS
Subject: [subject]
Style: [style prompt]
============================================================

TITLE: [Song Title]
----------------------------------------

STYLE:
  Genre: [genre]
  Tempo: [bpm] BPM
  Key: [key]
  Meter: [meter]
  Structure: [structure]

  Production Details:
    Guitars: [guitar arrangement]
    Bass: [bass details]
    Synths: [synth setup]
    Drums: [drum pattern]
    Vocals: [vocal style]
    FX: [effects]
    Mix Notes: [mixing instructions]

HOOK: "[memorable hook phrase]"

LYRICS:
--------------------

[Section Label]
Line 1
Line 2
...
```

### Album Art Output

When image generation is enabled:
- Images saved to `album_art/` directory
- Filenames: `{slugified-track-title}.png`
- Base64-decoded PNG format
- 1024x1024 resolution (standard album art size)

## Model-Specific Behavior

### GPT-5 Models

For models starting with "gpt-5":
- Uses `reasoning={"effort": "high"}` by default
- Supports `text={"verbosity": "medium"}` parameter
- Native web search with high context
- Native image generation capability

### O-Series Models

For O1, O3, O4 models:
- Uses `reasoning={"effort": ...}` format
- Omits verbosity parameter (not supported)
- Web search and image generation available

### Other Models

For models like gpt-4o:
- Omits reasoning parameter (not supported)
- Basic Responses API functionality only

## Security & Validation

### Input Validation

All user inputs (subject, style) are validated:
- Length limits (max 200 characters)
- Control character removal
- Basic XSS/injection pattern detection
- Empty string checks

### Path Security

Output file paths are validated:
- Resolved to absolute paths
- Parent directory existence checks
- Overwrite protection (requires `--force` or confirmation)

### API Key Protection

- Checks for `OPENAI_API_KEY` before API calls
- Clear error messages for missing credentials
- Explicit API key passing (not relying on defaults)

## Error Handling

Comprehensive error handling for:

- **Authentication Errors** - Invalid API key detection
- **Rate Limiting** - Graceful handling with clear messages
- **API Errors** - Specific error types caught and reported
- **JSON Parsing** - Multiple fallback strategies for malformed responses
- **File I/O** - Permission and disk space error handling
- **Validation** - Input, parameter, and response validation

## File Naming

Output files use slugified subjects with timestamps:

```
tracks_{slugified-subject}_{unix-timestamp}.json
tracks_{slugified-subject}_{unix-timestamp}.txt
album_art/{slugified-track-title}.png
```

## Performance

- **Generation Time**: ~30-90 seconds per track (depending on reasoning effort)
- **Token Usage**: Variable based on complexity (typically 10k-50k tokens)
- **Max Output**: 1 million tokens supported
- **API Calls**: Single request per generation (efficient)

