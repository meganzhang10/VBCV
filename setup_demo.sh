#!/bin/bash
# One-command setup for the VBCV volleyball tracker demo.
# Run this after cloning the repo:
#   git clone https://github.com/meganzhang10/VBCV && cd VBCV
#   bash setup_demo.sh

set -e

echo "=== VBCV Volleyball Tracker Setup ==="
echo ""

# 1. Python venv
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[1/4] Virtual environment already exists."
fi

source .venv/bin/activate

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install -q -r requirements.txt
pip install -q gradio anthropic

# 3. Check model weights
if [ -f "models/volleyball_ball_yolo/weights/best.pt" ]; then
    echo "[3/4] Model weights found."
else
    echo "[3/4] Model weights not found. Training required."
    echo "  Run: python3 -m scripts.setup_and_train --api-key YOUR_ROBOFLOW_KEY"
    echo "  Get a free key at: https://app.roboflow.com/settings/api"
fi

# 4. Download sample videos if not present
if [ -d "data/raw/videos/side_angle" ] && [ "$(ls data/raw/videos/side_angle/*.mp4 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[4/4] Sample videos found."
else
    echo "[4/4] Downloading sample volleyball clips..."
    pip install -q yt-dlp
    mkdir -p data/raw/videos/side_angle

    # Tokyo Olympic rallies
    yt-dlp --download-sections "*0:05-0:35" \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "data/raw/videos/side_angle/tokyo_olympic_rallies.%(ext)s" \
        "https://www.youtube.com/watch?v=6e5_A9QCoC0" 2>/dev/null || echo "  (clip 1 failed, skipping)"

    # VNL 2024
    yt-dlp --download-sections "*0:05-0:35" \
        -f "bestvideo[height<=720]+bestaudio/best[height<=720]" \
        --merge-output-format mp4 \
        -o "data/raw/videos/side_angle/vnl_2024_rally.%(ext)s" \
        "https://www.youtube.com/watch?v=0lN1HfFAYUY" 2>/dev/null || echo "  (clip 2 failed, skipping)"

    echo "  Done. Videos saved to data/raw/videos/side_angle/"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To launch the demo:"
echo "  source .venv/bin/activate"
echo "  python3 -m src.visualization.demo_gradio"
echo ""
echo "Then open http://localhost:7860 in your browser."
echo "Add share=True in demo_gradio.py to get a public link for remote demos."
