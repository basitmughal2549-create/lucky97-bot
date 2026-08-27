import os
import logging
import re
import asyncio
from collections import Counter
from typing import List, Dict, Tuple, Optional
import io

import cv2
import numpy as np
import easyocr
from PIL import Image
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

load_dotenv()

# ========================== CONFIGURATION ==========================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN not set in .env")

MAX_HISTORY = 20
CONFIDENCE_THRESHOLD = 60

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ========================== OCR ENGINE ==========================
reader = easyocr.Reader(['en'], gpu=False)

# ========================== UTILITIES ==========================

def download_image_as_opencv(file_obj) -> np.ndarray:
    """Download Telegram photo and convert to OpenCV BGR."""
    image_bytes = file_obj.download_as_bytearray()
    pil_image = Image.open(io.BytesIO(image_bytes))
    # Convert to RGB then BGR for OpenCV
    rgb = np.array(pil_image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def ocr_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> str:
    """Crop region and run OCR, return concatenated text."""
    if w <= 0 or h <= 0:
        return ""
    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        return ""
    # Resize if too small
    if crop.shape[0] < 30 or crop.shape[1] < 30:
        scale = max(2, int(40 / min(crop.shape[:2])))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    results = reader.readtext(crop, detail=0, paragraph=False)
    return " ".join(results).strip()

def classify_roll(n: int) -> str:
    if 2 <= n <= 6:
        return "Down"
    elif n == 7:
        return "Seven"
    elif 8 <= n <= 12:
        return "Up"
    return "Unknown"

def color_emoji_for_roll(n: int) -> str:
    cls = classify_roll(n)
    if cls == "Down":
        return "🔴"
    elif cls == "Seven":
        return "🔵"
    elif cls == "Up":
        return "🟢"
    return "⚪"

# ========================== IMAGE PROCESSING ==========================

def detect_ante_tier(image: np.ndarray) -> Tuple[str, int]:
    """Detect room tier from bottom chip area."""
    h, w, _ = image.shape
    bottom_roi = image[int(h*0.8):h, 0:w]
    text = ocr_region(image, 0, int(h*0.8), w, int(h*0.2))
    # Search for chip values
    match = re.search(r'\b(10|50|100|500|1000|5000)\b', text)
    if match:
        val = int(match.group(1))
        if val == 10:
            return "Room 1 (10 Ante)", 10
        elif val == 50:
            return "Room 2 (50 Ante)", 50
        elif val == 100:
            return "Room 3 (100 Ante)", 100
        else:
            return f"Room ({val} Ante)", val
    return "Unknown", 0

def extract_history_numbers(image: np.ndarray) -> List[int]:
    """Extract top history bar numbers."""
    h, w, _ = image.shape
    top_region = image[0:int(h*0.12), 0:w]
    text = ocr_region(image, 0, 0, w, int(h*0.12))
    # Find numbers 2-12
    numbers = re.findall(r'\b([2-9]|10|11|12)\b', text)
    return [int(n) for n in numbers]

def extract_player_fingerprint(image: np.ndarray) -> Dict:
    """Extract usernames, titles, balances for table hash."""
    h, w, _ = image.shape
    full_text = ocr_region(image, 0, 0, w, h)
    usernames = re.findall(r'name\d+', full_text)
    titles = re.findall(r'Richie Rich|Millionaire|Human Calculator|Naomi', full_text)
    balances = re.findall(r'\d{1,3}(?:,\d{3})*\.\d{2}', full_text)
    # Generate simple hash from sorted unique usernames and titles
    unique_usernames = sorted(set(usernames))[:5]
    unique_titles = sorted(set(titles))[:3]
    hash_str = f"{'_'.join(unique_usernames)}_{'_'.join(unique_titles)}"
    if not hash_str or hash_str == "_":
        hash_str = "unknown_table"
    return {
        "usernames": usernames,
        "titles": titles,
        "balances": balances,
        "hash": hash_str
    }

# ========================== PREDICTION ENGINE ==========================

def compute_prediction(history: List[int]) -> Dict:
    """Return prediction dict with zone, confidence, trend, frequencies."""
    if not history:
        return {
            "zone": "Unknown",
            "confidence": 0,
            "trend": "No history data",
            "frequencies": {}
        }
    recent = history[-MAX_HISTORY:]
    classes = [classify_roll(n) for n in recent]
    counts = Counter(classes)
    total = len(recent)
    freq = {z: counts.get(z, 0)/total for z in ["Down", "Seven", "Up"]}
    
    # Dominant zone
    dominant = max(freq, key=freq.get) if freq else "Unknown"
    confidence = int(freq.get(dominant, 0) * 100)
    
    # Streak detection on last 5
    streak = None
    if len(classes) >= 3:
        last5 = classes[-5:]
        if all(c == last5[0] for c in last5):
            streak = last5[0]
    if streak and streak == dominant:
        confidence = min(100, confidence + 15)
    elif streak and streak != dominant:
        confidence = max(0, confidence - 10)
    
    # Trend description
    if confidence >= 80:
        trend = f"Strong {dominant} trend ({confidence}%)"
    elif confidence >= 60:
        trend = f"Moderate {dominant} bias"
    else:
        trend = "Mixed / unclear pattern"
    if counts.get("Seven", 0) >= 3 and total >= 10:
        if freq.get("Seven", 0) > 0.25:
            trend += " | High Seven frequency"
    
    return {
        "zone": dominant,
        "confidence": confidence,
        "trend": trend,
        "frequencies": freq,
        "streak": streak
    }

def format_history(history: List[int]) -> str:
    """Format last up to 12 rolls with color emojis."""
    if not history:
        return "No history"
    last = history[-12:]
    parts = []
    for n in last:
        em = color_emoji_for_roll(n)
        parts.append(f"{n} {em}")
    return " -> ".join(parts)

# ========================== TELEGRAM HANDLERS ==========================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👑 Welcome to Frost AI 7 Up 7 Down Predictor!\n"
        "Send me a screenshot of your table and I'll analyze and predict."
    )

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logger.info(f"Photo from {user.username or user.id}")
    photo = update.message.photo[-1]
    file = await photo.get_file()
    
    try:
        image = download_image_as_opencv(file)
    except Exception as e:
        logger.error(f"Download error: {e}")
        await update.message.reply_text("❌ Failed to download image.")
        return
    
    try:
        # 1. Room tier
        tier_name, ante = detect_ante_tier(image)
        # 2. History
        history = extract_history_numbers(image)
        if not history:
            await update.message.reply_text("⚠️ Could not read history numbers. Ensure top bar is clear.")
            return
        # 3. Fingerprint
        fp = extract_player_fingerprint(image)
        table_hash = fp.get("hash", "unknown")
        high_rollers = ", ".join(fp.get("titles", [])) or "None detected"
        # 4. Prediction
        pred = compute_prediction(history)
        zone = pred["zone"]
        confidence = pred["confidence"]
        trend = pred["trend"]
        # 5. Build signal text
        if zone == "Down":
            signal = "🔴🔴🔴 BET ON 2-6 (DOWN / RED) 🔴🔴🔴"
        elif zone == "Seven":
            signal = "🔵🔵🔵 HIGH ALERT: BET ON 7 (EXACT / BLUE) 🔵🔵🔵"
        elif zone == "Up":
            signal = "🟢🟢🟢 BET ON 8-12 (UP / GREEN) 🟢🟢🟢"
        else:
            signal = "⚠️ No clear signal"
        # 6. Stake suggestion
        if confidence >= 80:
            stake = f"{ante} Coins (High confidence)"
        elif confidence >= 60:
            stake = f"{ante} Coins (Moderate)"
        else:
            stake = f"{ante//2 or 1} Coins (Low risk)"
        # 7. Format history display
        history_str = format_history(history)
        # 8. Build final message
        msg = (
            f"👑 FROST AI | 7 UP 7 DOWN PREDICTION 👑\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 ROOM DETAILS:\n"
            f"* Room Tier: {tier_name}\n"
            f"* Table Hash ID: #{table_hash[:12]}\n"
            f"* Active High-Rollers: {high_rollers}\n\n"
            f"📊 EXTRACTED HISTORY:\n"
            f"* Last Rolls: {history_str}\n"
            f"* Trend Analysis: {trend}\n\n"
            f"🎯 NEXT ROUND SIGNAL:\n"
            f"{signal}\n\n"
            f"🔥 CONFIDENCE LEVEL: {confidence}%\n"
            f"💰 SUGGESTED STAKE: {stake}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Status: 24/7 Real-Time Table Tracking"
        )
        await update.message.reply_text(msg)
    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await update.message.reply_text("❌ Error processing image. Try a clearer screenshot.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.warning(f"Update {update} caused error {context.error}")

# ========================== MAIN ==========================

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    app.add_error_handler(error_handler)
    logger.info("Bot started polling...")
    app.run_polling(allowed_updates=constants.UpdateType.MESSAGE)

if __name__ == "__main__":
    main()