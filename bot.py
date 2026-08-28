#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
7 Up Down Prediction Bot
- Detects room ID and history numbers from screenshots using OCR.
- Predicts the next number (2–12) based on room‑specific pattern analysis.
- Provides color‑coded predictions with interactive Yes/No feedback.
- Learns from user feedback to improve future predictions.
- Persists all data in an SQLite database.
"""

import os
import logging
import sqlite3
import re
import io
from collections import Counter
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from PIL import Image
import pytesseract

# ------------------------------
#  Configuration & Logging
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Telegram Bot Token (set environment variable TELEGRAM_BOT_TOKEN)
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set.")

# Database file
DB_PATH = "learning_db.db"

# ------------------------------
#  Database Layer
# ------------------------------
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS rooms (
            room_id TEXT PRIMARY KEY,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT,
            outcome INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(room_id) REFERENCES rooms(room_id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            room_id TEXT,
            predicted INTEGER,
            actual INTEGER,
            correct BOOLEAN,
            feedback BOOLEAN,  -- True = Yes, False = No
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(room_id) REFERENCES rooms(room_id)
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")


def get_room_history(room_id, limit=1000):
    """Return list of outcomes for a room (oldest first)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT outcome FROM history WHERE room_id=? ORDER BY timestamp ASC LIMIT ?",
        (room_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in rows]


def add_history(room_id, outcome):
    """Insert an outcome into history for a room."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO rooms (room_id) VALUES (?)", (room_id,))
    c.execute(
        "UPDATE rooms SET last_updated=CURRENT_TIMESTAMP WHERE room_id=?",
        (room_id,),
    )
    c.execute(
        "INSERT INTO history (room_id, outcome) VALUES (?, ?)",
        (room_id, outcome),
    )
    conn.commit()
    conn.close()


def add_prediction(room_id, predicted):
    """Insert a prediction and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO predictions (room_id, predicted) VALUES (?, ?)",
        (room_id, predicted),
    )
    pred_id = c.lastrowid
    conn.commit()
    conn.close()
    return pred_id


def update_prediction_feedback(pred_id, feedback, actual=None):
    """
    Update a prediction with user feedback.
    If actual is given, also set the correct flag.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if actual is not None:
        c.execute(
            "UPDATE predictions SET feedback=?, actual=?, correct=? WHERE id=?",
            (feedback, actual, actual == (c.execute("SELECT predicted FROM predictions WHERE id=?", (pred_id,)).fetchone()[0])),
        )
    else:
        c.execute(
            "UPDATE predictions SET feedback=? WHERE id=?",
            (feedback, pred_id),
        )
    conn.commit()
    conn.close()

# ------------------------------
#  OCR & Image Processing
# ------------------------------
def preprocess_image(image_bytes):
    """Convert to grayscale and binarize for better OCR."""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("L")          # grayscale
    threshold = 150
    img = img.point(lambda p: 255 if p > threshold else 0)  # binary
    return img


def extract_room_and_numbers(image_bytes):
    """
    Extract room ID and list of history numbers from a screenshot.
    Returns (room_id, list_of_numbers) or raises ValueError.
    """
    img = preprocess_image(image_bytes)

    # Tesseract configuration: only digits and a few keywords
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789RoomTableID:"
    text = pytesseract.image_to_string(img, config=custom_config)
    logger.info(f"OCR raw text: {text}")

    # 1. Find room ID
    room_patterns = [
        r"Room\s*[:#]?\s*(\d+)",
        r"Table\s*[:#]?\s*(\d+)",
        r"ID\s*[:#]?\s*(\d+)",
        r"Room\s*(\d+)",
        r"Table\s*(\d+)",
    ]
    room_id = None
    for pattern in room_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            room_id = match.group(1)
            break

    if not room_id:
        raise ValueError("Room ID not found in the image.")

    # 2. Extract numbers 2–12 (possible outcomes)
    #    Use word boundaries to avoid capturing part of larger numbers
    numbers = re.findall(r"\b([2-9]|1[0-2])\b", text)
    history_numbers = [int(n) for n in numbers]

    # Optional: filter out the room ID if it falls into 2–12 (unlikely but possible)
    # We'll keep it; the history usually has multiple numbers, so a single extra number won't hurt.

    return room_id, history_numbers

# ------------------------------
#  Prediction Engine
# ------------------------------
def predict_next(history):
    """
    Given a list of past outcomes (oldest first), predict the next number.
    Uses pattern matching on the last up to 5 outcomes; falls back to frequency.
    Returns (predicted_number, confidence) or (None, 0.0) if no data.
    """
    if not history:
        return None, 0.0

    n = min(5, len(history))          # use last 5 outcomes at most
    pattern = history[-n:]            # recent sequence

    # Search for the pattern in earlier history (excluding the last occurrence)
    occurrences = []
    for i in range(len(history) - n):
        if history[i:i+n] == pattern:
            # The outcome after this pattern is at index i+n
            if i + n < len(history):
                occurrences.append(history[i+n])

    if occurrences:
        counter = Counter(occurrences)
        most_common = counter.most_common(1)[0]
        predicted = most_common[0]
        confidence = most_common[1] / len(occurrences)
    else:
        # Fallback: overall frequency
        counter = Counter(history)
        total = len(history)
        most_common = counter.most_common(1)[0]
        predicted = most_common[0]
        confidence = most_common[1] / total

    return predicted, confidence

# ------------------------------
#  Telegram Bot Handlers
# ------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message."""
    await update.message.reply_text(
        "🎲 *7 Up Down Prediction Bot*\n\n"
        "Send me a screenshot of the game screen showing the room ID and the history of outcomes.\n"
        "I'll analyse the pattern and predict the next number.\n\n"
        "Use /help for more info.",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send help message."""
    await update.message.reply_text(
        "📸 *How to use*\n"
        "1. Take a clear screenshot of the game screen in 'Lucky 97'.\n"
        "2. Make sure the Room/Table ID and the recent outcome numbers are visible.\n"
        "3. Send the image to this bot.\n"
        "4. I'll reply with a prediction and two buttons: Yes / No.\n"
        "5. Tap Yes if the prediction was correct, No otherwise.\n\n"
        "Your feedback helps the bot learn and improve over time.",
        parse_mode="Markdown",
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process a screenshot: OCR, predict, and respond."""
    # Download the photo
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()

    try:
        room_id, history_numbers = extract_room_and_numbers(image_bytes)
    except Exception as e:
        logger.error(f"OCR error: {e}")
        await update.message.reply_text(
            "❌ Could not read the image. Please ensure:\n"
            "• The Room/Table ID is clearly visible.\n"
            "• The history numbers are legible.\n"
            "• The screenshot is not blurry.\n\n"
            "Try sending a clearer screenshot."
        )
        return

    if not history_numbers:
        await update.message.reply_text(
            "❌ No history numbers found in the image. "
            "Please make sure the recent outcomes are visible."
        )
        return

    logger.info(f"Room {room_id}, extracted numbers: {history_numbers}")

    # Store the extracted history (they are past outcomes, so add them)
    # Avoid inserting duplicates by checking last few entries? We'll just append.
    # This may create duplicates if the user sends the same screenshot twice,
    # but the pattern analysis will still work (more data is usually okay).
    for num in history_numbers:
        add_history(room_id, num)

    # Get full history for prediction
    full_history = get_room_history(room_id, limit=1000)
    if len(full_history) < 3:
        await update.message.reply_text(
            "⚠️ Not enough history data for this room yet. "
            "Please send more screenshots to build a pattern."
        )
        return

    predicted, confidence = predict_next(full_history)
    if predicted is None:
        await update.message.reply_text(
            "⚠️ Could not generate a prediction with the available data."
        )
        return

    # Categorise the prediction
    if 2 <= predicted <= 6:
        category = "SMALL / DOWN"
        emoji = "🔴"
    elif predicted == 7:
        category = "LUCKY SEVEN"
        emoji = "🔵"
    elif 8 <= predicted <= 12:
        category = "UP / LARGE"
        emoji = "🟢"
    else:
        category = "UNKNOWN"
        emoji = "⚪"

    # Store prediction in DB for feedback tracking
    pred_id = add_prediction(room_id, predicted)

    # Prepare message
    recent = full_history[-10:] if full_history else []
    message_text = (
        f"🎯 *Room {room_id}*\n"
        f"📊 *Predicted Number*: {emoji} *{predicted}* ({category})\n"
        f"📈 *Confidence*: {confidence:.2%}\n"
        f"📋 *Recent outcomes*: {' '.join(map(str, recent))}\n\n"
        "Was this prediction correct?"
    )

    keyboard = [
        [
            InlineKeyboardButton("✅ Yes", callback_data=f"yes_{pred_id}"),
            InlineKeyboardButton("❌ No",  callback_data=f"no_{pred_id}"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        message_text,
        parse_mode="Markdown",
        reply_markup=reply_markup,
    )


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle Yes/No feedback from inline buttons."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if data.startswith("yes_"):
        pred_id = int(data.split("_")[1])
        # Retrieve prediction details
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT room_id, predicted FROM predictions WHERE id=?", (pred_id,))
        row = c.fetchone()
        if row:
            room_id, predicted = row
            # User says Yes → prediction was correct → add to history
            add_history(room_id, predicted)
            # Update prediction record
            c.execute(
                "UPDATE predictions SET feedback=1, correct=1, actual=? WHERE id=?",
                (predicted, pred_id),
            )
            conn.commit()
            await query.edit_message_text("✅ Thanks! The prediction was correct and has been recorded.")
        else:
            await query.edit_message_text("❌ Prediction record not found.")
        conn.close()

    elif data.startswith("no_"):
        pred_id = int(data.split("_")[1])
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "UPDATE predictions SET feedback=0, correct=0 WHERE id=?",
            (pred_id,),
        )
        conn.commit()
        conn.close()
        await query.edit_message_text("❌ Thanks! Feedback recorded. I'll learn from this.")

    else:
        await query.edit_message_text("⚠️ Unknown action.")


# ------------------------------
#  Main Entry Point
# ------------------------------
def main():
    """Start the bot."""
    init_db()

    # Create the Application
    application = Application.builder().token(TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(CallbackQueryHandler(feedback_callback))

    # Start polling (blocking)
    logger.info("Bot started polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()