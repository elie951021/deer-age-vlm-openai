from io import BytesIO
import os
import base64
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime
from uuid import uuid4
from typing import Optional

from fastapi import UploadFile

from app.config import settings
from app.core.llm import LLMClient
from app.loggers.log_session import RequestSession
from app.loggers.logger import logger
from app.prompts.classify import DEER_CLASSIFY_PROMPT
from app.prompts.estimate import DEER_AGE_ESTIMATION_PROMPT
from app.prompts.observe import DEER_OBSERVATION_PROMPT
from app.schemas import EstimationResponse
from app.utils.image import normalize_image
from PIL import Image

llm = LLMClient(
    model=settings.openai_model,
    api_key=settings.openai_api_key,
    model_kwargs={"response_format": {"type": "json_object"}},
)

UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'upload')
SQLITE_DB_PATH = os.getenv('SQLITE_DB_PATH', 'database/app.db')
   
# Age class metadata for reliability assessment
# Scientific characteristics based on tooth replacement and wear patterns
AGE_CLASS_INFO = {
    "0.5": {
        "description": "Fawn (6 months)",
        "reliability": "high",
        "characteristics": "Fawns have 5 or less teeth present. The 3rd premolar (tooth 3) has 3 cusps. Tooth 6 has not yet erupted. In younger fawns, tooth 5 has not erupted and only 4 teeth will be visible.",
        "key_indicators": ["5 or fewer teeth", "Tooth 3 has 3 cusps", "Tooth 6 not erupted"],
        "note": None
    },
    "1.5": {
        "description": "Yearling (1.5 years)",
        "reliability": "high",
        "characteristics": "Tooth 3 (3rd premolar) has 3 cusps with heavy wear. Tooth 6 has erupted and is slightly visible just above the gum line. All 6 teeth are now present.",
        "key_indicators": ["3-cusp tooth 3 with heavy wear", "Tooth 6 just erupted", "6 teeth visible"],
        "note": None
    },
    "2.5": {
        "description": "2.5 years old",
        "reliability": "high",
        "characteristics": "Lingual crest on all molars are sharp and pointed. Tooth 3 now has 2 cusps (permanent replacement). Back cusp of tooth 6 is sharp and pointed. Enamel is wider than the dentine in teeth 4, 5, and 6.",
        "key_indicators": ["Tooth 3 now has 2 cusps", "Sharp lingual crests", "Enamel wider than dentine"],
        "note": None
    },
    "3.5": {
        "description": "3.5 years old",
        "reliability": "high",
        "characteristics": "Lingual crest on tooth 4 is blunt. The dentine is as wide or wider than the enamel in tooth 4. The back cusp on tooth 6 is forming a concavity.",
        "key_indicators": ["Blunt lingual crest on tooth 4", "Dentine equals enamel width in tooth 4", "Concavity forming on tooth 6"],
        "note": None
    },
    "4.5": {
        "description": "4.5 years old",
        "reliability": "moderate",
        "characteristics": "Lingual crest on tooth 4 is almost rounded off and lingual crest in tooth 5 is blunt. The dentine in tooth 4 is twice as wide as the enamel. The dentine in tooth 5 is wider than the enamel. The back cusp on tooth 6 slopes downward towards the cheek.",
        "key_indicators": ["Rounded lingual crest on tooth 4", "Dentine 2x wider than enamel in tooth 4", "Cusp slopes downward on tooth 6"],
        "note": None
    },
    "5+": {
        "description": "5+ years old (mature deer)",
        "reliability": "moderate",
        "characteristics": "Lingual crests show significant wear or are worn away. Dentine is wider than enamel on multiple teeth. Teeth may have 'dished out' appearance. For deer 5.5 years and older, precise aging becomes unreliable due to individual variation in tooth wear.",
        "key_indicators": ["Dentine wider than enamel", "Significant lingual crest wear", "Dished out appearance possible"],
        "note": "Visual aging becomes unreliable for deer 5+ years old. Tooth wear varies significantly by diet, habitat, and soil conditions. For precise aging of mature deer, cementum annuli analysis is strongly recommended."
    }
}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _parse_json_response(response, step: str) -> dict:
    content = response.content
    if not isinstance(content, str):
        raise ValueError(f"[{step}] Unexpected LLM response type: {type(content)}")
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("[%s] Failed to parse JSON: %s | raw: %s", step, str(e), content)
        raise

def get_reliability_level(age_class: str, confidence: float) -> str:
    """Determine overall reliability based on age class and model confidence."""
    base_reliability = AGE_CLASS_INFO.get(age_class, {}).get("reliability", "unknown")

    # Adjust based on confidence
    if confidence < 0.4:
        return "low"
    elif confidence < 0.6:
        if base_reliability == "high":
            return "moderate"
        return "low"
    elif confidence < 0.8:
        if base_reliability == "low":
            return "low"
        return base_reliability
    else:
        return base_reliability

async def estimate_deer_age(file: UploadFile, usermail: Optional[str] = None) -> dict:
    """Single-step pipeline: observe + classify in one LLM call."""
    session = RequestSession()
    try:
        image_bytes = await file.read()
        image_bytes = normalize_image(image_bytes)
        session.save_image(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # try:
        #     contents = await file.read()
        #     image = Image.open(BytesIO(contents)).convert('RGB')
        # except Exception:
        #     await file.seek(0)
        #     image = Image.open(file.file).convert('RGB')

        # Create date-named directory in upload folder
        current_date = datetime.now().strftime('%Y-%m')
        date_folder = Path(UPLOAD_DIR) / current_date
        date_folder.mkdir(parents=True, exist_ok=True)

        # Save the uploaded image with a random filename
        suffix = Path(file.filename).suffix.lower() if file.filename else ''
        if not suffix:
            suffix = '.jpg'
        random_filename = f"{uuid4().hex}{suffix}"
        image_path = date_folder / random_filename
        image_bytes.save(str(image_path))

        BASE_DIR = Path(__file__).resolve().parent
        few_shot_base64 = encode_image(BASE_DIR / "few_shot.jpg")

        messages = DEER_AGE_ESTIMATION_PROMPT.format_messages(
            image_data=base64_image,
            few_shot_base64=few_shot_base64,
        )
        session.save_prompt(messages)

        response = llm.invoke(messages)
        logger.debug("RAW CONTENT: %s", response.content)

        data = _parse_json_response(response, "single-step")
        data["cost"] = getattr(response, "cost", None)
        session.save_response(data)

        result = EstimationResponse(**data)

        match = re.search(r"\d+\.\d+|\d+", result.final_classification.estimated_age.replace(" ", ""))
        primary_class = float(match.group()) if match else None
        if primary_class >= 5.5:
            primary_class = "5+"
        primary_confidence = result.final_classification.confidence_score
        primary_info = AGE_CLASS_INFO.get(str(primary_class), {})
        class_info = AGE_CLASS_INFO.get(str(primary_class), {})

        response = {"prediction_id": 24, "warnings": []}
        response['prediction'] = {
            'age_estimate': result.final_classification.estimated_age, 
            'confidence': result.final_classification.confidence_score, 
            'logic_path': result.final_classification.logic_path,
            'characteristics': '',
            'reliability': primary_info.get("reliability", "unknown"),
        }
        
        warnings = []
        if primary_class == "5+":
            warnings.append(primary_info.get("note", "Visual aging is unreliable for deer 5+ years old. Cementum annuli analysis recommended."))
        response['prediction']['age_estimate'] = primary_class
            
        if primary_confidence < 0.5:
            warnings.append("Low confidence prediction. Image quality or angle may affect accuracy. Consider submitting a clearer image.")
        if primary_confidence < 0.3:
            warnings.append("Very low confidence. This prediction should not be relied upon.")

        if warnings:
            response["warnings"] = warnings
        response['prediction']['characteristics'] = class_info.get("characteristics", "")
        response['prediction']['reliability'] = get_reliability_level(str(primary_class), primary_confidence)

        prediction_id = save_prediction_result(
            db_path=SQLITE_DB_PATH,
            usermail=usermail,
            original_filename=file.filename,
            saved_filename=random_filename,
            saved_image_path=str(image_path),
            response_payload=response,
        )

        response["prediction_id"] = prediction_id
        return response

    except Exception as e:
        logger.error("ERROR in estimate_deer_age: %s", str(e))
        raise
    finally:
        session.close()


async def estimate_deer_age_two_step(image_bytes: bytes) -> EstimationResponse:
    """Two-step pipeline: Step 1 = structured observation, Step 2 = rule-based classification."""
    session = RequestSession()
    total_cost = 0.0
    try:
        image_bytes = normalize_image(image_bytes)
        session.save_image(image_bytes)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # --- Step 1: Observe ---
        observe_messages = DEER_OBSERVATION_PROMPT.format_messages(image_data=base64_image)
        session.save_prompt(observe_messages)

        observe_response = llm.invoke(observe_messages)
        logger.debug("OBSERVE RAW: %s", observe_response.content)
        total_cost += getattr(observe_response, "cost", 0.0) or 0.0

        observation = _parse_json_response(observe_response, "observe")
        logger.debug("OBSERVATION: %s", json.dumps(observation, indent=2))

        # --- Step 2: Classify ---
        classify_messages = DEER_CLASSIFY_PROMPT.format_messages(
            observation_json=json.dumps(observation, indent=2)
        )

        classify_response = llm.invoke(classify_messages)
        logger.debug("CLASSIFY RAW: %s", classify_response.content)
        total_cost += getattr(classify_response, "cost", 0.0) or 0.0

        data = _parse_json_response(classify_response, "classify")
        data["cost"] = total_cost
        data["observation"] = observation
        session.save_response(data)

        return EstimationResponse(**data)

    except Exception as e:
        logger.error("ERROR in estimate_deer_age_two_step: %s", str(e))
        raise
    finally:
        session.close()

def init_database(db_path: str):
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    print("creating db")

    with sqlite3.connect(str(db_file)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usermail TEXT,
                created_at TEXT NOT NULL,
                original_filename TEXT,
                saved_filename TEXT,
                saved_image_path TEXT,
                age_estimate TEXT,
                confidence REAL,
                reliability TEXT,
                exact_age TEXT,
                feedback TEXT,
                reply TEXT
            )
            """
        )


def save_prediction_result(
    db_path: str,
    usermail: str,
    original_filename: str,
    saved_filename: str,
    saved_image_path: str,
    response_payload: dict,
) -> int:
    prediction = response_payload.get("prediction", {})
    created_at = datetime.utcnow().isoformat(timespec='seconds') + "Z"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                created_at,
                usermail,
                original_filename,
                saved_filename,
                saved_image_path,
                age_estimate,
                confidence,
                reliability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                usermail,
                original_filename,
                saved_filename,
                saved_image_path,
                prediction.get("age_estimate"),
                prediction.get("confidence"),
                prediction.get("reliability")
            ),
        )
        conn.commit()
        return cursor.lastrowid


def save_prediction_feedback(
    db_path: str,
    prediction_id: int,
    exact_age: str,
    feedback: Optional[str],
) -> bool:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE predictions
            SET exact_age = ?,
                feedback = ?
            WHERE id = ?
            """,
            (exact_age, feedback, prediction_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def to_upload_url(saved_image_path: Optional[str]) -> Optional[str]:
    if not saved_image_path:
        return None

    normalized = saved_image_path.replace("\\", "/")
    if normalized.startswith("upload/"):
        return f"/{normalized}"
    if normalized.startswith("/upload/"):
        return normalized

    upload_index = normalized.find("/upload/")
    if upload_index >= 0:
        return normalized[upload_index:]

    return None


def get_prediction_history_by_usermail(db_path: str, usermail: str):
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                usermail,
                created_at,
                original_filename,
                saved_filename,
                saved_image_path,
                age_estimate,
                confidence,
                reliability,
                feedback,
                exact_age
            FROM predictions
            WHERE lower(usermail) = lower(?)
            ORDER BY created_at DESC, id DESC
            """,
            (usermail,),
        ).fetchall()

    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "usermail": row["usermail"],
            "created_at": row["created_at"],
            "original_filename": row["original_filename"],
            "saved_image_url": to_upload_url(row["saved_image_path"]),
            "feedback": row["feedback"],
            "exact_age": row["exact_age"],
            "prediction": {
                "age_estimate": row["age_estimate"],
                "confidence": row["confidence"],
                "reliability": row["reliability"],
            },
        })

    return history


def clear_prediction_history_by_usermail(db_path: str, usermail: str) -> int:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            DELETE FROM predictions
            WHERE lower(usermail) = lower(?)
            """,
            (usermail,),
        )
        conn.commit()
        return max(cursor.rowcount, 0)
