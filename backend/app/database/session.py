import sqlite3
import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple

try:
    from app.core.config import settings
except ImportError:
    from backend.app.core.config import settings

def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(settings.DATABASE_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_history (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        filename TEXT,
        species TEXT NOT NULL,
        species_confidence REAL NOT NULL,
        grade INTEGER NOT NULL,
        grade_name TEXT NOT NULL,
        grade_confidence REAL NOT NULL,
        discrete_probabilities TEXT NOT NULL,
        energy_score REAL NOT NULL,
        is_ood INTEGER NOT NULL DEFAULT 0,
        routing_action TEXT NOT NULL,
        latency_ms REAL NOT NULL,
        thumbnail_base64 TEXT,
        device TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()

def save_analysis(result: Any, original_filename: Optional[str] = None) -> str:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Extract values whether result is PredictionResponse object or dict
    if hasattr(result, "analysis_id"):
        rec_id = result.analysis_id
        timestamp = result.timestamp
        species = result.species.species
        species_conf = result.species.confidence
        grade = result.ordinal_grade.grade
        grade_name = result.ordinal_grade.grade_name
        grade_conf = result.ordinal_grade.confidence
        discrete_probs = json.dumps(result.ordinal_grade.discrete_probabilities)
        energy_score = result.ood.energy_score
        is_ood = 1 if result.ood.is_ood else 0
        routing_action = result.selective_routing.routing_action
        latency_ms = result.inference_latency_ms
        thumb = result.thumbnail_base64
        device = result.device_used
    else:
        rec_id = result.get("analysis_id", str(uuid.uuid4()))
        timestamp = result.get("timestamp", datetime.now(timezone.utc).isoformat())
        species = result["species"]["species"]
        species_conf = result["species"]["confidence"]
        grade = result["ordinal_grade"]["grade"]
        grade_name = result["ordinal_grade"]["grade_name"]
        grade_conf = result["ordinal_grade"]["confidence"]
        discrete_probs = json.dumps(result["ordinal_grade"]["discrete_probabilities"])
        energy_score = result["ood"]["energy_score"]
        is_ood = 1 if result["ood"]["is_ood"] else 0
        routing_action = result["selective_routing"]["routing_action"]
        latency_ms = result["inference_latency_ms"]
        thumb = result.get("thumbnail_base64")
        device = result.get("device_used", "cpu")

    cursor.execute("""
    INSERT INTO analysis_history (
        id, timestamp, filename, species, species_confidence,
        grade, grade_name, grade_confidence, discrete_probabilities,
        energy_score, is_ood, routing_action, latency_ms,
        thumbnail_base64, device
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rec_id, timestamp, original_filename or "unnamed_sample.jpg",
        species, species_conf, grade, grade_name, grade_conf,
        discrete_probs, energy_score, is_ood, routing_action,
        latency_ms, thumb, device
    ))
    conn.commit()
    conn.close()
    return rec_id

def get_history_items(
    limit: int = 50,
    offset: int = 0,
    search: Optional[str] = None,
    species: Optional[str] = None,
    grade: Optional[int] = None,
    is_ood: Optional[bool] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    where_clauses = []
    params: List[Any] = []
    
    if search:
        where_clauses.append("(species LIKE ? OR grade_name LIKE ? OR filename LIKE ?)")
        search_pattern = f"%{search}%"
        params.extend([search_pattern, search_pattern, search_pattern])
        
    if species:
        where_clauses.append("species = ?")
        params.append(species)
        
    if grade is not None:
        where_clauses.append("grade = ?")
        params.append(grade)
        
    if is_ood is not None:
        where_clauses.append("is_ood = ?")
        params.append(1 if is_ood else 0)
        
    where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    # Get total count matching query
    count_query = f"SELECT COUNT(*) FROM analysis_history{where_str}"
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]
    
    # Get paginated rows
    query = f"SELECT * FROM analysis_history{where_str} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
    query_params = list(params)
    query_params.extend([limit, offset])
    
    cursor.execute(query, query_params)
    rows = cursor.fetchall()
    conn.close()
    
    items = []
    for r in rows:
        d = dict(r)
        d["is_ood"] = bool(d["is_ood"])
        if "discrete_probabilities" in d and d["discrete_probabilities"]:
            d["discrete_probabilities"] = json.loads(d["discrete_probabilities"])
        items.append(d)
        
    return total, items

def get_history_item_by_id(analysis_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM analysis_history WHERE id = ?", (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    d["is_ood"] = bool(d["is_ood"])
    if "discrete_probabilities" in d and d["discrete_probabilities"]:
        d["discrete_probabilities"] = json.loads(d["discrete_probabilities"])
    return d

def delete_history_item(analysis_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted

def clear_all_history() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count
