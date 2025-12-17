"""
Helpers for working with feedback data.

- get_scores_for_message(message_id)
- export_feedback_aggregates_to_csv(output_path)

Run this file directly to export feedback to CSV:

    python feedback.py
"""

import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ✅ Same DB as app.py
DB_PATH = Path("data/shipcube.db")


# ---------------------------------------------------------------------
# 1) Per-message stats helper (unchanged)
# ---------------------------------------------------------------------
def get_scores_for_message(message_id: str) -> Optional[Dict[str, Any]]:
    """
    Return like/dislike stats for one specific answer (by message_id).

    Result example:
    {
        "message_id": "...",
        "question": "...",
        "answer": "...",
        "likes": 3,
        "dislikes": 1,
        "net_score": 2,
        "total_votes": 4,
    }
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            message_id,
            question,
            answer,
            SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END) AS likes,
            SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS dislikes,
            (SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END)
             - SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END)) AS net_score,
            COUNT(*) AS total_votes
        FROM feedback
        WHERE message_id = ?
        GROUP BY message_id, question, answer
        """,
        (message_id,),
    )

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "message_id": row[0],
        "question": row[1],
        "answer": row[2],
        "likes": row[3],
        "dislikes": row[4],
        "net_score": row[5],
        "total_votes": row[6],
    }


# ---------------------------------------------------------------------
# 2) Export ALL aggregated feedback to a CSV (Excel-readable)
#    👉 aggregated by ANSWER text, not by message_id / question
# ---------------------------------------------------------------------
def export_feedback_aggregates_to_csv(
    output_path: Path | str = Path("data/exports/feedback_aggregated_by_answer.csv"),
) -> Path:
    """
    Aggregate feedback (likes / dislikes / net_score) for every UNIQUE ANSWER
    (across all message_ids and questions) and write the result to a CSV.

    All thumbs for:
        - "who is founder"
        - "founder of shipcube"
    that produced the SAME 'answer' text will be combined into one row.

    Returns the Path to the created file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            MIN(question) AS example_question,   -- just one representative question
            answer,
            SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END) AS likes,
            SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) AS dislikes,
            (SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END)
             - SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END)) AS net_score,
            COUNT(*) AS total_votes
        FROM feedback
        GROUP BY answer          -- 🔹 aggregate across all questions / message_ids
        ORDER BY net_score DESC, total_votes DESC
        """
    )

    rows: List[Tuple] = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    conn.close()

    import csv

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(col_names)  # header
        writer.writerows(rows)

    return output_path


# ---------------------------------------------------------------------
# 3) Allow running as a script: `python feedback.py`
# ---------------------------------------------------------------------
if __name__ == "__main__":
    out_file = export_feedback_aggregates_to_csv()
    print(f"✅ Feedback aggregated (by answer) and exported to: {out_file}")
