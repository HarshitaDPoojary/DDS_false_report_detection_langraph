# pip install sentence-transformers numpy
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Model setup 
# ----------------------------
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, fast
_embed_model = SentenceTransformer(_EMBED_MODEL_NAME)

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [1, d], b: [N, d] (both already L2-normalized)
    return (a @ b.T).ravel()

# ----------------------------------------------------------
# Core API: similarity + hoax scoring on DB-returned reports
# ----------------------------------------------------------
def score_similarity_and_hoax(
    new_report: Dict,
    candidate_reports: List[Dict],
    *,
    k: int = 20,
    sim_threshold: float = 0.70,   # consider “similar” above this
    high_template_sim: float = 0.90 # near-duplicate / template-like
) -> Dict:
    """
    Parameters
    ----------
    new_report: {
        "report_id": str,
        "text": str,
        # optional fields you may already have:
        # "incident_time": datetime, "lat": float, "lon": float,
        # "reporter_name": Optional[str], "is_anonymous": Optional[bool]
    }
    candidate_reports: list of reports from your DB filter
        Each item should have keys:
            - "report_id": str
            - "text": str
            - "is_anonymous": bool    (or infer from reporter_name)
            - (optional extras: lat/lon/time/etc.)
    Returns
    -------
    {
      "topk": List[{"report_id", "similarity", "is_anonymous", "text"}],
      "summary": {
          "avg_similarity": float,
          "named_ratio": float,
          "anon_ratio": float,
          "cluster_size": int,
          "near_duplicate_count": int
      },
      "hoax_score": float  # 0..1 (higher => more hoax-leaning)
    }
    """
    if not candidate_reports:
        return {
            "topk": [],
            "summary": {
                "avg_similarity": 0.0,
                "named_ratio": 0.0,
                "anon_ratio": 0.0,
                "cluster_size": 0,
                "near_duplicate_count": 0,
            },
            "hoax_score": 0.0,  # with no corroboration we don't auto-accuse
        }

    # ----------------
    # Embedding stage
    # ----------------
    q_vec = _embed_model.encode([new_report["text"]], convert_to_numpy=True)
    q_vec = _l2_normalize(q_vec)

    texts = [r["text"] for r in candidate_reports]
    mat = _embed_model.encode(texts, convert_to_numpy=True, batch_size=64)
    mat = _l2_normalize(mat)

    sims = _cosine_sim(q_vec, mat)  # cosine similarities

    # Attach sims and basic identity info
    enriched = []
    for r, s in zip(candidate_reports, sims):
        is_anonymous = r.get("is_anonymous")
        if is_anonymous is None:
            # Fallback heuristic if you don’t store is_anonymous explicitly
            is_anonymous = not bool(r.get("reporter_name"))
        enriched.append({
            "report_id": r["report_id"],
            "text": r["text"],
            "is_anonymous": bool(is_anonymous),
            "similarity": float(s),
        })

    # ----------------
    # Top-k selection
    # ----------------
    enriched.sort(key=lambda x: x["similarity"], reverse=True)
    topk = enriched[:k]

    # Only consider “similar” ones (above threshold) as a cluster
    similar_hits = [x for x in topk if x["similarity"] >= sim_threshold]
    cluster_size = len(similar_hits)

    if cluster_size == 0:
        return {
            "topk": topk,
            "summary": {
                "avg_similarity": 0.0,
                "named_ratio": 0.0,
                "anon_ratio": 0.0,
                "cluster_size": 0,
                "near_duplicate_count": 0,
            },
            "hoax_score": 0.0,  # nothing similar in-window → no hoax bump
        }

    # -------------------------
    # Named vs Anonymous signal
    # -------------------------
    named_count = sum(1 for x in similar_hits if not x["is_anonymous"])
    anon_count  = cluster_size - named_count
    named_ratio = named_count / cluster_size
    anon_ratio  = anon_count / cluster_size

    # -------------------------
    # Similarity strength stats
    # -------------------------
    avg_sim = float(np.mean([x["similarity"] for x in similar_hits]))
    near_duplicate_count = sum(1 for x in similar_hits if x["similarity"] >= high_template_sim)

    # ----------------------------------------------
    # Hoax score (0..1): heuristic, simple + tunable
    #   Intuition:
    #     - More anonymous & more template-like → higher hoax
    #     - More named corroboration → lowers hoax
    #     - Larger cluster with high avg_sim → template wave risk
    # ----------------------------------------------
    # Component A: anonymity pressure
    #   - push hoax score up as anon_ratio increases
    anon_pressure = anon_ratio  # 0..1

    # Component B: template pressure
    #   - avg_sim above threshold boosts hoax suspicion
    #   - near duplicates add extra push
    template_pressure = max(0.0, (avg_sim - sim_threshold) / (1.0 - sim_threshold))
    if cluster_size > 1:
        template_pressure *= min(1.0, (cluster_size - 1) / 5.0)  # saturate after ~6 reports
    template_pressure += min(1.0, near_duplicate_count / 3.0) * 0.25  # extra bump for many near-dupes

    # Component C: named corroboration relief
    #   - meaningful presence of named reporters should reduce hoax suspicion
    named_relief = named_ratio  # 0..1; more named → more relief

    # Combine (weights are easy to tune)
    raw = 0.55 * anon_pressure + 0.45 * template_pressure - 0.35 * named_relief
    hoax_score = float(np.clip(raw, 0.0, 1.0))

    return {
        "topk": topk,
        "summary": {
            "avg_similarity": round(avg_sim, 4),
            "named_ratio": round(named_ratio, 4),
            "anon_ratio": round(anon_ratio, 4),
            "cluster_size": cluster_size,
            "near_duplicate_count": near_duplicate_count,
        },
        "hoax_score": hoax_score
    }


if __name__ == "__main__":
    # Pseudocode: you already ran a DB query for neighbors in ±h hours & radius R
    new_report = {
        "report_id": "NEW-123",
        "text": "There is a fight near the east gate by the parking kiosk.",
        # "reporter_name": "", "is_anonymous": True,  # optional if you have it
    }

    # candidate_reports: list of dicts from your DB (already filtered by geo/time)
    # required keys: "report_id", "text"
    # recommended: "is_anonymous" or "reporter_name" to infer it
    candidate_reports = [
        {"report_id":"A1","text":"Fight near east gate by parking kiosk","is_anonymous":True},
        {"report_id":"A2","text":"Argument at east parking kiosk, people gathering","is_anonymous":False},
        {"report_id":"A3","text":"Multiple people shouting near east gate","is_anonymous":True},
        # ...
    ]

    result = score_similarity_and_hoax(new_report, candidate_reports, k=20, sim_threshold=0.70)
    print(result["summary"])
    print("hoax_score:", result["hoax_score"])
    for r in result["topk"][:5]:
        print(r["report_id"], r["similarity"], "ANON" if r["is_anonymous"] else "NAMED")

