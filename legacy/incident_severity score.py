"""
Incident classification and severity scoring

Provides:
- keyword/regex based incident type classifier (deterministic, offline)
- optional LLM prompt builder and generic caller hook for semantic fallback
- severity mapping from types -> severity levels
- small demo showing Case 1 (local classifier) and Case 2 (LLM prompt + how to call)

This module is safe to import without network access. To enable LLM calls,
pass a callable `llm_client` (see `call_llm`) or install OpenAI and provide
an API key in your environment and run the demo section.
"""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# default model constant (avoid repeated literal)
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"

# demo sample text used in multiple places
DEMO_REPORT_TEXT = "Multiple people reporting smoke and a loud explosion near the market."

# ---------------------------------------------------------------------------
# Canonical taxonomy: incident types mapped to regex patterns (lowercased)
# ---------------------------------------------------------------------------
INCIDENT_KEYWORDS: Dict[str, List[str]] = {
	"shooting": [r"\bshoot(?:ing|s|er)\b", r"\bshots?\b", r"\bgunfire\b", r"\bopen fire\b", r"\bactive shooter\b"],
	"bombing": [r"\bbomb\b", r"\bexplosion\b", r"\bexplosive\b", r"\bied\b", r"\bsuspicious device\b"],
	"stabbing": [r"\bstab(?:bing|bed|ber)\b", r"\bknife\b", r"\bcut\b"],
	"assault": [r"\bassault\b", r"\bbeat(?:ing)?\b", r"\battack(?:ed|s)?\b"],
	"robbery": [r"\brobber(?:y)?\b", r"\bmugging\b", r"\barmed robbery\b"],
	"burglary": [r"\bburglary\b", r"\bbreak-?in\b", r"\bhome invasion\b", r"\bbroke into\b", r"\bbroke in\b", r"\bforced (?:entry|open)\b", r"\bforced open\b"],
	"theft": [r"\btheft\b", r"\bstolen\b", r"\bshoplift(?:ing|ed)?\b", r"\bsteal(?:ing|s)?\b", r"\bstole\b", r"\bsnatch(?:ed)?\b", r"\btook\b", r"\brobbed\b"],
	"kidnapping": [r"\bkidnap(?:ping|ped)?\b", r"\babduct(?:ed|ion)?\b", r"\btaken\b"],
	"vandalism": [r"\bvandal(?:ism|ize|ised)\b", r"\bgraffiti\b", r"\bproperty damage\b"],
	"arson": [r"\barson\b", r"\bset (on )?fire\b", r"\bsuspicious fire\b"],
	"hit_and_run": [r"\bhit and run\b", r"\bhit-and-run\b", r"\bhit ?and ?run\b", r"\bfled the scene\b", r"\bfled\b"],
	"traffic_accident": [r"\baccident\b", r"\bcrash\b", r"\bcollision\b", r"\bpileup\b", r"\broad traffic incident\b", r"\bwas hit\b", r"\bhit a\b", r"\bhit an?\b", r"\bhit\b"],
	"sexual_assault": [r"\bsexual assault\b", r"\brape\b"],
	"suspicious_package": [r"\bsuspicious package\b", r"\bpackage\b left\b", r"\bunattended bag\b", r"\bbag with wires\b"],
	"protest": [r"\bprotest\b", r"\briot\b", r"\bdemonstration\b"],
	"hazmat": [r"\bchemical spill\b", r"\bgas leak\b", r"\bhazardous material\b", r"\bhazmat\b"],
	"hostage": [r"\bhostage\b", r"\bheld hostage\b"],
	"medical_emergency": [r"\bcollapsed\b", r"\bnot breathing\b", r"\bunconscious\b", r"\bcardiac arrest\b", r"\bmedical emergency\b"],
	"natural_disaster": [r"\bearthquake\b", r"\bflood\b", r"\btornado\b", r"\bstorm\b"],
	"cyber_incident": [r"\bhack(?:ed|ing)?\b", r"\bdata breach\b", r"\bphish(?:ing)?\b"],
}

# A canonical list used in LLM prompts and validation
CANONICAL_TYPES = list(INCIDENT_KEYWORDS.keys()) + ["other"]


def get_incident_types(
	text: str,
	top_k: int = 3,
	min_score: float = 0.1,
	enable_llm_fallback: bool = False,
	llm_client: Optional[Callable[[str, str], Any]] = None,
	llm_model: str = DEFAULT_LLM_MODEL,
) -> List[Dict[str, Any]]:
	"""
	Classify an input report into one or more incident types.

	Returns a list of dicts: {"type": <canonical_type>, "confidence": float, "matches": [matched substrings]}

	- Uses rule-based regex matching and a simple scoring heuristic.
	- If no matches and enable_llm_fallback=True, will call `llm_client(prompt, model)` if provided.
	"""
	if not text or not text.strip():
		return []

	t = text.lower()
	scores: Dict[str, float] = defaultdict(float)
	matches: Dict[str, List[str]] = defaultdict(list)

	# Keyword matching
	for incident_type, patterns in INCIDENT_KEYWORDS.items():
		for pat in patterns:
			try:
				m = re.search(pat, t)
			except re.error:
				continue
			if m:
				matched_text = m.group(0).strip().lower()
				# preserve matched substrings per-type
				matches[incident_type].append(matched_text)
				# scoring heuristic: short strong tokens score higher
				if len(matched_text.split()) <= 2:
					scores[incident_type] += 0.6
				else:
					scores[incident_type] += 0.5
				# contextual bonuses
				if "armed" in matched_text or "active" in matched_text:
					scores[incident_type] += 0.2
				if any(k in matched_text for k in ("fled", "flee", "ran away")):
					scores[incident_type] += 0.3
				if any(k in matched_text for k in ("stole", "stolen", "snatch", "robbed", "took")):
					scores[incident_type] += 0.15

	# Normalize and prepare output
	cap = 1.6
	normalized = []
	for itype, score in scores.items():
		# deduplicate matches while preserving order
		seen = set()
		uniq_matches = []
		for m in matches.get(itype, []):
			if m not in seen:
				seen.add(m)
				uniq_matches.append(m)
		conf = min(score / cap, 1.0)
		normalized.append((itype, round(conf, 3), uniq_matches))

	# If no rule-based matches, optionally use LLM fallback
	if not normalized and enable_llm_fallback and llm_client is not None:
		prompt = build_llm_prompt_for_types(text)
		try:
			raw = call_llm(llm_client, prompt, llm_model)
			# Expect raw to be JSON-like or a dict with 'types'
			if isinstance(raw, str):
				parsed = json.loads(raw)
			elif isinstance(raw, dict):
				parsed = raw
			else:
				parsed = None

			if parsed and "types" in parsed and isinstance(parsed["types"], list):
				out = []
				for item in parsed["types"]:
					ttype = item.get("type", "other")
					if ttype not in CANONICAL_TYPES:
						ttype = "other"
					conf = float(item.get("confidence", 0.0))
					reason = item.get("reason") or "llm"
					out.append({"type": ttype, "confidence": round(conf, 3), "matches": [reason]})
				return out[:top_k]
		except Exception:
			# Fall through to returning 'other'
			pass

	# Sort and filter top_k
	normalized.sort(key=lambda x: x[1], reverse=True)
	results: List[Dict[str, Any]] = []
	for itype, conf, matched in normalized[:top_k]:
		if conf >= min_score:
			results.append({"type": itype, "confidence": conf, "matches": matched})

	if not results:
		results = [{"type": "other", "confidence": 0.2, "matches": []}]

	return results


def build_llm_prompt_for_types(text: str) -> str:
	"""
	Build a strict JSON-only LLM prompt for classifying incident types.
	"""
	canonical = ", ".join(CANONICAL_TYPES)
	instruction = (
		"You are an assistant that classifies short incident reports into one or more canonical incident types.\n"
		"Return ONLY a JSON object with a 'types' array. Each element must be an object with keys: 'type' (one of the canonical types),\n"
		"'confidence' (float 0.0-1.0), and 'reason' (short text explanation of what triggered the classification).\n\n"
		f"Canonical types: {canonical}.\n\n"
		"Rules:\n"
		"- Output must be parsable JSON and nothing else.\n"
		"- If multiple plausible types exist, include them with appropriate confidences.\n"
		"- If none apply, return [{\"type\":\"other\",\"confidence\":0.1,\"reason\":\"no clear indicators\"}].\n\n"
		"Examples:\n"
		"Input: 'Multiple shots fired outside the mall, people running, heard several pops and saw smoke.'\n"
		"Output: {\"types\":[{\"type\":\"shooting\",\"confidence\":0.93,\"reason\":\"mentions 'shots fired'\"}]}\n\n"
		"Input: 'There is a suspicious package at the subway entrance, unattended bag emitting an odor.'\n"
		"Output: {\"types\":[{\"type\":\"suspicious_package\",\"confidence\":0.88,\"reason\":\"'suspicious package' and 'unattended bag'\"}]}\n\n"
		"Now classify the following report and return only JSON:\n\n"
		f"Report:\n{text.strip()}\n"
	)
	return instruction


def call_llm(llm_client: Callable[[str, str], Any], prompt: str, model: str = DEFAULT_LLM_MODEL) -> Any:
	"""
	Generic wrapper to call an LLM client. Supports a few common shapes:

	- If llm_client is a callable that accepts (prompt, model) and returns a dict/string, it will be used.
	- If llm_client is the OpenAI module (imported), it will call openai.ChatCompletion.create or openai.ChatCompletion.create-like API.

	IMPORTANT: This function does not supply an API key. The caller must ensure credentials are configured.
	"""
	# If user passed a direct callable wrapper: try calling with (prompt, model)
	try:
		return llm_client(prompt, model)
	except TypeError:
		pass

	# If llm_client looks like the OpenAI module, try multiple API shapes
	try:
		# Shape 1: legacy OpenAI python (pre-1.0) -> openai.ChatCompletion.create(...)
		if hasattr(llm_client, "ChatCompletion") and hasattr(llm_client.ChatCompletion, "create"):
			resp = llm_client.ChatCompletion.create(
				model=model,
				messages=[{"role": "user", "content": prompt}],
				max_tokens=300,
				temperature=0.0,
			)
			# Extract text depending on API shape
			if isinstance(resp, dict):
				choices = resp.get("choices") or []
				if choices:
					return choices[0].get("message", {}).get("content") or choices[0].get("text")
			# try attribute access
			try:
				return resp.choices[0].message.content  # type: ignore[attr-defined]
			except Exception:
				return resp

		# Shape 2: openai>=1.0.0 style: client = openai.OpenAI(); client.chat.completions.create(...)
		if hasattr(llm_client, "OpenAI") or hasattr(llm_client, "openai") or hasattr(llm_client, "__all__"):
			try:
				# attempt to construct a client if available
				client_cls = getattr(llm_client, "OpenAI", None)
				if client_cls is not None:
					client = client_cls()
				else:
					# maybe the module itself is the client
					client = llm_client

				# Try the new chat completions path
				if hasattr(client, "chat") and hasattr(client.chat, "completions"):
					resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.0)
					# resp may be dict-like or object
					try:
						return resp.choices[0].message.content  # type: ignore[attr-defined]
					except Exception:
						if isinstance(resp, dict):
							choices = resp.get("choices") or []
							if choices:
								return choices[0].get("message", {}).get("content") or choices[0].get("text")
						return resp
			except Exception:
				# fall through to other strategies
				pass
	except Exception:
		# not an OpenAI-like client; continue
		pass

	# Last resort: try calling with single arg
	try:
		return llm_client(prompt)
	except Exception as exc:
		raise RuntimeError("LLM call failed; checked callable(prompt, model), OpenAI legacy and OpenAI new client shapes") from exc


# ---------------------------------------------------------------------------
# Severity mapping
# ---------------------------------------------------------------------------
SEVERITY_ORDER = ["low", "medium", "high", "critical"]

TYPE_TO_SEVERITY: Dict[str, str] = {
	"shooting": "critical",
	"bombing": "critical",
	"stabbing": "high",
	"assault": "high",
	"robbery": "high",
	"burglary": "medium",
	"theft": "low",
	"kidnapping": "critical",
	"vandalism": "low",
	"arson": "high",
	"hit_and_run": "high",
	"traffic_accident": "medium",
	"sexual_assault": "critical",
	"suspicious_package": "high",
	"protest": "medium",
	"hazmat": "critical",
	"hostage": "critical",
	"medical_emergency": "high",
	"natural_disaster": "critical",
	"cyber_incident": "medium",
	"other": "low",
}


def map_type_to_severity(itype: str) -> str:
	return TYPE_TO_SEVERITY.get(itype, "low")


def aggregate_severity(classifications: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Given classifications like [{type, confidence, matches}], return an aggregated severity result.

	Output: {"severity": <level>, "reason": <top_reason>, "details": classifications}
	"""
	if not classifications:
		return {"severity": "unknown", "reason": "no data", "details": []}

	# pick the highest severity among predicted types weighted by confidence
	best_weight = -1.0
	best_sev = "unknown"
	best_item = None
	for item in classifications:
		itype = item.get("type", "other")
		conf = float(item.get("confidence", 0.0))
		sev = map_type_to_severity(itype)
		idx = SEVERITY_ORDER.index(sev) if sev in SEVERITY_ORDER else 0
		# weight by confidence
		weighted = idx * conf
		if best_item is None or weighted > best_weight:
			best_weight = weighted
			best_item = item
			best_sev = sev

	if best_item:
		return {"severity": best_sev, "reason": best_item.get("matches") or best_item.get("type"), "details": classifications}

	return {"severity": "unknown", "reason": "no clear mapping", "details": classifications}


# -----------------------------
# API key helpers & LLM wrappers
# -----------------------------
def get_openai_client(api_key: Optional[str] = None):
	"""Import OpenAI via importlib and set API key (from env if not provided).

	Returns the imported openai module or raises ImportError.
	"""
	import importlib

	openai = importlib.import_module("openai")
	key = api_key or os.environ.get("OPENAI_API_KEY")
	if key:
		setattr(openai, "api_key", key)
	return openai


def call_openai(prompt: str, model: str = DEFAULT_LLM_MODEL, api_key: Optional[str] = None) -> str:
	"""Wrapper that calls OpenAI if available. Returns string content.

	Uses importlib to avoid import-time dependency.
	"""
	try:
		openai = get_openai_client(api_key)
	except Exception as exc:
		raise RuntimeError("OpenAI client not available") from exc

	# use ChatCompletion-like API shape
	resp = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=600)
	if isinstance(resp, dict):
		choices = resp.get("choices") or []
		if choices:
			return choices[0].get("message", {}).get("content") or choices[0].get("text") or ""
	# Fallback: try to coerce
	return str(resp)


def call_huggingface_inference(prompt: str, hf_token: Optional[str] = None, model: str = "bigscience/bloomz-1b1") -> str:
	"""Call HuggingFace Inference API (REST) if HF token is present in env.

	This is optional and will attempt a network call only if HF token is set.
	Returns the text result or raises RuntimeError on failure.
	"""
	hf_token = hf_token or os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN")
	if not hf_token:
		raise RuntimeError("HuggingFace API token not configured in HF_API_TOKEN or HUGGINGFACE_API_TOKEN")

	# Use requests to call inference endpoint
	try:
		import requests
	except Exception:
		raise RuntimeError("requests library required for HF inference wrapper")

	# New HuggingFace router endpoint (replaces api-inference.huggingface.co)
	url = f"https://router.huggingface.co/hf-inference/{model}"
	headers = {"Authorization": f"Bearer {hf_token}"}
	payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
	resp = requests.post(url, headers=headers, json=payload, timeout=30)
	# 410 indicates the old api-inference endpoint is deprecated for this account/model
	if resp.status_code == 410:
		raise RuntimeError(
			"HF inference returned 410: the api-inference endpoint is deprecated for this model/account. "
			"Use the router endpoint https://router.huggingface.co/hf-inference or check your model name/token. "
			f"Response: {resp.text}"
		)
	if resp.status_code != 200:
		raise RuntimeError(f"HF inference failed: {resp.status_code} {resp.text}")
	data = resp.json()
	# response may be a list of dicts or a dict
	if isinstance(data, list) and data:
		# some models return [{"generated_text": "..."}]
		first = data[0]
		return first.get("generated_text") or json.dumps(first)
	if isinstance(data, dict):
		return data.get("generated_text") or json.dumps(data)
	return str(data)


def local_llm_simulator(prompt: str) -> str:
	"""A tiny local simulator that returns a plausible JSON answer for demo purposes.

	This does not call any network service and is useful for printing example outputs.
	"""
	# super-simple heuristic for demo only
	out = {"types": []}
	t = prompt.lower()
	if "explosion" in t or "bomb" in t:
		out["types"].append({"type": "bombing", "confidence": 0.95, "reason": "mentions explosion/bomb"})
	if "shots" in t or "shoot" in t or "gunfire" in t:
		out["types"].append({"type": "shooting", "confidence": 0.9, "reason": "mentions shots/gunfire"})
	if not out["types"]:
		out["types"].append({"type": "other", "confidence": 0.2, "reason": "no strong indicators (simulated)"})
	return json.dumps(out)


# -----------------------------
# Urgency / weighted scoring
# -----------------------------

# Base scores per incident type (scale 0-10)
BASE_SCORE_BY_TYPE: Dict[str, float] = {
	"shooting": 9.0,
	"bombing": 10.0,
	"stabbing": 6.5,
	"assault": 6.0,
	"robbery": 5.5,
	"burglary": 4.0,  # example from your spec
	"theft": 2.0,
	"kidnapping": 9.5,
	"vandalism": 2.0,
	"arson": 8.0,
	"hit_and_run": 6.0,
	"traffic_accident": 5.0,
	"sexual_assault": 9.0,
	"suspicious_package": 8.5,
	"protest": 4.0,
	"hazmat": 9.5,
	"hostage": 10.0,
	"medical_emergency": 7.0,
	"natural_disaster": 10.0,
	"cyber_incident": 3.0,
	"other": 1.0,
}


def _extract_location_weight(text: str) -> Tuple[float, str]:
	"""Detect location type and return (weight, location_label)."""
	t = text.lower()
	# mapping from keywords to (label, weight)
	mapping = [
		(r"\bbank\b", ("bank", 4.0)),
		(r"\b(s?upermarket|mall|retail|store|shop)\b", ("retail", 2.5)),
		(r"\b(home|house|apartment|residence)\b", ("home", 2.0)),
		(r"\b(warehouse|office|closed office)\b", ("warehouse", 1.0)),
		(r"\b(school|hospital)\b", ("sensitive", 3.0)),
		(r"\b(subway|bus|train|transit|station)\b", ("transit", 3.0)),
	]
	for pat, (label, weight) in mapping:
		if re.search(pat, t):
			return weight, label
	return 0.0, "unknown"


def _extract_weapon_weight(text: str) -> Tuple[float, str]:
	t = text.lower()
	if re.search(r"\b(gun|firearm|rifle|pistol|shotgun|shooting)\b", t):
		return 3.0, "firearm"
	if re.search(r"\b(knife|blade|stabbed|stabbing|bat|club|blunt)\b", t):
		return 1.5, "knife/blunt"
	return 0.0, "none"


def _extract_suspect_count_weight(text: str) -> Tuple[float, int]:
	t = text.lower()
	# look for numeric counts
	m = re.search(r"(\b(\d+)\b) ?(suspects|people|persons|men|women|individuals)?", t)
	if m:
		try:
			n = int(m.group(2))
		except Exception:
			n = 1
		if n == 1:
			return 0.5, n
		if 2 <= n <= 3:
			return 1.0, n
		if n >= 4:
			return 2.0, n
	# words
	if re.search(r"\bone\b( suspect)?\b", t):
		return 0.5, 1
	if re.search(r"\btwo|three|2|3\b", t):
		return 1.0, 2
	if re.search(r"\bfour|five|several|many\b", t):
		return 2.0, 4
	return 0.0, 0


def _extract_time_weight(text: str) -> Tuple[float, str]:
	t = text.lower()
	# night: 9pm-5am
	if re.search(r"\b(night|tonight|after dark|after-dark|9pm|10pm|11pm|12am|1am|2am|3am|4am|5am)\b", t):
		return 1.5, "night"
	# explicit times daytime
	return 0.0, "day"


def _extract_vulnerability_weight(text: str) -> Tuple[float, str]:
	t = text.lower()
	if re.search(r"\b(children|child|kids|elderly|old people|senior|baby|infant)\b", t):
		return 2.0, "children/elderly"
	if re.search(r"\badults? (inside|home|present)|people inside|occupants|people home\b", t):
		return 1.0, "adults"
	if re.search(r"\b(unoccupied|vacant|empty|no one inside)\b", t):
		return 0.0, "unoccupied"
	return 0.0, "unknown"


def calculate_urgency(classifications: List[Dict[str, Any]], text: str) -> Dict[str, Any]:
	"""Calculate an urgency score between 0 and 10 using weights per incident type.

	Uses base score for the top predicted type (by confidence), then applies additive
	weights: location, weapon, suspect count, time, vulnerability. For 'bombing' and other
	immediate-catastrophic types, returns max urgency 10.0.
	"""
	if not classifications:
		return {"urgency": 0.0, "breakdown": [], "reason": "no classifications"}

	# choose top classification by confidence
	top = max(classifications, key=lambda x: float(x.get("confidence", 0.0)))
	ttype = top.get("type", "other")
	base = BASE_SCORE_BY_TYPE.get(ttype, 1.0)

	# immediate max cases
	if ttype in ("bombing", "hostage", "natural_disaster"):
		return {"urgency": 10.0, "reason": ttype, "breakdown": [{"base": base, "override": "max for type"}]}

	loc_w, loc_label = _extract_location_weight(text)
	weap_w, weap_label = _extract_weapon_weight(text)
	sus_w, sus_n = _extract_suspect_count_weight(text)
	time_w, time_label = _extract_time_weight(text)
	vuln_w, vuln_label = _extract_vulnerability_weight(text)

	# Total urgency = base + all weights (clamped to 10)
	total = base + loc_w + weap_w + sus_w + time_w + vuln_w
	total = min(total, 10.0)

	breakdown = [
		{"component": "base", "type": ttype, "value": base},
		{"component": "location", "label": loc_label, "value": loc_w},
		{"component": "weapon", "label": weap_label, "value": weap_w},
		{"component": "suspects", "count": sus_n, "value": sus_w},
		{"component": "time", "label": time_label, "value": time_w},
		{"component": "vulnerability", "label": vuln_label, "value": vuln_w},
	]

	return {"urgency": round(total, 2), "reason": ttype, "breakdown": breakdown}



# ---------------------------------------------------------------------------
# Demo / example usage
# ---------------------------------------------------------------------------
def _demo_local():
	samples = [
		"I just heard multiple shots fired near the corner of 5th and Main. People are screaming.",
		"There is a suspicious package at the subway entrance, an unattended bag with wires.",
		"Someone broke into the house and stole jewelry. The back door was forced open.",
		"Car hit a bicyclist and fled the scene.",
	]

	print("--- Case 1: Local deterministic classifier ---")
	for s in samples:
		types = get_incident_types(s)
		severity = aggregate_severity(types)
		urgency = calculate_urgency(types, s)
		print("Report:", s)
		print("Classification:", types)
		print("Severity:", severity)
		print("Urgency:", urgency)
		print()


def _demo_llm_prompt_only():
	text = DEMO_REPORT_TEXT
	prompt = build_llm_prompt_for_types(text)
	print("--- Case 2: LLM prompt (no network call) ---")
	print("Report:", text)
	print("Prompt to send to LLM (JSON-only expected):\n")
	print(prompt)
	print('\nExpected output example:')
	print(json.dumps({"types": [{"type": "bombing", "confidence": 0.92, "reason": "mentions explosion and smoke"}]}, indent=2))


if __name__ == "__main__":
	# Run demos. If OpenAI is available and an API key is configured, try a live LLM call (optional).
	_demo_local()

	# Show the prepared prompt and example for Case 2
	_demo_llm_prompt_only()

	# Optional: live LLM call example (only if user wants to enable). We do not run it automatically.
	openai_key = os.environ.get("OPENAI_API_KEY")
	if openai_key:
		try:
			import importlib

			openai = importlib.import_module("openai")
			openai.api_key = openai_key
			text = "Multiple people reporting smoke and a loud explosion near the market."
			prompt = build_llm_prompt_for_types(text)
			print('\nAttempting live LLM call using OpenAI...')
			resp = call_llm(openai, prompt, model=DEFAULT_LLM_MODEL)
			print('LLM raw response:', resp)
		except Exception as exc:
			print('OpenAI live call failed or not configured:', exc)

	# Demo: print outputs from multiple LLM functions (simulated, OpenAI if available, HuggingFace if token present)
	print("\n--- Case 2: LLM wrappers demo ---")
	text = "Multiple people reporting smoke and a loud explosion near the market."
	prompt = build_llm_prompt_for_types(text)

	# Local simulator
	print("\nLocal LLM simulator output:")
	try:
		print(local_llm_simulator(prompt))
	except Exception as exc:
		print("Local simulator failed:", exc)

	# OpenAI wrapper (if token present)
	if os.environ.get("OPENAI_API_KEY"):
		print("\nOpenAI wrapper output:")
		try:
			print(call_openai(prompt, model=DEFAULT_LLM_MODEL))
		except Exception as exc:
			print("OpenAI wrapper failed:", exc)

	# HuggingFace inference wrapper (if HF token present)
	if os.environ.get("HF_API_TOKEN") or os.environ.get("HUGGINGFACE_API_TOKEN"):
		print("\nHuggingFace inference output:")
		try:
			print(call_huggingface_inference(prompt))
		except Exception as exc:
			print("HuggingFace inference failed:", exc)

	# Print urgency for the demo text using local classifier
	print("\nDemo text urgency (from local classifier):")
	cls = get_incident_types(text)
	print(calculate_urgency(cls, text))

