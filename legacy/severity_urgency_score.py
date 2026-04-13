"""
Severity and Urgency Scoring System

This module calculates urgency scores (0.0 to 1.0) for incident reports based on:
1. Base score from incident type (from incident_severity_score)
2. Contextual factors: vulnerable people, suspects, weapons, time, location

Urgency Score Formula:
    urgency = (base_score + factor_weights) / max_possible_score
    normalized to 0.0 - 1.0 range

Usage:
    from severity_urgency_score import calculate_urgency_score
    
    report_text = "Armed robbery at bank with 3 suspects, children inside"
    urgency = calculate_urgency_score(report_text)
    print(f"Urgency: {urgency['score']:.2f}")
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from incident_severity_score import (
    get_incident_types,
    BASE_SCORE_BY_TYPE,
)


# -----------------------------------------
# Weight definitions for contextual factors
# -----------------------------------------

# Maximum possible score (for normalization)
# Base max = 10.0, all factors max = ~13.5
MAX_URGENCY_POINTS = 23.5

# Vulnerable people weights
VULNERABILITY_WEIGHTS = {
    "children": 2.5,           # kids, child, baby, infant
    "elderly": 2.5,            # senior, elderly, old people
    "disabled": 2.0,           # disabled, wheelchair
    "large_crowd": 1.5,        # crowd, many people, dozens
    "adults_present": 0.5,     # adults inside/present
    "no_one_present": 0.0,     # unoccupied, vacant, empty
}

# Suspect count weights
SUSPECT_WEIGHTS = {
    "1": 0.5,
    "2-3": 1.5,
    "4-6": 2.5,
    "7+": 3.5,
}

# Weapon type weights
WEAPON_WEIGHTS = {
    "firearm": 4.0,            # gun, rifle, pistol, armed, shooting
    "explosive": 5.0,          # bomb, explosive, IED
    "knife": 2.0,              # knife, blade, stabbing
    "vehicle": 2.5,            # vehicle as weapon, hit-and-run
    "blunt_object": 1.5,       # bat, club, blunt
    "unknown_weapon": 1.0,     # armed but unspecified
    "no_weapon": 0.0,
}

# Time of day weights
TIME_WEIGHTS = {
    "late_night": 2.0,         # 11pm-4am (highest risk)
    "night": 1.5,              # 9pm-11pm, 4am-6am
    "evening": 0.5,            # 6pm-9pm
    "early_morning": 0.5,      # 6am-8am
    "daytime": 0.0,            # 8am-6pm
}

# Location risk weights
LOCATION_WEIGHTS = {
    "school": 4.0,             # school, daycare, playground
    "hospital": 3.5,           # hospital, clinic, medical center
    "bank": 3.5,               # bank, financial institution
    "transit": 3.0,            # subway, bus, train, airport
    "mall_retail": 2.5,        # mall, shopping center, store
    "residential": 2.0,        # home, house, apartment
    "public_space": 1.5,       # park, street, plaza
    "office": 1.0,             # office building, workplace
    "warehouse": 0.5,          # warehouse, storage, closed facility
    "unknown": 0.0,
}


# -----------------------------------------
# Factor extraction functions
# -----------------------------------------

def _extract_vulnerability_factor(text: str) -> Dict[str, Any]:
    """Extract vulnerable people mentions and return weight + details."""
    t = text.lower()
    factors = []
    total_weight = 0.0
    
    # Check for specific vulnerable groups
    if re.search(r"\b(child(?:ren)?|kids?|baby|babies|infant|toddler)\b", t):
        factors.append("children")
        total_weight += VULNERABILITY_WEIGHTS["children"]
    
    if re.search(r"\b(elderly|senior|old (?:people|person)|aged)\b", t):
        factors.append("elderly")
        total_weight += VULNERABILITY_WEIGHTS["elderly"]
    
    if re.search(r"\b(disabled|wheelchair|handicap(?:ped)?)\b", t):
        factors.append("disabled")
        total_weight += VULNERABILITY_WEIGHTS["disabled"]
    
    # Check for crowd/occupancy
    if re.search(r"\b(crowd|dozens|many people|hundreds|packed|busy)\b", t):
        factors.append("large_crowd")
        total_weight += VULNERABILITY_WEIGHTS["large_crowd"]
    elif re.search(r"\b(people (?:inside|present|home)|occupants|adults? (?:inside|present))\b", t):
        if not factors:  # only if no specific vulnerable group found
            factors.append("adults_present")
            total_weight += VULNERABILITY_WEIGHTS["adults_present"]
    elif re.search(r"\b(unoccupied|vacant|empty|no one (?:inside|home|present))\b", t):
        factors.append("no_one_present")
        total_weight = 0.0
    
    return {
        "weight": total_weight,
        "factors": factors,
        "description": ", ".join(factors) if factors else "none detected"
    }


def _extract_suspect_factor(text: str) -> Dict[str, Any]:
    """Extract suspect count and return weight + details."""
    t = text.lower()
    
    # Try to find explicit numbers
    numbers = re.findall(r"\b(\d+)\s*(?:suspects?|people|persons?|men|individuals?|armed|attackers?)\b", t)
    if numbers:
        count = int(numbers[0])
    else:
        # Look for text indicators
        if re.search(r"\b(one|single|a suspect)\b", t):
            count = 1
        elif re.search(r"\b(two|three|couple|pair)\b", t):
            count = 2
        elif re.search(r"\b(several|four|five|six)\b", t):
            count = 5
        elif re.search(r"\b(many|multiple|numerous|gang|group)\b", t):
            count = 7
        else:
            count = 0
    
    # Determine weight category
    if count == 0:
        category = "unknown"
        weight = 0.0
    elif count == 1:
        category = "1"
        weight = SUSPECT_WEIGHTS["1"]
    elif 2 <= count <= 3:
        category = "2-3"
        weight = SUSPECT_WEIGHTS["2-3"]
    elif 4 <= count <= 6:
        category = "4-6"
        weight = SUSPECT_WEIGHTS["4-6"]
    else:  # 7+
        category = "7+"
        weight = SUSPECT_WEIGHTS["7+"]
    
    return {
        "weight": weight,
        "count": count if count > 0 else "unknown",
        "category": category,
        "description": f"{count} suspect(s)" if count > 0 else "count unknown"
    }


def _extract_weapon_factor(text: str) -> Dict[str, Any]:
    """Extract weapon type and return weight + details."""
    t = text.lower()
    weapons_found = []
    max_weight = 0.0
    
    # Check for explosives (highest priority)
    if re.search(r"\b(bomb|explosive|ied|detonat|blast|grenade)\b", t):
        weapons_found.append("explosive")
        max_weight = max(max_weight, WEAPON_WEIGHTS["explosive"])
    
    # Check for firearms
    if re.search(r"\b(gun|firearm|rifle|pistol|shotgun|armed|shoot(?:ing|er)?|shots?)\b", t):
        weapons_found.append("firearm")
        max_weight = max(max_weight, WEAPON_WEIGHTS["firearm"])
    
    # Check for vehicle as weapon
    if re.search(r"\b(hit and run|ran (?:over|into)|vehicle (?:weapon|attack)|drove into)\b", t):
        weapons_found.append("vehicle")
        max_weight = max(max_weight, WEAPON_WEIGHTS["vehicle"])
    
    # Check for knife/blade
    if re.search(r"\b(knife|knives|blade|stab(?:bing|bed)?|machete)\b", t):
        weapons_found.append("knife")
        max_weight = max(max_weight, WEAPON_WEIGHTS["knife"])
    
    # Check for blunt objects
    if re.search(r"\b(bat|club|pipe|hammer|blunt|bludgeon)\b", t):
        weapons_found.append("blunt_object")
        max_weight = max(max_weight, WEAPON_WEIGHTS["blunt_object"])
    
    # Check for generic "armed" without specifics
    if not weapons_found and re.search(r"\b(armed|weapon)\b", t):
        weapons_found.append("unknown_weapon")
        max_weight = WEAPON_WEIGHTS["unknown_weapon"]
    
    if not weapons_found:
        weapons_found.append("no_weapon")
        max_weight = 0.0
    
    return {
        "weight": max_weight,
        "types": weapons_found,
        "description": ", ".join(weapons_found)
    }


def _extract_time_factor(text: str) -> Dict[str, Any]:
    """Extract time of day and return weight + details."""
    t = text.lower()
    
    # Late night (highest risk): 11pm-4am
    if re.search(r"\b(11\s*pm|12\s*am|1\s*am|2\s*am|3\s*am|midnight|late night|after midnight)\b", t):
        category = "late_night"
        weight = TIME_WEIGHTS["late_night"]
    # Night: 9pm-11pm, 4am-6am
    elif re.search(r"\b(9\s*pm|10\s*pm|4\s*am|5\s*am|night|tonight|after dark)\b", t):
        category = "night"
        weight = TIME_WEIGHTS["night"]
    # Evening: 6pm-9pm
    elif re.search(r"\b(6\s*pm|7\s*pm|8\s*pm|evening|dusk)\b", t):
        category = "evening"
        weight = TIME_WEIGHTS["evening"]
    # Early morning: 6am-8am
    elif re.search(r"\b(6\s*am|7\s*am|8\s*am|early morning|dawn)\b", t):
        category = "early_morning"
        weight = TIME_WEIGHTS["early_morning"]
    # Daytime (default/lowest risk)
    else:
        category = "daytime"
        weight = TIME_WEIGHTS["daytime"]
    
    return {
        "weight": weight,
        "category": category,
        "description": category.replace("_", " ")
    }


def _extract_location_factor(text: str) -> Dict[str, Any]:
    """Extract location type and return weight + details."""
    t = text.lower()
    
    # Check locations in order of priority/risk
    if re.search(r"\b(school|daycare|kindergarten|playground|university|college|campus)\b", t):
        category = "school"
        weight = LOCATION_WEIGHTS["school"]
    elif re.search(r"\b(hospital|clinic|medical center|emergency room|er)\b", t):
        category = "hospital"
        weight = LOCATION_WEIGHTS["hospital"]
    elif re.search(r"\b(bank|atm|financial|credit union)\b", t):
        category = "bank"
        weight = LOCATION_WEIGHTS["bank"]
    elif re.search(r"\b(subway|metro|bus|train|station|airport|transit)\b", t):
        category = "transit"
        weight = LOCATION_WEIGHTS["transit"]
    elif re.search(r"\b(mall|shopping center|store|retail|market|supermarket)\b", t):
        category = "mall_retail"
        weight = LOCATION_WEIGHTS["mall_retail"]
    elif re.search(r"\b(home|house|apartment|residence|residential)\b", t):
        category = "residential"
        weight = LOCATION_WEIGHTS["residential"]
    elif re.search(r"\b(park|street|plaza|square|public)\b", t):
        category = "public_space"
        weight = LOCATION_WEIGHTS["public_space"]
    elif re.search(r"\b(office|building|workplace|business)\b", t):
        category = "office"
        weight = LOCATION_WEIGHTS["office"]
    elif re.search(r"\b(warehouse|storage|facility|closed)\b", t):
        category = "warehouse"
        weight = LOCATION_WEIGHTS["warehouse"]
    else:
        category = "unknown"
        weight = LOCATION_WEIGHTS["unknown"]
    
    return {
        "weight": weight,
        "category": category,
        "description": category.replace("_", " ")
    }


# -----------------------------------------
# Main urgency calculation function
# -----------------------------------------

def calculate_urgency_score(
    report_text: str,
    incident_types: Optional[List[Dict[str, Any]]] = None,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    Calculate urgency score (0.0-1.0) for an incident report.
    
    Args:
        report_text: The incident report text
        incident_types: Pre-computed incident types (optional; will compute if not provided)
        normalize: If True, normalize to 0-1 range; if False, return raw score
    
    Returns:
        {
            "score": float (0.0-1.0 if normalized),
            "raw_score": float (before normalization),
            "max_possible": float,
            "incident_type": str,
            "base_score": float,
            "factors": {
                "vulnerability": {...},
                "suspects": {...},
                "weapon": {...},
                "time": {...},
                "location": {...}
            },
            "breakdown": [...]  # list of all contributing factors
        }
    """
    
    # Step 1: Get incident type and base score
    if incident_types is None:
        incident_types = get_incident_types(report_text, top_k=1, min_score=0.0)
    
    if not incident_types:
        incident_type = "other"
        base_score = BASE_SCORE_BY_TYPE.get("other", 1.0)
    else:
        top_incident = incident_types[0]
        incident_type = top_incident.get("type", "other")
        base_score = BASE_SCORE_BY_TYPE.get(incident_type, 1.0)
    
    # Step 2: Extract contextual factors
    vulnerability = _extract_vulnerability_factor(report_text)
    suspects = _extract_suspect_factor(report_text)
    weapon = _extract_weapon_factor(report_text)
    time_of_day = _extract_time_factor(report_text)
    location = _extract_location_factor(report_text)
    
    # Step 3: Calculate total raw score
    raw_score = (
        base_score +
        vulnerability["weight"] +
        suspects["weight"] +
        weapon["weight"] +
        time_of_day["weight"] +
        location["weight"]
    )
    
    # Step 4: Normalize to 0-1 if requested
    if normalize:
        final_score = min(raw_score / MAX_URGENCY_POINTS, 1.0)
    else:
        final_score = raw_score
    
    # Step 5: Build breakdown for transparency
    breakdown = [
        {"component": "base_score", "type": incident_type, "value": base_score},
        {"component": "vulnerability", "value": vulnerability["weight"], "details": vulnerability["description"]},
        {"component": "suspects", "value": suspects["weight"], "details": suspects["description"]},
        {"component": "weapon", "value": weapon["weight"], "details": weapon["description"]},
        {"component": "time", "value": time_of_day["weight"], "details": time_of_day["description"]},
        {"component": "location", "value": location["weight"], "details": location["description"]},
    ]
    
    return {
        "score": round(final_score, 3),
        "raw_score": round(raw_score, 2),
        "max_possible": MAX_URGENCY_POINTS,
        "incident_type": incident_type,
        "base_score": base_score,
        "factors": {
            "vulnerability": vulnerability,
            "suspects": suspects,
            "weapon": weapon,
            "time": time_of_day,
            "location": location,
        },
        "breakdown": breakdown,
    }


def get_urgency_level(score: float) -> str:
    """Convert urgency score to human-readable level."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


# -----------------------------------------
# Demo / Testing
# -----------------------------------------

def _demo():
    """Demonstrate urgency scoring with various scenarios."""
    
    test_cases = [
        {
            "name": "Armed bank robbery with children",
            "report": "Armed robbery in progress at downtown bank. 3 suspects with guns, children and elderly inside.",
        },
        {
            "name": "School shooting",
            "report": "Active shooter at Lincoln High School. Multiple shots fired, students hiding.",
        },
        {
            "name": "Late night burglary",
            "report": "Someone broke into the house at 2am. Back door forced open. Family was asleep upstairs.",
        },
        {
            "name": "Daytime theft",
            "report": "Shoplifting at retail store. One suspect took merchandise and left. No weapons.",
        },
        {
            "name": "Hit and run with victim",
            "report": "Hit and run accident on Main Street at 3pm. Car hit a pedestrian and fled. Victim unconscious.",
        },
        {
            "name": "Bomb threat at transit",
            "report": "Suspicious package at subway station. Bomb squad called. Station evacuated, hundreds of people affected.",
        },
    ]
    
    print("=" * 80)
    print("URGENCY SCORING DEMO")
    print("=" * 80)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. {test['name']}")
        print(f"   Report: {test['report']}")
        print("-" * 80)
        
        result = calculate_urgency_score(test['report'])
        level = get_urgency_level(result['score'])
        
        print(f"   Urgency Score: {result['score']:.3f} ({level})")
        print(f"   Incident Type: {result['incident_type']} (base: {result['base_score']:.1f})")
        print(f"   Raw Score: {result['raw_score']:.2f} / {result['max_possible']:.1f}")
        print("\n   Factor Breakdown:")
        for item in result['breakdown']:
            if item['component'] != 'base_score':
                details = f" - {item.get('details', '')}" if item.get('details') else ""
                print(f"     • {item['component'].capitalize()}: +{item['value']:.1f}{details}")
        print()


if __name__ == "__main__":
    _demo()
