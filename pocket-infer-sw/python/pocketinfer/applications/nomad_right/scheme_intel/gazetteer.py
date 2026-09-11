"""
Offline place gazetteer: Indian states/UTs and the cities/districts migrant
workers most often name, each mapped to its state.

Plain geography, no scheme facts. Used by ScenarioEngine to turn "working in
Chennai" into current_city=Chennai / current_state=Tamil Nadu.
"""

import re
from typing import List, NamedTuple, Optional

STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    # Union Territories
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
)

# Alternative spellings / short forms (lower-case). Deliberately excludes
# ambiguous English words ("up", "mp", "tn") that would match ordinary speech.
STATE_ALIASES = {
    "tamilnadu": "Tamil Nadu", "tamil nadu state": "Tamil Nadu",
    "orissa": "Odisha", "pondicherry": "Puducherry", "pondy": "Puducherry",
    "bengal": "West Bengal", "w bengal": "West Bengal",
    "new delhi": "Delhi", "delhi ncr": "Delhi",
    "j&k": "Jammu and Kashmir", "jammu kashmir": "Jammu and Kashmir", "kashmir": "Jammu and Kashmir",
    "u.p.": "Uttar Pradesh", "u.p": "Uttar Pradesh", "m.p.": "Madhya Pradesh",
    "uttaranchal": "Uttarakhand", "chattisgarh": "Chhattisgarh", "chhatisgarh": "Chhattisgarh",
    "jharkand": "Jharkhand", "telengana": "Telangana", "karnatak": "Karnataka",
    "maharastra": "Maharashtra", "gujrat": "Gujarat", "rajastan": "Rajasthan",
    "andhra": "Andhra Pradesh", "himachal": "Himachal Pradesh", "arunachal": "Arunachal Pradesh",
    "daman": "Dadra and Nagar Haveli and Daman and Diu", "andaman": "Andaman and Nicobar Islands",
}

# city -> (display name, state, is_district_hq)
CITIES = {
    # Tamil Nadu
    "chennai": ("Chennai", "Tamil Nadu", True), "madras": ("Chennai", "Tamil Nadu", True),
    "coimbatore": ("Coimbatore", "Tamil Nadu", True), "madurai": ("Madurai", "Tamil Nadu", True),
    "tiruppur": ("Tiruppur", "Tamil Nadu", True), "tirupur": ("Tiruppur", "Tamil Nadu", True),
    "salem": ("Salem", "Tamil Nadu", True), "trichy": ("Tiruchirappalli", "Tamil Nadu", True),
    "tiruchirappalli": ("Tiruchirappalli", "Tamil Nadu", True), "erode": ("Erode", "Tamil Nadu", True),
    "vellore": ("Vellore", "Tamil Nadu", True), "hosur": ("Hosur", "Tamil Nadu", False),
    "kanchipuram": ("Kanchipuram", "Tamil Nadu", True), "sriperumbudur": ("Sriperumbudur", "Tamil Nadu", False),
    "chengalpattu": ("Chengalpattu", "Tamil Nadu", True), "tirunelveli": ("Tirunelveli", "Tamil Nadu", True),
    "thoothukudi": ("Thoothukudi", "Tamil Nadu", True), "tuticorin": ("Thoothukudi", "Tamil Nadu", True),
    "thanjavur": ("Thanjavur", "Tamil Nadu", True), "nagercoil": ("Nagercoil", "Tamil Nadu", False),
    "dindigul": ("Dindigul", "Tamil Nadu", True), "karur": ("Karur", "Tamil Nadu", True),
    "krishnagiri": ("Krishnagiri", "Tamil Nadu", True), "cuddalore": ("Cuddalore", "Tamil Nadu", True),
    "ambattur": ("Ambattur", "Tamil Nadu", False), "tambaram": ("Tambaram", "Tamil Nadu", False),
    # Karnataka
    "bengaluru": ("Bengaluru", "Karnataka", True), "bangalore": ("Bengaluru", "Karnataka", True),
    "mysuru": ("Mysuru", "Karnataka", True), "mysore": ("Mysuru", "Karnataka", True),
    "mangaluru": ("Mangaluru", "Karnataka", False), "mangalore": ("Mangaluru", "Karnataka", False),
    "hubli": ("Hubballi", "Karnataka", False), "hubballi": ("Hubballi", "Karnataka", False),
    "belagavi": ("Belagavi", "Karnataka", True), "belgaum": ("Belagavi", "Karnataka", True),
    # Kerala
    "kochi": ("Kochi", "Kerala", False), "cochin": ("Kochi", "Kerala", False),
    "ernakulam": ("Ernakulam", "Kerala", True), "thiruvananthapuram": ("Thiruvananthapuram", "Kerala", True),
    "trivandrum": ("Thiruvananthapuram", "Kerala", True), "kozhikode": ("Kozhikode", "Kerala", True),
    "calicut": ("Kozhikode", "Kerala", True), "thrissur": ("Thrissur", "Kerala", True),
    "perumbavoor": ("Perumbavoor", "Kerala", False),
    # Andhra Pradesh / Telangana
    "hyderabad": ("Hyderabad", "Telangana", True), "secunderabad": ("Secunderabad", "Telangana", False),
    "warangal": ("Warangal", "Telangana", True), "visakhapatnam": ("Visakhapatnam", "Andhra Pradesh", True),
    "vizag": ("Visakhapatnam", "Andhra Pradesh", True), "vijayawada": ("Vijayawada", "Andhra Pradesh", False),
    "guntur": ("Guntur", "Andhra Pradesh", True), "nellore": ("Nellore", "Andhra Pradesh", True),
    "tirupati": ("Tirupati", "Andhra Pradesh", True),
    # Maharashtra
    "mumbai": ("Mumbai", "Maharashtra", True), "bombay": ("Mumbai", "Maharashtra", True),
    "pune": ("Pune", "Maharashtra", True), "nagpur": ("Nagpur", "Maharashtra", True),
    "nashik": ("Nashik", "Maharashtra", True), "thane": ("Thane", "Maharashtra", True),
    "navi mumbai": ("Navi Mumbai", "Maharashtra", False), "aurangabad": ("Aurangabad", "Maharashtra", True),
    "solapur": ("Solapur", "Maharashtra", True), "kolhapur": ("Kolhapur", "Maharashtra", True),
    "bhiwandi": ("Bhiwandi", "Maharashtra", False),
    # Gujarat
    "ahmedabad": ("Ahmedabad", "Gujarat", True), "surat": ("Surat", "Gujarat", True),
    "vadodara": ("Vadodara", "Gujarat", True), "baroda": ("Vadodara", "Gujarat", True),
    "rajkot": ("Rajkot", "Gujarat", True), "morbi": ("Morbi", "Gujarat", True),
    "gandhinagar": ("Gandhinagar", "Gujarat", True), "jamnagar": ("Jamnagar", "Gujarat", True),
    # Delhi NCR / Haryana / Punjab
    "delhi": ("Delhi", "Delhi", False), "noida": ("Noida", "Uttar Pradesh", False),
    "greater noida": ("Greater Noida", "Uttar Pradesh", False), "ghaziabad": ("Ghaziabad", "Uttar Pradesh", True),
    "gurugram": ("Gurugram", "Haryana", True), "gurgaon": ("Gurugram", "Haryana", True),
    "faridabad": ("Faridabad", "Haryana", True), "panipat": ("Panipat", "Haryana", True),
    "sonipat": ("Sonipat", "Haryana", True), "ludhiana": ("Ludhiana", "Punjab", True),
    "amritsar": ("Amritsar", "Punjab", True), "jalandhar": ("Jalandhar", "Punjab", True),
    "chandigarh": ("Chandigarh", "Chandigarh", False),
    # Uttar Pradesh
    "lucknow": ("Lucknow", "Uttar Pradesh", True), "kanpur": ("Kanpur", "Uttar Pradesh", True),
    "varanasi": ("Varanasi", "Uttar Pradesh", True), "banaras": ("Varanasi", "Uttar Pradesh", True),
    "prayagraj": ("Prayagraj", "Uttar Pradesh", True), "allahabad": ("Prayagraj", "Uttar Pradesh", True),
    "gorakhpur": ("Gorakhpur", "Uttar Pradesh", True), "azamgarh": ("Azamgarh", "Uttar Pradesh", True),
    "jaunpur": ("Jaunpur", "Uttar Pradesh", True), "ballia": ("Ballia", "Uttar Pradesh", True),
    "deoria": ("Deoria", "Uttar Pradesh", True), "agra": ("Agra", "Uttar Pradesh", True),
    "meerut": ("Meerut", "Uttar Pradesh", True), "bareilly": ("Bareilly", "Uttar Pradesh", True),
    "aligarh": ("Aligarh", "Uttar Pradesh", True), "moradabad": ("Moradabad", "Uttar Pradesh", True),
    "sultanpur": ("Sultanpur", "Uttar Pradesh", True), "basti": ("Basti", "Uttar Pradesh", True),
    "bahraich": ("Bahraich", "Uttar Pradesh", True), "mau": ("Mau", "Uttar Pradesh", True),
    # Bihar
    "patna": ("Patna", "Bihar", True), "gaya": ("Gaya", "Bihar", True),
    "muzaffarpur": ("Muzaffarpur", "Bihar", True), "darbhanga": ("Darbhanga", "Bihar", True),
    "madhubani": ("Madhubani", "Bihar", True), "purnia": ("Purnia", "Bihar", True),
    "bhagalpur": ("Bhagalpur", "Bihar", True), "siwan": ("Siwan", "Bihar", True),
    "chhapra": ("Saran", "Bihar", True), "saran": ("Saran", "Bihar", True),
    "gopalganj": ("Gopalganj", "Bihar", True), "samastipur": ("Samastipur", "Bihar", True),
    "begusarai": ("Begusarai", "Bihar", True), "katihar": ("Katihar", "Bihar", True),
    "araria": ("Araria", "Bihar", True), "kishanganj": ("Kishanganj", "Bihar", True),
    "sitamarhi": ("Sitamarhi", "Bihar", True), "motihari": ("East Champaran", "Bihar", True),
    "bettiah": ("West Champaran", "Bihar", True), "hajipur": ("Vaishali", "Bihar", True),
    "vaishali": ("Vaishali", "Bihar", True), "nalanda": ("Nalanda", "Bihar", True),
    "buxar": ("Buxar", "Bihar", True), "sasaram": ("Rohtas", "Bihar", True),
    "arrah": ("Bhojpur", "Bihar", True), "saharsa": ("Saharsa", "Bihar", True),
    "supaul": ("Supaul", "Bihar", True), "nawada": ("Nawada", "Bihar", True),
    # Jharkhand
    "ranchi": ("Ranchi", "Jharkhand", True), "dhanbad": ("Dhanbad", "Jharkhand", True),
    "jamshedpur": ("Jamshedpur", "Jharkhand", False), "bokaro": ("Bokaro", "Jharkhand", True),
    "hazaribagh": ("Hazaribagh", "Jharkhand", True), "giridih": ("Giridih", "Jharkhand", True),
    "palamu": ("Palamu", "Jharkhand", True), "dumka": ("Dumka", "Jharkhand", True),
    # Odisha
    "bhubaneswar": ("Bhubaneswar", "Odisha", False), "cuttack": ("Cuttack", "Odisha", True),
    "berhampur": ("Berhampur", "Odisha", False), "ganjam": ("Ganjam", "Odisha", True),
    "balangir": ("Balangir", "Odisha", True), "bolangir": ("Balangir", "Odisha", True),
    "rourkela": ("Rourkela", "Odisha", False), "sambalpur": ("Sambalpur", "Odisha", True),
    "kalahandi": ("Kalahandi", "Odisha", True), "nuapada": ("Nuapada", "Odisha", True),
    # West Bengal / North East
    "kolkata": ("Kolkata", "West Bengal", True), "calcutta": ("Kolkata", "West Bengal", True),
    "howrah": ("Howrah", "West Bengal", True), "siliguri": ("Siliguri", "West Bengal", False),
    "murshidabad": ("Murshidabad", "West Bengal", True), "malda": ("Malda", "West Bengal", True),
    "guwahati": ("Guwahati", "Assam", False), "dibrugarh": ("Dibrugarh", "Assam", True),
    "agartala": ("Agartala", "Tripura", False), "imphal": ("Imphal", "Manipur", False),
    "shillong": ("Shillong", "Meghalaya", False),
    # Rajasthan / Madhya Pradesh / Chhattisgarh
    "jaipur": ("Jaipur", "Rajasthan", True), "jodhpur": ("Jodhpur", "Rajasthan", True),
    "udaipur": ("Udaipur", "Rajasthan", True), "kota": ("Kota", "Rajasthan", True),
    "ajmer": ("Ajmer", "Rajasthan", True), "bikaner": ("Bikaner", "Rajasthan", True),
    "bhopal": ("Bhopal", "Madhya Pradesh", True), "indore": ("Indore", "Madhya Pradesh", True),
    "jabalpur": ("Jabalpur", "Madhya Pradesh", True), "gwalior": ("Gwalior", "Madhya Pradesh", True),
    "rewa": ("Rewa", "Madhya Pradesh", True), "raipur": ("Raipur", "Chhattisgarh", True),
    "bilaspur": ("Bilaspur", "Chhattisgarh", True), "durg": ("Durg", "Chhattisgarh", True),
    "bhilai": ("Bhilai", "Chhattisgarh", False),
    # Others
    "dehradun": ("Dehradun", "Uttarakhand", True), "haridwar": ("Haridwar", "Uttarakhand", True),
    "shimla": ("Shimla", "Himachal Pradesh", True), "srinagar": ("Srinagar", "Jammu and Kashmir", True),
    "jammu": ("Jammu", "Jammu and Kashmir", True), "panaji": ("Panaji", "Goa", False),
    "margao": ("Margao", "Goa", False), "puducherry city": ("Puducherry", "Puducherry", True),
}


class PlaceMention(NamedTuple):
    start: int
    end: int
    surface: str
    state: str
    city: Optional[str]
    district: Optional[str]


def _compile(words):
    ordered = sorted(words, key=len, reverse=True)
    return re.compile(r"(?<![a-z])(" + "|".join(re.escape(w) for w in ordered) + r")(?![a-z])")


_STATE_KEYS = {s.lower(): s for s in STATES}
_STATE_KEYS.update(STATE_ALIASES)
_STATE_RE = _compile(_STATE_KEYS.keys())
_CITY_RE = _compile(CITIES.keys())


def find_places(text_lower: str) -> List[PlaceMention]:
    """
    All state and city mentions in lower-cased text, in order of appearance.
    A span that is both a state and a city ("delhi", "chandigarh") is
    reported once, as a city of that state.
    """
    found: List[PlaceMention] = []
    taken = []
    for m in _CITY_RE.finditer(text_lower):
        name, state, is_district = CITIES[m.group(1)]
        found.append(PlaceMention(m.start(), m.end(), m.group(1), state, name, name if is_district else None))
        taken.append((m.start(), m.end()))
    for m in _STATE_RE.finditer(text_lower):
        if any(s <= m.start() < e or s < m.end() <= e for s, e in taken):
            continue
        found.append(PlaceMention(m.start(), m.end(), m.group(1), _STATE_KEYS[m.group(1)], None, None))
    found.sort(key=lambda p: p.start)
    return found


def canonical_state(name: str) -> Optional[str]:
    if not name:
        return None
    key = name.strip().lower()
    if key in _STATE_KEYS:
        return _STATE_KEYS[key]
    if key in CITIES:
        return CITIES[key][1]
    return None
