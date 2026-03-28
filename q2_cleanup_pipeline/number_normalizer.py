import re

UNITS = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}
TENS = {
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
}
COMPOUND_TENS = {
    "इक्कीस": 21, "बाईस": 22, "तेईस": 23, "चौबीस": 24,
    "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28,
    "उनतीस": 29, "इकतीस": 31, "बत्तीस": 32, "तैंतीस": 33,
    "चौंतीस": 34, "पैंतीस": 35, "छत्तीस": 36, "सैंतीस": 37,
    "अड़तीस": 38, "उनतालीस": 39, "इकतालीस": 41, "बयालीस": 42,
    "तैंतालीस": 43, "चवालीस": 44, "पैंतालीस": 45, "छियालीस": 46,
    "सैंतालीस": 47, "अड़तालीस": 48, "उनचास": 49, "इक्यावन": 51,
    "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "इकसठ": 61, "बासठ": 62, "तिरसठ": 63, "चौंसठ": 64,
    "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67, "अड़सठ": 68,
    "उनहत्तर": 69, "इकहत्तर": 71, "बहत्तर": 72, "तिहत्तर": 73,
    "चौहत्तर": 74, "पचहत्तर": 75, "छिहत्तर": 76, "सतहत्तर": 77,
    "अठहत्तर": 78, "उनासी": 79, "इक्यासी": 81, "बयासी": 82,
    "तिरासी": 83, "चौरासी": 84, "पचासी": 85, "छियासी": 86,
    "सत्तासी": 87, "अट्ठासी": 88, "नवासी": 89, "इक्यानवे": 91,
    "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}
# Multipliers absorbed into digit
ABSORB_MULT = {"सौ": 100, "हज़ार": 1000, "हजार": 1000}
# Multipliers kept as words (Indian convention: "5 करोड़" not "50000000")
WORD_MULT = {"लाख": "लाख", "करोड़": "करोड़"}

ALL_NUM = {}
ALL_NUM.update(UNITS)
ALL_NUM.update(TENS)
ALL_NUM.update(COMPOUND_TENS)
ALL_NUM.update(ABSORB_MULT)

# Words after which "दो"/"एक" are confirmed as numbers
NUM_FOLLOWERS = {
    "किलो","किलोग्राम","ग्राम","लीटर","मीटर","किलोमीटर",
    "लोग","व्यक्ति","बच्चे","साल","वर्ष","महीने","दिन",
    "घंटे","मिनट","सेकंड","बजे","बार","किताब","किताबें",
    "पुस्तक","सेब","आम","प्रतिशत","गुना","सौ","हज़ार","लाख","करोड़",
}
# Words BEFORE which दो/एक are verbs/articles
VERB_BEFORE = {"को","मुझे","उसे","हमें","तुम्हें","आपको","रुपये","रुपया","पैसे"}
AMBIGUOUS = {"दो", "एक"}

IDIOMS = [
    "दो-चार","दो चार","चार-पाँच","चार पांच","पाँच-सात","पांच सात",
    "दो-तीन","दो तीन","तीन-चार","तीन चार","सात-आठ","आठ-दस",
    "एक-दो","एक दो","दस-बारह","छह-सात","पाँच-छह",
]

def is_verb_context(words, idx):
    prev = words[idx-1] if idx > 0 else None
    nxt  = words[idx+1] if idx+1 < len(words) else None
    # next word confirms number
    if nxt and nxt in NUM_FOLLOWERS:
        return False   # it's a number
    # prev word signals verb
    if prev and prev in VERB_BEFORE:
        return True    # it's a verb
    # last word in sentence → verb
    if nxt is None:
        return True
    # default: treat as verb (safe)
    return True

def parse_seq(words, start):
    ws = words[start:]
    total, current, consumed = 0, 0, 0
    word_mult_str = None
    i = 0
    while i < len(ws):
        w = ws[i]
        if w in WORD_MULT:
            total += current; current = 0
            word_mult_str = WORD_MULT[w]
            consumed = i + 1; i += 1; break
        if w in ABSORB_MULT:
            if current == 0: current = 1
            current *= ABSORB_MULT[w]
            total += current; current = 0
            consumed = i + 1; i += 1; continue
        if w in ALL_NUM:
            if w in AMBIGUOUS and is_verb_context(words, start + i):
                break
            current += ALL_NUM[w]
            consumed = i + 1; i += 1; continue
        break
    total += current
    if consumed == 0:
        return None, 0
    if word_mult_str:
        return f"{total} {word_mult_str}", consumed
    return str(total), consumed

def normalize_numbers(text):
    # protect idioms
    protected = text
    imap = {}
    for idx, idiom in enumerate(IDIOMS):
        if idiom in protected:
            ph = f"__I{idx}__"
            imap[ph] = idiom
            protected = protected.replace(idiom, ph)

    words = protected.split()
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ALL_NUM or w in WORD_MULT:
            val, consumed = parse_seq(words, i)
            if consumed > 0 and val is not None:
                out.append(val); i += consumed
            else:
                out.append(w); i += 1
        else:
            out.append(w); i += 1

    result = " ".join(out)
    for ph, orig in imap.items():
        result = result.replace(ph, orig)
    return result

# ── Tests ──────────────────────────────────────────────────────────────────
tests = [
    ("मुझे दस रुपये दो",           "मुझे 10 रुपये दो",          "दस=10, दो=verb(give)"),
    ("तीन सौ चौवन किताबें हैं",    "354 किताबें हैं",            "Compound 354"),
    ("एक हज़ार पाँच सौ रुपये",     "1500 रुपये",                 "1500"),
    ("पच्चीस लोग आए थे",           "25 लोग आए थे",               "25"),
    ("दो-चार बातें करो",           "दो-चार बातें करो",          "Idiom preserved"),
    ("दो तीन दिन बाद मिलना",       "दो तीन दिन बाद मिलना",      "Idiom preserved"),
    ("उसने चौदह किताबें खरीदीं",   "उसने 14 किताबें खरीदीं",    "14"),
    ("पाँच करोड़ की सम्पत्ति",     "5 करोड़ की सम्पत्ति",       "5 crore Indian format ✅"),
    ("नौ बजे आना",                  "9 बजे आना",                  "9 o'clock"),
    ("दो किलो चीनी दो",            "2 किलो चीनी दो",             "दो=2 then दो=verb"),
    ("एक बात बताओ",                 "एक बात बताओ",                "एक=article not 1"),
    ("छह सात किलोमीटर दूर है",     "6 7 किलोमीटर दूर है",        "two numbers"),
]

print("="*68)
print("NUMBER NORMALIZER v2 — TEST RESULTS")
print("="*68)
passed = 0
for inp, exp, desc in tests:
    got = normalize_numbers(inp)
    ok = got == exp
    if ok: passed += 1
    print(f"\n[{'✅' if ok else '❌'}] {desc}")
    print(f"  Input   : {inp}")
    print(f"  Expected: {exp}")
    print(f"  Got     : {got}")
print(f"\n{'='*68}")
print(f"Results: {passed}/{len(tests)} passed")
print("="*68)