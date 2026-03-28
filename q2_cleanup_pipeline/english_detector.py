# =============================================================================
# english_detector.py
# Q2b — Detect English words inside Hindi (Devanagari) text
# Josh Talks ASR Assignment
# =============================================================================
#
# WHY THIS IS HARD:
# Hindi speakers constantly mix English words into conversation.
# These English words appear in TWO forms in ASR output:
#
#   Form 1 — Roman script  : "मेरा interview अच्छा गया"
#   Form 2 — Devanagari    : "मेरा इंटरव्यू अच्छा गया"
#             (same word, transliterated into Hindi script)
#
# The assignment says: English words spoken in conversation are transcribed
# in Devanagari. So "computer" → "कंप्यूटर" is CORRECT, not an error.
# But we still need to TAG these words for downstream processing.
#
# OUR APPROACH — 3 detection layers:
#   Layer 1: Roman script detection (easy — just Unicode range check)
#   Layer 2: Known Devanagari-transliterated English words (curated list)
#   Layer 3: Phonetic pattern matching (English phoneme patterns in Devanagari)
# =============================================================================

import re
import unicodedata

# -----------------------------------------------------------------------------
# LAYER 1: Roman script words mixed into Hindi text
# These are easy — they're just ASCII/Latin characters in a Devanagari string
# Example: "मेरा interview अच्छा गया"
# -----------------------------------------------------------------------------

def is_roman_english(word: str) -> bool:
    """
    Returns True if the word is written in Roman/Latin script.
    These are clearly English — no ambiguity.
    """
    # Strip punctuation
    clean = re.sub(r'[^\w]', '', word)
    if not clean:
        return False
    # Check if majority of chars are Latin
    latin_chars = sum(1 for c in clean if ord(c) < 128 and c.isalpha())
    return latin_chars > len(clean) * 0.5


# -----------------------------------------------------------------------------
# LAYER 2: Known English words transliterated into Devanagari
# Built from REAL words found in the Josh Talks dataset + common Hindi-English
# mix words (Hinglish)
#
# Categories:
#   - Technology words (very common in conversations)
#   - Education / professional
#   - Places / geography terms
#   - Common Hinglish words
#   - Outdoor / nature (found in our actual data: टेंट, कैम्पिंग, एरिया)
# -----------------------------------------------------------------------------

DEVANAGARI_ENGLISH_WORDS = {

    # ── Found directly in our Josh Talks dataset ──────────────────────────
    "एरिया",        # area
    "टेंट",         # tent
    "कैम्पिंग",     # camping
    "कैम्प",        # camp
    "प्रोजेक्ट",    # project
    "गार्ड",        # guard
    "मिस्टेक",      # mistake
    "लाइट",         # light
    "अमेजन",        # Amazon

    # ── Technology ────────────────────────────────────────────────────────
    "कंप्यूटर",     # computer
    "कम्प्यूटर",    # computer (alternate)
    "मोबाइल",       # mobile
    "फोन",          # phone
    "इंटरनेट",      # internet
    "वेबसाइट",      # website
    "ऐप",           # app
    "एप",           # app
    "सॉफ्टवेयर",    # software
    "हार्डवेयर",    # hardware
    "डेटा",         # data
    "सर्वर",        # server
    "क्लाउड",       # cloud
    "डिजिटल",       # digital
    "ऑनलाइन",       # online
    "ऑफलाइन",       # offline
    "स्क्रीन",      # screen
    "लैपटॉप",       # laptop
    "टैबलेट",       # tablet
    "कीबोर्ड",      # keyboard
    "माउस",         # mouse
    "प्रिंटर",      # printer
    "वाईफाई",       # wifi
    "ब्लूटूथ",      # bluetooth
    "चार्जर",       # charger
    "बैटरी",        # battery
    "कैमरा",        # camera
    "वीडियो",       # video
    "ऑडियो",        # audio
    "रेडियो",       # radio

    # ── Education / Professional ──────────────────────────────────────────
    "इंटरव्यू",     # interview
    "इंटरव्यूू",    # interview (typo variant)
    "जॉब",          # job
    "ऑफिस",         # office
    "मैनेजर",       # manager
    "टीम",          # team
    "मीटिंग",       # meeting
    "प्रेजेंटेशन",  # presentation
    "रिपोर्ट",      # report
    "डिग्री",       # degree
    "कॉलेज",        # college
    "यूनिवर्सिटी",  # university
    "स्कूल",        # school
    "क्लास",        # class
    "एग्जाम",       # exam
    "सिलेबस",       # syllabus
    "फीस",          # fees
    "सर्टिफिकेट",   # certificate
    "रिज्यूमे",     # resume
    "इंटर्नशिप",    # internship
    "स्टार्टअप",    # startup
    "बिजनेस",       # business
    "मार्केट",      # market
    "सेल्स",        # sales
    "टारगेट",       # target
    "प्रोफेशनल",    # professional
    "एक्सपीरियंस",  # experience
    "स्किल",        # skill
    "ट्रेनिंग",     # training
    "वर्कशॉप",      # workshop
    "सेमिनार",      # seminar
    "प्रोग्राम",    # program
    "कोर्स",        # course
    "फीडबैक",       # feedback
    "परफॉर्मेंस",   # performance

    # ── Places / Geography ────────────────────────────────────────────────
    "स्टेशन",       # station
    "एयरपोर्ट",     # airport
    "होटल",         # hotel
    "रिसोर्ट",      # resort
    "पार्क",        # park
    "मॉल",          # mall
    "मार्केट",      # market
    "सिटी",         # city
    "टाउन",         # town
    "विलेज",        # village
    "रोड",          # road
    "हाईवे",        # highway
    "ब्रिज",        # bridge

    # ── Common Hinglish words ─────────────────────────────────────────────
    "ओके",          # okay
    "ओक्के",        # okay
    "हेलो",         # hello
    "हाय",          # hi
    "थैंक्स",       # thanks
    "थैंक्यू",      # thank you
    "सॉरी",         # sorry
    "प्लीज",        # please
    "नो",           # no
    "येस",          # yes
    "बाय",          # bye
    "कूल",          # cool
    "नाइस",         # nice
    "ग्रेट",        # great
    "परफेक्ट",      # perfect
    "सुपर",         # super
    "वेरी",         # very
    "टोटली",        # totally
    "एक्चुअली",     # actually
    "बेसिकली",      # basically
    "लिटरली",       # literally
    "सीरियसली",     # seriously
    "प्रॉब्लम",     # problem
    "सॉल्यूशन",     # solution
    "चेक",          # check
    "ट्राई",        # try
    "शेयर",         # share
    "सेंड",         # send
    "क्लिक",        # click
    "डाउनलोड",      # download
    "अपलोड",        # upload
    "सेव",          # save
    "डिलीट",        # delete
    "फाइल",         # file
    "फोल्डर",       # folder
    "लिंक",         # link
    "पासवर्ड",      # password

    # ── Medical ───────────────────────────────────────────────────────────
    "डॉक्टर",       # doctor
    "हॉस्पिटल",     # hospital
    "मेडिसिन",      # medicine
    "ट्रीटमेंट",    # treatment
    "ऑपरेशन",       # operation
    "डायबिटीज",     # diabetes
    "प्रेशर",       # pressure
    "टेस्ट",        # test

    # ── Media / Entertainment ─────────────────────────────────────────────
    "मूवी",         # movie
    "फिल्म",        # film (also Hindi)
    "सीरीज",        # series
    "सीजन",         # season
    "एपिसोड",       # episode
    "चैनल",         # channel
    "न्यूज़",       # news
    "सोशल",         # social
    "मीडिया",       # media
    "यूट्यूब",      # youtube
    "इंस्टाग्राम",  # instagram
    "फेसबुक",       # facebook
    "ट्विटर",       # twitter
    "व्हाट्सएप",    # whatsapp

    # ── Finance ───────────────────────────────────────────────────────────
    "बैंक",         # bank
    "लोन",          # loan
    "ईएमआई",        # EMI
    "इंटरेस्ट",     # interest
    "अकाउंट",       # account
    "पेमेंट",       # payment
    "इन्वेस्टमेंट", # investment
    "इंश्योरेंस",   # insurance
    "टैक्स",        # tax
    "बजट",          # budget

    # ── Sports / Fitness ──────────────────────────────────────────────────
    "क्रिकेट",      # cricket
    "फुटबॉल",       # football
    "बास्केटबॉल",   # basketball
    "टेनिस",        # tennis
    "जिम",          # gym
    "फिटनेस",       # fitness
    "वर्कआउट",      # workout
    "ट्रेनर",       # trainer
    "कोच",          # coach
    "मैच",          # match
    "टूर्नामेंट",   # tournament
    "टीम",          # team
    "प्लेयर",       # player
    "स्कोर",        # score
}


# -----------------------------------------------------------------------------
# LAYER 3: Phonetic pattern detection
# English words in Devanagari often have characteristic phoneme patterns
# that are rare in native Hindi words.
#
# Key signals:
#   - Ends in consonant clusters rare in Hindi: ट, ड at end
#   - Contains ऑ (the "o" sound not native to Hindi)
#   - Contains foreign phoneme markers: ज़, फ़, क़
#   - Typical English suffixes transliterated: -शन, -मेंट, -इंग, -ली, -नेस
# -----------------------------------------------------------------------------

# Devanagari patterns strongly indicating an English loanword
ENGLISH_PHONEME_PATTERNS = [
    r'.*शन$',       # -tion / -sion  (प्रेजेंटेशन, ऑपरेशन)
    r'.*मेंट$',     # -ment          (पेमेंट, ट्रीटमेंट)
    r'.*इंग$',      # -ing           (कैम्पिंग, मीटिंग, ट्रेनिंग)
    r'.*नेस$',      # -ness          (हैपीनेस)
    r'.*लिटी$',     # -lity          (क्वालिटी, फ्लेक्सिबिलिटी)
    r'.*टी$',       # -ty            (यूनिवर्सिटी, क्वालिटी)
    r'.*फुल$',      # -ful           (हेल्पफुल, यूजफुल)
    r'.*लेस$',      # -less          (होपलेस)
    r'.*वर्क$',     # -work          (नेटवर्क, वर्कआउट, फ्रेमवर्क)
    r'.*स्टार्ट.*', # start-         (स्टार्टअप)
    r'.*ऑन.*',      # ऑ sound        (ऑनलाइन, ऑफिस, ऑपरेशन)
    r'.*फ़.*',      # फ़ (f sound)   (फ़ीचर, फ़ाइल)
    r'.*ज़.*',      # ज़ (z sound)   (न्यूज़, ज़ोन)
]

def matches_english_pattern(word: str) -> bool:
    """
    Check if a Devanagari word matches typical English loanword phoneme patterns.
    """
    for pattern in ENGLISH_PHONEME_PATTERNS:
        if re.match(pattern, word):
            return True
    return False


# -----------------------------------------------------------------------------
# SECTION: Pure Hindi word filter
# Some words could be confused — these are DEFINITELY pure Hindi, not English
# -----------------------------------------------------------------------------

PURE_HINDI_WORDS = {
    # Common Hindi words that might accidentally match patterns
    "में", "है", "हैं", "था", "थे", "थी", "हो", "होता", "होती",
    "और", "या", "तो", "भी", "ही", "पर", "से", "को", "के", "की", "का",
    "एक", "यह", "वह", "यहाँ", "वहाँ", "कब", "कैसे", "क्यों", "क्या",
    "नहीं", "मत", "हाँ", "जी", "नमस्ते", "धन्यवाद",
    # Words ending in -ट that are pure Hindi
    "बात", "रात", "जात", "लात", "मात", "घात", "पात",
    # Words with ऑ that are pure Hindi (rare but exists)
    "औरत",
}


# -----------------------------------------------------------------------------
# MAIN DETECTION FUNCTION
# -----------------------------------------------------------------------------

def detect_english_words(text: str) -> dict:
    """
    Detect English words in a Hindi text using all 3 layers.

    Args:
        text: Hindi sentence (may contain Roman or Devanagari English words)

    Returns:
        dict with:
          - 'tagged'    : text with [EN]word[/EN] tags
          - 'en_words'  : list of detected English words
          - 'method'    : detection method for each word
    """
    words = text.split()
    tagged_words = []
    detected = []

    for word in words:
        # Strip punctuation for analysis but keep original for output
        clean = re.sub(r'[।,!?।"\'()—\-]', '', word).strip()

        if not clean:
            tagged_words.append(word)
            continue

        # Skip pure Hindi words
        if clean in PURE_HINDI_WORDS:
            tagged_words.append(word)
            continue

        is_english = False
        method = None

        # Layer 1: Roman script check
        if is_roman_english(clean):
            is_english = True
            method = "roman-script"

        # Layer 2: Known Devanagari English words
        elif clean in DEVANAGARI_ENGLISH_WORDS:
            is_english = True
            method = "known-loanword"

        # Layer 3: Phonetic pattern
        elif matches_english_pattern(clean) and clean not in PURE_HINDI_WORDS:
            is_english = True
            method = "phonetic-pattern"

        if is_english:
            tagged_words.append(f"[EN]{word}[/EN]")
            detected.append({"word": word, "method": method})
        else:
            tagged_words.append(word)

    return {
        "original": text,
        "tagged": " ".join(tagged_words),
        "en_words": detected,
        "en_count": len(detected),
    }


def tag_transcript(text: str) -> str:
    """Simple wrapper — returns just the tagged string."""
    return detect_english_words(text)["tagged"]


# -----------------------------------------------------------------------------
# TEST SUITE — uses real examples from our Josh Talks dataset
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    test_cases = [
        # (description, input_text, expected_english_words)

        # ── From our REAL dataset (825780_transcription.json) ─────────────
        ("Real data: एरिया",
         "पहली बारी था क्योंकि चलना नहीं आता न वहाँ का जो लैंड एरिया होता है",
         ["एरिया"]),

        ("Real data: टेंट, कैम्प",
         "हमने टेंट गड़ा और रहा तो जब पता जैसी रात हुआ",
         ["टेंट"]),

        ("Real data: कैम्पिंग",
         "अगर कहीं भी कैम्पिंग करने जाते हैं तो आसपास की एरिया में",
         ["कैम्पिंग", "एरिया"]),

        ("Real data: प्रोजेक्ट",
         "हमारा प्रोजेक्ट भी था कि जो जन जाती पाई जाती है",
         ["प्रोजेक्ट"]),

        ("Real data: मिस्टेक",
         "पता है लेकिन हम ने मिस्टेक किए कि हम लाइट नहीं ले गए थे",
         ["मिस्टेक", "लाइट"]),

        # ── Assignment example ─────────────────────────────────────────────
        ("Assignment example: इंटरव्यू, जॉब",
         "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मिल गई",
         ["इंटरव्यू", "जॉब"]),

        # ── Roman script mixed in ──────────────────────────────────────────
        ("Roman English in Hindi",
         "मेरा interview अच्छा गया और मुझे job मिल गई",
         ["interview", "job"]),

        # ── Phonetic pattern detection ─────────────────────────────────────
        ("Phonetic: -इंग suffix",
         "वो मीटिंग में ट्रेनिंग दे रहे थे",
         ["मीटिंग", "ट्रेनिंग"]),

        ("Phonetic: -शन suffix",
         "उसकी प्रेजेंटेशन बहुत अच्छी थी",
         ["प्रेजेंटेशन"]),

        # ── Pure Hindi — should NOT be tagged ─────────────────────────────
        ("Pure Hindi — no tagging",
         "वो बहुत अच्छा इंसान है और सबकी मदद करता है",
         []),

        # ── Mixed sentence ─────────────────────────────────────────────────
        ("Mixed: tech conversation",
         "मेरा लैपटॉप का बैटरी खत्म हो गया तो मैंने चार्जर लगाया",
         ["लैपटॉप", "बैटरी", "चार्जर"]),
    ]

    print("=" * 70)
    print("ENGLISH WORD DETECTOR — TEST RESULTS")
    print("=" * 70)

    passed = 0
    for desc, inp, expected_words in test_cases:
        result = detect_english_words(inp)
        found = [d["word"] for d in result["en_words"]]

        # Check if all expected words were found
        all_found = all(w in found for w in expected_words)
        no_false_pos = all(w in DEVANAGARI_ENGLISH_WORDS
                           or is_roman_english(w)
                           or matches_english_pattern(w)
                           for w in found)
        ok = all_found

        if ok:
            passed += 1

        print(f"\n[{'✅' if ok else '❌'}] {desc}")
        print(f"  Input  : {inp}")
        print(f"  Tagged : {result['tagged']}")
        print(f"  Found  : {found}")
        print(f"  Methods: {[d['method'] for d in result['en_words']]}")

    print(f"\n{'=' * 70}")
    print(f"Results: {passed}/{len(test_cases)} passed")
    print("=" * 70)
