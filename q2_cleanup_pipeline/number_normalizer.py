# =============================================================================
# number_normalizer.py
# Q2a — Convert Hindi number words into digits
# Josh Talks ASR Assignment
# =============================================================================

import re

# -----------------------------------------------------------------------------
# SECTION 1: Core number word dictionaries
# These are the building blocks. Every Hindi number word maps to a value.
# -----------------------------------------------------------------------------

# Basic units: 0-19 (these are "atomic" — can't be broken down further)
UNITS = {
    "शून्य": 0,
    "एक": 1,
    "दो": 2,
    "तीन": 3,
    "चार": 4,
    "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6,
    "सात": 7,
    "आठ": 8,
    "नौ": 9,
    "दस": 10,
    "ग्यारह": 11,
    "बारह": 12,
    "तेरह": 13,
    "चौदह": 14,
    "पंद्रह": 15,
    "सोलह": 16,
    "सत्रह": 17,
    "अठारह": 18,
    "उन्नीस": 19,
}

# Tens: 20, 30, 40 ... 90
TENS = {
    "बीस": 20,
    "तीस": 30,
    "चालीस": 40,
    "पचास": 50,
    "साठ": 60,
    "सत्तर": 70,
    "अस्सी": 80,
    "नब्बे": 90,
}

# Hindi has unique words for 21-99 (unlike English)
# We map all of them explicitly for accuracy
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

# Multipliers: sai (100), hazaar (1000), lakh (100000), crore (10000000)
MULTIPLIERS = {
    "सौ": 100,
    "हज़ार": 1000, "हजार": 1000,
    "लाख": 100000,
    "करोड़": 10000000,
}

# Merge all into one master lookup
ALL_NUMBER_WORDS = {}
ALL_NUMBER_WORDS.update(UNITS)
ALL_NUMBER_WORDS.update(TENS)
ALL_NUMBER_WORDS.update(COMPOUND_TENS)
ALL_NUMBER_WORDS.update(MULTIPLIERS)

# -----------------------------------------------------------------------------
# SECTION 2: Idiom / Phrase blocklist
# These phrases LOOK like numbers but should NOT be converted.
# "दो-चार" means "a few" in Hindi — converting to "2-4" changes the meaning.
# -----------------------------------------------------------------------------

IDIOM_PHRASES = [
    "दो-चार",        # a few
    "दो चार",        # a few (without hyphen)
    "चार-पाँच",      # four or five (used idiomatically for "some")
    "चार पांच",
    "पाँच-सात",      # five or seven (idiom for "several")
    "दो-तीन",        # two or three (idiom for "a couple")
    "दो तीन",
    "तीन-चार",       # three or four
    "सात-आठ",        # seven or eight
    "आठ-दस",         # eight or ten
    "एक-दो",         # one or two
    "एक दो",
    "दस-बारह",       # ten or twelve
]

# -----------------------------------------------------------------------------
# SECTION 3: Main conversion logic
# -----------------------------------------------------------------------------

def is_idiom(text: str) -> bool:
    """
    Check if a piece of text is an idiomatic number phrase.
    These should NOT be converted to digits.
    
    Example:
        "दो-चार बातें" → True (it's an idiom meaning "a few things")
        "दो सौ" → False (it literally means 200)
    """
    for idiom in IDIOM_PHRASES:
        if idiom in text:
            return True
    return False


def parse_number_sequence(words: list) -> tuple:
    """
    Given a list of Hindi words, try to parse as many consecutive
    number words as possible from the front of the list.
    
    Returns:
        (numeric_value, number_of_words_consumed)
    
    Example:
        ["तीन", "सौ", "चौवन", "किताबें"] → (354, 3)
        (consumed "तीन", "सौ", "चौवन" = 354, left "किताबें" untouched)
    """
    total = 0          # final result
    current = 0        # running partial result
    consumed = 0       # how many words we've used

    i = 0
    while i < len(words):
        word = words[i]

        # Is this word a number word at all?
        if word not in ALL_NUMBER_WORDS:
            break  # stop — non-number word found

        value = ALL_NUMBER_WORDS[word]

        if word in MULTIPLIERS:
            # Multipliers scale what came before
            # e.g., "तीन सौ" → current=3, then *100 = 300
            if word == "सौ" or word in ("हज़ार", "हजार"):
                if current == 0:
                    current = 1  # "सौ" alone means 100
                current *= value
                total += current
                current = 0
            elif word in ("लाख", "करोड़"):
                if current == 0:
                    current = 1
                current *= value
                total += current
                current = 0
        else:
            # It's a unit/tens/compound — just add to current
            current += value

        consumed = i + 1
        i += 1

    total += current  # add any remaining partial

    if consumed == 0:
        return None, 0
    return total, consumed


def normalize_numbers(text: str) -> str:
    """
    Main function: takes a Hindi sentence and converts number words to digits.
    Respects idioms and preserves non-number words.
    
    Args:
        text: raw Hindi ASR output string
    
    Returns:
        cleaned string with number words replaced by digits
    
    Example:
        "मुझे तीन सौ चौवन रुपये दो" → "मुझे 354 रुपये दो"
        "दो-चार बातें करो" → "दो-चार बातें करो"  (idiom preserved!)
    """
    # Step 1: Protect idioms by temporarily replacing them
    # We use a placeholder so they survive tokenization
    protected = text
    idiom_map = {}
    for idx, idiom in enumerate(IDIOM_PHRASES):
        if idiom in protected:
            placeholder = f"__IDIOM_{idx}__"
            idiom_map[placeholder] = idiom
            protected = protected.replace(idiom, placeholder)

    # Step 2: Tokenize on spaces
    words = protected.split()
    result_tokens = []
    i = 0

    while i < len(words):
        word = words[i]

        # Is this word a number word?
        if word in ALL_NUMBER_WORDS:
            # Try to parse a full number sequence starting here
            value, consumed = parse_number_sequence(words[i:])
            if consumed > 0:
                result_tokens.append(str(value))
                i += consumed
            else:
                result_tokens.append(word)
                i += 1
        else:
            result_tokens.append(word)
            i += 1

    result = " ".join(result_tokens)

    # Step 3: Restore idioms
    for placeholder, original in idiom_map.items():
        result = result.replace(placeholder, original)

    return result


# -----------------------------------------------------------------------------
# SECTION 4: Demo / test when you run this file directly
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    test_cases = [
        # (input, expected_output, description)
        ("मुझे दस रुपये दो",                     "मुझे 10 रुपये दो",        "Simple unit"),
        ("तीन सौ चौवन किताबें हैं",               "354 किताबें हैं",         "Compound: 354"),
        ("एक हज़ार पाँच सौ रुपये",                "1500 रुपये",              "1500"),
        ("पच्चीस लोग आए थे",                     "25 लोग आए थे",           "25"),
        ("दो-चार बातें करो",                     "दो-चार बातें करो",       "Idiom — no change"),
        ("दो तीन दिन बाद मिलना",                 "दो तीन दिन बाद मिलना",   "Idiom — no change"),
        ("उसने चौदह किताबें खरीदीं",              "उसने 14 किताबें खरीदीं", "14 books"),
        ("पाँच करोड़ की सम्पत्ति",                "5 करोड़ की सम्पत्ति",    "5 crore — multiplier"),
    ]

    print("=" * 65)
    print("NUMBER NORMALIZER — TEST RESULTS")
    print("=" * 65)

    passed = 0
    for inp, expected, desc in test_cases:
        output = normalize_numbers(inp)
        status = "✅ PASS" if output == expected else "❌ FAIL"
        if output == expected:
            passed += 1
        print(f"\n[{status}] {desc}")
        print(f"  Input   : {inp}")
        print(f"  Expected: {expected}")
        print(f"  Got     : {output}")

    print(f"\n{'='*65}")
    print(f"Results: {passed}/{len(test_cases)} passed")
    print("=" * 65)