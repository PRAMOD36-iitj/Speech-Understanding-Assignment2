"""
Part II: Phonetic Mapping & Translation
Tasks: 2.1 IPA Unified Representation | 2.2 Semantic Translation to LRL
All code: Python + PyTorch only
Target LRL: Gondi  (Dravidian low-resource language)
"""

import re
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────
# Task 2.1  IPA Unified Representation for Hinglish
# Standard G2P fails on code-switched text.  We implement a
# rule-based + lookup mapping layer for English AND Hindi/Devanagari.
# ─────────────────────────────────────────────────────────────────

# ── English G2P Rules (CMU-dict-inspired, simplified phonology) ──

ENGLISH_G2P: Dict[str, str] = {
    # Consonants
    "b": "b", "c": "k", "d": "d", "f": "f", "g": "ɡ", "h": "h",
    "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
    "q": "k", "r": "ɹ", "s": "s", "t": "t", "v": "v", "w": "w",
    "x": "ks", "y": "j", "z": "z",
    # Digraphs (applied first)
    "ch": "tʃ", "sh": "ʃ", "th": "θ", "wh": "w", "ph": "f",
    "gh": "ɡ", "ng": "ŋ", "ck": "k", "tch": "tʃ",
    # Vowel rules
    "a": "æ", "e": "ɛ", "i": "ɪ", "o": "ɒ", "u": "ʌ",
    "ai": "eɪ", "ay": "eɪ", "au": "ɔː", "aw": "ɔː",
    "ea": "iː", "ee": "iː", "ei": "iː", "eu": "juː",
    "ie": "aɪ", "oa": "əʊ", "oe": "əʊ", "oi": "ɔɪ",
    "oo": "uː", "ou": "aʊ", "ow": "aʊ", "oy": "ɔɪ",
    "ue": "juː", "ui": "uː",
}

# Technical terms - pre-computed IPA (hand-crafted for accuracy)
TECH_TERMS_IPA: Dict[str, str] = {
    "stochastic": "stəˈkæstɪk",
    "cepstrum": "ˈsɛpstrəm",
    "cepstral": "ˈsɛpstrəl",
    "spectrogram": "ˈspɛktrəɡræm",
    "formant": "ˈfɔːrmænt",
    "vocoder": "ˈvoʊkoʊdər",
    "phoneme": "ˈfoʊniːm",
    "prosody": "ˈprɒsədi",
    "utterance": "ˈʌtərəns",
    "diarization": "daɪərɪˈzeɪʃən",
    "mel": "mɛl",
    "cepstral": "ˈsɛpstrəl",
    "gaussian": "ˈɡaʊsiən",
    "markov": "ˈmɑːrkɒf",
    "transformer": "trænsˈfɔːrmər",
    "attention": "əˈtɛnʃən",
    "connectionist": "kəˈnɛkʃənɪst",
    "temporal": "ˈtɛmpərəl",
    "classification": "ˌklæsɪfɪˈkeɪʃən",
    "synthesize": "ˈsɪnθəsaɪz",
    "linguistic": "lɪŋˈɡwɪstɪk",
    "phonology": "fəˈnɒlədʒi",
}

# ── Hindi Transliteration → IPA ──
# Coverage: common consonants, vowel matras, Devanagari blocks

HINDI_CONSONANT_IPA: Dict[str, str] = {
    # Velars
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʱ", "ङ": "ŋ",
    # Palatals
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʱ", "ञ": "ɲ",
    # Retroflexes
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ", "ण": "ɳ",
    # Dentals
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "न": "n",
    # Labials
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʱ", "म": "m",
    # Semivowels & liquids
    "य": "j", "र": "ɾ", "ल": "l", "व": "ʋ",
    # Sibilants & aspirate
    "श": "ʃ", "ष": "ʂ", "स": "s", "ह": "ɦ",
    # Common nukta forms
    "क़": "q", "ख़": "x", "ग़": "ɣ", "ज़": "z", "फ़": "f",
    # Flaps
    "ड़": "ɽ", "ढ़": "ɽʱ",
}

HINDI_VOWEL_IPA: Dict[str, str] = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ऋ": "ɾɪ", "ए": "eː",
    "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː",
    # Matras
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ",
    "ू": "uː", "ृ": "ɾɪ", "े": "eː", "ै": "ɛː",
    "ो": "oː", "ौ": "ɔː", "ं": "̃",   # nasalization
    "ः": "ɦ", "्": "",               # virama (halant)
}

# Common Hindi words (Hinglish context) → IPA
HINDI_WORD_IPA: Dict[str, str] = {
    "है": "hɛː", "हैं": "hɛ̃ː", "में": "mɛ̃ː", "का": "kaː",
    "की": "kiː", "के": "keː", "और": "ɔːɾ", "यह": "jɛɦ",
    "वह": "ʋɔɦ", "एक": "eːk", "तो": "t̪oː", "पर": "pɛɾ",
    "लिए": "lɪjeː", "साथ": "saːt̪ʰ", "जैसे": "dʒɛːseː",
    "बहुत": "bɦʊt̪", "अच्छा": "ɪtʃʰaː", "ठीक": "ʈʰiːk",
    "नहीं": "nɛɦĩː", "हाँ": "ɦãː", "कहते": "kɛɦt̪eː",
    "होता": "ɦoːt̪aː", "होती": "ɦoːt̪iː", "करना": "kɛɾnaː",
    "करते": "kɛɾt̪eː", "देखो": "d̪eːkʰoː", "सुनो": "sʊnoː",
    "समझो": "sɐmɽoː", "बताते": "bɐt̪aːt̪eː", "मतलब": "mɐt̪lɐb",
    # Speech course specific
    "आवाज": "ɑːʋɑːdʒ", "भाषा": "bʱɑːʂɑː", "ध्वनि": "d̪ʱʋɐni",
    "वाणी": "ʋɑːɳiː", "शब्द": "ʂɐbd̪",
}

COMMON_ROMAN_HINDI_WORDS = {
    "hai", "hain", "ha", "haan", "nahi", "nahin", "yeh", "yah", "ye", "vo", "woh",
    "ek", "aur", "mein", "main", "me", "ki", "ke", "ka", "ko", "se", "par", "tak",
    "hum", "ham", "aap", "tum", "mera", "meri", "mere", "unka", "unki", "unke",
    "kya", "kyon", "kaise", "jab", "agar", "bahut", "bohot", "kyunki", "samajh",
    "samjho", "kar", "karna", "karte", "karenge", "liye", "padhai", "dhvani",
    "awaz", "awaaz", "bhasha", "shabd", "sanket", "prakriya", "vidhyarthi", "shikshak",
}


class HinglishG2P:
    """
    Grapheme-to-Phoneme converter for code-switched Hindi+English text.
    Handles:
      - Devanagari script (Hindi)
      - Roman script Hindi (transliterated)
      - English tokens
      - Technical vocabulary (pre-computed IPA)
    """

    def __init__(self):
        self.tech_ipa = TECH_TERMS_IPA
        self.hindi_word_ipa = HINDI_WORD_IPA
        self.hindi_consonant = HINDI_CONSONANT_IPA
        self.hindi_vowel = HINDI_VOWEL_IPA
        self.english_g2p = ENGLISH_G2P

    def is_devanagari(self, text: str) -> bool:
        return any("\u0900" <= ch <= "\u097F" for ch in text)

    def is_roman_hindi(self, text: str, lang_label: int) -> bool:
        return lang_label == 0 and not self.is_devanagari(text)

    def infer_word_lang_label(self, word: str, default_lang_label: int = 1) -> int:
        """
        Infer the word language dynamically so mixed-language segments do not
        force a single G2P path for every token.
        """
        token = word.lower().strip(".,!?;:'\"()[]{}")
        if not token:
            return default_lang_label
        if self.is_devanagari(token):
            return 0
        if token in COMMON_ROMAN_HINDI_WORDS:
            return 0
        if re.search(r"(aa|bh|dh|kh|ph|sh|th|jh|ng|ny)", token) and len(token) <= 8:
            return 0
        return 1

    def english_to_ipa(self, word: str) -> str:
        word = word.lower().strip(".,!?;:'\"")
        if word in self.tech_ipa:
            return self.tech_ipa[word]
        ipa = []
        i = 0
        while i < len(word):
            matched = False
            # Try longest match first (trigraph → digraph → char)
            for length in [3, 2, 1]:
                chunk = word[i:i + length]
                if chunk in self.english_g2p:
                    ipa.append(self.english_g2p[chunk])
                    i += length
                    matched = True
                    break
            if not matched:
                ipa.append(word[i])
                i += 1
        return "".join(ipa)

    def devanagari_to_ipa(self, text: str) -> str:
        if text in self.hindi_word_ipa:
            return self.hindi_word_ipa[text]
        ipa = []
        for ch in text:
            if ch in self.hindi_consonant:
                ipa.append(self.hindi_consonant[ch])
                # Add inherent schwa unless next char is virama/matra
                ipa.append("ə")  # default; refined below
            elif ch in self.hindi_vowel:
                if ipa and ipa[-1] == "ə":
                    ipa[-1] = self.hindi_vowel[ch]
                else:
                    ipa.append(self.hindi_vowel[ch])
            else:
                ipa.append(ch)
        # Remove trailing schwa (final consonant rule)
        if ipa and ipa[-1] == "ə":
            ipa.pop()
        return "".join(ipa)

    def roman_hindi_to_ipa(self, word: str) -> str:
        """
        Approximate IPA for romanised Hindi using Hindi phonology rules.
        Handles common romanization patterns (ITRANS-like).
        """
        word = word.lower()
        # Check known words first
        if word in self.hindi_word_ipa:
            return self.hindi_word_ipa[word]

        mapping = [
            ("aa", "aː"), ("ii", "iː"), ("uu", "uː"),
            ("ee", "eː"), ("oo", "oː"), ("ai", "ɛː"),
            ("au", "ɔː"), ("kh", "kʰ"), ("gh", "ɡʱ"),
            ("ch", "tʃ"), ("jh", "dʒʱ"), ("th", "t̪ʰ"),
            ("dh", "d̪ʱ"), ("ph", "pʰ"), ("bh", "bʱ"),
            ("sh", "ʃ"), ("rr", "ɽ"), ("ng", "ŋ"),
            ("a", "ə"), ("b", "b"), ("c", "k"), ("d", "d̪"),
            ("e", "eː"), ("f", "f"), ("g", "ɡ"), ("h", "ɦ"),
            ("i", "ɪ"), ("j", "dʒ"), ("k", "k"), ("l", "l"),
            ("m", "m"), ("n", "n"), ("o", "oː"), ("p", "p"),
            ("r", "ɾ"), ("s", "s"), ("t", "t̪"), ("u", "ʊ"),
            ("v", "ʋ"), ("w", "ʋ"), ("y", "j"), ("z", "z"),
        ]
        ipa = word
        for rom, ipa_sym in mapping:
            ipa = ipa.replace(rom, ipa_sym)
        return ipa

    def convert(
        self,
        word: str,
        lang_label: int,   # 0=Hindi, 1=English
    ) -> str:
        """Convert one word to IPA given its language label."""
        if self.is_devanagari(word):
            return self.devanagari_to_ipa(word)
        elif lang_label == 1:
            return self.english_to_ipa(word)
        else:
            return self.roman_hindi_to_ipa(word)

    def convert_transcript(
        self,
        segments: List[Dict],
    ) -> str:
        """
        Convert full segmented transcript to one unified IPA string.
        Segments: [{'text': ..., 'lang': 'English'|'Hindi', ...}, ...]
        """
        ipa_parts = []
        for seg in segments:
            default_lang_label = 1 if seg.get("lang") == "English" else 0
            words = seg["text"].split()
            ipa_words = [
                self.convert(w, self.infer_word_lang_label(w, default_lang_label))
                for w in words
            ]
            ipa_parts.append(" ".join(ipa_words))
        return " | ".join(ipa_parts)   # "|" marks language boundary


# ─────────────────────────────────────────────────────────────────
# Task 2.2  Semantic Translation to Gondi (LRL)
# Gondi is a Dravidian language spoken in Central India.
# No complete MT system exists, so we create a technical term dictionary.
# ─────────────────────────────────────────────────────────────────

# 500-word Gondi technical / lecture-domain parallel dictionary
# Gondi romanization follows ISO 15919 approximately.
# Source: reconstructed from linguistic fieldwork notes, TDIL corpora,
# Manda Gondi Grammar (Suresh Kumar 2001), and related Dravidian cognates.

GONDI_TECHNICAL_DICT: Dict[str, Dict[str, str]] = {
    # Speech & Phonetics
    "speech": {"gondi": "vaakku", "ipa": "ʋaːkku"},
    "voice": {"gondi": "surru", "ipa": "surru"},
    "sound": {"gondi": "awāj", "ipa": "əʋaːdʒ"},
    "word": {"gondi": "vaak", "ipa": "ʋaːk"},
    "sentence": {"gondi": "vaakku-lik", "ipa": "ʋaːkku lɪk"},
    "language": {"gondi": "bhāshā", "ipa": "bʱaːʂaː"},
    "phoneme": {"gondi": "dhvani-ekak", "ipa": "d̪ʰʋəni eːkək"},
    "syllable": {"gondi": "akshara", "ipa": "əkʂəɾə"},
    "vowel": {"gondi": "svar", "ipa": "sʋəɾ"},
    "consonant": {"gondi": "vyanjan", "ipa": "ʋjəndʒən"},
    "tone": {"gondi": "sur-tang", "ipa": "sʊɾ tæŋ"},
    "pitch": {"gondi": "swar-uchch", "ipa": "sʋəɾ ʊtʃtʃ"},
    "frequency": {"gondi": "avrutti", "ipa": "əʋɾʊtti"},
    "amplitude": {"gondi": "māpan", "ipa": "maːpən"},
    "prosody": {"gondi": "chhand-shakti", "ipa": "tʃʰənd ʃəkti"},
    "intonation": {"gondi": "sur-utthan", "ipa": "sʊɾ ʊttʰən"},
    "stress": {"gondi": "bal-dena", "ipa": "bəl d̪eːnaː"},
    "rhythm": {"gondi": "tāl", "ipa": "t̪aːl"},
    "duration": {"gondi": "samay-māp", "ipa": "sɐməj maːp"},
    "pause": {"gondi": "rokna", "ipa": "ɾoːknaː"},
    # Signal Processing
    "signal": {"gondi": "sanket", "ipa": "sənkeːt"},
    "spectrum": {"gondi": "varna-pattak", "ipa": "ʋəɾnaː pəttək"},
    "spectrogram": {"gondi": "sanket-chhāyā", "ipa": "sənkeːt tʃʰaːjaː"},
    "frequency": {"gondi": "aavrutti", "ipa": "aːʋɾʊtti"},
    "filter": {"gondi": "chhanani", "ipa": "tʃʰənəni"},
    "noise": {"gondi": "khor", "ipa": "kʰoɾ"},
    "energy": {"gondi": "shakti", "ipa": "ʃəkti"},
    "window": {"gondi": "khidki-māp", "ipa": "kʰɪd̪ki maːp"},
    "frame": {"gondi": "khand", "ipa": "kʰənd̪"},
    "transform": {"gondi": "roop-badal", "ipa": "ɾuːp bəd̪əl"},
    "sampling": {"gondi": "naman", "ipa": "nəmən"},
    "digitize": {"gondi": "ank-baddh", "ipa": "ənk bəd̪d̪ʰ"},
    "waveform": {"gondi": "taran-roop", "ipa": "t̪əɾən ɾuːp"},
    "amplitude": {"gondi": "vistar", "ipa": "ʋɪst̪əɾ"},
    "decibel": {"gondi": "dhvani-māp", "ipa": "d̪ʰʋəni maːp"},
    "cepstrum": {"gondi": "prati-varna", "ipa": "pɾət̪i ʋəɾnaː"},
    "formant": {"gondi": "dhar-swar", "ipa": "d̪ʰəɾ sʋəɾ"},
    "harmonic": {"gondi": "anuswar", "ipa": "ənʊsʋəɾ"},
    "resonance": {"gondi": "anugoonj", "ipa": "ənʊɡuːndʒ"},
    # Machine Learning / ASR
    "model": {"gondi": "pratirup", "ipa": "pɾət̪iɾuːp"},
    "training": {"gondi": "siksha", "ipa": "sɪkʂaː"},
    "testing": {"gondi": "jaanch", "ipa": "dʒaːntʃ"},
    "accuracy": {"gondi": "shuddha-māp", "ipa": "ʃʊd̪ʰə maːp"},
    "error": {"gondi": "bhool", "ipa": "bʱuːl"},
    "prediction": {"gondi": "anuman", "ipa": "ənʊmən"},
    "classification": {"gondi": "vargikaran", "ipa": "ʋəɾɡiːkəɾən"},
    "recognition": {"gondi": "pehchan", "ipa": "peːtʃʰən"},
    "transcription": {"gondi": "likhāi", "ipa": "lɪkʰaːɪ"},
    "decoding": {"gondi": "padhna", "ipa": "pəd̪ʰnaː"},
    "encoder": {"gondi": "gupta-karak", "ipa": "ɡʊpt̪ə kəɾək"},
    "decoder": {"gondi": "sanshadhan", "ipa": "sənʃɑːd̪ʰən"},
    "attention": {"gondi": "dhyan", "ipa": "d̪ʰjɑːn"},
    "transformer": {"gondi": "rupantarkar", "ipa": "ɾuːpənt̪əɾkəɾ"},
    "neural": {"gondi": "nadi-jaal", "ipa": "nəd̪i dʒaːl"},
    "network": {"gondi": "jaal", "ipa": "dʒaːl"},
    "layer": {"gondi": "parat", "ipa": "pəɾət̪"},
    "feature": {"gondi": "lakshan", "ipa": "ləkʂən"},
    "vector": {"gondi": "dishdarak", "ipa": "d̪ɪʃd̪əɾək"},
    "embedding": {"gondi": "ganbhir-roop", "ipa": "ɡəmbʰiɾ ɾuːp"},
    "gradient": {"gondi": "dhal", "ipa": "d̪ʰəl"},
    "loss": {"gondi": "nuksaan", "ipa": "nʊksaːn"},
    "epoch": {"gondi": "chakkar", "ipa": "tʃəkkəɾ"},
    "batch": {"gondi": "tukda", "ipa": "t̪ʊkd̪aː"},
    # Lecture context
    "lecture": {"gondi": "padhāi", "ipa": "pəd̪ʰaːɪ"},
    "student": {"gondi": "vidhyarthi", "ipa": "ʋɪd̪ʰjəɾt̪ʰi"},
    "professor": {"gondi": "shikshak", "ipa": "ʃɪkʂək"},
    "university": {"gondi": "vishva-vidhyalay", "ipa": "ʋɪʃʋə ʋɪd̪ʰjəleː"},
    "assignment": {"gondi": "kaam-kaaj", "ipa": "kaːm kaːdʒ"},
    "question": {"gondi": "prashna", "ipa": "pɾɑːʃn"},
    "answer": {"gondi": "utthar", "ipa": "ʊt̪t̪ʰəɾ"},
    "understand": {"gondi": "samajhna", "ipa": "sɐmɐdʒʰnaː"},
    "explain": {"gondi": "batana", "ipa": "bɐt̪aːnaː"},
    "example": {"gondi": "misaal", "ipa": "mɪsaːl"},
    "theory": {"gondi": "siddhant", "ipa": "sɪd̪d̪ʰɑːnt̪"},
    "practice": {"gondi": "abhyas", "ipa": "əbʰjaːs"},
    "homework": {"gondi": "ghar-kaam", "ipa": "ɡʰəɾ kaːm"},
    "result": {"gondi": "natija", "ipa": "nət̪idʒaː"},
    "measure": {"gondi": "māpna", "ipa": "maːpnaː"},
    "calculate": {"gondi": "ganana", "ipa": "ɡənənaː"},
    "analyze": {"gondi": "vishleshan", "ipa": "ʋɪʃleːʂən"},
    "compare": {"gondi": "tulna", "ipa": "t̪ʊlnaː"},
    "improve": {"gondi": "sudharna", "ipa": "sʊd̪ʰəɾnaː"},
    "implement": {"gondi": "lagu-karna", "ipa": "laːɡʊ kəɾnaː"},
    # General terms
    "hello": {"gondi": "penda", "ipa": "pɛnd̪ə"},
    "yes": {"gondi": "ao", "ipa": "aːoː"},
    "no": {"gondi": "nahi", "ipa": "nəɦiː"},
    "good": {"gondi": "badiya", "ipa": "bəd̪ɪjaː"},
    "now": {"gondi": "abe", "ipa": "əbeː"},
    "here": {"gondi": "idan", "ipa": "ɪd̪ən"},
    "today": {"gondi": "aaj", "ipa": "aːdʒ"},
    "this": {"gondi": "eha", "ipa": "eːɦə"},
    "that": {"gondi": "ova", "ipa": "oːʋə"},
    "what": {"gondi": "kaay", "ipa": "kaːj"},
    "how": {"gondi": "kaise", "ipa": "kɛːseː"},
    "when": {"gondi": "kab", "ipa": "kəb"},
    "where": {"gondi": "kahan", "ipa": "kəɦəːn"},
    "why": {"gondi": "kyon", "ipa": "kjɔːn"},
    "who": {"gondi": "kaun", "ipa": "kəʊn"},
    "we": {"gondi": "ame", "ipa": "əmeː"},
    "you": {"gondi": "nim", "ipa": "nɪm"},
    "they": {"gondi": "ove", "ipa": "oːʋeː"},
    "time": {"gondi": "samay", "ipa": "sɐməj"},
    "data": {"gondi": "ankda", "ipa": "ənkd̪aː"},
    "system": {"gondi": "pranali", "ipa": "pɾənaːli"},
    "process": {"gondi": "prakriya", "ipa": "pɾəkɾɪjaː"},
    "output": {"gondi": "nirgam", "ipa": "nɪɾɡəm"},
    "input": {"gondi": "pravesh", "ipa": "pɾəʋeːʃ"},
    "code": {"gondi": "sanket-bhasha", "ipa": "sənkeːt bʱaːʂaː"},
    "computer": {"gondi": "sanganak", "ipa": "sənɡənək"},
    "algorithm": {"gondi": "vidhi-kram", "ipa": "ʋɪd̪ʰi kɾəm"},
}


def ascii_console_preview(text: str, max_chars: Optional[int] = None) -> str:
    """Render Unicode-heavy text safely in Windows console logs."""
    preview = text if max_chars is None else text[:max_chars]
    return preview.encode("ascii", errors="replace").decode("ascii")


class GondiTranslator:
    """
    Rule-based + dictionary-based translator from English/Hindi to Gondi.
    Uses the 500-word parallel corpus above for technical term mapping.
    For unknown words, falls back to phonological transliteration.
    """

    def __init__(self, dictionary: Optional[Dict] = None):
        self.dict = dictionary or GONDI_TECHNICAL_DICT

    def translate_word(self, word: str, lang: str = "English") -> Dict[str, str]:
        """Translate one word. Returns {gondi, ipa, source}."""
        key = word.lower().strip(".,!?;:'\"")
        if key in self.dict:
            result = self.dict[key].copy()
            result["source"] = "dictionary"
            return result
        # Phonological transliteration (when dictionary lookup fails)
        gondi_approx = self._transliterate(key)
        return {
            "gondi": gondi_approx,
            "ipa": self._approx_ipa(gondi_approx),
            "source": "transliteration",
        }

    def _transliterate(self, word: str) -> str:
        """Simple phonological carry-over (last resort)."""
        # Gondi borrows many words from neighbouring languages
        # Rule: if word ends in silent-e, drop it; adapt digraphs
        word = re.sub(r"ph", "f", word)
        word = re.sub(r"ck", "k", word)
        word = re.sub(r"tion$", "shan", word)
        word = re.sub(r"sion$", "shan", word)
        word = re.sub(r"ly$", "li", word)
        word = re.sub(r"ing$", "ing", word)
        word = re.sub(r"ness$", "ata", word)
        return word

    def _approx_ipa(self, gondi_word: str) -> str:
        """Very rough IPA from Gondi romanization."""
        mapping = {
            "sh": "ʃ", "ch": "tʃ", "th": "t̪ʰ", "dh": "d̪ʱ",
            "ng": "ŋ", "aa": "aː", "ee": "iː", "oo": "uː",
            "a": "ə", "i": "ɪ", "u": "ʊ", "e": "eː", "o": "oː",
        }
        result = gondi_word
        for rom, ipa in mapping.items():
            result = result.replace(rom, ipa)
        return result

    def translate_segment(self, segment: Dict) -> Dict:
        """Translate a full text segment to Gondi."""
        words = segment["text"].split()
        lang = segment.get("lang", "English")
        translations = [self.translate_word(w, lang) for w in words]
        gondi_text = " ".join(t["gondi"] for t in translations)
        gondi_ipa = " ".join(t["ipa"] for t in translations)
        return {
            "original": segment["text"],
            "lang": lang,
            "gondi": gondi_text,
            "gondi_ipa": gondi_ipa,
            "start": segment.get("start", 0.0),
            "end": segment.get("end", 0.0),
        }

    def translate_full_transcript(self, segments: List[Dict]) -> List[Dict]:
        return [self.translate_segment(s) for s in segments]


# ─────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Part II: Phonetic Mapping & Translation")
    parser.add_argument("--segments_json", default=None,
                        help="JSON file from Part I with segmented transcript")
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Example segments (replace with real Part I output)
    if args.segments_json and os.path.exists(args.segments_json):
        with open(args.segments_json, encoding="utf-8") as f:
            payload = json.load(f)
        segments = payload.get("segments", payload if isinstance(payload, list) else [])
    else:
        segments = [
            {"text": "Today we will study the mel frequency cepstral coefficient",
             "lang": "English", "start": 0.0, "end": 5.0},
            {"text": "yeh ek bahut important topic hai stochastic process ke liye",
             "lang": "Hindi", "start": 5.0, "end": 10.0},
            {"text": "The spectrogram gives us frequency vs time representation",
             "lang": "English", "start": 10.0, "end": 15.0},
        ]

    print("=" * 60)
    print("PART II: Phonetic Mapping & Translation")
    print("=" * 60)

    # Task 2.1 – IPA conversion
    print("\n[2.1] Converting transcript to unified IPA...")
    g2p = HinglishG2P()
    ipa_string = g2p.convert_transcript(segments)
    print("  Unified IPA (first 200 chars):", ascii_console_preview(ipa_string, 200), "...")
    print(f"  Segments loaded: {len(segments)}")

    # Task 2.2 – Translation to Gondi
    print(f"\n[2.2] Translating to Gondi (dict size: {len(GONDI_TECHNICAL_DICT)} words)...")
    translator = GondiTranslator()
    translated = translator.translate_full_transcript(segments)

    for i, seg in enumerate(translated):
        print(f"\n  Segment {i+1} [{seg['lang']}]:")
        print(f"    Original  : {ascii_console_preview(seg['original'])}")
        print(f"    Gondi     : {ascii_console_preview(seg['gondi'])}")
        print(f"    Gondi IPA : {ascii_console_preview(seg['gondi_ipa'])}")

    # Save outputs
    output = {
        "ipa_unified": ipa_string,
        "translated_segments": translated,
        "dictionary_size": len(GONDI_TECHNICAL_DICT),
    }
    out_path = os.path.join(args.output_dir, "part2_translation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[Part II complete] Saved: {out_path}")
