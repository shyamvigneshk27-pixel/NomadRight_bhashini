#!/usr/bin/env python3
"""
Builds forms.json - the deterministic definitions of the paper forms the kiosk can
fill for a citizen: which form it is (title patterns / keywords for OCR matching),
which scheme it belongs to, and the fields in the EXACT order they appear on the
form, each with its type, whether it is required, how it is validated, and the
predefined question in every supported language. Nothing here is translated at
runtime: the Hindi and Tamil questions below are spoken as written.

Sources (official PDFs in ./sources, fetched 2026-09-12):
  PMSBY / PMJJBY consent-cum-declaration forms (jansuraksha.gov.in, en/hi/ta)
  APY subscriber registration form (jansuraksha.gov.in, en/hi/ta)
  PMUY KYC application form v8 (pmuy.gov.in)            PM SVANidhi common loan application form (mohua)
  PM-KISAN self-declaration form (central format, scanned)
  MGNREGA "Application for registration" (Operational Guidelines 2013, Annexure 3)
  Sukanya Samriddhi Account Scheme 2019 Form-1 (SSA-1)

usage: python3 build_forms.py   (writes forms.json next to this file)

Native-speaker review of the Hindi/Tamil strings is still pending - they were written
to be short and plain, the way a clerk would ask.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LANGS = ["en", "hi", "ta"]

# ── Shared spoken prompts (per language) ───────────────────────────────────
PROMPTS = {
    "confirm_form": {"en": "I think this is the {title} form. Is that right? Please say yes or no.",
                     "hi": "यह {title} का फ़ॉर्म लगता है। क्या यह सही है? हाँ या नहीं बोलिए।",
                     "ta": "இது {title} படிவம் என்று நினைக்கிறேன். சரியா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."},
    "choose_form": {"en": "Which form is this: {a}, or {b}? Please say the name.",
                    "hi": "यह कौन सा फ़ॉर्म है: {a}, या {b}? नाम बोलिए।",
                    "ta": "இது எந்தப் படிவம்: {a}, அல்லது {b}? பெயரைச் சொல்லுங்கள்."},
    "not_readable": {"en": "I could not read the form. Please hold it flat and close to the camera, or tell me which scheme it is for.",
                     "hi": "मैं फ़ॉर्म पढ़ नहीं पाया। कृपया इसे सीधा और कैमरे के पास रखें, या बताइए यह किस योजना का फ़ॉर्म है।",
                     "ta": "படிவத்தைப் படிக்க முடியவில்லை. தயவுசெய்து அதை நேராகவும் கேமராவுக்கு அருகிலும் காட்டுங்கள், அல்லது இது எந்தத் திட்டத்திற்கான படிவம் என்று சொல்லுங்கள்."},
    "unsupported": {"en": "Sorry, I cannot fill this form yet. I can help with these forms: {forms}.",
                    "hi": "माफ़ कीजिए, यह फ़ॉर्म मैं अभी नहीं भर सकता। मैं इन फ़ॉर्मों में मदद कर सकता हूँ: {forms}।",
                    "ta": "மன்னிக்கவும், இந்தப் படிவத்தை இப்போது நிரப்ப முடியாது. இந்தப் படிவங்களுக்கு உதவ முடியும்: {forms}."},
    "start": {"en": "I will now ask the questions on this form one by one. After each question, tap the button, say your answer, then tap again to stop. Each answer is saved as soon as you give it.",
              "hi": "अब मैं इस फ़ॉर्म के सवाल एक-एक करके पूछूँगा। हर सवाल के बाद बटन दबाइए, अपना जवाब बोलिए, फिर रुकने के लिए दोबारा दबाइए। हर जवाब बोलते ही सुरक्षित हो जाता है।",
              "ta": "இப்போது இந்தப் படிவத்தின் கேள்விகளை ஒவ்வொன்றாகக் கேட்பேன். ஒவ்வொரு கேள்விக்குப் பிறகும் பொத்தானைத் தட்டி, பதிலைச் சொல்லி, நிறுத்த மீண்டும் தட்டுங்கள். ஒவ்வொரு பதிலும் சொன்னவுடன் சேமிக்கப்படும்."},
    "confirm_value": {"en": "You said {value}. Is that correct? Say yes or no.",
                      "hi": "आपने {value} बताया। क्या यह सही है? हाँ या नहीं बोलिए।",
                      "ta": "நீங்கள் {value} என்று சொன்னீர்கள். சரியா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."},
    "retry": {"en": "Sorry, I did not understand. Please say it again.",
              "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। कृपया फिर से बोलिए।",
              "ta": "மன்னிக்கவும், புரியவில்லை. மீண்டும் சொல்லுங்கள்."},
    "invalid_number": {"en": "That does not look like a valid number. Please say the digits slowly, one by one.",
                       "hi": "यह सही नंबर नहीं लगता। कृपया अंक धीरे-धीरे, एक-एक करके बोलिए।",
                       "ta": "இது சரியான எண்ணாகத் தெரியவில்லை. இலக்கங்களை ஒவ்வொன்றாக மெதுவாகச் சொல்லுங்கள்."},
    "invalid_date": {"en": "I need the date as day, month and year. Please say it again.",
                     "hi": "मुझे तारीख दिन, महीना और साल में चाहिए। कृपया फिर से बोलिए।",
                     "ta": "தேதியை நாள், மாதம், வருடம் என்று சொல்லுங்கள். மீண்டும் சொல்லுங்கள்."},
    "invalid_choice": {"en": "Please choose one of: {options}.",
                       "hi": "कृपया इनमें से एक चुनिए: {options}।",
                       "ta": "இவற்றில் ஒன்றைச் சொல்லுங்கள்: {options}."},
    "optional_hint": {"en": "If you do not know, say skip.",
                      "hi": "अगर पता नहीं है तो 'छोड़ो' बोलिए।",
                      "ta": "தெரியாவிட்டால் 'தவிர்' என்று சொல்லுங்கள்."},
    "show_document": {"en": "Please show your {doc} to the camera and hold it steady.",
                      "hi": "कृपया अपना {doc} कैमरे के सामने स्थिर रखिए।",
                      "ta": "தயவுசெய்து உங்கள் {doc} கேமரா முன் அசையாமல் காட்டுங்கள்."},
    "doc_read_ok": {"en": "I read {value} from the document. Is that correct? Say yes or no.",
                    "hi": "दस्तावेज़ से मैंने {value} पढ़ा। क्या यह सही है? हाँ या नहीं बोलिए।",
                    "ta": "ஆவணத்திலிருந்து {value} என்று படித்தேன். சரியா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."},
    "doc_read_fail_skip": {"en": "I could not read it from the document; the officer will fill it in.",
                           "hi": "मैं दस्तावेज़ से इसे नहीं पढ़ पाया; यह अधिकारी भर देंगे।",
                           "ta": "ஆவணத்திலிருந்து அதைப் படிக்க முடியவில்லை; அதை அதிகாரி நிரப்புவார்."},
    "doc_read_fail": {"en": "I could not read it from the document. Please say it slowly, close to the microphone.",
                      "hi": "मैं दस्तावेज़ से इसे नहीं पढ़ पाया। कृपया माइक के पास धीरे-धीरे बोलिए।",
                      "ta": "ஆவணத்திலிருந்து அதைப் படிக்க முடியவில்லை. மைக்கிற்கு அருகில் மெதுவாகச் சொல்லுங்கள்."},
    "review_intro": {"en": "I will now read back your answers.",
                     "hi": "अब मैं आपके जवाब पढ़कर सुनाता हूँ।",
                     "ta": "இப்போது உங்கள் பதில்களைப் படித்துக் காட்டுகிறேன்."},
    "review_confirm": {"en": "Is everything correct? Say yes, or press the Yes button, to send it to the office; or say change.",
                       "hi": "क्या सब सही है? दफ़्तर भेजने के लिए 'हाँ' बोलिए या हाँ का बटन दबाइए, या 'बदलो' बोलिए।",
                       "ta": "எல்லாம் சரியா? அலுவலகத்திற்கு அனுப்ப 'ஆம்' என்று சொல்லுங்கள் அல்லது ஆம் பொத்தானை அழுத்துங்கள்; அல்லது 'மாற்று' என்று சொல்லுங்கள்."},
    "didnt_hear": {"en": "I did not catch that. Please say yes or no, or press the Yes or No button on the screen.",
                   "hi": "मुझे सुनाई नहीं दिया। कृपया हाँ या नहीं बोलिए, या स्क्रीन पर हाँ या नहीं का बटन दबाइए।",
                   "ta": "எனக்குக் கேட்கவில்லை. தயவுசெய்து ஆம் அல்லது இல்லை என்று சொல்லுங்கள், அல்லது திரையில் ஆம் அல்லது இல்லை பொத்தானை அழுத்துங்கள்."},
    "saved_unconfirmed": {"en": "I could not hear a yes. Your answers have already been saved for the officer, who will check them with you. Thank you.",
                          "hi": "मुझे 'हाँ' सुनाई नहीं दिया। आपके जवाब अधिकारी के लिए पहले ही सुरक्षित कर दिए गए हैं, वे आपके साथ जाँच लेंगे। धन्यवाद।",
                          "ta": "'ஆம்' என்று கேட்கவில்லை. உங்கள் பதில்கள் ஏற்கனவே அதிகாரிக்காகச் சேமிக்கப்பட்டுள்ளன; அவர் உங்களுடன் சரிபார்ப்பார். நன்றி."},
    "cancelled_saved": {"en": "Form filling stopped. The answers you gave so far have been saved for the officer.",
                        "hi": "फ़ॉर्म भरना रोक दिया गया। अब तक के आपके जवाब अधिकारी के लिए सुरक्षित कर दिए गए हैं।",
                        "ta": "படிவம் நிரப்புவது நிறுத்தப்பட்டது. இதுவரை நீங்கள் சொன்ன பதில்கள் அதிகாரிக்காகச் சேமிக்கப்பட்டுள்ளன."},
    "which_field": {"en": "Which answer do you want to change? Say its number.",
                    "hi": "कौन सा जवाब बदलना है? उसका नंबर बोलिए।",
                    "ta": "எந்தப் பதிலை மாற்ற வேண்டும்? அதன் எண்ணைச் சொல்லுங்கள்."},
    "sending": {"en": "Sending your form to the office computer.",
                "hi": "आपका फ़ॉर्म दफ़्तर के कंप्यूटर पर भेजा जा रहा है।",
                "ta": "உங்கள் படிவம் அலுவலக கணினிக்கு அனுப்பப்படுகிறது."},
    "sent": {"en": "Your form has been sent. Thank you.",
             "hi": "आपका फ़ॉर्म भेज दिया गया है। धन्यवाद।",
             "ta": "உங்கள் படிவம் அனுப்பப்பட்டது. நன்றி."},
    "send_pending": {"en": "The office computer is not reachable right now. Your form is saved safely and will be sent automatically.",
                     "hi": "दफ़्तर का कंप्यूटर अभी नहीं मिल रहा है। आपका फ़ॉर्म सुरक्षित रखा गया है और अपने आप भेज दिया जाएगा।",
                     "ta": "அலுவலக கணினியை இப்போது அணுக முடியவில்லை. உங்கள் படிவம் பாதுகாப்பாகச் சேமிக்கப்பட்டது, தானாக அனுப்பப்படும்."},
    "cancelled": {"en": "Form filling cancelled.", "hi": "फ़ॉर्म भरना रद्द कर दिया गया।", "ta": "படிவம் நிரப்புவது ரத்து செய்யப்பட்டது."},
    "need_language": {"en": "Please choose Hindi or Tamil to fill a form by voice.",
                      "hi": "फ़ॉर्म भरने के लिए कृपया हिंदी या तमिल चुनिए।",
                      "ta": "படிவம் நிரப்ப தயவுசெய்து இந்தி அல்லது தமிழ் தேர்வு செய்யுங்கள்."},
    "field_number": {"en": "Question {n} of {total}.", "hi": "सवाल {n}, कुल {total}।", "ta": "கேள்வி {n}, மொத்தம் {total}."},
}

# Words the kiosk understands as answers/commands (lower-case, matched as whole words).
WORDS = {
    "yes": {"en": ["yes", "yeah", "yep", "yup", "correct", "right", "ok", "okay", "alright", "confirm", "confirmed", "send", "send it", "go ahead", "that's right", "yes please"],
            "hi": ["हाँ", "हां", "हा", "जी", "जी हाँ", "जी हां", "हां जी", "सही", "ठीक", "ठीक है", "ठीक हैं", "हाँ जी", "बिल्कुल", "यस", "येस", "ओके"],
            "ta": ["ஆம்", "ஆமா", "ஆமாம்", "ஆமாமா", "ஆமான்", "சரி", "சரிங்க", "ஓகே", "சரிதான்", "ஆமாங்க", "எஸ்", "யெஸ்"]},
    "no": {"en": ["no", "nope", "nah", "wrong", "incorrect", "not correct", "not right"],
           "hi": ["नहीं", "नही", "ना", "नो", "गलत", "गलत है", "नहीं है", "जी नहीं", "नहीं जी"],
           "ta": ["இல்லை", "இல்ல", "இல்லையே", "நோ", "தவறு", "தப்பு", "இல்லைங்க", "வேண்டாம்"]},
    "repeat": {"en": ["repeat", "again", "say again", "pardon"],
               "hi": ["दोबारा", "फिर से", "फिर", "दुबारा", "फिर बोलो", "फिर से बोलो"],
               "ta": ["மீண்டும்", "திரும்ப", "திரும்பச் சொல்லுங்கள்", "மறுபடியும்"]},
    "change": {"en": ["change", "back", "go back", "correct it", "wrong one"],
               "hi": ["बदलो", "बदलना", "पीछे", "वापस", "पिछला", "गलत हो गया"],
               "ta": ["மாற்று", "மாத்து", "பின்னால்", "திருத்து", "மாற்ற வேண்டும்"]},
    "skip": {"en": ["skip", "leave it", "next", "don't know", "not known"],
             "hi": ["छोड़ो", "छोड़ दो", "छोड़", "अगला", "पता नहीं", "मालूम नहीं"],
             "ta": ["தவிர்", "விடு", "விட்டுவிடு", "அடுத்தது", "தெரியாது", "தெரியல"]},
    "cancel": {"en": ["cancel", "stop", "quit"],
               "hi": ["रद्द", "बंद करो", "रोको", "बंद"],
               "ta": ["ரத்து", "நிறுத்து", "வேண்டாம் நிறுத்து"]},
}

# ── Question bank: every field the pilot forms ask, in the three languages ──
Q = {
    "full_name": ("What is your full name?", "आपका पूरा नाम क्या है?", "உங்கள் முழு பெயர் என்ன?"),
    "father_or_husband_name": ("What is your father's or husband's name?", "आपके पिता या पति का नाम क्या है?", "உங்கள் தந்தை அல்லது கணவரின் பெயர் என்ன?"),
    "father_name": ("What is your father's name?", "आपके पिता का नाम क्या है?", "உங்கள் தந்தையின் பெயர் என்ன?"),
    "father_or_spouse_name": ("What is your father's or spouse's name?", "आपके पिता या जीवनसाथी का नाम क्या है?", "உங்கள் தந்தை அல்லது வாழ்க்கைத் துணையின் பெயர் என்ன?"),
    "dob": ("What is your date of birth? Say the day, the month and the year.", "आपकी जन्म तिथि क्या है? दिन, महीना और साल बताइए।", "உங்கள் பிறந்த தேதி என்ன? நாள், மாதம், வருடம் சொல்லுங்கள்."),
    "age": ("How old are you? Say your age in years.", "आपकी उम्र कितनी है? साल में बताइए।", "உங்கள் வயது என்ன? வருடங்களில் சொல்லுங்கள்."),
    "gender": ("Are you male or female?", "आप पुरुष हैं या महिला?", "நீங்கள் ஆண் அல்லது பெண்?"),
    "mobile": ("What is your mobile number? Say the ten digits slowly.", "आपका मोबाइल नंबर क्या है? दस अंक धीरे-धीरे बोलिए।", "உங்கள் மொபைல் எண் என்ன? பத்து இலக்கங்களையும் மெதுவாகச் சொல்லுங்கள்."),
    "email": ("What is your email address? If you have none, say skip.", "आपका ईमेल क्या है? न हो तो 'छोड़ो' बोलिए।", "உங்கள் மின்னஞ்சல் என்ன? இல்லையென்றால் 'தவிர்' என்று சொல்லுங்கள்."),
    "address": ("What is your address? Say the house, the street and the area.", "आपका पता क्या है? मकान, गली और इलाका बताइए।", "உங்கள் முகவரி என்ன? வீடு, தெரு, பகுதி சொல்லுங்கள்."),
    "village_or_town": ("Which village, town or city do you live in?", "आप किस गाँव, कस्बे या शहर में रहते हैं?", "நீங்கள் எந்த கிராமம், நகரம் அல்லது ஊரில் வசிக்கிறீர்கள்?"),
    "district": ("Which district?", "कौन सा ज़िला?", "எந்த மாவட்டம்?"),
    "state": ("Which state?", "कौन सा राज्य?", "எந்த மாநிலம்?"),
    "pincode": ("What is the PIN code of your address? Say the six digits.", "आपके पते का पिन कोड क्या है? छह अंक बोलिए।", "உங்கள் முகவரியின் பின் குறியீடு என்ன? ஆறு இலக்கங்களைச் சொல்லுங்கள்."),
    "aadhaar": ("Please show your Aadhaar card, or a copy of it, to the camera.", "कृपया अपना आधार कार्ड या उसकी फोटोकॉपी कैमरे के सामने दिखाइए।", "தயவுசெய்து உங்கள் ஆதார் அட்டையை அல்லது அதன் நகலை கேமரா முன் காட்டுங்கள்."),
    "aadhaar_spoken": ("Please say the twelve digits of your Aadhaar number slowly.", "आधार नंबर के बारह अंक धीरे-धीरे बोलिए।", "ஆதார் எண்ணின் பன்னிரண்டு இலக்கங்களையும் மெதுவாகச் சொல்லுங்கள்."),
    "bank_passbook": ("Please show the first page of your bank passbook to the camera.", "कृपया अपनी बैंक पासबुक का पहला पन्ना कैमरे के सामने दिखाइए।", "தயவுசெய்து உங்கள் வங்கி பாஸ்புக்கின் முதல் பக்கத்தை கேமரா முன் காட்டுங்கள்."),
    "account_number_spoken": ("Please say your bank account number slowly, digit by digit.", "अपना बैंक खाता नंबर धीरे-धीरे, एक-एक अंक करके बोलिए।", "உங்கள் வங்கிக் கணக்கு எண்ணை ஒவ்வொரு இலக்கமாக மெதுவாகச் சொல்லுங்கள்."),
    "bank_name": ("Which bank is your account in?", "आपका खाता किस बैंक में है?", "உங்கள் கணக்கு எந்த வங்கியில் உள்ளது?"),
    "branch_name": ("Which branch of the bank?", "बैंक की कौन सी शाखा?", "வங்கியின் எந்தக் கிளை?"),
    "ifsc": ("Please show the page of your passbook with the IFSC code to the camera.", "कृपया पासबुक का वह पन्ना दिखाइए जिस पर IFSC कोड लिखा है।", "IFSC குறியீடு உள்ள பாஸ்புக் பக்கத்தை கேமரா முன் காட்டுங்கள்."),
    "pan": ("Do you have a PAN card? If yes, please show it to the camera; otherwise say skip.", "क्या आपके पास पैन कार्ड है? हो तो कैमरे के सामने दिखाइए, नहीं तो 'छोड़ो' बोलिए।", "உங்களிடம் பான் கார்டு உள்ளதா? இருந்தால் கேமரா முன் காட்டுங்கள்; இல்லையென்றால் 'தவிர்' என்று சொல்லுங்கள்."),
    "kyc_document": ("Which identity document will you attach: Aadhaar, voter card, ration card, or driving licence?", "पहचान के लिए कौन सा दस्तावेज़ देंगे: आधार, वोटर कार्ड, राशन कार्ड, या ड्राइविंग लाइसेंस?", "அடையாளத்திற்கு எந்த ஆவணம் தருவீர்கள்: ஆதார், வாக்காளர் அட்டை, ரேஷன் அட்டை, அல்லது ஓட்டுநர் உரிமம்?"),
    "disability": ("Do you have any disability? Say yes or no.", "क्या आपको कोई विकलांगता है? हाँ या नहीं बोलिए।", "உங்களுக்கு ஏதேனும் மாற்றுத்திறன் உள்ளதா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "disability_details": ("What is the disability?", "कौन सी विकलांगता है?", "என்ன மாற்றுத்திறன்?"),
    "nominee_name": ("Who is your nominee? Say their full name.", "आपका नॉमिनी कौन है? पूरा नाम बताइए।", "உங்கள் நாமினி யார்? முழு பெயரைச் சொல்லுங்கள்."),
    "nominee_relation": ("What is the nominee's relation to you: wife, husband, son, daughter, father, mother, or other?", "नॉमिनी का आपसे क्या रिश्ता है: पत्नी, पति, बेटा, बेटी, पिता, माता, या अन्य?", "நாமினிக்கும் உங்களுக்கும் என்ன உறவு: மனைவி, கணவர், மகன், மகள், தந்தை, தாய், அல்லது வேறு?"),
    "nominee_dob": ("What is the nominee's date of birth? Say the day, the month and the year.", "नॉमिनी की जन्म तिथि क्या है? दिन, महीना और साल बताइए।", "நாமினியின் பிறந்த தேதி என்ன? நாள், மாதம், வருடம் சொல்லுங்கள்."),
    "nominee_address": ("What is the nominee's address?", "नॉमिनी का पता क्या है?", "நாமினியின் முகவரி என்ன?"),
    "nominee_mobile": ("What is the nominee's mobile number? If none, say skip.", "नॉमिनी का मोबाइल नंबर क्या है? न हो तो 'छोड़ो' बोलिए।", "நாமினியின் மொபைல் எண் என்ன? இல்லையென்றால் 'தவிர்' என்று சொல்லுங்கள்."),
    "guardian_name": ("The nominee is a minor. Who is their guardian? Say the full name.", "नॉमिनी नाबालिग है। उनका अभिभावक कौन है? पूरा नाम बताइए।", "நாமினி சிறியவர். அவர்களின் பாதுகாவலர் யார்? முழு பெயரைச் சொல்லுங்கள்."),
    "guardian_relation": ("What is the guardian's relation to the nominee?", "अभिभावक का नॉमिनी से क्या रिश्ता है?", "பாதுகாவலருக்கும் நாமினிக்கும் என்ன உறவு?"),
    "married": ("Are you married? Say yes or no.", "क्या आप शादीशुदा हैं? हाँ या नहीं बोलिए।", "நீங்கள் திருமணமானவரா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "spouse_name": ("What is your spouse's name?", "आपके जीवनसाथी का नाम क्या है?", "உங்கள் வாழ்க்கைத் துணையின் பெயர் என்ன?"),
    "other_social_scheme": ("Are you already getting a pension or benefit from any other government social security scheme? Say yes or no.", "क्या आपको पहले से किसी और सरकारी सामाजिक सुरक्षा योजना से पेंशन या लाभ मिलता है? हाँ या नहीं बोलिए।", "வேறு ஏதேனும் அரசு சமூகப் பாதுகாப்புத் திட்டத்தில் ஏற்கனவே ஓய்வூதியம் அல்லது பலன் பெறுகிறீர்களா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "income_tax_payer": ("Do you pay income tax? Say yes or no.", "क्या आप आयकर भरते हैं? हाँ या नहीं बोलिए।", "நீங்கள் வருமான வரி செலுத்துகிறீர்களா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "apy_pension_amount": ("How much monthly pension do you want at sixty: one thousand, two thousand, three thousand, four thousand, or five thousand rupees?", "साठ साल के बाद कितनी मासिक पेंशन चाहिए: एक हज़ार, दो हज़ार, तीन हज़ार, चार हज़ार, या पाँच हज़ार रुपये?", "அறுபது வயதுக்குப் பிறகு மாதம் எவ்வளவு ஓய்வூதியம் வேண்டும்: ஆயிரம், இரண்டாயிரம், மூவாயிரம், நான்காயிரம், அல்லது ஐயாயிரம் ரூபாய்?"),
    "contribution_period": ("How often will you pay the contribution: every month, every three months, or every six months?", "आप योगदान कब-कब देंगे: हर महीने, हर तीन महीने, या हर छह महीने?", "பங்களிப்பை எப்படிச் செலுத்துவீர்கள்: ஒவ்வொரு மாதமும், மூன்று மாதங்களுக்கு ஒருமுறை, அல்லது ஆறு மாதங்களுக்கு ஒருமுறை?"),
    "caste_category": ("Which category do you belong to: general, SC, ST, OBC, or other?", "आप किस श्रेणी में आते हैं: सामान्य, अनुसूचित जाति, अनुसूचित जनजाति, ओबीसी, या अन्य?", "நீங்கள் எந்தப் பிரிவு: பொது, எஸ்சி, எஸ்டி, ஓபிசி, அல்லது வேறு?"),
    "is_migrant": ("Are you a migrant, living away from your home state? Say yes or no.", "क्या आप प्रवासी हैं, यानी अपने गृह राज्य से बाहर रहते हैं? हाँ या नहीं बोलिए।", "நீங்கள் புலம்பெயர்ந்தவரா, அதாவது சொந்த மாநிலத்தை விட்டு வெளியே வசிக்கிறீர்களா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "ration_card": ("Please show your ration card to the camera.", "कृपया अपना राशन कार्ड कैमरे के सामने दिखाइए।", "தயவுசெய்து உங்கள் ரேஷன் அட்டையை கேமரா முன் காட்டுங்கள்."),
    "ration_card_spoken": ("Please say your ration card number slowly.", "अपना राशन कार्ड नंबर धीरे-धीरे बोलिए।", "உங்கள் ரேஷன் அட்டை எண்ணை மெதுவாகச் சொல்லுங்கள்."),
    "ration_card_state": ("Which state issued your ration card?", "आपका राशन कार्ड किस राज्य ने जारी किया है?", "உங்கள் ரேஷன் அட்டையை எந்த மாநிலம் வழங்கியது?"),
    "household_adults": ("How many people aged eighteen or more live in your household, including you?", "आपके घर में आपको मिलाकर अठारह साल या उससे ऊपर के कितने लोग रहते हैं?", "உங்களையும் சேர்த்து, உங்கள் வீட்டில் பதினெட்டு வயது அல்லது அதற்கு மேற்பட்டவர்கள் எத்தனை பேர்?"),
    "member_name": ("What is the name of household member {n}?", "घर के सदस्य {n} का नाम क्या है?", "வீட்டு உறுப்பினர் {n} இன் பெயர் என்ன?"),
    "member_relation": ("What is their relation to you?", "उनका आपसे क्या रिश्ता है?", "அவருக்கும் உங்களுக்கும் என்ன உறவு?"),
    "member_gender": ("Is this member male or female?", "यह सदस्य पुरुष है या महिला?", "இந்த உறுப்பினர் ஆணா அல்லது பெண்ணா?"),
    "member_age": ("How old is this member?", "इस सदस्य की उम्र कितनी है?", "இந்த உறுப்பினரின் வயது என்ன?"),
    "lpg_connection_type": ("Which connection do you want: a single five kilo cylinder, a double five kilo cylinder, or a fourteen kilo cylinder?", "कौन सा कनेक्शन चाहिए: पाँच किलो का एक सिलेंडर, पाँच किलो के दो सिलेंडर, या चौदह किलो का सिलेंडर?", "எந்த இணைப்பு வேண்டும்: ஐந்து கிலோ ஒரு சிலிண்டர், ஐந்து கிலோ இரண்டு சிலிண்டர், அல்லது பதினான்கு கிலோ சிலிண்டர்?"),
    "marital_status": ("What is your marital status: married, unmarried, widowed, or divorced?", "आपकी वैवाहिक स्थिति क्या है: विवाहित, अविवाहित, विधवा या विधुर, या तलाकशुदा?", "உங்கள் திருமண நிலை என்ன: திருமணமானவர், திருமணமாகாதவர், விதவை அல்லது விதவர், அல்லது விவாகரத்தானவர்?"),
    "minority": ("Do you belong to a minority community? Say yes or no.", "क्या आप अल्पसंख्यक समुदाय से हैं? हाँ या नहीं बोलिए।", "நீங்கள் சிறுபான்மை சமூகத்தைச் சேர்ந்தவரா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "family_member_name": ("Name one family member for the form. What is their full name?", "फ़ॉर्म के लिए परिवार के एक सदस्य का नाम बताइए। पूरा नाम क्या है?", "படிவத்திற்காக குடும்ப உறுப்பினர் ஒருவரின் பெயரைச் சொல்லுங்கள். முழு பெயர் என்ன?"),
    "vending_activity": ("What do you sell or do as a street vendor?", "आप फेरी में क्या बेचते हैं या क्या काम करते हैं?", "நீங்கள் தெருவோர வியாபாரியாக என்ன விற்கிறீர்கள் அல்லது என்ன செய்கிறீர்கள்?"),
    "vending_place": ("Where do you sell? Say the place or the market.", "आप कहाँ बेचते हैं? जगह या बाज़ार का नाम बताइए।", "நீங்கள் எங்கே விற்கிறீர்கள்? இடம் அல்லது சந்தையின் பெயரைச் சொல்லுங்கள்."),
    "vending_years": ("For how many years have you been vending?", "आप कितने साल से फेरी लगा रहे हैं?", "எத்தனை வருடங்களாக வியாபாரம் செய்கிறீர்கள்?"),
    "vendor_type": ("Do you sell from one fixed place, or do you move around?", "आप एक ही जगह पर बेचते हैं, या घूम-घूम कर?", "ஒரே இடத்தில் விற்கிறீர்களா, அல்லது நடமாடி விற்கிறீர்களா?"),
    "monthly_sales": ("About how much do you sell in a month, in rupees?", "महीने में लगभग कितने रुपये की बिक्री होती है?", "மாதத்திற்கு சுமார் எவ்வளவு ரூபாய் விற்பனை ஆகும்?"),
    "loan_amount": ("How much loan do you want: ten thousand, twenty thousand, or fifty thousand rupees?", "कितना कर्ज़ चाहिए: दस हज़ार, बीस हज़ार, या पचास हज़ार रुपये?", "எவ்வளவு கடன் வேண்டும்: பத்தாயிரம், இருபதாயிரம், அல்லது ஐம்பதாயிரம் ரூபாய்?"),
    "loan_purpose": ("What will you use the loan for?", "कर्ज़ का इस्तेमाल किसलिए करेंगे?", "கடனை எதற்குப் பயன்படுத்துவீர்கள்?"),
    "land_village": ("In which village is your farm land?", "आपकी खेती की ज़मीन किस गाँव में है?", "உங்கள் விவசாய நிலம் எந்த கிராமத்தில் உள்ளது?"),
    "land_survey_number": ("What is the survey or khata number of the land? If you do not know, say skip.", "ज़मीन का सर्वे या खाता नंबर क्या है? पता न हो तो 'छोड़ो' बोलिए।", "நிலத்தின் சர்வே அல்லது கட்டா எண் என்ன? தெரியாவிட்டால் 'தவிர்' என்று சொல்லுங்கள்."),
    "land_area": ("How much land do you own, in acres?", "आपके पास कितनी ज़मीन है, एकड़ में?", "உங்களுக்கு எவ்வளவு நிலம் உள்ளது, ஏக்கரில்?"),
    "govt_employee_family": ("Does anyone in your family hold a government job, or a pension of ten thousand rupees or more a month? Say yes or no.", "क्या आपके परिवार में कोई सरकारी नौकरी में है, या दस हज़ार रुपये महीने या उससे ज़्यादा पेंशन पाता है? हाँ या नहीं बोलिए।", "உங்கள் குடும்பத்தில் யாரேனும் அரசு வேலையில் இருக்கிறார்களா, அல்லது மாதம் பத்தாயிரம் ரூபாய் அல்லது அதற்கு மேல் ஓய்வூதியம் பெறுகிறார்களா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "house_number": ("What is your house number?", "आपका मकान नंबर क्या है?", "உங்கள் வீட்டு எண் என்ன?"),
    "gram_panchayat": ("Which gram panchayat?", "कौन सी ग्राम पंचायत?", "எந்த கிராம பஞ்சாயத்து?"),
    "block": ("Which block?", "कौन सा ब्लॉक?", "எந்த வட்டாரம்?"),
    "head_of_household": ("What is the name of the head of your household?", "आपके घर के मुखिया का नाम क्या है?", "உங்கள் குடும்பத் தலைவரின் பெயர் என்ன?"),
    "relation_to_head": ("What is your relation to the head of the household?", "घर के मुखिया से आपका क्या रिश्ता है?", "குடும்பத் தலைவருக்கும் உங்களுக்கும் என்ன உறவு?"),
    "bpl": ("Is your family a BPL family? Say yes or no.", "क्या आपका परिवार बीपीएल परिवार है? हाँ या नहीं बोलिए।", "உங்கள் குடும்பம் பிபிஎல் குடும்பமா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "voter_id": ("Please show your voter card to the camera. If you have none, say skip.", "कृपया अपना वोटर कार्ड कैमरे के सामने दिखाइए। न हो तो 'छोड़ो' बोलिए।", "தயவுசெய்து உங்கள் வாக்காளர் அட்டையை கேமரா முன் காட்டுங்கள். இல்லையென்றால் 'தவிர்' என்று சொல்லுங்கள்."),
    "workers_count": ("How many adults in your household want to work under the scheme?", "आपके घर के कितने वयस्क इस योजना में काम करना चाहते हैं?", "உங்கள் வீட்டில் எத்தனை பெரியவர்கள் இந்தத் திட்டத்தில் வேலை செய்ய விரும்புகிறார்கள்?"),
    "worker_name": ("What is the name of worker {n}?", "काम करने वाले सदस्य {n} का नाम क्या है?", "வேலை செய்யும் உறுப்பினர் {n} இன் பெயர் என்ன?"),
    "girl_name": ("What is the girl child's full name?", "बच्ची का पूरा नाम क्या है?", "பெண் குழந்தையின் முழு பெயர் என்ன?"),
    "girl_dob": ("What is the girl's date of birth? Say the day, the month and the year.", "बच्ची की जन्म तिथि क्या है? दिन, महीना और साल बताइए।", "குழந்தையின் பிறந்த தேதி என்ன? நாள், மாதம், வருடம் சொல்லுங்கள்."),
    "guardian_full_name": ("What is your full name, as the parent or guardian?", "माता-पिता या अभिभावक के रूप में आपका पूरा नाम क्या है?", "பெற்றோர் அல்லது பாதுகாவலராக உங்கள் முழு பெயர் என்ன?"),
    "guardian_dob": ("What is your own date of birth? Say the day, the month and the year.", "आपकी अपनी जन्म तिथि क्या है? दिन, महीना और साल बताइए।", "உங்கள் சொந்தப் பிறந்த தேதி என்ன? நாள், மாதம், வருடம் சொல்லுங்கள்."),
    "same_permanent_address": ("Is your permanent address the same as this address? Say yes or no.", "क्या आपका स्थायी पता यही है? हाँ या नहीं बोलिए।", "உங்கள் நிரந்தர முகவரியும் இதுவேதானா? ஆம் அல்லது இல்லை என்று சொல்லுங்கள்."),
    "permanent_address": ("What is your permanent address?", "आपका स्थायी पता क्या है?", "உங்கள் நிரந்தர முகவரி என்ன?"),
    "birth_certificate": ("Please show the girl's birth certificate to the camera.", "कृपया बच्ची का जन्म प्रमाण पत्र कैमरे के सामने दिखाइए।", "தயவுசெய்து குழந்தையின் பிறப்புச் சான்றிதழை கேமரா முன் காட்டுங்கள்."),
    "initial_deposit": ("How much will you deposit to open the account, in rupees? Minimum two hundred and fifty.", "खाता खोलने के लिए कितने रुपये जमा करेंगे? कम से कम दो सौ पचास।", "கணக்கு தொடங்க எவ்வளவு ரூபாய் செலுத்துவீர்கள்? குறைந்தது இருநூற்று ஐம்பது."),
}

# Document names spoken in show_document prompts
DOCS = {
    "aadhaar": ("Aadhaar card", "आधार कार्ड", "ஆதார் அட்டை"),
    "bank_passbook": ("bank passbook", "बैंक पासबुक", "வங்கி பாஸ்புக்"),
    "ration_card": ("ration card", "राशन कार्ड", "ரேஷன் அட்டை"),
    "voter_id": ("voter card", "वोटर कार्ड", "வாக்காளர் அட்டை"),
    "pan": ("PAN card", "पैन कार्ड", "பான் கார்டு"),
    "birth_certificate": ("birth certificate", "जन्म प्रमाण पत्र", "பிறப்புச் சான்றிதழ்"),
}

# Spoken labels for read-back / "which answer to change"
LABELS = {
    "full_name": ("name", "नाम", "பெயர்"), "father_or_husband_name": ("father's or husband's name", "पिता या पति का नाम", "தந்தை அல்லது கணவர் பெயர்"),
    "father_name": ("father's name", "पिता का नाम", "தந்தை பெயர்"), "father_or_spouse_name": ("father's or spouse's name", "पिता या जीवनसाथी का नाम", "தந்தை அல்லது துணை பெயர்"),
    "dob": ("date of birth", "जन्म तिथि", "பிறந்த தேதி"), "age": ("age", "उम्र", "வயது"), "gender": ("gender", "लिंग", "பாலினம்"),
    "mobile": ("mobile number", "मोबाइल नंबर", "மொபைல் எண்"), "email": ("email", "ईमेल", "மின்னஞ்சல்"), "address": ("address", "पता", "முகவரி"),
    "village_or_town": ("village or town", "गाँव या शहर", "கிராமம் அல்லது நகரம்"), "district": ("district", "ज़िला", "மாவட்டம்"), "state": ("state", "राज्य", "மாநிலம்"),
    "pincode": ("PIN code", "पिन कोड", "பின் குறியீடு"), "aadhaar": ("Aadhaar number", "आधार नंबर", "ஆதார் எண்"),
    "account_number": ("bank account number", "बैंक खाता नंबर", "வங்கிக் கணக்கு எண்"), "bank_name": ("bank", "बैंक", "வங்கி"), "branch_name": ("branch", "शाखा", "கிளை"),
    "ifsc": ("IFSC code", "IFSC कोड", "IFSC குறியீடு"), "pan": ("PAN", "पैन", "பான்"), "kyc_document": ("identity document", "पहचान दस्तावेज़", "அடையாள ஆவணம்"),
    "disability": ("disability", "विकलांगता", "மாற்றுத்திறன்"), "disability_details": ("disability details", "विकलांगता का विवरण", "மாற்றுத்திறன் விவரம்"),
    "nominee_name": ("nominee", "नॉमिनी", "நாமினி"), "nominee_relation": ("nominee's relation", "नॉमिनी का रिश्ता", "நாமினி உறவு"),
    "nominee_dob": ("nominee's date of birth", "नॉमिनी की जन्म तिथि", "நாமினி பிறந்த தேதி"), "nominee_address": ("nominee's address", "नॉमिनी का पता", "நாமினி முகவரி"),
    "nominee_mobile": ("nominee's mobile", "नॉमिनी का मोबाइल", "நாமினி மொபைல்"), "guardian_name": ("guardian", "अभिभावक", "பாதுகாவலர்"),
    "guardian_relation": ("guardian's relation", "अभिभावक का रिश्ता", "பாதுகாவலர் உறவு"), "married": ("married", "शादीशुदा", "திருமணம்"),
    "spouse_name": ("spouse's name", "जीवनसाथी का नाम", "துணையின் பெயர்"), "other_social_scheme": ("other scheme benefit", "अन्य योजना का लाभ", "வேறு திட்டப் பலன்"),
    "income_tax_payer": ("income tax", "आयकर", "வருமான வரி"), "apy_pension_amount": ("pension amount", "पेंशन राशि", "ஓய்வூதியத் தொகை"),
    "contribution_period": ("contribution period", "योगदान की अवधि", "பங்களிப்புக் காலம்"), "caste_category": ("category", "श्रेणी", "பிரிவு"),
    "is_migrant": ("migrant", "प्रवासी", "புலம்பெயர்ந்தவர்"), "ration_card": ("ration card number", "राशन कार्ड नंबर", "ரேஷன் அட்டை எண்"),
    "ration_card_state": ("ration card state", "राशन कार्ड का राज्य", "ரேஷன் அட்டை மாநிலம்"), "household_adults": ("adults in household", "घर के वयस्क", "வீட்டுப் பெரியவர்கள்"),
    "household_members": ("household members", "घर के सदस्य", "வீட்டு உறுப்பினர்கள்"), "lpg_connection_type": ("connection type", "कनेक्शन का प्रकार", "இணைப்பு வகை"),
    "marital_status": ("marital status", "वैवाहिक स्थिति", "திருமண நிலை"), "minority": ("minority", "अल्पसंख्यक", "சிறுபான்மை"),
    "family_member": ("family member", "परिवार का सदस्य", "குடும்ப உறுப்பினர்"), "vending_activity": ("vending work", "फेरी का काम", "வியாபாரம்"),
    "vending_place": ("vending place", "बेचने की जगह", "வியாபார இடம்"), "vending_years": ("years of vending", "फेरी के साल", "வியாபார வருடங்கள்"),
    "vendor_type": ("vendor type", "विक्रेता का प्रकार", "வியாபாரி வகை"), "monthly_sales": ("monthly sales", "मासिक बिक्री", "மாத விற்பனை"),
    "loan_amount": ("loan amount", "कर्ज़ की राशि", "கடன் தொகை"), "loan_purpose": ("loan purpose", "कर्ज़ का उद्देश्य", "கடன் நோக்கம்"),
    "land_village": ("land village", "ज़मीन का गाँव", "நிலக் கிராமம்"), "land_survey_number": ("survey number", "सर्वे नंबर", "சர்வே எண்"),
    "land_area": ("land area", "ज़मीन का क्षेत्रफल", "நில அளவு"), "govt_employee_family": ("government job in family", "परिवार में सरकारी नौकरी", "குடும்பத்தில் அரசு வேலை"),
    "house_number": ("house number", "मकान नंबर", "வீட்டு எண்"), "gram_panchayat": ("gram panchayat", "ग्राम पंचायत", "கிராம பஞ்சாயத்து"), "block": ("block", "ब्लॉक", "வட்டாரம்"),
    "head_of_household": ("head of household", "घर का मुखिया", "குடும்பத் தலைவர்"), "relation_to_head": ("relation to head", "मुखिया से रिश्ता", "தலைவருடன் உறவு"),
    "bpl": ("BPL", "बीपीएल", "பிபிஎல்"), "voter_id": ("voter card number", "वोटर कार्ड नंबर", "வாக்காளர் அட்டை எண்"), "workers": ("workers", "काम करने वाले", "வேலை செய்பவர்கள்"),
    "workers_count": ("number of workers", "काम करने वालों की संख्या", "வேலை செய்பவர்கள் எண்ணிக்கை"),
    "girl_name": ("girl's name", "बच्ची का नाम", "குழந்தையின் பெயர்"), "girl_dob": ("girl's date of birth", "बच्ची की जन्म तिथि", "குழந்தையின் பிறந்த தேதி"),
    "guardian_full_name": ("guardian's name", "अभिभावक का नाम", "பாதுகாவலர் பெயர்"), "guardian_dob": ("guardian's date of birth", "अभिभावक की जन्म तिथि", "பாதுகாவலர் பிறந்த தேதி"),
    "same_permanent_address": ("same permanent address", "स्थायी पता वही", "அதே நிரந்தர முகவரி"), "permanent_address": ("permanent address", "स्थायी पता", "நிரந்தர முகவரி"),
    "birth_certificate": ("birth certificate number", "जन्म प्रमाण पत्र नंबर", "பிறப்புச் சான்றிதழ் எண்"), "initial_deposit": ("initial deposit", "पहली जमा राशि", "முதல் வைப்புத் தொகை"),
}

# Choice fields: canonical value -> spoken synonyms per language
CHOICES = {
    "gender": {"male": (["male", "man", "boy"], ["पुरुष", "आदमी", "मर्द", "लड़का", "मेल"], ["ஆண்", "ஆம்பள", "ஆண்பால்"]),
               "female": (["female", "woman", "girl", "lady"], ["महिला", "औरत", "स्त्री", "लड़की", "फीमेल"], ["பெண்", "பொண்ணு", "பெண்பால்"]),
               "other": (["other", "transgender"], ["अन्य", "ट्रांसजेंडर", "किन्नर"], ["மற்றவை", "திருநங்கை", "வேறு"])},
    "relation": {"wife": (["wife"], ["पत्नी", "बीवी", "घरवाली"], ["மனைவி"]), "husband": (["husband"], ["पति", "शौहर"], ["கணவர்", "கணவன்", "புருஷன்"]),
                 "son": (["son"], ["बेटा", "पुत्र", "लड़का"], ["மகன்", "பையன்"]), "daughter": (["daughter"], ["बेटी", "पुत्री", "लड़की"], ["மகள்", "பொண்ணு"]),
                 "father": (["father"], ["पिता", "पापा", "बाप"], ["தந்தை", "அப்பா"]), "mother": (["mother"], ["माता", "माँ", "मां", "अम्मा"], ["தாய்", "அம்மா"]),
                 "brother": (["brother"], ["भाई"], ["சகோதரன்", "அண்ணன்", "தம்பி"]), "sister": (["sister"], ["बहन"], ["சகோதரி", "அக்கா", "தங்கை"]),
                 "self": (["self", "myself"], ["खुद", "स्वयं", "मैं"], ["நானே", "சுயம்"]), "other": (["other"], ["अन्य", "दूसरा"], ["வேறு", "மற்றவர்"])},
    "caste_category": {"general": (["general"], ["सामान्य", "जनरल"], ["பொது", "ஜெனரல்"]), "sc": (["sc", "scheduled caste"], ["अनुसूचित जाति", "एससी", "एस सी"], ["எஸ்சி", "எஸ் சி", "பட்டியல் சாதி", "ஆதிதிராவிடர்"]),
                       "st": (["st", "scheduled tribe"], ["अनुसूचित जनजाति", "एसटी", "एस टी"], ["எஸ்டி", "எஸ் டி", "பழங்குடி"]),
                       "obc": (["obc", "backward"], ["ओबीसी", "ओ बी सी", "पिछड़ा"], ["ஓபிசி", "ஓ பி சி", "பிற்படுத்தப்பட்ட", "எம்பிசி", "பிசி"]),
                       "other": (["other", "others"], ["अन्य"], ["வேறு", "மற்றவை"])},
    "yesno": {"yes": (WORDS["yes"]["en"], WORDS["yes"]["hi"], WORDS["yes"]["ta"]), "no": (WORDS["no"]["en"], WORDS["no"]["hi"], WORDS["no"]["ta"])},
    "apy_pension_amount": {"1000": (["one thousand", "1000", "thousand"], ["एक हज़ार", "एक हजार", "हज़ार", "1000"], ["ஆயிரம்", "ஒரு ஆயிரம்", "1000"]),
                           "2000": (["two thousand", "2000"], ["दो हज़ार", "दो हजार", "2000"], ["இரண்டாயிரம்", "ரெண்டாயிரம்", "2000"]),
                           "3000": (["three thousand", "3000"], ["तीन हज़ार", "तीन हजार", "3000"], ["மூவாயிரம்", "மூணாயிரம்", "3000"]),
                           "4000": (["four thousand", "4000"], ["चार हज़ार", "चार हजार", "4000"], ["நான்காயிரம்", "நாலாயிரம்", "4000"]),
                           "5000": (["five thousand", "5000"], ["पाँच हज़ार", "पांच हजार", "पाँच हजार", "5000"], ["ஐயாயிரம்", "அஞ்சாயிரம்", "5000"])},
    "contribution_period": {"monthly": (["monthly", "every month", "month"], ["हर महीने", "मासिक", "महीने"], ["ஒவ்வொரு மாதமும்", "மாதம்", "மாதாந்திர"]),
                            "quarterly": (["quarterly", "three months"], ["हर तीन महीने", "तीन महीने", "तिमाही"], ["மூன்று மாதங்களுக்கு ஒருமுறை", "மூன்று மாதம்", "காலாண்டு"]),
                            "half_yearly": (["half yearly", "six months"], ["हर छह महीने", "छह महीने", "छमाही"], ["ஆறு மாதங்களுக்கு ஒருமுறை", "ஆறு மாதம்", "அரையாண்டு"])},
    "lpg_connection_type": {"5kg_single": (["single five", "single", "one cylinder"], ["पाँच किलो का एक", "एक सिलेंडर", "सिंगल"], ["ஐந்து கிலோ ஒரு", "ஒரு சிலிண்டர்", "சிங்கிள்"]),
                            "5kg_double": (["double five", "double", "two cylinders"], ["पाँच किलो के दो", "दो सिलेंडर", "डबल"], ["ஐந்து கிலோ இரண்டு", "இரண்டு சிலிண்டர்", "டபுள்"]),
                            "14kg": (["fourteen", "14", "big cylinder"], ["चौदह किलो", "चौदह", "बड़ा सिलेंडर"], ["பதினான்கு கிலோ", "பதினான்கு", "பெரிய சிலிண்டர்"])},
    "marital_status": {"married": (["married"], ["विवाहित", "शादीशुदा"], ["திருமணமானவர்", "திருமணம் ஆனவர்"]), "unmarried": (["unmarried", "single"], ["अविवाहित", "कुँवारा", "कुंवारी"], ["திருமணமாகாதவர்", "திருமணம் ஆகாதவர்"]),
                       "widowed": (["widow", "widowed", "widower"], ["विधवा", "विधुर"], ["விதவை", "விதவர்"]), "divorced": (["divorced", "separated"], ["तलाकशुदा", "तलाक"], ["விவாகரத்தானவர்", "விவாகரத்து"])},
    "vendor_type": {"stationary": (["fixed", "one place", "stationary"], ["एक ही जगह", "एक जगह", "स्थिर"], ["ஒரே இடம்", "ஒரே இடத்தில்", "நிலையான"]),
                    "mobile": (["move", "moving", "mobile", "around"], ["घूम", "घूम-घूम", "चलते-फिरते", "मोबाइल"], ["நடமாடி", "நடமாடும்", "சுற்றி"])},
    "loan_amount": {"10000": (["ten thousand", "10000"], ["दस हज़ार", "दस हजार", "10000"], ["பத்தாயிரம்", "10000"]), "20000": (["twenty thousand", "20000"], ["बीस हज़ार", "बीस हजार", "20000"], ["இருபதாயிரம்", "20000"]),
                    "50000": (["fifty thousand", "50000"], ["पचास हज़ार", "पचास हजार", "50000"], ["ஐம்பதாயிரம்", "50000"])},
    "kyc_document": {"aadhaar": (["aadhaar", "aadhar"], ["आधार"], ["ஆதார்"]), "voter_id": (["voter"], ["वोटर", "मतदाता"], ["வாக்காளர்", "ஓட்டர்"]),
                     "ration_card": (["ration"], ["राशन"], ["ரேஷன்"]), "driving_licence": (["driving", "licence", "license"], ["ड्राइविंग", "लाइसेंस"], ["ஓட்டுநர்", "லைசென்ஸ்"])},
}


def q(key, **kw):
    """Field question in the three languages (from the bank), with format placeholders kept."""
    en, hi, ta = Q[key]
    return {"en": en, "hi": hi, "ta": ta}


def field(key, ftype, required=True, question=None, label=None, **extra):
    f = {"key": key, "type": ftype, "required": required, "question": q(question or key)}
    lab = LABELS.get(label or key)
    if lab:
        f["label"] = {"en": lab[0], "hi": lab[1], "ta": lab[2]}
    f.update(extra)
    return f


def doc_field(key, ftype, doc, spoken_question, required=True, **extra):
    """A sensitive field read from a document the citizen shows (OCR); spoken fallback if that fails."""
    f = field(key, ftype, required=required, question=key if key in Q else spoken_question, **extra)
    f["source_document"] = doc
    f["document_name"] = {"en": DOCS[doc][0], "hi": DOCS[doc][1], "ta": DOCS[doc][2]}
    f["spoken_question"] = q(spoken_question)
    f["confirm"] = True
    return f


def choice(key, name, required=True, question=None, **extra):
    opts = CHOICES[name]
    return field(key, "choice", required=required, question=question or key,
                 options=[{"value": v, "en": s[0], "hi": s[1], "ta": s[2]} for v, s in opts.items()], **extra)


NOMINEE = [
    field("nominee_name", "name"),
    choice("nominee_relation", "relation"),
    field("nominee_dob", "date"),
    field("nominee_address", "address", required=False),
    field("guardian_name", "name", required=False, condition={"field": "nominee_dob", "minor": True}),
    field("guardian_relation", "text", required=False, condition={"field": "nominee_dob", "minor": True}),
]

FORMS = [
    {
        "form_id": "FORM_PMSBY_CONSENT", "scheme_id": "SCH_PMSBY",
        "title": {"en": "PM Suraksha Bima Yojana consent form", "hi": "प्रधानमंत्री सुरक्षा बीमा योजना सहमति फ़ॉर्म", "ta": "பிரதம மந்திரி சுரக்ஷா பீமா யோஜனா ஒப்புதல் படிவம்"},
        "spoken_name": {"en": "PM Suraksha Bima Yojana", "hi": "प्रधानमंत्री सुरक्षा बीमा योजना", "ta": "பிரதம மந்திரி சுரக்ஷா பீமா யோஜனா"},
        "title_patterns": {"en": ["pradhan mantri suraksha bima yojana", "suraksha bima yojana", "pmsby", "consent cum declaration form"],
                           "hi": ["प्रधानमंत्री सुरक्षा बीमा योजना", "सुरक्षा बीमा योजना", "सहमति-सह-घोषणा"],
                           "ta": ["பிரதம மந்திரி சுரக்ஷா பீமா யோஜனா", "சுரக்ஷா பீமா"]},
        "keywords": {"suraksha": 3.0, "accident": 1.5, "accidental": 1.5, "rupees twenty": 1.5, "disability": 0.5, "master policy": 0.5, "pmsby": 3.0, "सुरक्षा बीमा": 3.0, "दुर्घटना": 1.5, "बीस रुपये": 1.0, "சுரக்ஷா": 3.0, "விபத்து": 1.5},
        "discriminators": {"against": {"FORM_PMJJBY_CONSENT": ["suraksha", "accident"]}},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/pmsby_en.pdf (jansuraksha.gov.in)",
        "fields": [
            field("full_name", "name"), field("father_or_husband_name", "name"), field("address", "address"), field("village_or_town", "text"),
            field("district", "text"), field("state", "text"), field("pincode", "pincode"), field("mobile", "phone"),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            doc_field("ifsc", "ifsc", "bank_passbook", "ifsc", required=False),
            choice("kyc_document", "kyc_document"),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"),
            field("dob", "date"), field("email", "email", required=False),
            choice("disability", "yesno"), field("disability_details", "text", required=False, condition={"field": "disability", "equals": "yes"}),
        ] + NOMINEE,
    },
    {
        "form_id": "FORM_PMJJBY_CONSENT", "scheme_id": "SCH_PMJJBY",
        "title": {"en": "PM Jeevan Jyoti Bima Yojana consent form", "hi": "प्रधानमंत्री जीवन ज्योति बीमा योजना सहमति फ़ॉर्म", "ta": "பிரதம மந்திரி ஜீவன் ஜோதி பீமா யோஜனா ஒப்புதல் படிவம்"},
        "spoken_name": {"en": "PM Jeevan Jyoti Bima Yojana", "hi": "प्रधानमंत्री जीवन ज्योति बीमा योजना", "ta": "பிரதம மந்திரி ஜீவன் ஜோதி பீமா யோஜனா"},
        "title_patterns": {"en": ["pradhan mantri jeevan jyoti bima yojana", "jeevan jyoti bima yojana", "pmjjby", "consent cum declaration form"],
                           "hi": ["प्रधान मंत्री जीवन ज्योति बीमा योजना", "जीवन ज्योति बीमा योजना", "जीवन ज्योति"],
                           "ta": ["பிரதம மந்திரி ஜீவன் ஜோதி பீமா யோஜனா", "ஜீவன் ஜோதி"]},
        "keywords": {"jeevan": 3.0, "jyoti": 3.0, "life insurance": 1.5, "436": 1.5, "lien period": 1.0, "pmjjby": 3.0, "जीवन ज्योति": 3.0, "जीवन बीमा": 1.0, "ஜீவன் ஜோதி": 3.0, "ஆயுள் காப்பீடு": 1.0},
        "discriminators": {"against": {"FORM_PMSBY_CONSENT": ["jeevan", "jyoti", "436"]}},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/pmjjby_en.pdf (jansuraksha.gov.in)",
        "fields": [
            field("full_name", "name"), field("father_or_husband_name", "name"), field("address", "address"), field("village_or_town", "text"),
            field("district", "text"), field("state", "text"), field("pincode", "pincode"), field("mobile", "phone"),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            doc_field("ifsc", "ifsc", "bank_passbook", "ifsc", required=False),
            choice("kyc_document", "kyc_document"),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"),
            field("dob", "date"), field("email", "email", required=False),
        ] + NOMINEE + [field("nominee_mobile", "phone", required=False)],
    },
    {
        "form_id": "FORM_APY_REGISTRATION", "scheme_id": "SCH_APY",
        "title": {"en": "Atal Pension Yojana subscriber registration form", "hi": "अटल पेंशन योजना अभिदाता पंजीकरण फ़ॉर्म", "ta": "அடல் ஓய்வூதிய திட்டம் சந்தாதாரர் பதிவுப் படிவம்"},
        "spoken_name": {"en": "Atal Pension Yojana", "hi": "अटल पेंशन योजना", "ta": "அடல் ஓய்வூதிய திட்டம்"},
        "title_patterns": {"en": ["atal pension yojana", "apy subscriber registration form", "subscriber registration form", "national pension system"],
                           "hi": ["अटल पेंशन योजना", "अटल पेंशन योजना अभिदाता पंजीकरण"],
                           "ta": ["அடல் ஓய்வூதிய திட்டம்", "atal pension yojana"]},
        "keywords": {"atal": 3.0, "pension": 1.0, "apy": 2.0, "apy account": 2.0, "pran": 1.5, "pfrda": 1.5, "national pension system": 2.0, "nps": 1.0, "pension amount": 1.5, "guaranteed pension": 1.5, "contribution amount": 1.0, "subscriber registration": 1.5, "अटल": 3.0, "पेंशन": 1.0, "अभिदाता": 1.5, "एपीवाई": 2.0, "அடல்": 3.0, "ஓய்வூதிய": 1.5, "சந்தாதாரர்": 1.5},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/apy_en.pdf, apy_hi.pdf, apy_ta.pdf (jansuraksha.gov.in)",
        "fields": [
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            field("bank_name", "text"), field("branch_name", "text", required=False),
            field("full_name", "name"), field("dob", "date"), field("mobile", "phone"), field("email", "email", required=False),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"),
            choice("married", "yesno"), field("spouse_name", "name", required=False, condition={"field": "married", "equals": "yes"}),
            field("nominee_name", "name"), choice("nominee_relation", "relation"),
            field("nominee_dob", "date", required=False), field("guardian_name", "name", required=False, condition={"field": "nominee_dob", "minor": True}),
            choice("other_social_scheme", "yesno"), choice("income_tax_payer", "yesno"),
            choice("apy_pension_amount", "apy_pension_amount"), choice("contribution_period", "contribution_period"),
        ],
    },
    {
        "form_id": "FORM_PMUY_KYC", "scheme_id": "SCH_PMUY",
        "title": {"en": "PM Ujjwala Yojana KYC application form", "hi": "प्रधानमंत्री उज्ज्वला योजना केवाईसी आवेदन फ़ॉर्म", "ta": "பிரதம மந்திரி உஜ்ஜ்வலா யோஜனா கேஒய்சி விண்ணப்பப் படிவம்"},
        "spoken_name": {"en": "PM Ujjwala Yojana", "hi": "प्रधानमंत्री उज्ज्वला योजना", "ta": "பிரதம மந்திரி உஜ்ஜ்வலா யோஜனா"},
        "title_patterns": {"en": ["pradhan mantri ujjwala yojana", "ujjwala kyc application", "ujjwala kyc application form"],
                           "hi": ["प्रधानमंत्री उज्ज्वला योजना", "उज्ज्वला योजना केवाईसी"],
                           "ta": ["பிரதம மந்திரி உஜ்ஜ்வலா யோஜனா"]},
        "keywords": {"ujjwala": 3.0, "lpg": 2.0, "kyc application": 1.5, "ioc/ bpc/ hpc": 1.5, "petroleum": 1.5, "natural gas": 1.0, "cylinder": 1.0, "consumer details": 1.0, "उज्ज्वला": 3.0, "एलपीजी": 1.5, "केवाईसी": 1.0, "உஜ்ஜ்வலா": 3.0},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/pmuy_kyc_en.pdf (pmuy.gov.in, v8)",
        "fields": [
            choice("is_migrant", "yesno"), field("full_name", "name"), field("dob", "date"),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"),
            choice("caste_category", "caste_category"),
            field("address", "address"), field("village_or_town", "text"), field("district", "text"), field("state", "text"), field("pincode", "pincode"),
            field("mobile", "phone"), field("email", "email", required=False),
            field("household_adults", "integer", validation={"min": 1, "max": 12}),
            {"key": "household_members", "type": "group", "required": True, "count_from": "household_adults", "max_items": 6,
             "label": {"en": LABELS["household_members"][0], "hi": LABELS["household_members"][1], "ta": LABELS["household_members"][2]},
             "question": q("member_name"),
             "fields": [field("name", "name", question="member_name", label="full_name"), choice("relation", "relation", question="member_relation", label="nominee_relation"),
                        choice("gender", "gender", question="member_gender", label="gender"), field("age", "integer", question="member_age", label="age", validation={"min": 18, "max": 110})]},
            doc_field("ration_card", "text", "ration_card", "ration_card_spoken", required=False), field("ration_card_state", "text", required=False),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            field("bank_name", "text"), field("branch_name", "text", required=False), doc_field("ifsc", "ifsc", "bank_passbook", "ifsc", required=False),
            choice("lpg_connection_type", "lpg_connection_type"),
        ],
    },
    {
        "form_id": "FORM_PMSVANIDHI_LAF", "scheme_id": "SCH_SVANIDHI",
        "title": {"en": "PM SVANidhi loan application form", "hi": "पीएम स्वनिधि ऋण आवेदन फ़ॉर्म", "ta": "பிஎம் ஸ்வநிதி கடன் விண்ணப்பப் படிவம்"},
        "spoken_name": {"en": "PM SVANidhi street vendor loan", "hi": "पीएम स्वनिधि", "ta": "பிஎம் ஸ்வநிதி"},
        "title_patterns": {"en": ["pm street vendor's atmanirbhar nidhi", "pm svanidhi", "svanidhi", "common loan application form", "street vendor"],
                           "hi": ["पीएम स्वनिधि", "स्वनिधि", "स्ट्रीट वेंडर"], "ta": ["ஸ்வநிதி", "pm svanidhi"]},
        "keywords": {"svanidhi": 3.0, "street vendor": 2.0, "vending": 2.0, "atmanirbhar nidhi": 2.0, "loan application": 1.0, "ulb": 1.0, "lender": 0.5, "स्वनिधि": 3.0, "रेहड़ी": 1.5, "ஸ்வநிதி": 3.0, "தெருவோர": 1.5},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/pmsvanidhi_laf.pdf (pmsvanidhi.mohua.gov.in)",
        "fields": [
            field("full_name", "name"), field("father_or_spouse_name", "name"), field("dob", "date"), choice("marital_status", "marital_status"),
            field("mobile", "phone"), choice("gender", "gender"), choice("caste_category", "caste_category"),
            choice("disability", "yesno"), choice("minority", "yesno"),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"), doc_field("voter_id", "text", "voter_id", "voter_id", required=False),
            field("family_member_name", "name", label="family_member"), choice("family_member_relation", "relation", question="member_relation", label="nominee_relation"),
            field("family_member_age", "integer", question="member_age", label="age", validation={"min": 0, "max": 110}),
            field("address", "address"), field("village_or_town", "text"), field("district", "text"), field("state", "text"), field("pincode", "pincode"),
            field("vending_activity", "text"), field("vending_place", "text"), field("vending_years", "integer", validation={"min": 0, "max": 80}),
            choice("vendor_type", "vendor_type"), field("monthly_sales", "amount", validation={"min": 0, "max": 10000000}),
            field("bank_name", "text"), field("branch_name", "text", required=False), doc_field("ifsc", "ifsc", "bank_passbook", "ifsc", required=False),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            choice("loan_amount", "loan_amount"), field("loan_purpose", "text", required=False),
        ],
    },
    {
        "form_id": "FORM_PMKISAN_SELF_DECLARATION", "scheme_id": "SCH_PMKISAN",
        "title": {"en": "PM-KISAN self declaration form", "hi": "पीएम-किसान स्व-घोषणा फ़ॉर्म", "ta": "பிஎம்-கிசான் சுய அறிவிப்புப் படிவம்"},
        "spoken_name": {"en": "PM Kisan Samman Nidhi", "hi": "पीएम किसान सम्मान निधि", "ta": "பிஎம் கிசான் சம்மான் நிதி"},
        "title_patterns": {"en": ["pm-kisan", "pm kisan", "self declaration form", "pradhan mantri kisan samman nidhi", "kisan samman nidhi"],
                           "hi": ["प्रधानमंत्री किसान सम्मान निधि", "पीएम-किसान", "स्व घोषणा", "किसान सम्मान"], "ta": ["பிஎம்-கிசான்", "கிசான் சம்மான் நிதி"]},
        "keywords": {"kisan": 3.0, "samman nidhi": 2.0, "self declaration": 2.0, "farmer": 1.0, "pm-kisan": 3.0, "khata": 0.5, "किसान": 3.0, "सम्मान निधि": 2.0, "स्व घोषणा": 1.5, "किसान सम्मान": 2.0, "கிசான்": 3.0, "சம்மான்": 1.5},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/pmkisan_selfdecl.pdf (central format, scanned)",
        "fields": [
            field("full_name", "name"), field("father_name", "name"), field("gender", "choice", question="gender", options=[{"value": v, "en": s[0], "hi": s[1], "ta": s[2]} for v, s in CHOICES["gender"].items()]),
            choice("caste_category", "caste_category"), field("dob", "date"),
            field("village_or_town", "text"), field("district", "text"), field("state", "text"), field("pincode", "pincode", required=False),
            field("mobile", "phone"), doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"),
            field("bank_name", "text"), doc_field("ifsc", "ifsc", "bank_passbook", "ifsc", required=False),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken"),
            field("land_village", "text"), field("land_survey_number", "text", required=False), field("land_area", "number", validation={"min": 0, "max": 1000}),
            choice("income_tax_payer", "yesno"), choice("govt_employee_family", "yesno"),
        ],
    },
    {
        "form_id": "FORM_MGNREGA_REGISTRATION", "scheme_id": "SCH_MGNREGA",
        "title": {"en": "MGNREGA application for registration (job card)", "hi": "मनरेगा पंजीकरण आवेदन (जॉब कार्ड)", "ta": "எம்ஜிஎன்ஆர்இஜிஏ பதிவு விண்ணப்பம் (வேலை அட்டை)"},
        "spoken_name": {"en": "MGNREGA job card", "hi": "मनरेगा जॉब कार्ड", "ta": "நூறு நாள் வேலை அட்டை"},
        "title_patterns": {"en": ["application for registration under mgnregs", "mahatma gandhi national rural employment guarantee act", "national rural employment guarantee", "detail format for application for registration"],
                           "hi": ["महात्मा गांधी राष्ट्रीय ग्रामीण रोजगार गारंटी", "मनरेगा पंजीकरण आवेदन"], "ta": ["மகாத்மா காந்தி தேசிய ஊரக வேலை உறுதி"]},
        "keywords": {"mgnrega": 3.0, "mgnregs": 3.0, "employment guarantee": 2.0, "gram panchayat": 1.5, "job card": 1.5, "willing to work": 1.5, "head of household": 1.0, "operational guidelines": 1.0, "मनरेगा": 3.0, "रोजगार गारंटी": 2.0, "ग्राम पंचायत": 1.0, "வேலை உறுதி": 2.0, "கிராம பஞ்சாயத்து": 1.0},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/mgnrega_registration.pdf (Operational Guidelines 2013, Annexure 3)",
        "fields": [
            field("workers_count", "integer", validation={"min": 1, "max": 8}),
            {"key": "workers", "type": "group", "required": True, "count_from": "workers_count", "max_items": 6,
             "label": {"en": LABELS["workers"][0], "hi": LABELS["workers"][1], "ta": LABELS["workers"][2]}, "question": q("worker_name"),
             "fields": [field("name", "name", question="worker_name", label="full_name"), field("age", "integer", question="member_age", label="age", validation={"min": 18, "max": 100}),
                        choice("gender", "gender", question="member_gender", label="gender")]},
            field("house_number", "text", required=False), field("village_or_town", "text"), field("gram_panchayat", "text"), field("block", "text"),
            field("head_of_household", "name"), choice("relation_to_head", "relation"), field("father_or_husband_name", "name"),
            choice("disability", "yesno"), choice("caste_category", "caste_category"), choice("minority", "yesno"), choice("bpl", "yesno"),
            doc_field("voter_id", "text", "voter_id", "voter_id", required=False),
            doc_field("account_number", "account_number", "bank_passbook", "account_number_spoken", required=False),
            doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken", required=False), field("mobile", "phone", required=False),
        ],
    },
    {
        "form_id": "FORM_SSA1_ACCOUNT_OPENING", "scheme_id": "SCH_SSY",
        "title": {"en": "Sukanya Samriddhi Account opening form (SSA-1)", "hi": "सुकन्या समृद्धि खाता खोलने का फ़ॉर्म (एसएसए-1)", "ta": "சுகன்யா சம்ரிதி கணக்கு தொடங்கும் படிவம் (எஸ்எஸ்ஏ-1)"},
        "spoken_name": {"en": "Sukanya Samriddhi account", "hi": "सुकन्या समृद्धि खाता", "ta": "சுகன்யா சம்ரிதி கணக்கு"},
        "title_patterns": {"en": ["sukanya samriddhi account scheme", "sukanya samriddhi", "application for opening an account"],
                           "hi": ["सुकन्या समृद्धि खाता", "सुकन्या समृद्धि योजना"], "ta": ["சுகன்யா சம்ரிதி கணக்கு"]},
        "keywords": {"sukanya": 3.0, "samriddhi": 3.0, "girl child": 1.5, "depositor": 1.5, "postmaster": 1.0, "initial deposit": 1.0, "ssa-1": 1.5, "सुकन्या": 3.0, "समृद्धि": 2.0, "சுகன்யா": 3.0},
        "discriminators": {},
        "supported_languages": ["hi", "ta", "en"], "source": "sources/ssa1_centralbank.pdf (Sukanya Samriddhi Account Scheme 2019, Form-1)",
        "fields": [
            field("girl_name", "name"), field("girl_dob", "date"), field("guardian_full_name", "name"), field("father_or_husband_name", "name"),
            field("guardian_dob", "date"), doc_field("aadhaar", "aadhaar", "aadhaar", "aadhaar_spoken"), doc_field("pan", "text", "pan", "pan", required=False),
            field("address", "address"), field("village_or_town", "text"), field("district", "text"), field("state", "text"), field("pincode", "pincode"),
            choice("same_permanent_address", "yesno"), field("permanent_address", "address", required=False, condition={"field": "same_permanent_address", "equals": "no"}),
            field("mobile", "phone"), field("email", "email", required=False),
            doc_field("birth_certificate", "text", "birth_certificate", "birth_certificate", required=False),
            field("initial_deposit", "amount", validation={"min": 250, "max": 150000}),
        ],
    },
]


def main():
    out = {"version": 2, "built": "2026-09-13", "languages": LANGS, "prompts": PROMPTS, "words": WORDS, "forms": FORMS}
    # sanity: every question has all languages, keys unique per form
    for f in FORMS:
        keys = [x["key"] for x in f["fields"]]
        assert len(keys) == len(set(keys)), f"duplicate keys in {f['form_id']}"
        for x in f["fields"]:
            assert all(l in x["question"] for l in LANGS), (f["form_id"], x["key"])
    path = os.path.join(HERE, "forms.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_fields = sum(len(f["fields"]) for f in FORMS)
    print(f"wrote {path}: {len(FORMS)} forms, {n_fields} fields, {len(PROMPTS)} prompts, {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
