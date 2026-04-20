""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.
'''
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Bengali character baseline. This keeps the first Bengali model simple:
# text is represented as Unicode characters, not phonemes.
_bengali_letters = 'অআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়ৎ'
_bengali_marks = 'ািীুূৃৄেৈোৌ্ৎংঃঁ়ৗৢৣ'
_bengali_digits = '০১২৩৪৫৬৭৮৯'
_bengali_punctuation = '।॥‘’“”—–-()[]{}0123456789'
_bengali_block = ''.join(chr(codepoint) for codepoint in range(0x0980, 0x0A00))


# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

for symbol in list(_bengali_letters) + list(_bengali_marks) \
    + list(_bengali_digits) + list(_bengali_punctuation) \
    + list(_bengali_block):
  if symbol not in symbols:
    symbols.append(symbol)

# Special symbol ids
SPACE_ID = symbols.index(" ")
