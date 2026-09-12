"""
Assisted form filling (the camera's primary use): identify the paper form the
citizen shows (tesseract OCR + the local form catalogue, no language model), ask
the form's questions one by one in the selected language through the existing
TTS/ASR, validate and keep the answers in memory, seal the completed form and
hand it to the fixed office PC over the local network. See formfill/README in
docs/ and the module docstrings.
"""
