import warnings
warnings.simplefilter('ignore')

import whisper

model = whisper.load_model("small")
audio_file = ""
result = model.transcribe(audio_file)
translate_result = model.transcribe(audio_file,task="translate")

if result["language"] == "ja":
  lan = "Japanese"
  print(lan)
else:
  print(result["language"])

print(result["text"])
print("")
print("English")
print(translate_result["text"])