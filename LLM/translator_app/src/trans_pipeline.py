from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

class TranslatePipeline():
    # model load
    def models(self):
        checkpoint = "facebook/nllb-200-distilled-600M"
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        return model,tokenizer

    def translate_txt(self,translator,text:str) ->str:
        translate = translator(text)
        # fetch from a list
        translate_list = translate[0]
        #Delete elements by specifying dictionary keys and get only translation results
        translate_select = translate_list.pop("translation_text")

        return translate_select

# class inheritance
class ApiResponse():
    # JSON Serialization
    def tojson(self):
        return json.dumps(self, default=lambda o: o.__dict__,sort_keys=True, indent=4,ensure_ascii=False)

class TranslationAPiOutput(ApiResponse):
    def __init__(self) ->None:
        self.translation = ""

class ErrorOutput(ApiResponse):
    def __init__(self) ->None:
        self.error_code = 422
        self.error_message  = ""

