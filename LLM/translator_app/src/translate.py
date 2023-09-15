import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from transformers import pipeline
from enum import Enum
# Pipeline
from trans_pipeline import TranslatePipeline,TranslationAPiOutput,ErrorOutput


class ETranslationLanguages(Enum):

    ENGLISH2JAPANESE= "JP"
    JAPANESE2ENGLISH= "EN"

def translate(language:ETranslationLanguages,text:str,model,tokenizer):
    trans_pipeline = TranslatePipeline()
    translationoutput= TranslationAPiOutput()
    erroroutput = ErrorOutput()

    try:
        # english→japanese
        if language == ETranslationLanguages.ENGLISH2JAPANESE.value:
            translator = pipeline('translation', model=model, tokenizer=tokenizer,src_lang="eng_Latn", tgt_lang="jpn_Jpan")

            translatedtext  = trans_pipeline.translate_txt(translator,text)
            print(translatedtext)
            # JSON Serialization
            translationoutput.translation = translatedtext 
            # JSON Serialization/Return value as res_json
            res_json = translationoutput.tojson()
   
            return res_json

        # japanese→english
        elif language == ETranslationLanguages.JAPANESE2ENGLISH.value:

            translator = pipeline('translation', model=model, tokenizer=tokenizer, src_lang="jpn_Jpan", tgt_lang="eng_Latn")
            translatedtext  = trans_pipeline.translate_txt(translator,text)
            # JSON Serialization
            translationoutput.translation = translatedtext
            # JSON Serialization/Return value as res_json
            res_json = translationoutput.tojson()

            return res_json

        else:
            # JSON Serialization
            erroroutput.error_code = 422
            erroroutput.error_message = "Language not found, languages supported [EN,JP]"
            res_json = erroroutput.tojson()

            return res_json

    except Exception as errormessage:
        erroroutput.error_code = 422
        erroroutput.error_message = str(errormessage)
        res_json = erroroutput.tojson()

        return res_json