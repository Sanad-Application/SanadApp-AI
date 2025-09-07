import os

class TemplateParser:

    def __init__(self, lang: str=None, default_lang='ar'):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_lang = default_lang
        self.lang = None

        self.set_language(lang)
    
    def set_language(self, lang: str):
        if lang:
            language_path = os.path.join(self.current_path, "locales", lang)
            if os.path.exists(language_path):
                self.lang = lang
                return
        # fallback
        self.lang = self.default_lang

    def get(self, group: str, key: str, vars: dict={}):
        if not group or not key:
            return None
        
        group_path = os.path.join(self.current_path, "locales", self.lang, f"{group}.py" )
        targeted_lang = self.lang
        if not os.path.exists(group_path):
            group_path = os.path.join(self.current_path, "locales", self.default_lang, f"{group}.py" )
            targeted_lang = self.default_lang

        if not os.path.exists(group_path):
            return None
        
        # import group module
        module = __import__(f"stores.LLM.templates.locales.{targeted_lang}.{group}", fromlist=[group])

        if not module:
            return None
        
        key_attribute = getattr(module, key)
        return key_attribute.substitute(vars)