from convert_case import camel_case, pascal_case, snake_case
from pluralizer import Pluralizer

pluralizer = Pluralizer()
class CaseConvertUtil:
    @staticmethod
    def camel_case(text: str) -> str:
        try:
            return camel_case(text)
        except Exception as e:
            words = text.replace('-', ' ').replace('_', ' ').split()
            if not words:
                return text
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    @staticmethod
    def pascal_case(text: str) -> str:
        try:
            return pascal_case(text)
        except Exception as e:
            words = text.replace('-', ' ').replace('_', ' ').split()
            if not words:
                return text
            return ''.join(word.capitalize() for word in words)
    
    @staticmethod
    def snake_case(text: str) -> str:
        try:
            return snake_case(text)
        except Exception as e:
            return text.replace('-', '_')

    @staticmethod
    def plural(text: str) -> str:
        try:
            return pluralizer.plural(camel_case(text))
        except Exception as e:
            try:
                camel = CaseConvertUtil.camel_case(text)
                if camel.endswith('y'):
                    return camel[:-1] + 'ies'
                elif camel.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    return camel + 'es'
                else:
                    return camel + 's'
            except:
                return text