# 외부 라이브러리 함수는 별칭 import — 정적분석기가 동명 정적메서드의 위임 호출을 자기재귀로 오인하지 않도록 함
from convert_case import camel_case as _camel_case, pascal_case as _pascal_case, snake_case as _snake_case
from pluralizer import Pluralizer
from project_generator.utils.catchable_exceptions import CATCHABLE_EXCEPTIONS

pluralizer = Pluralizer()
class CaseConvertUtil:
    @staticmethod
    def camel_case(text: str) -> str:
        try:
            return _camel_case(text)
        except CATCHABLE_EXCEPTIONS as e:
            words = text.replace('-', ' ').replace('_', ' ').split()
            if not words:
                return text
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    
    @staticmethod
    def pascal_case(text: str) -> str:
        try:
            return _pascal_case(text)
        except CATCHABLE_EXCEPTIONS as e:
            words = text.replace('-', ' ').replace('_', ' ').split()
            if not words:
                return text
            return ''.join(word.capitalize() for word in words)
    
    @staticmethod
    def snake_case(text: str) -> str:
        try:
            return _snake_case(text)
        except CATCHABLE_EXCEPTIONS as e:
            return text.replace('-', '_')

    @staticmethod
    def plural(text: str) -> str:
        try:
            return pluralizer.plural(_camel_case(text))
        except CATCHABLE_EXCEPTIONS as e:
            try:
                camel = CaseConvertUtil.camel_case(text)
                if camel.endswith('y'):
                    return camel[:-1] + 'ies'
                elif camel.endswith(('s', 'x', 'z', 'ch', 'sh')):
                    return camel + 'es'
                else:
                    return camel + 's'
            except CATCHABLE_EXCEPTIONS:
                return text