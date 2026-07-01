import json

from .logging_util import LoggingUtil

class JsonUtil:
    @staticmethod
    def convert_to_json(data: any, indent: int = 4) -> str:
        # 데이터 변환 함수
        def convert_data(item):
            try :
                
                if hasattr(item, 'model_dump_json'):
                    return json.loads(item.model_dump_json())
                elif isinstance(item, dict):
                    return {k: convert_data(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [convert_data(i) for i in item]
                else:
                    return item if isinstance(item, (str, int, float, bool, type(None))) else str(item)
                
            except Exception as e:
                LoggingUtil.exception("json_util", f"Error converting to json", e)
                return item
        
        # 데이터 타입에 따른 처리
        if isinstance(data, list):
            processed_data = [convert_data(item) for item in data]
            json_data = json.dumps(processed_data, indent=indent, ensure_ascii=False)
        # Pydantic BaseModel인 경우
        elif hasattr(data, 'model_dump_json'):
            json_data = data.model_dump_json(indent=indent)
        # 딕셔너리인 경우
        elif isinstance(data, dict):
            processed_data = convert_data(data)
            json_data = json.dumps(processed_data, indent=indent, ensure_ascii=False)
        # 그 외의 경우
        else:
            json_data = str(data)

        return json_data

    @staticmethod
    def convert_to_dict(json_str: str) -> dict:
        try :     
            return json.loads(json_str)
        except Exception as e:
            LoggingUtil.exception("json_util", f"Error converting to dict", e)
            return {}

