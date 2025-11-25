from pydantic import BaseModel, Field, field_validator
from collections.abc import MutableMapping
import json

class Md(BaseModel, MutableMapping):
    key1: str = Field(default='')
    key2: int = Field(default=0)
    key3: float = Field(default=0.0)

    class Config:
        extra = 'allow'
        validate_assignment = True

    @field_validator('*', mode='before')
    def check_serializable(cls, value, info):
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value for {info.field_name} is not serializable: {e}")
        return value

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __len__(self):
        return len(self.__dict__) + len(self.__pydantic_extra__)

    def __iter__(self):
        return iter(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__