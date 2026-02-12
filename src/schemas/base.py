from src.schemas.compat import BaseModelLike, PYDANTIC_AVAILABLE, PydanticBaseModel


class BaseEvent(BaseModelLike):
    schema_version: str = "v1"

    if PYDANTIC_AVAILABLE and PydanticBaseModel and issubclass(BaseModelLike, PydanticBaseModel):
        class Config:
            extra = "allow"
            validate_assignment = True
