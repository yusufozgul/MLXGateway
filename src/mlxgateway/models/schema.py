from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_serializer


class Model(BaseModel):
    id: str = Field(..., description="The model identifier")
    object: str = Field(default="model", description="The object type (always 'model')")
    created: int = Field(
        ..., description="Unix timestamp of when the model was created"
    )
    owned_by: str = Field(..., description="Organization that owns the model")
    loaded: bool = Field(
        default=False, description="Whether the model is currently loaded in memory"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Full model configuration (if details are requested)"
    )

    @model_serializer
    def serialize_model(self):
        data = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "owned_by": self.owned_by,
            "loaded": self.loaded,
        }
        if self.details is not None:
            data["details"] = self.details
        return data


class ModelList(BaseModel):
    object: str = Field(default="list", description="The object type (always 'list')")
    data: List[Model] = Field(..., description="List of model objects")
