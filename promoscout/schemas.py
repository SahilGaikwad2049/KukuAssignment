from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

EventType = Literal["dialogue", "action", "camera", "sound_fx", "other"]

class ScreenplayEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    event_type: EventType
    label: str
    speaker: str | None = None
    delivery: str | None = None
    start_timestamp: str
    end_timestamp: str
    start_seconds: float
    end_seconds: float
    content: str
    raw_block: str
    source_line_start: int
    source_line_end: int

    @model_validator(mode="after")
    def check_times(self):
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("Event timestamps must be non-negative and end >= start")
        return self

class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scene_number: int
    heading: str | None = None
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    events: list[ScreenplayEvent] = Field(default_factory=list)

class ParsedEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    logline: str | None = None
    continuity_notes: str | None = None
    source_file: str | None = None
    scenes: list[Scene] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def events(self) -> list[ScreenplayEvent]:
        return [event for scene in self.scenes for event in scene.events]
