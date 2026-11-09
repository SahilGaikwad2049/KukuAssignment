"""Validated response models shared by episode analysis and event discovery."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HookCategory = Literal[
    "confrontation", "revelation", "mystery", "threat", "betrayal",
    "emotional_shock", "power_reversal", "unexpected_action", "visual_intrigue",
]
EndingCategory = Literal[
    "unanswered_question", "unrevealed_identity", "unexpected_discovery",
    "imminent_danger", "interrupted_revelation", "unresolved_confrontation",
    "uncertain_outcome",
]


class DiscoveredEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str
    role: Literal["hook", "potential_ending"]
    category: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    narrative_context: str

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        if value not in set(HookCategory.__args__) | set(EndingCategory.__args__):
            raise ValueError(f"Unsupported event category: {value}")
        return value

    @model_validator(mode="after")
    def category_matches_role(self):
        hook_categories = set(HookCategory.__args__)
        ending_categories = set(EndingCategory.__args__)
        allowed = hook_categories if self.role == "hook" else ending_categories
        if self.category not in allowed:
            raise ValueError(f"Category {self.category!r} does not match role {self.role!r}")
        return self


class EpisodeAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    episode_summary: str
    important_characters: list[str] = Field(default_factory=list)
    character_relationships: list[str] = Field(default_factory=list)
    main_conflict: str
    important_story_developments: list[str] = Field(default_factory=list)
    major_revelations: list[str] = Field(default_factory=list)
    emotional_turning_points: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    narrative_progression: list[str] = Field(default_factory=list)


class EpisodeDiscoveryResponse(EpisodeAnalysis):
    potential_events: list[DiscoveredEvent] = Field(default_factory=list)


class AnalysisEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_analysis: EpisodeAnalysis
    potential_events: list[DiscoveredEvent] = Field(default_factory=list)

