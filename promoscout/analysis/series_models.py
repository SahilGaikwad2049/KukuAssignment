"""Pydantic schemas for source-grounded, cross-episode narrative analysis."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    event_id: str


class RelationshipChange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    description: str
    source_references: list[SourceReference] = Field(min_length=1)


class CharacterRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")
    other_character: str
    relationship: str
    source_references: list[SourceReference] = Field(default_factory=list)
    changes: list[RelationshipChange] = Field(default_factory=list)


class CharacterProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    role: str
    source_references: list[SourceReference] = Field(default_factory=list)
    relationships: list[CharacterRelationship] = Field(default_factory=list)


class StoryDevelopment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    development: str
    development_type: str
    source_references: list[SourceReference] = Field(min_length=1)
    spoiler_level: Literal["low", "moderate", "major", "unknown"] = "unknown"


class ContinuityIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    source_references: list[SourceReference] = Field(min_length=1)


class PromoBeat(BaseModel):
    model_config = ConfigDict(extra="forbid")
    episode_number: int
    anchor_event_ids: list[str] = Field(min_length=1)
    beat_role: Literal["hook", "setup", "escalation", "revelation", "ending"]
    reason: str
    spoiler_level: Literal["low", "moderate", "major", "unknown"] = "unknown"


class CompositePromoOpportunity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    story_arc_id: str
    narrative_theme: str
    narrative_rationale: str
    beats: list[PromoBeat] = Field(min_length=2)
    hook_beat_index: int = Field(ge=0)
    ending_beat_index: int = Field(ge=0)
    spoiler_warnings: list[str] = Field(default_factory=list)
    uncertainty_warnings: list[str] = Field(default_factory=list)


class StoryArc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    story_arc_id: str
    title: str
    overview: str
    characters: list[str] = Field(default_factory=list)
    developments: list[StoryDevelopment] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    spoiler_sensitive_developments: list[StoryDevelopment] = Field(default_factory=list)
    continuity_uncertainties: list[ContinuityIssue] = Field(default_factory=list)


class SeriesAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analyzed_episode_numbers: list[int]
    series_synopsis: str
    character_registry: list[CharacterProfile] = Field(default_factory=list)
    story_arcs: list[StoryArc] = Field(default_factory=list)
    episode_developments: list[StoryDevelopment] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    chronological_narrative_structure: list[StoryDevelopment] = Field(default_factory=list)
    spoiler_sensitive_developments: list[StoryDevelopment] = Field(default_factory=list)
    continuity_uncertainties: list[ContinuityIssue] = Field(default_factory=list)
    composite_promo_opportunities: list[CompositePromoOpportunity] = Field(default_factory=list)
