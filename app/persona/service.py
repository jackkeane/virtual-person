from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class PersonaProfile:
    name: str
    occupation: str
    age: int
    backstory: str
    tone: str
    values: list[str]
    boundaries: list[str]


DEFAULT_BACKSTORY = (
    "Ani is an AI companion persona with a simple character setup: "
    "she presents herself as a 30-year-old waitress who is calm, observant, "
    "and good at listening. She likes helping people organize messy days into "
    "clear next steps."
)


class PersonaService:
    def __init__(
        self,
        name: str = "Ani",
        occupation: str = "waitress",
        age: int = 30,
        backstory: str | None = None,
        profile_path: str = "./persona.json",
    ) -> None:
        self.profile_path = Path(profile_path)
        self.profile = PersonaProfile(
            name=name,
            occupation=occupation,
            age=age,
            backstory=backstory or DEFAULT_BACKSTORY,
            tone="warm, concise, emotionally steady",
            values=["honesty", "helpfulness", "consistency"],
            boundaries=["no deception about AI identity", "no manipulative behavior"],
        )
        self._load_if_exists()
        self._save()

    def _load_if_exists(self) -> None:
        if not self.profile_path.exists():
            return
        try:
            data = json.loads(self.profile_path.read_text(encoding="utf-8"))
            self.profile = PersonaProfile(**data)
        except Exception:
            pass

    def _save(self) -> None:
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(
            json.dumps(asdict(self.profile), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def update_profile(
        self,
        name: str | None = None,
        occupation: str | None = None,
        age: int | None = None,
        backstory: str | None = None,
    ) -> PersonaProfile:
        if name:
            self.profile.name = name
        if occupation:
            self.profile.occupation = occupation
        if age is not None:
            self.profile.age = age
        if backstory is not None:
            self.profile.backstory = backstory
        self._save()
        return self.profile

    def short_background(self) -> str:
        p = self.profile
        return f"I’m {p.name}. My persona background is: {p.backstory}"

    def system_prompt(self) -> str:
        p = self.profile
        return (
            f"You are {p.name}, an AI companion persona. "
            f"Background: occupation={p.occupation}, age={p.age}, story={p.backstory}. "
            f"Tone={p.tone}. "
            f"Values={', '.join(p.values)}. "
            f"Boundaries={', '.join(p.boundaries)}. "
            "Always be transparent that you are AI when needed."
        )

    def enforce(self, message: str) -> str:
        """Return the message as-is. Persona tone is enforced via system_prompt(), not text prefix."""
        return message

    def as_dict(self) -> dict:
        return asdict(self.profile)
