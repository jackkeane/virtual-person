"""
Safety service: input + output moderation with blocked terms and policy checks.
"""


class SafetyService:
    def __init__(self) -> None:
        self.blocked_input_terms = {
            "make malware", "harm someone", "bypass safeguards",
            "how to hack", "build a weapon", "steal identity",
        }
        self.blocked_output_terms = {
            "here is how to hack", "here is malware code",
            "instructions to harm", "bypass security",
        }

    def check(self, text: str) -> tuple[bool, str | None]:
        """Check user input for safety violations."""
        lower = text.lower()
        for term in self.blocked_input_terms:
            if term in lower:
                return False, "I can't help with harmful or unsafe requests."
        return True, None

    def check_output(self, text: str) -> tuple[bool, str | None]:
        """Check LLM output for safety violations."""
        lower = text.lower()
        for term in self.blocked_output_terms:
            if term in lower:
                return False, "Output contained unsafe content."
        return True, None
