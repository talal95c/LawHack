from lawhack.schema import Decision
from lawhack.system_one import SystemOneResult


def noul(p: float) -> Decision:
    return Decision(value="true" if p >= 0.5 else "false", probabilities={"true": p, "false": 1 - p}, confidence=max(p, 1 - p))


def choice(value: str, confidence: float = 0.9) -> Decision:
    return Decision(value=value, probabilities={value: confidence}, confidence=confidence)


class ZoneJev:
    """Registry builder: attributes speakers from the zone in the state."""

    SPEAKERS = {"moyens": "DEMANDEUR", "expose": "JURIDICTION_FOND", "motivations": "JURIDICTION_FOND"}

    async def decide(self, state, questions):
        zone = state["zone"]
        out = {}
        for key in questions:
            if key.startswith("speaker:"):
                out[key] = choice(self.SPEAKERS.get(zone, "INDETERMINE"), 0.92)
            elif key.startswith("type:"):
                out[key] = choice("MOYEN" if zone == "moyens" else "MOTIF", 1.0)
            else:
                out[key] = choice("ALLEGUE" if zone == "moyens" else "CONSTATE", 1.0)
        return SystemOneResult(decisions=out, model="fake", input_tokens=1)


class ScriptedJev:
    """Answers by key prefix (`route`, `answerable`, `relevant`, `support`, `source`, `claimed`)."""

    def __init__(self, **answers):
        self.answers = answers
        self.calls: list[dict] = []

    async def decide(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        out = {}
        for key in questions:
            prefix = key.split(":")[0]
            answer = self.answers[prefix]
            out[key] = answer(state, key) if callable(answer) else answer
        return SystemOneResult(decisions=out, model="fake", input_tokens=5)


class FakeChat:
    def __init__(self, text: str):
        self.text = text
        self.calls: list[list[dict]] = []

    async def complete(self, messages, json_mode=False):
        from lawhack.llm import ChatResult

        self.calls.append(messages)
        return ChatResult(text=self.text, model="fake-mistral", input_tokens=100, output_tokens=20)
