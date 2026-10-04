"""Step 6: Jev decides what the question is about, whether the decision answers it, and which hits matter."""

import asyncio
from dataclasses import dataclass, field

from typesafe_sdk import Choice, Noul

from lawhack.schema import Decision, RegistryEntry
from lawhack.system_one import SystemOneClient


class Route:
    COUR_DECISION = "COUR_DECISION"
    LOWER_COURT = "LOWER_COURT"
    PARTY_ARGUMENT = "PARTY_ARGUMENT"
    FACTS = "FACTS"
    NOT_IN_DECISION = "NOT_IN_DECISION"


ROUTE_CRITERIA = {
    Route.COUR_DECISION: "The question asks what the Cour de cassation itself decided, held or reasoned",
    Route.LOWER_COURT: "The question asks what the lower court (cour d'appel) found or decided",
    Route.PARTY_ARGUMENT: "The question asks what a party argued, claimed or requested",
    Route.FACTS: "The question asks about the facts or the procedural history of the case",
    Route.NOT_IN_DECISION: "The question is unrelated to this decision or cannot be answered from it",
}


@dataclass
class Routing:
    route: Decision
    answerable: float
    relevant: list[RegistryEntry] = field(default_factory=list)
    input_tokens: int = 0


async def route_question(question: str, candidates: list[RegistryEntry], client: SystemOneClient, relevance_min: float,
                         keep: frozenset[str] = frozenset()) -> Routing:
    """`keep`: segment ids always passed on (the Cour's own words). One question per call: on OpenJev,
    questions asked together in one call influence each other."""
    state = {"question": question, "segments": {e.id: e.text for e in candidates}}
    asks = {
        "route": Choice(instructions="What is `question` about, with respect to this court decision?", criteria=ROUTE_CRITERIA),
        "answerable": Noul(instructions="Does at least one of `segments` contain information that answers `question`?"),
    }
    for e in candidates:
        if e.id not in keep:
            asks[f"relevant:{e.id}"] = Noul(instructions=f"Is segment `segments.{e.id}` useful to answer `question`?")
    results = await asyncio.gather(*(client.decide(state, {k: q}) for k, q in asks.items()))
    d = {k: r.decisions[k] for r in results for k in r.decisions}
    relevant = [e for e in candidates if e.id in keep or _yes(d[f"relevant:{e.id}"]) >= relevance_min]
    return Routing(route=d["route"], answerable=_yes(d["answerable"]), relevant=relevant,
                   input_tokens=sum(r.input_tokens for r in results))


def _yes(decision: Decision) -> float:
    return decision.probabilities.get("true", 1.0 if decision.value == "true" else 0.0)
