"""Step 9: check each claim against the registry with Jev, then apply the honesty policy (README §7)."""

import asyncio

from typesafe_sdk import Choice, Noul

from lawhack.attributor import SPEAKER_CRITERIA
from lawhack.config import Settings
from lawhack.retrieval import BM25Index
from lawhack.schema import (
    SPEAKER_LABELS, ZONE_LABELS, Badge, CheckedClaim, Citation, Claim, Registry, RegistryEntry, Speaker, Status, Zone,
)
from lawhack.segmenter import split_sentences
from lawhack.system_one import SystemOneClient

NO_SOURCE = "NONE"


def citation(e: RegistryEntry) -> Citation:
    label = f"§{e.paragraph}, {e.heading or ZONE_LABELS[e.zone]}" if e.paragraph else (e.heading or ZONE_LABELS[e.zone])
    return Citation(segment_id=e.id, paragraph=e.paragraph, zone=e.zone, label=label, speaker=e.speaker)


def _where(e: RegistryEntry) -> str:
    return f"§{e.paragraph}" if e.paragraph else ZONE_LABELS[e.zone]


def judge(claim: Claim, cited: list[RegistryEntry], support: float | None, settings: Settings) -> CheckedClaim:
    """Deterministic honesty policy: the code, not a model, decides what to do with the confidence."""
    base = claim.model_dump()
    if not cited or (support is not None and support < settings.support_min):
        return CheckedClaim(**base, badge=Badge.NON_SOUTENU, support=support, citations=[citation(e) for e in cited],
                            note="Aucun passage de l'arrêt ne soutient cette affirmation.")
    primary = max(cited, key=lambda e: e.speaker.confidence)
    actual = Speaker(primary.speaker.value)
    confidence = min(e.speaker.confidence for e in cited)
    speakers = {e.speaker.value for e in cited}
    out = dict(base, citations=[citation(e) for e in cited], actual_speaker=actual, confidence=confidence, support=support)

    only_argued = all(e.zone is Zone.MOYENS or e.status.value == Status.ALLEGUE.value for e in cited)
    if claim.claimed_speaker is Speaker.COUR_CASSATION and only_argued:
        return CheckedClaim(**out, badge=Badge.NON_TRANCHE,
                            note=f"La Cour ne tranche pas ce point ; c'est l'argument du {SPEAKER_LABELS[actual].lower()} ({_where(primary)}).")
    if claim.claimed_speaker is Speaker.JURIDICTION_FOND and only_argued:
        # A moyen quotes the decision under appeal, but only as the party reports it.
        return CheckedClaim(**out, badge=Badge.A_VERIFIER,
                            note=f"Rapporté par le {SPEAKER_LABELS[actual].lower()} ({_where(primary)}) : à confirmer dans l'arrêt attaqué.")
    if claim.claimed_speaker is not Speaker.INDETERMINE and claim.claimed_speaker.value not in speakers:
        return CheckedClaim(**out, badge=Badge.MAL_ATTRIBUE,
                            note=f"Attribué à : {SPEAKER_LABELS[claim.claimed_speaker]} — en réalité : {SPEAKER_LABELS[actual]} ({_where(primary)}).")
    if actual is Speaker.INDETERMINE or confidence < settings.confidence_high:
        return CheckedClaim(**out, badge=Badge.A_VERIFIER,
                            note=f"Attribution incertaine ({confidence:.0%}) : probablement {SPEAKER_LABELS[actual]}, à vérifier au {_where(primary)}.")
    return CheckedClaim(**out, badge=Badge.OK)


async def check_claims(claims: list[Claim], registry: Registry, client: SystemOneClient, settings: Settings) -> tuple[list[CheckedClaim], int]:
    """Claims drafted by Système 2: Jev checks that the cited segments support each one."""
    cited = {i: [e for e in map(registry.by_id, c.segments) if e] for i, c in enumerate(claims)}
    checkable = {i: es for i, es in cited.items() if es}
    support: dict[int, float] = {}
    tokens = 0

    async def one(i: int, es: list[RegistryEntry]) -> None:
        nonlocal tokens
        state = {"claim": claims[i].text, "segments": {e.id: e.text for e in es}}
        question = Noul(instructions="Is `claim` directly supported by at least one of `segments`? Ignore who said it; judge the content.")
        result = await client.decide(state, {"support": question})  # one claim per call, see routing.py
        tokens += result.input_tokens
        support[i] = result.decisions["support"].probabilities.get("true", 0.0)

    await asyncio.gather(*(one(i, es) for i, es in checkable.items()))
    return [judge(c, cited[i], support.get(i), settings) for i, c in enumerate(claims)], tokens


async def verify_text(text: str, registry: Registry, client: SystemOneClient, settings: Settings) -> tuple[list[CheckedClaim], int]:
    """Any text about the decision (e.g. Astra's answer): find each sentence's source and who it is attributed to."""
    index = BM25Index(registry)
    sentences = [text[s:e].strip() for s, e in split_sentences(text) if text[s:e].strip()]

    async def one(sentence: str) -> tuple[CheckedClaim, int]:
        hits = [e for e, _ in index.search(sentence, settings.top_k)]
        if not hits:
            return judge(Claim(text=sentence), [], None, settings), 0
        state = {"sentence": sentence, "segments": {e.id: e.text for e in hits}}
        sources = {e.id: f"Segment `segments.{e.id}` states or reports the content of `sentence`" for e in hits}
        sources[NO_SOURCE] = "No segment states or reports the content of `sentence`"
        # Two separate calls: on OpenJev, a second question in the same call shifts the source answer.
        found, attributed = await asyncio.gather(
            client.decide(state, {"source": Choice(
                instructions="Which segment states or reports the content of `sentence`, whoever says it?", criteria=sources)}),
            client.decide({"sentence": sentence}, {"claimed": Choice(
                instructions="Who does `sentence` present as the author of the statement or decision?", criteria=SPEAKER_CRITERIA)}),
        )
        source, claimed = found.decisions["source"], attributed.decisions["claimed"]
        cited = [registry.by_id(source.value)] if source.value != NO_SOURCE else []
        claim = Claim(text=sentence, segments=[e.id for e in cited], claimed_speaker=Speaker(claimed.value))
        return judge(claim, cited, source.confidence if cited else None, settings), found.input_tokens + attributed.input_tokens

    results = await asyncio.gather(*(one(s) for s in sentences))
    return [c for c, _ in results], sum(t for _, t in results)
