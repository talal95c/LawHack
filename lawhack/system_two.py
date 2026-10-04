"""Système 2 orchestration: route (Jev) → retrieve → draft (Mistral) → verify (Jev) → honest answer."""

import time

from lawhack.answer import draft_answer
from lawhack.config import Settings
from lawhack.llm import ChatClient, ChatResult
from lawhack.retrieval import pinned, retrieve
from lawhack.routing import Route, route_question
from lawhack.schema import Answer, AnswerStatus, Badge, Registry
from lawhack.system_one import SystemOneClient
from lawhack.verify import check_claims

NOT_ADDRESSED = "L'arrêt ne traite pas cette question."


async def ask(question: str, registry: Registry, system_one: SystemOneClient, chat: ChatClient,
              settings: Settings | None = None, solution: str | None = None) -> Answer:
    settings = settings or Settings.from_env()
    started = time.perf_counter()
    candidates = retrieve(registry, question, settings.top_k)
    keep = frozenset(e.id for e in pinned(registry))
    routing = await route_question(question, candidates, system_one, settings.relevance_min, keep)
    tokens = routing.input_tokens

    def done(answer: Answer) -> Answer:
        answer.seconds = round(time.perf_counter() - started, 2)
        return answer

    off_topic = routing.route.value == Route.NOT_IN_DECISION and routing.route.confidence >= settings.route_abstain
    if off_topic or routing.answerable < settings.answerable_min or not routing.relevant:
        # Abstain before calling Système 2: nothing to cite, so nothing to say.
        return done(Answer(question=question, status=AnswerStatus.HORS_ARRET, message=NOT_ADDRESSED,
                           route=routing.route, input_tokens=tokens))

    draft, result = await draft_answer(question, routing.relevant, chat, solution, routing.route.value)
    claims, verify_tokens = await check_claims(draft.claims, registry, system_one, settings)
    status = draft.status
    message = None
    supported = [c for c in claims if c.badge is not Badge.NON_SOUTENU]
    if not supported:
        status, message = AnswerStatus.HORS_ARRET, NOT_ADDRESSED
    elif status is AnswerStatus.NON_TRANCHE or all(c.badge is Badge.NON_TRANCHE for c in supported):
        status, message = AnswerStatus.NON_TRANCHE, "La Cour ne tranche pas ce point."
    return done(Answer(
        question=question, status=status, claims=claims, message=message, route=routing.route, model=result.model,
        input_tokens=tokens + verify_tokens + result.input_tokens, output_tokens=result.output_tokens,
    ))


BASELINE_PROMPT = "Tu es un assistant juridique. Réponds en français, brièvement, à la question sur l'arrêt fourni."


async def ask_baseline(question: str, decision_text: str, chat: ChatClient) -> ChatResult:
    """Demo split screen: the same question to a frontier model alone (Astra), without the registry."""
    return await chat.complete([
        {"role": "system", "content": BASELINE_PROMPT},
        {"role": "user", "content": f"Arrêt :\n\n{decision_text}\n\nQuestion : {question}"},
    ])
