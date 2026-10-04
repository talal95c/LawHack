"""Step 8: Système 2 (Mistral) drafts an answer citing registry segments as [S-xxx]."""

import json
import re

from pydantic import BaseModel, Field, ValidationError

from lawhack.llm import ChatClient, ChatResult
from lawhack.schema import AnswerStatus, Claim, RegistryEntry, Speaker, ZONE_LABELS

SYSTEM_PROMPT = """Tu es l'assistant d'un avocat. Tu réponds à une question sur UN arrêt de la Cour de cassation,
uniquement à partir des segments fournis. Chaque segment porte un identifiant [S-xxx], son paragraphe, sa rubrique
et son LOCUTEUR (qui parle) déterminé à l'avance. Le locuteur fait foi.

Règles :
1. Chaque affirmation cite au moins un identifiant de segment qui la soutient directement.
2. Le verbe d'attribution suit le locuteur : « la Cour juge / décide » seulement pour COUR_CASSATION ;
   « la cour d'appel a retenu » pour JURIDICTION_FOND ; « le demandeur soutient » pour DEMANDEUR, etc.
3. Un argument d'une partie (moyen) n'est jamais une décision de la Cour.
4. Si la Cour ne tranche pas le point (il n'apparaît que dans les moyens), status = "NON_TRANCHE"
   et les affirmations disent qui l'a soutenu.
5. Si aucun segment ne répond, status = "HORS_ARRET" et claims = []. N'invente rien.
6. Réponds en français, affirmations courtes, une idée par affirmation, sans mettre les identifiants dans le texte.

Réponds uniquement en JSON :
{"status": "ANSWER" | "NON_TRANCHE" | "HORS_ARRET",
 "claims": [{"text": "...", "segments": ["S-014"], "claimed_speaker": "COUR_CASSATION" | "JURIDICTION_FOND" | "DEMANDEUR" | "DEFENDEUR" | "MINISTERE_PUBLIC" | "LOI" | "INDETERMINE"}]}"""

_SEGMENT_ID = re.compile(r"S-\d+")


class Draft(BaseModel):
    status: AnswerStatus = AnswerStatus.HORS_ARRET
    claims: list[Claim] = Field(default_factory=list)


def format_segment(e: RegistryEntry) -> str:
    chain = " → ".join(s.value for s in e.chain)
    where = f"§{e.paragraph}" if e.paragraph else "—"
    return (
        f"[{e.id}] {where} · rubrique={e.heading or ZONE_LABELS[e.zone]} · zone={e.zone.value} · "
        f"locuteur={e.speaker.value} (chaîne {chain}, confiance {e.speaker.confidence:.2f}) · statut={e.status.value}\n"
        f"« {e.text} »"
    )


def build_messages(question: str, entries: list[RegistryEntry], solution: str | None = None, route: str | None = None) -> list[dict]:
    context = "\n\n".join(format_segment(e) for e in entries)
    header = f"Solution de l'arrêt (dispositif) : {solution}\n" if solution else ""
    if route:
        header += f"Nature de la question (Système 1) : {route}\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{header}\nSegments :\n\n{context}\n\nQuestion : {question}"},
    ]


def parse_draft(text: str) -> Draft:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
    except ValueError:
        return Draft()
    claims = []
    for c in data.get("claims") or []:
        if not isinstance(c, dict) or not c.get("text"):
            continue
        ids = _SEGMENT_ID.findall(" ".join(map(str, c.get("segments") or [])))
        speaker = c.get("claimed_speaker")
        claims.append(Claim(
            text=str(c["text"]).strip(),
            segments=list(dict.fromkeys(ids)),
            claimed_speaker=speaker if speaker in Speaker.__members__ else Speaker.INDETERMINE,
        ))
    try:
        return Draft(status=data.get("status") or AnswerStatus.ANSWER, claims=claims)
    except ValidationError:
        return Draft(status=AnswerStatus.ANSWER, claims=claims)


async def draft_answer(question: str, entries: list[RegistryEntry], client: ChatClient, solution: str | None = None, route: str | None = None) -> tuple[Draft, ChatResult]:
    result = await client.complete(build_messages(question, entries, solution, route), json_mode=True)
    return parse_draft(result.text), result
