from enum import Enum

from pydantic import BaseModel, Field


class Speaker(str, Enum):
    COUR_CASSATION = "COUR_CASSATION"
    JURIDICTION_FOND = "JURIDICTION_FOND"
    DEMANDEUR = "DEMANDEUR"
    DEFENDEUR = "DEFENDEUR"
    MINISTERE_PUBLIC = "MINISTERE_PUBLIC"
    LOI = "LOI"
    INDETERMINE = "INDETERMINE"


class StatementType(str, Enum):
    FAIT = "FAIT"
    PROCEDURE = "PROCEDURE"
    MOYEN = "MOYEN"
    MOTIF = "MOTIF"
    VISA = "VISA"
    DISPOSITIF = "DISPOSITIF"
    CITATION = "CITATION"


class Status(str, Enum):
    CONSTATE = "CONSTATE"
    ALLEGUE = "ALLEGUE"
    CONTESTE = "CONTESTE"
    DECIDE = "DECIDE"


class Zone(str, Enum):
    INTRODUCTION = "introduction"
    EXPOSE = "expose"
    MOYENS = "moyens"
    MOTIVATIONS = "motivations"
    DISPOSITIF = "dispositif"
    METADONNEES = "metadonnees"  # Légifrance trailer: ECLI, titrages, textes appliqués


class Block(BaseModel):
    """A paragraph of the decision: numbered (§n) or not, with its zone and heading."""

    index: int
    text: str
    start: int
    end: int
    zone: Zone
    heading: str | None = None
    paragraph: int | None = None
    page: int | None = None


class Segment(BaseModel):
    id: str
    text: str
    start: int
    end: int
    block: int
    zone: Zone
    heading: str | None = None
    paragraph: int | None = None
    page: int | None = None


class Decision(BaseModel):
    """A probabilistic decision over one label set."""

    value: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = 1.0


class RegistryEntry(Segment):
    speaker: Decision
    type: Decision
    status: Decision
    chain: list[Speaker] = Field(default_factory=list)
    source: str  # "rule" | "jev"


class Document(BaseModel):
    id: str
    text: str
    pages: list[tuple[int, int]] = Field(default_factory=list)  # (start, end) offsets per page


class Registry(BaseModel):
    document_id: str
    model: str | None = None
    version: str = "1"
    entries: list[RegistryEntry]
    input_tokens: int = 0
    seconds: float = 0.0

    def by_id(self, segment_id: str) -> RegistryEntry | None:
        return next((e for e in self.entries if e.id == segment_id), None)


SPEAKER_LABELS = {
    Speaker.COUR_CASSATION: "Cour",
    Speaker.JURIDICTION_FOND: "Cour d'appel",
    Speaker.DEMANDEUR: "Demandeur",
    Speaker.DEFENDEUR: "Défendeur",
    Speaker.MINISTERE_PUBLIC: "Min. public",
    Speaker.LOI: "Loi",
    Speaker.INDETERMINE: "?",
}

ZONE_LABELS = {
    Zone.INTRODUCTION: "En-tête",
    Zone.EXPOSE: "Faits et procédure",
    Zone.MOYENS: "Énoncé du moyen",
    Zone.MOTIVATIONS: "Réponse de la Cour",
    Zone.DISPOSITIF: "Dispositif",
    Zone.METADONNEES: "Métadonnées",
}


class Badge(str, Enum):
    OK = "OK"
    A_VERIFIER = "A_VERIFIER"  # attribution confidence between the low and high thresholds
    NON_TRANCHE = "NON_TRANCHE"  # presented as the Cour's view but only argued by a party
    MAL_ATTRIBUE = "MAL_ATTRIBUE"  # claimed speaker differs from the registry
    NON_SOUTENU = "NON_SOUTENU"  # no cited segment supports the claim


class AnswerStatus(str, Enum):
    ANSWER = "ANSWER"
    NON_TRANCHE = "NON_TRANCHE"
    HORS_ARRET = "HORS_ARRET"


class Citation(BaseModel):
    segment_id: str
    paragraph: int | None = None
    zone: Zone
    label: str  # "§8, Réponse de la Cour"
    speaker: Decision


class Claim(BaseModel):
    text: str
    segments: list[str] = Field(default_factory=list)
    claimed_speaker: Speaker = Speaker.INDETERMINE


class CheckedClaim(Claim):
    badge: Badge
    citations: list[Citation] = Field(default_factory=list)
    actual_speaker: Speaker | None = None
    confidence: float = 0.0
    support: float | None = None
    note: str | None = None


class Answer(BaseModel):
    question: str
    status: AnswerStatus
    claims: list[CheckedClaim] = Field(default_factory=list)
    message: str | None = None
    route: Decision | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def render(self) -> str:
        if self.status is AnswerStatus.HORS_ARRET and not self.claims:
            return self.message or "L'arrêt ne traite pas cette question."
        lines = [self.message] if self.message else []
        lines += [f"{c.text} {render_badge(c)}" for c in self.claims]
        return "\n".join(lines)


def render_badge(claim: CheckedClaim) -> str:
    if claim.badge is Badge.NON_SOUTENU or not claim.citations:
        return "[non soutenu par l'arrêt ✗]"
    first = claim.citations[0]
    speaker = SPEAKER_LABELS[claim.actual_speaker or Speaker.INDETERMINE]
    para = f"§{first.paragraph}" if first.paragraph else ZONE_LABELS[first.zone]
    flag = {Badge.OK: "", Badge.A_VERIFIER: " ⚠", Badge.NON_TRANCHE: " · non tranché", Badge.MAL_ATTRIBUE: " ✗"}[claim.badge]
    return f"[{speaker} · {para} · {claim.confidence:.0%}{flag}]"
