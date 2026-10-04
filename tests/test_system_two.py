import asyncio
import json

from fakes import FakeChat, ScriptedJev, choice, noul
from lawhack.answer import build_messages, parse_draft
from lawhack.config import Settings
from lawhack.retrieval import retrieve
from lawhack.schema import AnswerStatus, Badge, Speaker
from lawhack.system_two import ask
from lawhack.verify import verify_text

SETTINGS = Settings()


def seg(registry, paragraph):
    return next(e for e in registry.entries if e.paragraph == paragraph)


def answering_jev(**overrides):
    answers = dict(route=choice("COUR_DECISION"), answerable=noul(0.95), relevant=noul(0.9), support=noul(0.95))
    answers.update(overrides)
    return ScriptedJev(**answers)


def run(question, registry, jev, chat):
    return asyncio.run(ask(question, registry, jev, chat, SETTINGS, "REJET"))


def test_retrieval_finds_paragraph_and_dispositif(registry):
    hits = retrieve(registry, "montant maximal du prêt dans la promesse")
    assert seg(registry, 9).id in [e.id for e in hits]
    assert any(e.zone.value == "dispositif" for e in hits)


def test_off_topic_question_abstains_without_calling_mistral(registry):
    jev = answering_jev(route=choice("NOT_IN_DECISION", 0.9), answerable=noul(0.1))
    chat = FakeChat("{}")
    answer = run("Quel est le salaire minimum en 2024 ?", registry, jev, chat)
    assert answer.status is AnswerStatus.HORS_ARRET and not chat.calls
    assert answer.render() == "L'arrêt ne traite pas cette question."


def test_correct_answer_gets_ok_badge(registry):
    s9 = seg(registry, 9)
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La Cour approuve : le montant maximal n'était pas une condition.", "segments": [s9.id], "claimed_speaker": "COUR_CASSATION"}]}))
    answer = run("Le montant maximal du prêt était-il une condition ?", registry, answering_jev(), chat)
    claim = answer.claims[0]
    assert answer.status is AnswerStatus.ANSWER and claim.badge is Badge.OK
    assert claim.citations[0].label == "§9, Réponse de la Cour"
    assert "[Cour · §9 · 100%]" in answer.render()
    assert s9.id in chat.calls[0][1]["content"]


def test_party_argument_presented_as_cour_decision_is_not_decided(registry):
    s7 = seg(registry, 7)
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La Cour juge que l'indemnité d'immobilisation est due.", "segments": [s7.id], "claimed_speaker": "COUR_CASSATION"}]}))
    answer = run("L'indemnité d'immobilisation est-elle due ?", registry, answering_jev(), chat)
    assert answer.claims[0].badge is Badge.NON_TRANCHE
    assert answer.claims[0].actual_speaker is Speaker.DEMANDEUR
    assert answer.status is AnswerStatus.NON_TRANCHE


def test_lower_court_finding_attributed_to_cour_is_flagged(registry):
    s8 = seg(registry, 8)
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La Cour constate que la demande de prêt était conforme.", "segments": [s8.id], "claimed_speaker": "COUR_CASSATION"}]}))
    answer = run("La demande de prêt était-elle conforme ?", registry, answering_jev(), chat)
    claim = answer.claims[0]
    assert claim.badge is Badge.MAL_ATTRIBUE and claim.actual_speaker is Speaker.JURIDICTION_FOND
    assert "Cour d'appel (§8)" in claim.note


def test_lower_court_quoted_in_a_moyen_is_to_be_checked(registry):
    s7 = seg(registry, 7)
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La cour d'appel a déclaré la promesse caduque.", "segments": [s7.id], "claimed_speaker": "JURIDICTION_FOND"}]}))
    claim = run("Qu'a décidé la cour d'appel ?", registry, answering_jev(), chat).claims[0]
    assert claim.badge is Badge.A_VERIFIER and "arrêt attaqué" in claim.note


def test_jev_gets_one_question_per_call(registry):
    jev = answering_jev()
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La Cour rejette le pourvoi.", "segments": [seg(registry, 9).id], "claimed_speaker": "COUR_CASSATION"}]}))
    run("Pourquoi la Cour rejette-t-elle le pourvoi ?", registry, jev, chat)
    assert jev.calls and all(len(c["questions"]) == 1 for c in jev.calls)


def test_unknown_or_unsupported_segments_lead_to_abstention(registry):
    chat = FakeChat(json.dumps({"status": "ANSWER", "claims": [
        {"text": "La Cour accorde 50 000 euros.", "segments": ["S-999"], "claimed_speaker": "COUR_CASSATION"}]}))
    answer = run("Combien la Cour accorde-t-elle ?", registry, answering_jev(), chat)
    assert answer.claims[0].badge is Badge.NON_SOUTENU and answer.status is AnswerStatus.HORS_ARRET


def test_low_confidence_attribution_is_a_warning(registry):
    s8 = seg(registry, 8)
    s8.speaker = choice("JURIDICTION_FOND", 0.6)
    chat = FakeChat(json.dumps({"claims": [
        {"text": "La cour d'appel a relevé une demande conforme.", "segments": [s8.id], "claimed_speaker": "JURIDICTION_FOND"}]}))
    claim = run("Qu'a relevé la cour d'appel ?", registry, answering_jev(), chat).claims[0]
    assert claim.badge is Badge.A_VERIFIER and claim.note.startswith("Attribution incertaine")
    assert "60% ⚠" in run("Qu'a relevé la cour d'appel ?", registry, answering_jev(), chat).render()


def test_verify_external_text_catches_misattribution(registry):
    s7 = seg(registry, 7)
    jev = ScriptedJev(source=choice(s7.id, 0.9), claimed=choice("COUR_CASSATION", 0.9))
    claims, _ = asyncio.run(verify_text("La Cour juge que l'indemnité d'immobilisation est due au vendeur.", registry, jev, SETTINGS))
    assert claims[0].badge is Badge.NON_TRANCHE


def test_parse_draft_is_robust():
    draft = parse_draft('```json\n{"status": "ANSWER", "claims": [{"text": "x", "segments": ["[S-014]", "S-014"], "claimed_speaker": "NOPE"}]}\n```')
    assert draft.claims[0].segments == ["S-014"] and draft.claims[0].claimed_speaker is Speaker.INDETERMINE
    assert parse_draft("pas du json").status is AnswerStatus.HORS_ARRET


def test_prompt_shows_speaker_and_paragraph(registry):
    content = build_messages("q", [seg(registry, 8)])[1]["content"]
    assert "§8" in content and "locuteur=JURIDICTION_FOND" in content


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CONFIDENCE_HIGH", "0.9")
    monkeypatch.setenv("ANSWER_MODEL", "mistral-large-latest")
    s = Settings.from_env()
    assert s.confidence_high == 0.9 and s.answer_model == "mistral-large-latest" and s.confidence_low == 0.5
