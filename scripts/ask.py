"""Usage: python scripts/ask.py <arret.pdf> "question" [--baseline] [--verify "texte à vérifier"]"""

import argparse
import asyncio

from dotenv import load_dotenv

from lawhack.config import Settings
from lawhack.llm import answer_client, baseline_client
from lawhack.pipeline import analyse, load
from lawhack.schema import render_badge
from lawhack.system_one import TypeSafeSystemOne
from lawhack.system_two import ask, ask_baseline
from lawhack.verify import verify_text


def show(claims) -> None:
    for c in claims:
        print(f"  {c.text} {render_badge(c)}" + (f"\n    ↳ {c.note}" if c.note else ""))


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("decision")
    p.add_argument("question", nargs="?")
    p.add_argument("--baseline", action="store_true", help="poser aussi la question au modèle de référence et le vérifier")
    p.add_argument("--verify", help="vérifier un texte externe (ex. réponse d'Astra)")
    args = p.parse_args()

    settings = Settings.from_env()
    doc = load(args.decision)
    registry, solution = analyse(doc)
    jev = TypeSafeSystemOne()
    if args.question:
        answer = asyncio.run(ask(args.question, registry, jev, answer_client(settings), settings, solution.value))
        print(f"LawHack [{answer.status.value} · {answer.seconds}s · {answer.input_tokens} tokens]")
        if answer.message:
            print(f"  {answer.message}")
        show(answer.claims)
        if args.baseline:
            base = asyncio.run(ask_baseline(args.question, doc.text, baseline_client(settings)))
            print(f"\nRéférence ({base.model}) :\n  {base.text}\nVérification :")
            show(asyncio.run(verify_text(base.text, registry, jev, settings))[0])
    if args.verify:
        print("Vérification :")
        show(asyncio.run(verify_text(args.verify, registry, jev, settings))[0])


if __name__ == "__main__":
    main()
