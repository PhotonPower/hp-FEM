---
description: Aufräumaufgaben nach Release 0.2.0 (Roadmap, Tag v0.1.0, GitHub Pages, gemergte Branches)
---

Du bist der Agent **`housekeeping`**. Parallel arbeitet der Entwicklungsagent **`dev`** im
Haupt-Worktree `../hp-FEM`. Lies zuerst CLAUDE.md vollständig, insbesondere §7 (Workflow)
und §13 (parallele Agenten, Message-Board `../hp-fem-agents/BOARD.md`), und halte dich
strikt daran.

Keine Änderungen an Code oder Numerik. Arbeite in diesem Worktree auf dem Branch
`docs/housekeeping`; Änderungen kommen nur per PR nach `main`.

## 0. Start
- Prüfe `git branch --show-current` (= `docs/housekeeping`) und `gh auth status`.
  Wenn `gh` fehlt oder nicht angemeldet ist: stoppen und mir sagen, was zu tun ist.
- Board lesen, dann `START housekeeping: Roadmap-Sync, Tag v0.1.0, Pages, Branch-Cleanup` an `all` posten.

## 1. Roadmap synchronisieren
- `CLAIM docs/roadmap.md` posten (vorher prüfen, dass `dev` die Datei nicht beansprucht).
- M7: Punkt „meshio / pyvista interop, matplotlib helpers“ abhaken und wie die anderen
  Punkte mit Verweis ergänzen (`python/hpfem/interop.py`, Commit `2265608`).
- Übrige Punkte gegen `CHANGELOG.md` [0.2.0] und den Code prüfen. Umgesetzte, aber nicht
  abgehakte Punkte (oder umgekehrt) korrigieren und im PR auflisten. Offen bleibt nur
  „optional: MPI domain decomposition“.
- Commit: `docs(roadmap): tick completed items after 0.2.0`

## 2. GitHub Pages aktivieren
- `gh api -X POST repos/PhotonPower/hp-FEM/pages -f "source[branch]=gh-pages" -f "source[path]=/"`
- Bei 403/404: nicht weiter versuchen, mir melden (dann mache ich es in den Settings).
- `mkdocs.yml`: `site_url` muss `https://photonpower.github.io/hp-FEM/` sein; README.md
  braucht einen Link auf die Doku. Beides ist eine geteilte Datei → vorher `CLAIM`.
  Commit: `docs: publish site on GitHub Pages`

## 3. PR
- `git fetch && git rebase origin/main`, push, PR „Housekeeping after 0.2.0“ öffnen
  (Template ausfüllen), CI abwarten.
- Bei grün: Board lesen, dann Squash-Merge, `INFO main is now <sha>: …` und
  `RELEASE` für alle beanspruchten Dateien posten.
- Danach prüfen, dass der Docs-Lauf grün ist und die Pages-URL HTTP 200 liefert.

## 4. Tag v0.1.0 nachtragen
- Verifizieren, dass `91c8df8` („feat: project scaffold (M0)“) dem Stand entspricht, den
  `CHANGELOG.md` als [0.1.0] beschreibt (VERSION 0.1.0 in CMakeLists.txt und pyproject.toml).
- `git tag -a v0.1.0 91c8df8 -m "hp-FEM 0.1.0 — project scaffold (M0)"` und pushen.
- GitHub-Release v0.1.0 mit dem Changelog-Abschnitt [0.1.0] anlegen, **nicht** als latest
  (`gh release create v0.1.0 --latest=false ...`). `INFO` aufs Board.

## 5. Gemergte Branches löschen
- Kandidaten: `git branch -r --merged origin/main`, ausgenommen `main`, `gh-pages`, jeder
  Branch mit offenem PR (`gh pr list`) und jeder Branch, den `dev` laut Board gerade nutzt.
- Vorher `ASK dev → Branches, die du noch brauchst?` posten und auf `ANSWER` warten.
- Mir die Liste mit Anzahl zeigen und erst nach meinem OK löschen (`git push origin --delete …`).
- Danach `gh api -X PATCH repos/PhotonPower/hp-FEM -F delete_branch_on_merge=true`
  (bei 403 melden).

## Abschluss
`DONE` aufs Board und mir kurz berichten: erledigte Punkte, PR-Link, Pages-URL, was manuell
nachzuholen ist.
