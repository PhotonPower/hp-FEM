---
description: Meilenstein M10 — Validierung gegen Literatur-Benchmarks (Rippenwellenleiter, Metallgitter, Mie-Kugel, Spalt-Rille mit geschichtetem Hintergrund)
---

Du bist der Agent **`validation`**. Parallel arbeitet der Entwicklungsagent **`dev`** im
Haupt-Worktree `../hp-FEM`. Lies zuerst CLAUDE.md vollständig, besonders §2 (Prinzipien),
§6 (Konventionen), §7 (Workflow, Definition of Done), §8 (Tests) und §13 (parallele Agenten,
Board `../hp-fem-agents/BOARD.md`). Halte dich strikt daran.

Ziel: hp-FEM gegen **unabhängige, publizierte Referenzwerte** prüfen, nicht gegen eigene
Implementierungen. Alle Parameter und Referenzzahlen stehen unten; die Paper selbst brauchst
du nicht.

## Regeln zu Literatur und Urheberrecht (nicht verhandelbar)
- Es werden **keine Paper, PDFs, Abbildungen oder Textpassagen** ins Repo übernommen.
  `.gitignore` schließt `*.pdf` und `/papers/` aus; lege nichts davon an.
- Erlaubt sind Zahlenwerte (Parameter, Referenzergebnisse) mit vollständiger Quellenangabe
  (Autoren, Zeitschrift, Band, Seite, Jahr, DOI) im Code-Kommentar und in `docs/validation.md`.
  Beschreibe die Benchmarks in eigenen Worten.
- **Nie Parameter an die Referenz anpassen.** Wo unten eine Annahme markiert ist, wird sie
  geprüft, nicht gefittet. Stimmt etwas nicht, stoppen und melden.

## 0. Start
- Board lesen, `START validation: M10 Literatur-Validierung` an `all` posten.
- `ASK dev → Arbeitest du gerade an physics/scattering*, physics/sources*, physics/mie*,
  physics/propagating_mode* oder physics/diffraction*?` und auf `ANSWER` warten, bevor du
  diese Dateien anfasst. Neue Dateien sind unkritisch.
- `CLAIM docs/roadmap.md`, `CLAIM CHANGELOG.md` erst unmittelbar vor der jeweiligen Änderung.
- Jede Teilaufgabe = eigener Branch + PR (CLAUDE.md §7), Reihenfolge wie unten (nach Risiko).
  Nach jedem Merge `INFO main is now <sha>: …` posten und `RELEASE` für beanspruchte Dateien.

## Allgemeine Vorgaben für alle Benchmarks
- Konvention des Codes: Zeitabhängigkeit e^{−iωt}, Verluste haben Im ε > 0 (ADR-0002).
  Werte aus Quellen mit e^{+iωt} werden konjugiert (ist unten schon erledigt).
- Jeder Benchmark wird ein Konvergenztest in `tests/convergence/` mit ctest-Label
  `validation` (zusätzlich `convergence`). Laufzeit im CI je Test ≲ 2 min (release-Preset);
  aufwändigere Varianten bekommen das Label `validation-long` und laufen nicht im
  Standard-CI.
- Dokumentiere für jeden Benchmark in `docs/validation.md` (neue Seite, in `mkdocs.yml` ins
  Nav eintragen – geteilte Datei, also CLAIM): Aufbau in eigenen Worten, Quelle, Referenzwert,
  unser Ergebnis, Konvergenztabelle (DoF, p, Fehler, Laufzeit) und eine ehrliche Bewertung.
- Die Konvergenzläufe (auch die langen) als JSON unter `benchmarks/results/` ablegen.
- Zeige echte Konvergenz (Folge über p bzw. Verfeinerung), nicht nur einen Einzelwert.
  Ecken in Metallen sind singulär: hp-Adaptivität bzw. geometrische Verfeinerung zu den
  Ecken ist gewollt und darf Teil des Tests sein.

## A. Rippenwellenleiter (Vassallo 1997) — Branch `test/rib-waveguide`
Quelle: C. Vassallo, „1993–1995 Optical mode solvers", Opt. Quantum Electron. 29, 95–114
(1997), Abschnitt 7 und Tabelle I (Referenzspalte MTRM; laut Autor vier exakte Stellen).

Geometrie (2D-Querschnitt, x horizontal, y vertikal, alle Längen in µm):
- Substrat n = 3.40, halbunendlich nach unten.
- Wellenleitende Schicht n = 3.44: unter der Rippe Gesamtdicke 1.0 (Rippe inklusive);
  seitlich der Rippe Restdicke t. Rippenbreite 3.0, Rippe zentriert bei x = 0.
- Darüber Luft n = 1.0. Wellenlänge λ = 1.15.
- Kein Material ist verlustbehaftet → `physics::PropagatingMode<2>` passt direkt.

Referenz: normierte Ausbreitungskonstante B = (n_eff² − 3.40²) / (3.44² − 3.40²)

| t   | B (quasi-TE) | B (quasi-TM) |
|-----|--------------|--------------|
| 0.1 | 0.3019 | 0.2674 |
| 0.3 | 0.3110 | 0.2751 |
| 0.5 | 0.3270 | 0.2890 |
| 0.7 | 0.3512 | 0.3107 |
| 0.9 | 0.3883 | 0.3455 |

- Quasi-TE = Grundmode mit dominantem E parallel zum Substrat (E_x); quasi-TM = dominantes
  E_y. Moden über den Polarisationsanteil zuordnen, nicht über die Reihenfolge.
  Plausibilität: B_TE > B_TM für jedes t.
- **TM bei t = 0.9 ist leaky** (n_eff unter dem TE-Slabmode seitlich). Nicht in den harten
  Test aufnehmen; nur dokumentieren, was der Löser dort liefert.
- Rand: PEC-Box. Die Referenz ist gegen die Wandlage stabil (Wände dort 0.5 über der Rippe,
  3 unter der Schicht). Prüfe selbst, dass oben, unten und seitlich die Wandlage die
  4. Stelle von B nicht ändert (seitlich v. a. bei t = 0.9 großzügig, das Feld klingt in der
  Seitenplatte langsam ab). Symmetrie x = 0 darf ausgenutzt werden.
- Toleranz im Test: |B − B_ref| ≤ 2·10⁻⁴ für alle 9 harten Fälle. Hinweis: ΔB = 10⁻⁴
  entspricht Δn_eff ≈ 4·10⁻⁶; das braucht hohe Ordnung und Verfeinerung zu den Rippenecken.

## B. Metallisches Lamellengitter, H parallel zu den Stegen (Granet & Guizal 1996)
Branch `test/metal-grating`.
Quelle: G. Granet, B. Guizal, „Efficient implementation of the coupled-wave method for
metallic lamellar gratings in TM polarization", J. Opt. Soc. Am. A 13, 1019–1023 (1996),
Tabelle 1 (Spalte „Exact", Werte von L. Li mit einer Modalmethode berechnet); Aufbau aus
L. Li, C. W. Haggans, J. Opt. Soc. Am. A 10, 1184–1189 (1993).

Aufbau (Längen in µm):
- Periode d = 1, Wellenlänge λ = 1, Einfall aus Luft (ε = 1) unter θ = 30° zur Normalen.
- Polarisation: H parallel zu den Stegen, E in der Einfallsebene (im Code: in-plane-E, die
  H_z-Polarisation wie im bestehenden Test `lamellar_grating`).
- Gitterschicht der Höhe h auf einem **halbunendlichen Metallsubstrat** aus demselben Metall.
  In der Schicht: Metallsteg der Breite f·d, Rest Luft.
- Metall: Brechzahl 0.22 + 6.71i (in unserer Konvention), also
  **ε = (0.22 + 6.71i)² = −44.9757 + 2.9524i**.
- **Annahme f = 0.5** (Füllfaktor steht nicht im Text). Validierung der Annahme: alle drei
  Tiefen müssen mit f = 0.5 innerhalb der Toleranz passen. Passen sie nicht: stoppen, nicht
  fitten, mir die Werte melden.
- Gemessen: Reflexions-Beugungseffizienzen (Anteil der einfallenden Leistung) der beiden
  propagierenden Ordnungen. Ordnung 0 = spiegelnd. Ordnung −1 ist die Ordnung mit
  k_x = k₀(sin θ − λ/d) = −k₀/2, d. h. sie läuft genau in Einfallsrichtung zurück (Littrow).
  Ordnungen über k_x zuordnen, nicht über ein Vorzeichen-Label.

| h [µm] | η₋₁ (Referenz) | η₀ (Referenz) |
|--------|----------------|----------------|
| 0.1    | 0.3408 | 0.6312 |
| 1.0    | 0.1024 | 0.8477 |
| 4.8    | 0.0503 | 0.4985 |

- Umsetzung: Bloch-periodisch, Scattered-Field-Formulierung mit der ebenen Welle im Vakuum
  wie im EUV-Beispiel; oben PML in Luft. Unten genügt Metall mit einigen Skintiefen
  (δ ≈ 24 nm) plus PEC oder PML – Unempfindlichkeit prüfen.
- Toleranz: |η − η_ref| ≤ 3·10⁻⁴ (die Referenz hat vier Stellen). Energiebilanz
  η₋₁ + η₀ + Absorption = 1 als zusätzlicher Check (Absorption aus dem Volumenintegral).
- Der bestehende Test `lamellar_grating` vergleicht mit einer RCWA im Test selbst; dieser
  Benchmark ist der unabhängige Vergleich. Den alten Test nicht entfernen.

## C. Mie-Streuung an der Kugel (3D, analytisch) — Branch `feat/mie-sphere`
- Neue Funktionen in `physics::mie` (neben dem Zylinder): Mie-Koeffizienten a_n, b_n,
  Q_ext, Q_sca, Q_abs und das Feld innen/außen für eine Kugel mit **komplexem** ε in einem
  verlustfreien Hintergrund (Bohren & Huffman, Kap. 4). Sphärische Bessel-/Hankelfunktionen
  für komplexe Argumente stabil (Abwärtsrekursion bzw. logarithmische Ableitung D_n).
- Unit-Tests der Reihe ohne FEM: optisches Theorem (Q_ext aus der Vorwärtsamplitude =
  Q_sca + Q_abs), Rayleigh-Grenzfall für ka → 0, Konvergenz in der Ordnung. Optionaler
  Quercheck in den Python-Tests gegen das Paket `miepython`, nur wenn installiert
  (`pytest.importorskip`), keine neue Pflichtabhängigkeit.
- FEM-Konvergenztest `mie_sphere`: 3D-Streuung mit gekrümmten Elementen (Kugelgenerator),
  PML, zwei Fälle: dielektrisch (n = 2, ka = 2) und metallisch (ε = −10 + 1i, ka = 0.6).
  Vergleich von Q_sca und Q_abs sowie Feldwerten an einigen Punkten innen und außen.
  Exponentielle Konvergenz in p dokumentieren. Klein halten (CI-Zeit), große Läufe als
  `validation-long`.
- Python-Bindings für die neuen Funktionen.

## D. Geschichteter Hintergrund + Spalt-Rille in Silber — Branch `feat/layered-background`
Größte Teilaufgabe, vor dem Start eine ADR schreiben (`docs/adr/0008-layered-background.md`).

D1. Feature: Scattered-Field-Formulierung mit **geschichtetem Hintergrund** (Schichten
senkrecht zu y in 2D bzw. z in 3D).
- Analytisches Hintergrundfeld eines Schichtstapels für ebene Wellen, beide Polarisationen,
  schräger Einfall, mit numerisch stabiler Rekursion (S-Matrix oder Impedanzrekursion – dicke
  Metallschichten dürfen nicht überlaufen).
- Quelle der Streuformulierung: k₀² (ε(x) − ε_bg(y)) E_bg, d. h. nur dort, wo die Struktur
  vom Stapel abweicht. PML in allen Schichten (gestreckte Tensoren je Material).
- Postprocessing muss das Gesamtfeld E_bg + E_sc liefern können (Flüsse!).
- Tests: Transfermatrix gegen Fresnel (eine Grenzfläche), Energiebilanz R + T + A = 1,
  FEM ohne Störung liefert E_sc ≈ 0, Fluss des Gesamtfelds durch eine Ebene im Substrat =
  analytische Transmission.
- Python-Bindings, Theorieabschnitt in `docs/theory/maxwell.md` oder eigene Seite.

D2. Benchmark Spalt-Rille (2D). Quellen:
M. Besbes, J. P. Hugonin, P. Lalanne et al., „Numerical analysis of a slit-groove
diffraction problem", J. Eur. Opt. Soc. Rapid Publ. 2, 07022 (2007), DOI
10.2971/jeos.2007.07022 (Open Access); S. Burger, L. Zschiedrich, J. Pomplun, F. Schmidt,
„Finite-element based electromagnetic field simulations: Benchmark results for isolated
structures", Proc. SPIE 8880, 88801Z (2013), arXiv:1310.2732.

Aufbau (x horizontal, y vertikal, Längen in nm):
- Luft (ε = 1) für y > 0, Silberfilm −400 < y < 0 mit ε_Ag = −33.22 + 1.1700i,
  Substrat y < −400.
- Spalt: Breite 100, zentriert bei x = 0, durch den ganzen Film (Luft gefüllt).
- Rille: Breite 100, Tiefe 100 von der Oberseite (−100 < y < 0), Mitte bei x = −500
  (d = 500 ist der Abstand **Mitte zu Mitte**). Seite egal (Symmetrie).
- Beleuchtung: ebene Welle senkrecht von oben aus Luft, λ₀ = 852 nm, H parallel zu Spalt und
  Rille (in-plane-E).
- Detektor: Strecke bei y = −800 (400 unter der Grenzfläche Silber/Substrat),
  −100 ≤ x ≤ 100. S = Poynting-Fluss des **Gesamtfelds** nach unten durch diese Strecke.
  S₀ = dasselbe ohne Rille (zweite Rechnung). Benchmarkgröße S/S₀.
- Die Oberflächenplasmonen auf der Silberoberfläche sind schwach gedämpft: seitliche und
  obere PML entsprechend dick bzw. stark wählen und die Unempfindlichkeit zeigen.

**Wichtige Unstimmigkeit in den Quellen – beide Fälle rechnen:**
- Besbes et al. 2007 geben das Substrat mit n = 1.45 an → ε_sub = 2.1025. Beste Werte dort
  (Methoden mit der höchsten Genauigkeit): S/S₀ = 2.200952 (MM3), 2.200940 (HYB),
  2.200904 (MM2), 2.201143 (FEM2); interne Genauigkeit etwa 10⁻⁵ bis 10⁻⁴.
- Burger et al. 2013 geben ε_sub = 2.25 an und erhalten S/S₀ = 2.198825944 ± 2·10⁻⁹.
- Erwartung (Hypothese, prüfen!): Der Unterschied von ~0.1 % zwischen den beiden Quellen
  kommt vom Substrat-ε. Rechne mit beiden ε_sub. Melde das Ergebnis, auch wenn es die
  Hypothese widerlegt.
- Toleranzen: gegen Burger (ε_sub = 2.25) relativ ≤ 10⁻⁶ im `validation-long`-Test und
  ≤ 10⁻⁴ im CI-Test; gegen Besbes (ε_sub = 2.1025) relativ ≤ 2·10⁻⁴ zum Mittel aus MM3/HYB.
- Burger zeigt eine Konvergenztabelle über p und adaptive Schritte (z. B. p = 4 ohne
  Adaption: rel. Fehler ≈ 4·10⁻⁶ bei ≈ 1.2·10⁵ Unbekannten; p = 5…7 mit drei adaptiven
  Schritten < 10⁻⁹). Dokumentiere unsere Kurve Fehler vs. DoF und Laufzeit daneben.

## E. Roadmap und Doku
- Neuer Meilenstein in `docs/roadmap.md`: „M10 — Validierung gegen Literatur" mit den
  Punkten A–D (Häkchen erst nach Merge) und einem offenen Punkt:
  „Gold-Kugeldimer (Hoffmann et al., Proc. SPIE 7390, 73900J, 2009; 80 nm Kugeln, 1 nm Spalt,
  632 nm, Referenz |E|² im Spaltzentrum = 5.47624·10⁵ V²/m² bei |E_inc| = 1 V/m) –
  **blockiert**: das verwendete ε von Gold ist in der Quelle nicht angegeben."
  Diesen Punkt **nicht** mit einem selbst gewählten ε bearbeiten.
- CHANGELOG-Einträge unter *Unreleased*, CLAUDE.md §12 (Status) nachziehen.

## Abschluss
`DONE` aufs Board und mir berichten: je Benchmark Referenz, unser Wert, Fehler, DoF,
Laufzeit, PR-Links, offene Punkte und alles, was nicht gepasst hat.
