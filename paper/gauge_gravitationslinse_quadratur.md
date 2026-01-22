# Gravitation als geometrische Indexlinse im Radial-Scaling-Gauge

## 1. Kuhauge-Analogie (präzise)

Eine optische Linse (z.B. Auge) fokussiert Licht, weil der **Brechungsindex n(x)** räumlich variiert. Der optische Weg

$$\mathcal{L}_{\text{opt}} = \int n(\mathbf{x}) \, d\ell$$

ist pfadabhängig; nach dem Fermat-Prinzip sind die beobachteten Strahlen genau jene Pfade, für die $\delta \mathcal{L}_{\text{opt}} = 0$ (stationärer optischer Weg).

Im Radial-Scaling-Gauge ist der physikalische Mechanismus analog, aber **ohne Medium**: anstelle von $n(\mathbf{x})$ wirkt eine **geometrische Skalierung**

$$s(r, \varphi) = 1 + \Xi(r, \varphi)$$

so, dass das phasenakkumulierende Wegelement (Eikonal-/Optik-Limit) pfadabhängig wird. Formal entspricht dies einem effektiven "Index"

$$n_{\text{eff}}(r, \varphi) \;\widehat{=}\; s(r, \varphi)$$

und dieselbe Variationslogik gilt:

$$\delta \int s \, d\ell = 0$$

**Kuhauge-Analogie (präzise):** *Im Auge ist es ein materieller Indexgradient; in der Gravitation ein Geometrie-/Skalierungsgradient — die Abbildungslogik entsteht in beiden Fällen aus dem stationären-Pfad-Prinzip.*

---

## 2. Dünnlinsenformel im Gauge

Im Kleinwinkel-/Dünnlinsen-Limit kann die Ablenkung als (reduziertes) Ablenkungsfeld geschrieben werden:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) \;\approx\; \int \nabla_\perp \ln s \; d\ell \;\approx\; \int \nabla_\perp \Xi \; d\ell \quad(\text{für } s = 1 + \Xi,\ |\Xi| \ll 1)$$

Die Linsengleichung lautet dann:

$$\boxed{\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}_{\text{red}}(\boldsymbol{\theta})}, \qquad \boldsymbol{\alpha}_{\text{red}} = \frac{D_{ds}}{D_s} \boldsymbol{\alpha}$$

Hierbei ist $\boldsymbol{\theta}$ die Bildposition, $\boldsymbol{\beta}$ die Quellposition (Winkelkoordinaten), und $D_d, D_s, D_{ds}$ die üblichen Abstandsfaktoren.

---

## 3. Einsteinring als "Quadratur-Kalibrierung"

Der **Optimalfall** ist nicht "Kreisquadratur" im klassischen Sinn, sondern **rotationssymmetrische Degeneration**:

- Rotationssymmetrische Linse: $s = s(r)$ (bzw. $\Xi = \Xi(r)$)
- Perfekte Ausrichtung: $\boldsymbol{\beta} = 0$

Dann gibt es keinen bevorzugten Azimut $\varphi$, und die stationären Lösungen bilden eine **kontinuierliche** Familie: den Einsteinring.

Die Ring-Bedingung ist exakt:

$$\boxed{\theta_E = \alpha_{\text{red}}(\theta_E)} \quad\Longleftrightarrow\quad \theta_E = \frac{D_{ds}}{D_s} \alpha(b_E), \; b_E = D_d \theta_E$$

**Interpretation:** $\theta_E$ ist die **Skalierung**, die als "Quadratur-Kalibrierung" dient. Sobald $\theta_E$ bekannt ist, wird jede Abweichung davon zur Information über Quelle und Anisotropie.

---

## 4. Warum 4 Bilder? (Versatz + Quadrupol, nicht Versatz allein)

Ein **reiner Versatz** $\boldsymbol{\beta} \neq 0$ bricht immer den Ring — aber für eine glatte rotationssymmetrische Linse führt dies typischerweise zu **2 Bildern** (plus evtl. ein stark abgeschwächtes zentrales Bild).

Das **Einsteinkreuz (4 helle Bilder)** entsteht generisch nur, wenn zusätzlich ein **Quadrupol** aktiv ist (Linsen-Elliptizität oder externe Scherung). In Gauge-Sprache: $s$ (bzw. $\Xi$) hat einen dominanten $m=2$ Winkelterm:

$$\Xi(r, \varphi) = \Xi_0(r) + \Xi_2(r) \cos 2(\varphi - \varphi_\gamma) + \dots$$

- $\varphi_\gamma$: Orientierung der Quadrupol-Achse
- $\boldsymbol{\beta}$: Versatzrichtung

**Warum typischerweise 4 und nicht 6/8?**
Weil bei realistischen Linsen der erste stabile nicht-runde Beitrag meist der **Quadrupol (m=2)** ist. Höhere Moden (m=3,4,...) sind möglich, aber typischerweise subdominant.

---

## 5. Lokales Minimalmodell am Ringradius

Nahe $\theta \approx \theta_E$ (Bilder liegen nahe der kritischen Kurve für ein Kreuz) modellieren wir die Ablenkung in Polarform als:

$$\boldsymbol{\alpha}(\theta, \varphi) = \underbrace{\theta_E \hat{\mathbf{e}}_r}_{\text{Ringskala}} + \underbrace{a \cos 2\Delta \; \hat{\mathbf{e}}_r + b \sin 2\Delta \; \hat{\mathbf{e}}_\varphi}_{\text{Quadrupol}}, \quad \Delta = \varphi - \varphi_\gamma$$

Die Linsengleichung $\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}$ ergibt zwei praktische Gleichungen:

**(1) Winkelbedingung (stationäre Azimute):**

$$\boxed{\beta \sin(\varphi - \varphi_\beta) + b \sin 2(\varphi - \varphi_\gamma) = 0}$$

→ dies ergibt (im Kreuz-Regime) typischerweise **4** Lösungen $\varphi_i$.

**(2) Radialbedingung (Radien der vier Bilder):**

$$\boxed{r_i = \theta_E + a \cos 2(\varphi_i - \varphi_\gamma) + \beta \cos(\varphi_i - \varphi_\beta)}$$

→ gibt die leichten radialen Abweichungen der Bilder um $\theta_E$.

---

## 6. No-Fit Inversion (Algorithmus)

**Gegeben:** Linsenzentrum M und vier Bildpunkte $(x_i, y_i)$.

**Unbekannt:** $(β_x, β_y, θ_E, a, b, φ_γ)$

**Strategie:**

1. Für ein festes $\varphi_\gamma$ ist das System linear in $p = (\beta_x, \beta_y, \theta_E, a, b)$:
   - 8 skalare Gleichungen (2 pro Punkt)
   - 5 Unbekannte
   - Wähle 5 Zeilen → exakte Lösung via $p = A^{-1}b$

2. Bestimme $\varphi_\gamma$ **nicht durch Fit**, sondern via **Rootfinding**:
   - Definiere $h(\varphi_\gamma)$ = Residuum einer 6. Gleichung
   - Finde Nullstelle(n) via Bisection: $h(\varphi_\gamma) = 0$
   - Keine Optimierung, kein Least Squares

3. Validierung:
   - Berechne alle 8 Residuen: $r = Ap - b$
   - Berichte max|r| und RMS
   - Synthetische Daten: Residuen ~$10^{-15}$ (Maschinengenauigkeit)

**Symmetrie-Hinweis:** $\varphi_\gamma$ ist bei reinem $m=2$ äquivalent modulo $90°$ (Achse), und entsprechend ändern sich die Vorzeichen von $a, b$. Das ist Physik/Symmetrie, kein Fehler.

---

## 7. Physical Consistency Box

| Aspekt | Status |
|--------|--------|
| Stationärer Weg/Phase | ✓ Konsistent mit Fermat-Prinzip |
| Multipol-Bruch | ✓ m=2 erklärt 4 Bilder |
| Caustic-Logik | ✓ Kreuz nahe kritischer Kurve |
| Winkelinformation | ✓ β-Richtung + Quadrupol-Achse rekonstruierbar |
| **Tiefendistanz** | ✗ Nicht aus Winkeln allein (braucht z, Time Delays, Masse) |

---

## 8. Referenzen

- Schneider, P., Ehlers, J., Falco, E. E., *Gravitational Lenses*, Springer (1992).
- Born, M., Wolf, E., *Principles of Optics*, 7th ed., Cambridge University Press (1999).
- Wrede, C. N., Casu, L. P., Bingsi, *Radial Scaling Gauge for Maxwell Fields* (2025).

---

*Autoren: Carmen N. Wrede & Lino P. Casu*
