# Tabletop Wargame — Rules Reference

A condensed reference of the tabletop miniatures wargame mechanics this project aims to model.

## The Basics

The game is a **tabletop miniatures wargame** played on a physical table (typically 44"×60" for a standard game). Two players each command an **army** of miniature models and compete to outscore each other through a mixture of tactical movement, shooting, close combat, and objective control.

### Players & Armies

- The game is played between **two players** (larger multiplayer games exist but require custom rules — we focus on two-player games).
- Each player controls an **army**. An army is organised as a list of **units** (groups). Each unit is composed of one or more **models**. A model is a single physical miniature on the table.
- Models vary in both physical size/shape and in-game statistics. A single infantry trooper might have 1 wound and a basic weapon, while a tank could have 14 wounds, heavy armour, and devastating firepower.

### Pre-Game Setup

Before the first turn begins, several setup stages are resolved in order:

1. **Muster Armies** — Both players build armies to an agreed points limit.
2. **Select Mission** — Determines objectives, board layout, and special rules.
3. **Create the Battlefield** — Set up terrain features and objective markers on the table.
4. **Determine Attacker/Defender** — Players roll off; the winner chooses their role.
5. **Declare Battle Formations** — Secretly decide which units attach Leaders, embark in Transports, or start in Reserves.
6. **Deploy Armies** — Players **alternate placing units**, one at a time, into their respective **deployment zones** (designated areas on opposite sides of the table).
7. **Determine First Turn** — Players roll off to decide who goes first.
8. **Resolve Pre-battle Rules** — Handle abilities like scout moves or forward deployment.

### Game Timing

- The game lasts a fixed number of **battle rounds** (standard: **5 rounds**).
- Each battle round consists of two **player turns** — the first player completes their entire turn before the second player takes theirs.
- Each player turn is divided into **phases** (see below), during which various interactions between armies and the game state take place.
- The game does not end early if all of the enemy models are destroyed. The game carries through all of the turns, even though a player cannot take any actions.

### Winning the Game

Victory is determined by **Victory Points (VP)**. Players accumulate VP throughout the game by controlling objectives, completing mission-specific goals, and destroying enemy units. After all battle rounds are complete (or one army is destroyed), the player with the most VP wins.

## Turn Phases

Each player turn proceeds through five phases in order:

1. **Command Phase** — Both players gain 1 Command Point (CP); the active player takes morale tests for units below half-strength.
2. **Movement Phase** — Move units (Normal, Advance, Fall Back, or Remain Stationary); set up Reinforcements from Reserves.
3. **Shooting Phase** — Eligible units fire ranged weapons at visible targets within range.
4. **Charge Phase** — Units within 12" of enemies can attempt a charge (2D6" roll).
5. **Fight Phase** — Units in engagement range make melee attacks (Pile In → Attack → Consolidate).

## Models & Datasheets

Every model belongs to a unit, and every unit has a **datasheet** listing its characteristics:

| Stat | Meaning |
|------|---------|
| **M** (Move) | Inches the model can move per turn |
| **T** (Toughness) | Resilience against wounds |
| **Sv** (Save) | Armour save value (lower is better) |
| **W** (Wounds) | Hit points before destruction |
| **Ld** (Leadership) | Used for morale tests (lower is better) |
| **OC** (Objective Control) | Weight when contesting objectives |

Models are equipped with **weapons**, each with its own profile: **Range**, **Attacks (A)**, **BS/WS** (Ballistic/Weapon Skill), **Strength (S)**, **AP** (Armour Penetration), **Damage (D)**. A single model can carry multiple weapons (e.g. a rifle and a melee blade).

## Movement

Movement is freeform — a player physically picks up a model and moves it in any direction across the table, up to a distance equal to its **Move (M)** characteristic measured in inches. The path does not need to be a straight line; a model can weave, curve, or change direction at any point during its move. A straight line yields the maximum distance, but navigating around other models, terrain, or obstacles often requires a longer, indirect path. The total distance travelled along the path is what counts against the model's allowance.

Models cannot move through enemy models or off the table edge. Friendly models can be moved through (except large models moving through other large models), but a model cannot end its move on top of another model.

### Move Types

- **Normal Move**: Up to M", cannot end within engagement range (1") of enemies.
- **Advance**: M" + D6", but the unit cannot shoot or charge that turn.
- **Fall Back**: Up to M" to disengage from melee; cannot shoot or charge that turn. Desperate Escape tests required when moving over enemy models.

### Terrain & Flying

- **Terrain**: Models move freely over features ≤2" tall; taller features require climbing (vertical distance counts against the move allowance).
- **Flying**: Models with the FLY keyword can move over enemy models and measure distance through the air when starting/ending on terrain.

## Shooting (Ranged Attacks)

1. Select an eligible unit (didn't Advance or Fall Back).
2. For each model, select targets — at least one enemy model must be **visible** and **within range**.
3. Resolve attacks using the attack sequence (see below).

Units **locked in combat** (within engagement range) cannot shoot and cannot be shot at, except large models like monsters and vehicles (with a -1 to-hit penalty).

## Attack Sequence

Used for both ranged and melee attacks:

1. **Hit Roll**: D6 ≥ BS (ranged) or WS (melee). Unmodified 6 = Critical Hit (always succeeds). Unmodified 1 always fails.
2. **Wound Roll**: Compare attack S vs target T:

   | Comparison | Required Roll |
   |-----------|--------------|
   | S ≥ 2×T | 2+ |
   | S > T | 3+ |
   | S = T | 4+ |
   | S < T | 5+ |
   | S ≤ T/2 | 6+ |

   Unmodified 6 = Critical Wound (always succeeds). Unmodified 1 always fails.

3. **Allocate Attack**: Defender assigns the wound to a model (must allocate to already-damaged models first).
4. **Saving Throw**: D6 + AP modifier ≥ Sv to save. Invulnerable saves ignore AP. Unmodified 1 always fails.
5. **Inflict Damage**: Failed save → model loses wounds equal to weapon's Damage. Reduced to 0 wounds = destroyed. Excess damage from a single attack is lost.

**Mortal Wounds** bypass saving throws entirely; each mortal wound = 1 damage, and excess carries over to the next model.

## Charging & Melee

1. Declare charge targets (must be within 12", visibility not required).
2. Roll 2D6" — charge succeeds only if the unit can end within engagement range (1") of all targets without touching non-targets.
3. Charging units gain **Fights First** for the turn.
4. In the Fight phase: **Pile In** (up to 3" toward closest enemy) → **Make Melee Attacks** → **Consolidate** (up to 3" toward closest enemy or nearest objective).

## Unit Coherency

Models in a unit must stay within 2" of at least one other model (or two others if the unit has 7+ models). Units that break coherency at end of turn lose models until coherency is restored.

## Objective Control

- **Objective markers** are placed on the battlefield during setup. A model is within range of an objective if it is within 3".
- Each model has an **OC** (Objective Control) characteristic. A player's **Level of Control** over an objective is the sum of OC values of all their models within range.
- The player with the higher Level of Control controls the objective. Equal levels = contested (no one controls it).
- Controlling objectives is the primary way to earn Victory Points throughout the game.

## Morale

- Units **below half-strength** test in the Command phase: roll 2D6 ≥ Leadership to pass.
- **Shaken** units: OC becomes 0, cannot benefit from special tactics, must take Desperate Escape tests when Falling Back.

## Terrain Features

| Type | Key Effect |
|------|-----------|
| **Craters/Rubble** | Benefit of Cover for Infantry wholly on top |
| **Barricades** | Cover for Infantry within 3" if not fully visible through it |
| **Debris/Statuary** | Cover if not fully visible through it |
| **Hills/Sealed Buildings** | Cover if not fully visible; models can stand on top |
| **Woods** (Area Terrain) | Models wholly inside are never fully visible; Cover |
| **Ruins** (Area Terrain) | Block LOS entirely from outside; Infantry can move through walls/floors; Cover; Plunging Fire (+1 AP from 6"+ elevation) |

**Benefit of Cover**: +1 to armour saves against ranged attacks (not invulnerable saves). Does not apply to models with Sv 3+ or better against AP 0 weapons.

## Key Weapon Abilities

| Ability | Effect |
|---------|--------|
| **Assault** | Can shoot after Advancing |
| **Heavy** | +1 to hit if unit Remained Stationary |
| **Rapid Fire X** | +X attacks at half range |
| **Pistol** | Can shoot in engagement range (at engaged enemy only) |
| **Torrent** | Auto-hits |
| **Blast** | +1 attack per 5 models in target |
| **Melta X** | +X damage at half range |
| **Lethal Hits** | Critical Hits auto-wound |
| **Sustained Hits X** | Critical Hits score X additional hits |
| **Devastating Wounds** | Critical Wounds become mortal wounds (no saves) |
| **Indirect Fire** | Can target non-visible units (-1 to hit, 1-3 always fails, target gets Cover) |
| **Precision** | Can allocate wounds to Characters in Attached units |
| **Hazardous** | After shooting/fighting, roll D6 per weapon used; on 1, bearer's unit suffers 3 mortal wounds |

## Deployment Abilities

| Ability | Effect |
|---------|--------|
| **Deep Strike** | Start in Reserves; arrive in Reinforcements step, >9" from all enemies |
| **Infiltrators** | Deploy anywhere >9" from enemy deployment zone and all enemy models |
| **Scouts X"** | Free Normal move of up to X" before the first turn (>9" from enemies) |

## Stratagems (Core)

Stratagems are special tactical actions spent from Command Points (1 CP gained per Command phase per player). The same Stratagem cannot be used twice in one phase.

| Stratagem | CP | Effect |
|-----------|---:|--------|
| **Command Re-roll** | 1 | Re-roll any single dice roll |
| **Overwatch Fire** | 1 | Shoot at an enemy during their Movement/Charge phase (only hits on 6s); once per turn |
| **Counter-Offensive** | 2 | Fight next after an enemy unit fights |
| **Heroic Intervention** | 1 | Declare a charge in opponent's Charge phase against a unit that just charged within 6" |
| **Go to Ground** | 1 | Infantry gains 6+ invulnerable save and Benefit of Cover for the phase |
| **Grenade** | 1 | One model rolls 6D6 against enemy within 8"; each 4+ = 1 mortal wound |
| **Ram** | 1 | After a vehicle charges, roll D6s equal to its Toughness; each 5+ = 1 mortal wound (max 6) |
| **Duel** | 1 | Character's melee attacks gain Precision |
| **Stand Firm** | 1 | Auto-pass one morale test (once per battle) |
| **Smokescreen** | 1 | Unit gains Benefit of Cover and Stealth for the phase |

## Relevance to This Project

The phases, attack sequence, terrain rules, and objective mechanics form the core loop that the RL environment aims to model. The current implementation covers movement and objective capture (phases 2 and partial 1). The [roadmap](goals-and-roadmap.md) outlines the path toward shooting (phase 3), charging/melee (phases 4-5), morale, terrain, and eventually stratagems.
