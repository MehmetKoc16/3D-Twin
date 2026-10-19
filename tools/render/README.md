# Outfit render harness

Playwright scripts that drive the running web app and save screenshots of outfits, for visual QA of garment fitting.
Screenshots go to `tools/render/out/` (gitignored).

## Setup

1. Start the dev server: `npm run dev -w @dt/web -- --port 5201 --strictPort`
2. For twin renders, the twin GLB must exist at `user-data/twin/out/hybrid/twin.glb` (personal data, gitignored), or
   pass another path with `GLB=...`. Use `STD=1` to render the standard MakeHuman body instead.

## `outfits.cjs`

Run from this folder: `ONLY=3,4 UPPER=1 node outfits.cjs`

Each selected outfit is worn in A-pose and saved as `<PFX>k<n>-apose-front.png` and `-34.png` (3/4 view).

| Env var | Meaning |
| --- | --- |
| `ONLY` | Outfit numbers to render (default `1,2,3`), see `kombins` in the script |
| `PFX` | File name prefix |
| `OUT` | Output folder (default `./out`) |
| `GLB` | Twin GLB path (default `../../user-data/twin/out/hybrid/twin.glb`) |
| `STD=1` | Standard body instead of the twin |
| `BODY` | JSON of body-panel fields to set first, e.g. `{"gender":0,"hipCm":100}` (keys as in BodyPanel) |
| `UPPER=1` | Extra upper-body close-ups (`k<n>-upper-front/-34`) |
| `FEET=1` | Extra feet close-ups (`k<n>-feet-34/-front`) |
| `POSES` | Comma-separated pose button labels, e.g. `Rahat,Yan profil,Yürüyüş,Eller belde` |

Outfits: 1 tee + trousers-01 + low-poly sneakers, 2 polo + trousers-02 + dad sneaker, 3 tee + denim jacket +
trousers-01 + Converse-style, 4 polo + denim jacket + trousers-02 + low-poly sneakers, 5 tee + trousers-02 +
comfortable sneakers, 6 polo + trousers-01 + Converse-style, 7 denim jacket + trousers-01 + Converse-style (no top).

`native.json` holds each template's native measurements, used to fill the size chart (96 / 100 / 104 %).

## `qa.sh`

A full visual QA sweep: the twin in Rahat / Yan profil, the standard male body (with feet), and two female bodies.
