#!/usr/bin/env bash
# Visual QA sweep: the twin in extra poses, the standard male body and two female bodies.
# Needs the dev server on http://localhost:5201 (npm run dev -w @dt/web -- --port 5201 --strictPort).
cd "$(dirname "$0")" || exit 1
for i in $(seq 1 60); do
  curl -s -o /dev/null -w "%{http_code}" http://localhost:5201/ 2>/dev/null | grep -q 200 && break
  timeout 1 cat </dev/zero >/dev/null 2>&1
done
echo "A twin poses"
ONLY=1,2,3,4,5,6 POSES="Rahat,Yan profil" PFX=qaA- timeout 900 node outfits.cjs 2>&1 | tail -2
echo "B standard male"
STD=1 ONLY=1,2,3,4,5,6 FEET=1 PFX=qaB- timeout 900 node outfits.cjs 2>&1 | tail -2
echo "C1 female"
STD=1 BODY='{"gender":0,"heightCm":165,"weightKg":60,"chestCm":90,"waistCm":72,"hipCm":100}' \
  ONLY=1,2,3,4 UPPER=1 FEET=1 PFX=qaC1- timeout 900 node outfits.cjs 2>&1 | tail -2
echo "C2 female, wide hips"
STD=1 BODY='{"gender":0,"heightCm":168,"weightKg":72,"chestCm":96,"waistCm":80,"hipCm":110}' \
  ONLY=3,4,5 UPPER=1 POSES="Eller belde" PFX=qaC2- timeout 900 node outfits.cjs 2>&1 | tail -2
echo done
