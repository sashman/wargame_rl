# Held-out tables

The nine tables `configs/golden/25v25_maps_coherency.yaml` does **not** train on
— every table whose number is divisible by five. The other 36 are the training
pool, listed explicitly in that config's `map_pool.names`.

## Why this directory exists

`just measure-maps <policy|ckpt> <config> [n] [maps_dir]` scores **every map in
the directory it is given** and ignores the config's pool. Pointing it at
`configs/evaluation/maps/` therefore mixes all 36 training tables into what gets
reported as a held-out number. That mistake has been made here before.

    just measure-maps <ckpt> <config> 100 configs/evaluation/maps_heldout

## Reading the result

A map is the unit this generalises over, so **error bars go across the nine
maps, not across episodes** — nine maps at n=100 is nine samples of "an unseen
table", not nine hundred.

Only the `.yaml` layouts are copied; the `.png` previews stay in `maps/`.
