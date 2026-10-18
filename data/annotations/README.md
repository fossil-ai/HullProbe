# Hull/superstructure annotations

1,014 MARVEL-B vessels with an annotator-placed hull/superstructure boundary (paper App. K).

- `labels/<id>.png`: uint8 label map at the image's resolution, `0` background, `1` hull,
  `2` superstructure. The vessel region (`label > 0`) is the zero-shot silhouette shipped with the
  release assets; inside it, the annotator positioned a deck line, starting from a geometric estimate
  and shifting or rotating it (88% of labels were adjusted).
- `annotations.json`: one entry per vessel, in the order the segmenter's seeded split is drawn from:
  `id`, `image` (path under the MARVEL image root), `label`, `class`, `class_idx`,
  `marvel_split` (the classifier's train/test split), and `segformer_split` (train/val).

No imagery is included; images come from the MARVEL dataset.
