# Multishot Movie Pipeline

This folder contains an end-to-end pipeline for building scene-consistent
multishot movie samples.

## Why not use `detect_hybrid.py` directly?

`F:\dataset\movie\TransNetV2\detect_hybrid.py` can generate shot videos, but it is
not ideal as the main dataset builder:

- it uses PySceneDetect detectors, not the TransNetV2 model;
- it writes shot videos but no global shot manifest;
- its timestamp sorter is fragile for names like `00.00.04.663-00.00.09.179`;
- it has no character identity metadata;
- it cannot score or select mergeable multishot candidates.

The scripts here keep the useful hybrid shot-splitting idea and add structured
metadata for the downstream character and merge stages.

## Stages

1. `split_scene_shots.py`
   - input: MovieBench clips plus `movies_scenes.json`
   - output: `movie/scene/shot_*.mp4`
   - metadata: `shots/_manifests/shots.jsonl` and per-scene `scene_manifest.json`

2. `character_cluster.py`
   - samples frames from each shot
   - detects faces with MTCNN
   - embeds faces with FaceNet
   - clusters identities per scene with DBSCAN
   - metadata: `characters/_manifests/character_shots.jsonl`

3. `build_multishot_samples.py`
   - selects ordered shot groups inside the same scene
   - scores candidates by character consistency, background similarity, temporal
     proximity, duration, blur, and brightness
   - optionally writes concatenated sample videos
   - metadata: `samples/samples.jsonl`

## Example commands

Split shots:

```powershell
python H:\dataset\multishot_pipeline\split_scene_shots.py `
  --scene-json F:\dataset\movie\movies_scenes.json `
  --moviebench-root F:\dataset\movie\moviebench `
  --output-root H:\dataset\movie_multishot_output\shots
```

Cluster characters:

```powershell
python H:\dataset\multishot_pipeline\character_cluster.py `
  --shots-root H:\dataset\movie_multishot_output\shots `
  --shot-manifest H:\dataset\movie_multishot_output\shots\_manifests\shots.jsonl `
  --output-root H:\dataset\movie_multishot_output\characters
```

Build sample metadata and merged videos:

```powershell
python H:\dataset\multishot_pipeline\build_multishot_samples.py `
  --character-manifest H:\dataset\movie_multishot_output\characters\_manifests\character_shots.jsonl `
  --output-root H:\dataset\movie_multishot_output\samples `
  --write-videos
```

Run all stages:

```powershell
python H:\dataset\multishot_pipeline\run_pipeline.py --write-videos
```

For a quick smoke test on one movie:

```powershell
python H:\dataset\multishot_pipeline\run_pipeline.py `
  --only-movie 1037_The_Curious_Case_Of_Benjamin_Button `
  --write-videos
```

## Notes

- Character IDs are scene-level IDs by default. This is safer than forcing
  movie-level IDs early, because movies often contain costume, age, and lighting
  changes.
- `build_multishot_samples.py` preserves original scene order and limits gaps
  between selected shots. This avoids arbitrary combinations that share a face
  but do not look like a coherent multishot scene.
- The default output root is `H:\dataset\movie_multishot_output`, so it will not
  overwrite your current `H:\dataset\movie_shot`.

