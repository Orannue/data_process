python multishot_pipeline/run_scene_pipeline_parallel.py \
  --scene-json movies_scenes.json \
  --moviebench-root moviedataset \
  --work-root moviedataset_extracted \
  --scene-workers 8 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --write-videos